import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
from search_r1.llm_agent.econ_generation_manager import EconGenerationManager, EconGenerationConfig
from search_r1.llm_agent.econ_reward_adapter import build_replica_token_level_scores

WorkerType = Type[Worker]


def _econ_enabled(config) -> bool:
    return bool(OmegaConf.select(config, "econ.enabled", default=False))


def _econ_attach_log_prob_meta(dp: DataProto, rollout_cfg) -> None:
    """Econ 跳过 generate_sequences，需补全 actor.compute_log_prob 依赖的 meta（与 fsdp_workers 一致）。"""
    if dp.meta_info is None:
        dp.meta_info = {}
    mi = dp.meta_info
    mi.setdefault("micro_batch_size", int(rollout_cfg.log_prob_micro_batch_size))
    mi.setdefault("max_token_len", int(rollout_cfg.log_prob_max_token_len_per_gpu))
    mi.setdefault("use_dynamic_bsz", bool(rollout_cfg.log_prob_use_dynamic_bsz))
    mi.setdefault("temperature", float(rollout_cfg.temperature))


class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
#  GPU 资源分配表
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    # 先从数据箱子里取出模型的回答，量一下回答有多少个token
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    # 找到"哪些token是真正需要计算的"。按优先级：loss_mask > action_mask > info_mask > attention_mask
    if 'loss_mask' in data.batch:
        response_mask = data.batch['loss_mask'][:, -response_length:].float()
    elif 'action_mask' in data.batch:
        response_mask = data.batch['action_mask'][:, -response_length:].float()
    elif 'info_mask' in data.batch:
        response_mask = data.batch['info_mask'][:, -response_length:].float()
    else:
        response_mask = data.batch['attention_mask'][:, -response_length:].float()

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    # token-level reward 保持为原始 score；KL 由 actor KL loss 项单独处理，不在此重复扣进 reward。
    token_level_rewards = token_level_scores

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics

# 这次的决策比平均水平好多少还是差多少
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)

        if 'loss_mask' in data.batch:
            response_mask = data.batch['loss_mask'][:, -response_length:]
        elif 'action_mask' in data.batch:
            response_mask = data.batch['action_mask'][:, -response_length:]
        else:
            response_mask = data.batch['attention_mask'][:, -response_length:]

        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

# 把所有指标求平均
def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics

# 测量每条数据里：prompt有多少个token，response里有效的action token有多少个
def _compute_response_info(batch):
    response_width = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_width]
    raw_response_mask = batch.batch['attention_mask'][:, -response_width:]

    if 'loss_mask' in batch.batch:
        response_mask = batch.batch['loss_mask'][:, -response_width:].bool()
    elif 'action_mask' in batch.batch:
        response_mask = batch.batch['action_mask'][:, -response_width:].bool()
    else:
        response_mask = raw_response_mask.bool()

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
        raw_response_mask=raw_response_mask,
    )


# 计算一堆监控指标，供wandb显示用
def compute_data_metrics(batch, use_critic=True):
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    response_width = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_width].bool()
    raw_response_mask = batch.batch['attention_mask'][:, -response_width:].bool()

    if 'loss_mask' in batch.batch:
        metric_mask = batch.batch['loss_mask'][:, -response_width:].bool()
    elif 'action_mask' in batch.batch:
        metric_mask = batch.batch['action_mask'][:, -response_width:].bool()
    else:
        metric_mask = raw_response_mask

    max_prompt_length = prompt_mask.size(-1)
    max_response_length = metric_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, metric_mask)
    valid_returns = torch.masked_select(returns, metric_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, metric_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        'critic/score/mean': torch.mean(sequence_score).detach().item(),
        'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),
        'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max': torch.max(sequence_reward).detach().item(),
        'critic/rewards/min': torch.min(sequence_reward).detach().item(),
        'critic/advantages/mean': torch.mean(valid_adv).detach().item(),
        'critic/advantages/max': torch.max(valid_adv).detach().item(),
        'critic/advantages/min': torch.min(valid_adv).detach().item(),
        'critic/returns/mean': torch.mean(valid_returns).detach().item(),
        'critic/returns/max': torch.max(valid_returns).detach().item(),
        'critic/returns/min': torch.min(valid_returns).detach().item(),
        **({
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        'response_length/raw_response_coverage': torch.mean(
            response_length / response_info['raw_response_mask'].sum(-1).clamp(min=1).float()
        ).detach().item(),
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio': torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())

    return metrics

# 计算时间效率指标
def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()

    # timing 按模型真实处理过的 token 数算，不按 train-mask 后的 action token 数
    num_raw_response_tokens = torch.sum(response_info['raw_response_mask'].sum(-1)).item()
    num_train_response_tokens = torch.sum(response_info['response_length']).item()

    num_overall_tokens = num_prompt_tokens + num_raw_response_tokens

    num_tokens_of_section = {
        'gen': num_raw_response_tokens,
        'adv': max(num_train_response_tokens, 1),
        **{
            name: max(num_overall_tokens, 1)
            for name in ['ref', 'values', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last

# 主训练器类
class RayPPOTrainer(object):
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
    # 
    def _create_dataloader(self):
        from torch.utils.data import DataLoader, Dataset

        if _econ_enabled(self.config):
            # 先把 econ 项目路径加进 sys.path，否则 build_state_bank 导入失败
            import sys as _sys

            _econ_root = str(
                OmegaConf.select(
                    self.config,
                    "econ.project_root",
                    default="/workspace/QWEN2.5_42_7b_main",
                )
            )
            if _econ_root not in _sys.path:
                _sys.path.insert(0, _econ_root)

            class EconStateBankDataset(Dataset):
                def __init__(self, state_bank_path, default_year_index):
                    from build_state_bank import load_state_bank
                    self.records = load_state_bank(state_bank_path)
                    self.default_year_index = int(default_year_index)

                def __len__(self):
                    return len(self.records)
                # 这是第几年的初始状态、属于哪个状态组、编号是多少
                def __getitem__(self, idx):
                    rec = self.records[idx]
                    return {
                        "year_index": int(rec.get("year_index", self.default_year_index)),
                        "state_group_id": str(rec.get("state_group_id", f"group_{idx}")),
                        "episode_id": int(idx),
                    }

            class EconStubDataset(Dataset):
                """无 state bank 时提供与 EconStateBankDataset 同结构的 meta batch（smoke / 随机起点）。"""

                def __init__(self, n_items: int, default_year_index: int):
                    self.n_items = max(1, int(n_items))
                    self.default_year_index = int(default_year_index)

                def __len__(self):
                    return self.n_items

                def __getitem__(self, idx):
                    return {
                        "year_index": self.default_year_index,
                        "state_group_id": f"stub_{idx}",
                        "episode_id": int(idx),
                    }

            # 把多条数据拼成一个批次
            def econ_collate(xs):
                return {
                    "year_index": np.array([x["year_index"] for x in xs], dtype=object),
                    "state_group_id": np.array([x["state_group_id"] for x in xs], dtype=object),
                    "episode_id": np.array([x["episode_id"] for x in xs], dtype=object),
                }

            _sbp = OmegaConf.select(self.config, "econ.state_bank_path", default=None)
            _sbp_s = str(_sbp).strip() if _sbp is not None else ""
            _def_y = int(OmegaConf.select(self.config, "econ.default_year_index", default=0))
            if _sbp_s:
                if not os.path.isfile(_sbp_s):
                    raise FileNotFoundError(
                        f"econ.state_bank_path is set but not a readable file: {_sbp_s!r}"
                    )
                self.train_dataset = EconStateBankDataset(_sbp_s, _def_y)
                self.val_dataset = EconStateBankDataset(_sbp_s, _def_y)
            else:
                _stub_n = int(OmegaConf.select(self.config, "econ.stub_dataset_size", default=1))
                print(
                    f"[econ] No state_bank_path: using EconStubDataset (size={_stub_n}, "
                    f"default_year_index={_def_y}) for train/val meta batches."
                )
                self.train_dataset = EconStubDataset(_stub_n, _def_y)
                self.val_dataset = EconStubDataset(max(1, _stub_n), _def_y)
            self.train_dataloader = DataLoader(
                self.train_dataset, batch_size=self.config.data.train_batch_size,
                shuffle=True, drop_last=True, collate_fn=econ_collate,
            )
            self.val_dataloader = DataLoader(
                self.val_dataset, batch_size=self.config.data.val_batch_size,
                shuffle=False, drop_last=True, collate_fn=econ_collate,
            )
            assert len(self.train_dataloader) >= 1, "econ train DataLoader is empty (check batch_size vs dataset size)"
            assert len(self.val_dataloader) >= 1, "econ val DataLoader is empty (check batch_size vs dataset size)"
            # 计算总训练步数 = state bank里有多少个初始状态 × 训练epoch数
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
            if self.config.trainer.total_training_steps is not None:
                total_training_steps = self.config.trainer.total_training_steps
            self.total_training_steps = total_training_steps
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            return

        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    # 验证模式
    def _validate(self):
        import torch
        reward_tensor_lst = []
        data_source_lst = []

        econ_enabled = _econ_enabled(self.config)

        if econ_enabled:
            econ_cfg = EconGenerationConfig(
                num_months=int(OmegaConf.select(self.config, "econ.num_months", default=12)),
                num_agents=int(OmegaConf.select(self.config, "econ.num_agents", default=100)),
                max_start_length=OmegaConf.select(self.config, "data.max_start_length", default=256),
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                econ_project_root=OmegaConf.select(
                    self.config, "econ.project_root", default="/workspace/QWEN2.5_42_7b_main"
                ),
                qwen_config_path=OmegaConf.select(self.config, "econ.qwen_config_path", default=None),
                state_bank_path=OmegaConf.select(self.config, "econ.state_bank_path", default=None),
                replica_group_size=int(OmegaConf.select(self.config, "econ.replica_group_size", default=4)),
                default_year_index=int(OmegaConf.select(self.config, "econ.default_year_index", default=2)),
                state_bank_year_index=OmegaConf.select(self.config, "econ.state_bank_year_index", default=None),
                rollout_agent_micro_batch=int(
                    OmegaConf.select(self.config, "econ.rollout_agent_micro_batch", default=16)
                ),
                rollout_prompt_length=int(
                    OmegaConf.select(self.config, "econ.rollout_prompt_length", default=512)
                ),
            )

            generation_manager = EconGenerationManager(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=econ_cfg,
                is_validation=True,
            )

            for batch_dict in self.val_dataloader:
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)

                final_gen_batch_output = generation_manager.run_econ_year_loop(
                    meta_batch=test_batch,
                    temperature=0.0,
                )

                test_batch = final_gen_batch_output

                # 验证时把temperature设为0（贪心解码，不随机），跑一遍经济仿真，计算reward，记录到wandb
                # 把所有batch的reward汇总，算平均，返回给logger记录
                reward_tensor = build_replica_token_level_scores(test_batch)
                reward_tensor_lst.append(reward_tensor)

                if "data_source" in test_batch.non_tensor_batch:
                    data_source_lst.append(test_batch.non_tensor_batch["data_source"])
                else:
                    data_source_lst.append(
                        np.array(["econ"] * reward_tensor.shape[0], dtype=object)
                    )

        else:
            gen_config = GenerationConfig(
                max_turns=self.config.max_turns,
                max_start_length=self.config.data.max_start_length,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                no_think_rl=self.config.algorithm.no_think_rl,
                search_url=self.config.retriever.url,
                topk=self.config.retriever.topk,
            )

            generation_manager = LLMGenerationManager(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=gen_config,
                is_validation=True,
            )

            if not OmegaConf.select(self.config, 'do_search', default=False):
                for test_data in self.val_dataloader:
                    test_batch = DataProto.from_single_dict(test_data)

                    if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                        return {}

                    test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                    test_gen_batch.meta_info = {
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'recompute_log_prob': False,
                        'do_sample': False,
                        'validate': True,
                    }

                    test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                        test_gen_batch, self.actor_rollout_wg.world_size
                    )
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                    test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                    test_batch = test_batch.union(test_output_gen_batch)
                    reward_tensor = self.val_reward_fn(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(
                        test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])
                    )
            else:
                for batch_dict in self.val_dataloader:
                    test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                    test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                    test_gen_batch.meta_info = {
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'recompute_log_prob': False,
                        'do_sample': False,
                        'validate': True,
                    }

                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    final_gen_batch_output = generation_manager.run_llm_loop(
                        gen_batch=test_gen_batch,
                        initial_input_ids=first_input_ids,
                    )

                    test_batch = test_batch.union(final_gen_batch_output)
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()

                    reward_tensor = self.val_reward_fn(test_batch)
                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(
                        test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])
                    )

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # 核心训练循环
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # 训练开始前先跑一次验证，记录初始baseline，方便后续对比模型是否真的在进步
        if _econ_enabled(self.config):
            ec = OmegaConf.select(self.config, 'econ', default={})
            econ_gen_config = EconGenerationConfig(
                num_months=int(OmegaConf.select(ec, 'num_months', default=12)),
                num_agents=int(OmegaConf.select(ec, 'num_agents', default=100)),
                max_start_length=OmegaConf.select(self.config, "data.max_start_length", default=256),
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                econ_project_root=str(OmegaConf.select(ec, 'project_root', default='/workspace/QWEN2.5_42_7b_main')),
                qwen_config_path=OmegaConf.select(ec, 'qwen_config_path', default=None),
                state_bank_path=OmegaConf.select(ec, 'state_bank_path', default=None),
                replica_group_size=int(OmegaConf.select(ec, 'replica_group_size', default=4)),
                default_year_index=int(OmegaConf.select(ec, 'default_year_index', default=2)),
                state_bank_year_index=OmegaConf.select(ec, 'state_bank_year_index', default=None),
                rollout_agent_micro_batch=int(
                    OmegaConf.select(self.config, "econ.rollout_agent_micro_batch", default=16)
                ),
                rollout_prompt_length=int(
                    OmegaConf.select(self.config, "econ.rollout_prompt_length", default=512)
                ),
            )
            econ_generation_manager = EconGenerationManager(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=econ_gen_config,
            )
            gen_config = None
            generation_manager = None
        else:
            gen_config = GenerationConfig(
                max_turns=self.config.max_turns,
                max_start_length=self.config.data.max_start_length,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                no_think_rl=self.config.algorithm.no_think_rl,
                search_url=self.config.retriever.url,
                topk=self.config.retriever.topk,
            )

            generation_manager = LLMGenerationManager(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=gen_config,
            )
            econ_generation_manager = None
            econ_gen_config = None

        # 外层是epoch，内层是每个batch（每个经济初始状态）
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                econ_enabled = _econ_enabled(self.config)

                # Search-R1 原逻辑里 n_agent 是多 agent / 多 rollout 扩展；
                # Econ 模式下不要把 dataset batch 先重复 100 次，否则会白白放大显存。
                if not econ_enabled:
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n_agent,
                        interleave=True,
                    )

                gen_batch = None
                if not econ_enabled:
                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):
                    # 如果是Econ模式，就跑经济仿真，计算reward
                    if econ_enabled:
                        with _timer('gen', timing_raw):
                            econ_generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = econ_generation_manager.run_econ_year_loop(
                                meta_batch=batch,
                                temperature=float(self.config.actor_rollout_ref.rollout.temperature),
                            )

                        _FLOAT_KEYS = ('annual_reward',)
                        for key in final_gen_batch_output.batch.keys():
                            if key in _FLOAT_KEYS:
                                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].float()
                            else:
                                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        _econ_attach_log_prob_meta(
                            final_gen_batch_output,
                            self.config.actor_rollout_ref.rollout,
                        )
                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        batch = final_gen_batch_output

                    elif not OmegaConf.select(self.config, 'do_search', default=False):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()

                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    _skip_long = {'old_log_probs', 'annual_reward'}
                    for key in batch.batch.keys():
                        if key in _skip_long:
                            continue
                        batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # 经济模式下，把annual_reward（一个年度标量）拆分并分配到每个token上，得到token_level_scores
                        if econ_enabled:
                            # Log reward breakdown from econ_group_payload
                            _payload = batch.meta_info.get('econ_group_payload')
                            if _payload:
                                _pl = _payload if isinstance(_payload, dict) else (_payload[0] if isinstance(_payload, list) and _payload else {})
                                _bd = _pl.get('mean_reward_breakdown', {})
                                for _k, _v in _bd.items():
                                    metrics[f'econ_reward/{_k}'] = float(_v)
                                if 'mean_R_macro' in _pl:
                                    metrics['econ_reward/R_macro'] = float(_pl['mean_R_macro'])
                                if 'mean_R_micro' in _pl:
                                    metrics['econ_reward/R_micro'] = float(_pl['mean_R_micro'])
                                # Also log raw macro metrics from first replica
                                _reps = _pl.get('replicas', [])
                                if _reps:
                                    _m = _reps[0].get('env_stats', {})
                                    _ms = _m.get('monthly_series', [])
                                    if _ms:
                                        metrics['econ_env/format_fail_count'] = float(_m.get('format_fail_count', 0))
                                        metrics['econ_env/invalid_action_count'] = float(_m.get('invalid_action_count', 0))
                                        metrics['econ_env/out_of_range_count'] = float(_m.get('out_of_range_count', 0))
                                        metrics['econ_env/negative_wealth_count'] = float(_m.get('negative_wealth_count', 0))
                                        metrics['econ_env/fallback_action_count'] = float(_m.get('fallback_action_count', 0))

                            reward_tensor = build_replica_token_level_scores(batch)
                            batch.batch['token_level_scores'] = reward_tensor
                            if not self.config.actor_rollout_ref.actor.use_kl_loss:
                                batch, kl_metrics = apply_kl_penalty(batch,
                                                                     kl_ctrl=self.kl_ctrl,
                                                                     kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                            # 在4个replica之间做GRPO归一化，得到每个token的优势值
                            batch = compute_advantage(
                                batch,
                                adv_estimator='grpo',
                                gamma=1.0,
                                lam=1.0,
                                num_repeat=1,
                            )
                        else:
                            # compute scores. Support both model and function-based.
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            reward_tensor = self.reward_fn(batch)
                            batch.batch['token_level_scores'] = reward_tensor

                            if not self.config.actor_rollout_ref.actor.use_kl_loss:
                                batch, kl_metrics = apply_kl_penalty(batch,
                                                                     kl_ctrl=self.kl_ctrl,
                                                                     kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                            batch = compute_advantage(batch,
                                                      adv_estimator=self.config.algorithm.adv_estimator,
                                                      gamma=self.config.algorithm.gamma,
                                                      lam=self.config.algorithm.lam,
                                                      num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # 如果开了state_masking，先建loss_mask（只让action token的梯度通过，observation部分不参与反向传播），然后更新actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if (OmegaConf.select(self.config, 'do_search', default=False) or econ_enabled) and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    # 在response部分里，找到真正是"模型action"的那些token（即JSON决策部分），标记为1；observation部分标记为0。这样反向传播时，梯度只会流过action token，observation token不会被错误地"优化
    def _create_loss_mask(self, batch, metrics):
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]

        if 'action_mask' in batch.batch:
            loss_mask = batch.batch['action_mask'][:, -response_length:]
        elif 'info_mask' in batch.batch:
            loss_mask = batch.batch['info_mask'][:, -response_length:]
        else:
            loss_mask = response_mask

        batch.batch['loss_mask'] = loss_mask
        metrics.update({
            'action_tokens/total': loss_mask.sum().item(),
            'action_tokens/coverage': (loss_mask.sum() / response_mask.sum().clamp(min=1)).item(),
        })
        return batch, metrics
