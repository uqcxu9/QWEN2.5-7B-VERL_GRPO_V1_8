from __future__ import annotations

import copy
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml
from dateutil.relativedelta import relativedelta
from verl import DataProto


class EconVLLMGenerateWrapper:
    """Wraps a vLLM async server handle to provide a synchronous generate_sequences()
    interface compatible with EconGenerationManager._generate_with_gpu_padding().

    Instead of going through fsdp_workers (which has async event loop conflicts in verl 0.7.1),
    this wrapper calls the vLLM server's per-sequence generate() directly via Ray remote.
    Weight sync is handled externally by checkpoint_manager before the econ year loop starts.
    """

    def __init__(self, server_handles: list, tokenizer, rollout_config):
        self.server_handles = server_handles  # list of Ray actor handles for vLLM servers
        self.tokenizer = tokenizer
        self.response_length = getattr(rollout_config, 'response_length', 96)
        self.prompt_length = getattr(rollout_config, 'prompt_length', 768)

    def generate_sequences(self, active_batch: DataProto) -> DataProto:
        import ray

        input_ids = active_batch.batch["input_ids"]
        attention_mask = active_batch.batch["attention_mask"]
        batch_size = input_ids.shape[0]
        pad_id = active_batch.meta_info.get("pad_token_id",
                    self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)

        temperature = active_batch.meta_info.get("temperature", 1.0)
        do_sample = active_batch.meta_info.get("do_sample", True)

        # DEBUG: log actual temperature (first call only)
        if not hasattr(self, "_temp_logged"):
            print(f"[econ-debug-temp] vLLM sampling: temperature={temperature}, do_sample={do_sample}")
            self._temp_logged = True

        sampling_params = {
            "temperature": float(temperature) if do_sample else 0.0,
            "top_p": float(active_batch.meta_info.get("top_p", 1.0)),
            "top_k": int(active_batch.meta_info.get("top_k", -1)),
            "max_tokens": self.response_length,
            "logprobs": True,
        }

        server = self.server_handles[0]  # single GPU = single server

        # Submit all sequences as Ray remote calls
        futures = []
        for i in range(batch_size):
            mask = attention_mask[i]
            ids = input_ids[i][mask.bool()].tolist()
            futures.append(
                server.generate.remote(
                    prompt_ids=ids,
                    sampling_params=dict(sampling_params),
                    request_id=f"econ_{i}_{id(active_batch)}",
                )
            )

        results = ray.get(futures)

        # Assemble into DataProto
        response_list = []
        for r in results:
            tokens = list(r.token_ids) if hasattr(r, 'token_ids') else []
            tokens = tokens[:self.response_length]
            pad_len = self.response_length - len(tokens)
            tokens = tokens + [pad_id] * pad_len
            response_list.append(tokens)

        responses = torch.tensor(response_list, dtype=torch.long)
        resp_attn = (responses != pad_id).long()

        output = DataProto.from_dict(tensors={
            "responses": responses,
            "input_ids": torch.cat([input_ids, responses], dim=1),
            "attention_mask": torch.cat([attention_mask, resp_attn], dim=1),
        })
        output.meta_info = dict(active_batch.meta_info)
        return output


def _ensure_econ_on_path(econ_project_root: str) -> str:
    root = os.path.abspath(econ_project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root

# 从同一个初始状态派生多个独立 replica，每个 replica 必须有自己的环境副本，互不干扰。
def _clone_env_and_obs(env: Any, obs: Any) -> Tuple[Any, Any]:
    try:
        return copy.deepcopy(env), copy.deepcopy(obs)
    except Exception:
        blob = pickle.dumps((env, obs), protocol=pickle.HIGHEST_PROTOCOL)
        return pickle.loads(blob)

# 构建经济仿真环境的配置字典
def _build_qwen_rollout_env_config(
    num_agents: int,
    episode_length: int,
    config_path: Optional[str],
    econ_root: str,
) -> Dict[str, Any]:
    cfg_path = config_path or os.path.join(econ_root, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        run_configuration = yaml.safe_load(f)
    env_config = dict(run_configuration["env"])
    env_config["n_agents"] = int(num_agents)
    env_config["episode_length"] = int(episode_length)
    env_config["flatten_masks"] = False
    env_config["flatten_observations"] = False
    env_config["components"][0]["SimpleLabor"]["scale_obs"] = False
    env_config["components"][1]["PeriodicBracketTax"]["scale_obs"] = False
    env_config["components"][3]["SimpleSaving"]["scale_obs"] = False
    env_config["components"][2]["SimpleConsumption"]["max_price_inflation"] = 0.1
    env_config["components"][2]["SimpleConsumption"]["max_wage_inflation"] = 0.05
    return env_config

# 检查state bank记录是否有效：记录不能为空，而且必须包含四个路径字段（环境快照、observation、memory状态、历史状态），每个字段不能是空字符串。全部通过才返回True
def _record_restore_ok(rec: Optional[Mapping[str, Any]]) -> bool:
    if not rec:
        return False
    for k in ("snapshot_path", "obs_path", "memory_state_path", "history_state_path"):
        p = rec.get(k)
        if p is None or (isinstance(p, str) and not p.strip()):
            return False
    return True


def _restore_files_exist(bank_dir: str, rec: Mapping[str, Any]) -> bool:
    if not _record_restore_ok(rec):
        return False
    for k in ("snapshot_path", "obs_path", "memory_state_path", "history_state_path"):
        rel = rec[k]
        path = rel if os.path.isabs(str(rel)) else os.path.join(bank_dir, str(rel))
        if not os.path.isfile(path):
            return False
    return True


@dataclass
class EconGenerationConfig:
    num_months: int = 12
    num_agents: int = 100
    max_start_length: int = 1024
    max_prompt_length: int = 1536
    max_response_length: int = 96
    max_obs_length: int = 256
    num_gpus: int = 1
    econ_project_root: str = "/workspace/QWEN2.5_42_7b_main"
    qwen_config_path: Optional[str] = None
    state_bank_path: Optional[str] = None
    replica_group_size: int = 4
    default_year_index: int = 2
    state_bank_year_index: Optional[int] = None
    # Optional: supply one JSON row + its directory without reading state_bank.json again
    inline_state_bank_record: Optional[Dict[str, Any]] = None
    inline_bank_base_dir: str = ""
    rollout_agent_micro_batch: int = 16
    rollout_prompt_length: int = 512

# 经济仿真管理器
class EconGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: EconGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.econ_root = _ensure_econ_on_path(config.econ_project_root)
        self.timing_raw: Optional[Dict[str, Any]] = None

    # 生成器默认配置
    # 是否验证模式（True: 不随机，贪心解码；False: 随机采样）
    def _default_gen_meta(self) -> Dict[str, Any]:
        return {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": not self.is_validation,
            "validate": self.is_validation,
            "temperature": getattr(self, "_rollout_temperature", 0.6),
            "top_p": 0.95,
            "top_k": 20,
        }

    # 和Search-R1模式一样，用GPU并行加速推理
    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """Same logic as Search-R1 ``LLMGenerationManager._generate_with_gpu_padding``."""
        num_gpus = int(self.config.num_gpus)
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch["input_ids"].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        padding_size = num_gpus - remainder
        padded_batch = {}
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
        padded_active_batch.meta_info = dict(active_batch.meta_info)

        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        padded_output.batch = trimmed_batch
        return padded_output


    def _build_gen_dataproto_from_dialogs(self, dialog_lists: List[List[Dict[str, str]]]) -> DataProto:
        """Build DataProto from multi-turn dialog lists (ACL24 style).
        Each element is a list of {"role": "user"/"assistant", "content": "..."} messages.
        """
        pad_id = self.tokenizer.pad_token_id
        hard_cap = int(self.config.rollout_prompt_length)

        rows = []
        for messages in dialog_lists:
            prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            if len(ids) > hard_cap:
                ids = ids[-hard_cap:]
            rows.append(torch.tensor(ids, dtype=torch.long))

        return self._pad_and_stack_rows(rows, hard_cap)

    def _build_gen_dataproto_from_texts(self, prompt_texts: List[str]) -> DataProto:
        pad_id = self.tokenizer.pad_token_id
        hard_cap = int(self.config.rollout_prompt_length)

        #对每条prompt文本，套上Qwen的chat template，变成标准的输入格式
        rows = []
        for pt in prompt_texts:
            prompt_str = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": pt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            # tokenize后如果超过长度上限，从末尾截取（保留最近的上下文），转成tensor存入rows
            ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            if len(ids) > hard_cap:
                ids = ids[-hard_cap:]
            rows.append(torch.tensor(ids, dtype=torch.long))

        return self._pad_and_stack_rows(rows, hard_cap)

    def _pad_and_stack_rows(self, rows: List[torch.Tensor], hard_cap: int) -> DataProto:
        """Left-pad token rows and stack into DataProto with attention_mask and position_ids."""
        pad_id = self.tokenizer.pad_token_id
        max_len = max(int(x.shape[0]) for x in rows)
        max_len = min(max_len, hard_cap)
        input_rows, attn_rows, pos_rows = [], [], []
        for ids in rows:
            L = int(ids.shape[0])
            if L > max_len:
                ids = ids[-max_len:]
                L = max_len
            pad_len = max_len - L
            input_rows.append(torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), ids]))
            attn_rows.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(L, dtype=torch.long)]))
            pos_rows.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.arange(L, dtype=torch.long)]))
        tensors = {
            "input_ids": torch.stack(input_rows, dim=0),
            "attention_mask": torch.stack(attn_rows, dim=0),
            "position_ids": torch.stack(pos_rows, dim=0),
        }
        return DataProto.from_dict(tensors=tensors, meta_info=self._default_gen_meta())
    
    # 加载状态银行记录
    def _load_bank_record(self, year_index: int) -> Tuple[Optional[Dict[str, Any]], str]:
        path = self.config.state_bank_path
        if not path or not os.path.isfile(path):
            return None, ""
        from build_state_bank import get_state_bank_record, load_state_bank, state_bank_base_dir

        # 获取状态银行记录的年份
        bank_y = (
            int(self.config.state_bank_year_index)
            if self.config.state_bank_year_index is not None
            else int(year_index)
        )
        # 加载状态银行记录
        recs = load_state_bank(path)
        # 从状态银行记录中获取指定年份的记录
        rec = get_state_bank_record(recs, bank_y)
        # 返回记录和状态银行记录的目录
        return rec, state_bank_base_dir(path)

    # 构建这次rollout用的模板环境。返回环境对象、初始observation、memory状态、初始价格、以及state bank记录
    def _materialize_template(
        self, year_index: int
    ) -> Tuple[Any, Any, List[Dict[str, Any]], float, Optional[Dict[str, Any]]]:
        # 如果指定了inline_state_bank_record，直接用它；否则从状态银行记录中获取指定年份的记录
        if self.config.inline_state_bank_record is not None:
            rec = dict(self.config.inline_state_bank_record)
            bank_dir = str(self.config.inline_bank_base_dir or "")
        else:
            rec, bank_dir = self._load_bank_record(year_index)

        if self.config.state_bank_path:
            if not rec:
                raise RuntimeError(
                    f"econ state bank record not found for year_index={year_index} "
                    f"(state_bank_path={self.config.state_bank_path!r})"
                )
            if not bank_dir or not _restore_files_exist(bank_dir, rec):
                raise RuntimeError(
                    f"econ state bank assets missing or invalid for year_index={year_index}: {rec}"
                )

            from simulate import restore_env_snapshot

            env_s, obs_s, mem_s, ip = restore_env_snapshot(
                bank_dir, rec, restore_rng=False
            )
            env_t, obs_t = _clone_env_and_obs(env_s, obs_s)
            return env_t, obs_t, copy.deepcopy(mem_s), float(ip), rec

        import ai_economist.foundation as foundation
        from rollout_collector import default_memory_state

        ecfg = _build_qwen_rollout_env_config(
            self.config.num_agents,
            self.config.num_months,
            self.config.qwen_config_path,
            self.econ_root,
        )
        env = foundation.make_env_instance(**ecfg)
        obs = env.reset()
        mem = [default_memory_state() for _ in range(self.config.num_agents)]
        ip = float(env.world.price[0])
        env_t, obs_t = _clone_env_and_obs(env, obs)
        return env_t, obs_t, mem, ip, rec

    # 运行单个replica的一年仿真
    def run_single_replica_year(
        self,
        env: Any,
        obs: Any,
        # memory状态列表，每个元素是一个字典，包含agent的memory状态
        memory_states: List[Dict[str, Any]],
        # 初始价格
        initial_price: float,   # 初始价格
        *,
        episode_id: int,
        replica_id: int,
        # CRN (Common Random Numbers) for GRPO variance control:
        # bernoulli_crn: pre-drawn U[0,1] table, shape (num_months, num_agents),
        #   shared across replicas so Bernoulli work decisions differ only
        #   due to policy-output work probabilities, not random noise.
        # env_rng_seed: seed for env-internal RNG (price/wage/permutation),
        #   shared across replicas so env dynamics are identical given same actions.
        bernoulli_crn: Optional[np.ndarray] = None,
        env_rng_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        from simulate import build_monthly_prompt_from_state
        from simulate_utils import world_start_time
        # 构建月度prompt
        from rollout_collector import (
            build_quarterly_reflection_prompt,
            build_sample_meta,
            collect_env_stats_step,
            convert_raw_action_to_env_action,
            parse_action_json,
            update_memory_state,
        )

        episode_samples: List[Dict[str, Any]] = []
        global_counters = {
            "format_fail_count": 0,
            "invalid_action_count": 0,
            "fallback_action_count": 0,
            "out_of_range_count": 0,
            "negative_wealth_count": 0,
        }
        monthly_series: List[Dict[str, Any]] = []
        n_agents = int(self.config.num_agents)

        # ACL24 style: maintain per-agent dialog history for multi-turn generation
        from collections import deque
        dialog_queue = [deque(maxlen=3) for _ in range(n_agents)]
        dialog4ref_queue = [deque(maxlen=7) for _ in range(n_agents)]

        # System prompt (ACL24 baseline style) — fixed context for all agents
        _ECON_SYSTEM_PROMPT = {
            "role": "system",
            "content": (
                "You are an economic agent making monthly decisions. "
                "You must respond with ONLY a JSON object with exactly two keys: "
                '"work" and "consumption", both floats between 0 and 1 (in steps of 0.02).\n'
                'Example: {"work": 0.80, "consumption": 0.30}\n'
                "Do NOT output any explanation, only the JSON object."
            ),
        }

        # If env_rng_seed is provided, reset env RNG so all replicas in the
        # same GRPO group see identical env stochasticity (price/wage shocks,
        # agent iteration order). This ensures within-group reward differences
        # are driven by policy outputs, not env noise.
        if env_rng_seed is not None:
            np.random.seed(env_rng_seed)

        # 进入12个月的循环，t是当前时间步
        for epi in range(int(self.config.num_months)):
            t = int(env.world.timestep)
            # 为每个agent构建这个月的prompt文本（包含当前经济状态、agent自身状态等）
            prompt_texts: List[str] = []
            for idx in range(n_agents):
                pt = build_monthly_prompt_from_state(
                    env,
                    obs,
                    idx,
                    memory_states[idx],
                    world_start_time,
                    relativedelta,
                )
                prompt_texts.append(pt)
                # Append to dialog queues (ACL24 style)
                dialog_queue[idx].append({"role": "user", "content": pt})
                dialog4ref_queue[idx].append({"role": "user", "content": pt})

            # Build multi-turn dialog messages for generation (ACL24 style)
            # Quarterly months: splice in reflection from dialog4ref_queue
            is_quarterly = (t % 3 == 0 and t > 0)
            agent_dialogs: List[List[Dict[str, str]]] = []
            for idx in range(n_agents):
                if is_quarterly:
                    dq = list(dialog_queue[idx])
                    dr = list(dialog4ref_queue[idx])
                    # ACL24: first 2 msgs + last 2 reflection msgs + last user msg
                    msgs = [_ECON_SYSTEM_PROMPT] + dq[:2] + dr[-3:-1] + dq[-1:]
                else:
                    msgs = [_ECON_SYSTEM_PROMPT] + list(dialog_queue[idx])
                agent_dialogs.append(msgs)

            # 把100个agent分成每批16个，送给vLLM生成决策文本。batch_decode把token id转回文字
            agent_mb = int(getattr(self.config, "rollout_agent_micro_batch", 16))
            texts = []
            for start in range(0, n_agents, agent_mb):
                end = min(start + agent_mb, n_agents)
                sub_dp = self._build_gen_dataproto_from_dialogs(agent_dialogs[start:end])
                sub_out = self._generate_with_gpu_padding(sub_dp)
                sub_texts = self.tokenizer.batch_decode(
                    sub_out.batch["responses"], skip_special_tokens=True
                )
                texts.extend(sub_texts)

            # Append assistant responses to dialog queues (ACL24 style)
            for idx in range(n_agents):
                dialog_queue[idx].append({"role": "assistant", "content": texts[idx]})
                dialog4ref_queue[idx].append({"role": "assistant", "content": texts[idx]})

            # DEBUG: print first 3 agents' responses each month
            _n_show = min(3, len(texts))
            print(f"[econ-debug] replica={replica_id} month={epi} "
                  f"n_responses={len(texts)} showing first {_n_show}:")
            for _di in range(_n_show):
                _t = texts[_di]
                print(f"  agent[{_di}] ({len(_t)} chars): {_t[:200]!r}"
                      + ("..." if len(_t) > 200 else ""))
            # Count format stats for this month
            _fmt_ok = sum(1 for t in texts if '{' in t and '}' in t)
            print(f"  >>> {_fmt_ok}/{len(texts)} responses contain braces")

            # Build the actual prompt strings sent to vLLM (for training replay)
            # Must match exactly what the model saw during rollout, including truncation
            hard_cap = int(self.config.rollout_prompt_length)
            rollout_prompt_strs: List[str] = []
            for idx in range(n_agents):
                full_str = self.tokenizer.apply_chat_template(
                    agent_dialogs[idx],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                ids = self.tokenizer.encode(full_str, add_special_tokens=False)
                if len(ids) > hard_cap:
                    ids = ids[-hard_cap:]
                # Decode back to get the truncated string (matches what vLLM actually saw)
                rollout_prompt_strs.append(
                    self.tokenizer.decode(ids, skip_special_tokens=False)
                )

            # 解析每个agent的输出文本，提取JSON决策
            before = {k: global_counters[k] for k in global_counters}
            actions: Dict[str, Any] = {"p": [0]}
            parsed_rows = []
            for idx, text in enumerate(texts):
                raw, valid_fmt, used_fb = parse_action_json(text, stats=global_counters)
                # CRN: use pre-drawn Bernoulli random number for this (month, agent)
                # so that all replicas share the same u-value. Differences in the
                # binary work decision then come only from different work_prob outputs.
                b_u = float(bernoulli_crn[epi, idx]) if bernoulli_crn is not None else None
                env_act = convert_raw_action_to_env_action(raw, bernoulli_u=b_u)
                actions[str(idx)] = env_act
                meta = build_sample_meta(env, obs, idx)
                parsed_rows.append((idx, text, raw, valid_fmt, used_fb, env_act, meta))
                episode_samples.append(
                    {
                        "agent_id": int(idx),
                        "month": int(t),
                        "prompt_text": rollout_prompt_strs[idx],
                        "response_text": text,
                        "parsed_action_raw": dict(raw),
                        "env_action": [int(env_act[0]), float(env_act[1])],
                        "valid_format": bool(valid_fmt),
                        "used_fallback": bool(used_fb),
                        "meta": meta,
                    }
                )
            # DIAG: log pre-Bernoulli work_prob and consumption distributions
            _work_probs = [r[2]["work"] for r in parsed_rows]   # raw work prob
            _cons_vals  = [r[2]["consumption"] for r in parsed_rows]
            _labor_bits = [r[5][0] for r in parsed_rows]        # binary work after Bernoulli
            _price = float(env.world.price[-1])
            _demand_approx = sum(
                float(env.get_agent(str(r[0])).inventory.get("Coin", 0)) * r[2]["consumption"] / (_price + 1e-8)
                for r in parsed_rows
            )
            _supply_approx = sum(r[5][0] for r in parsed_rows) * float(
                env._components_dict["SimpleLabor"].num_labor_hours
            )
            print(f"[econ-diag] rep={replica_id} m={epi} | "
                  f"work_prob: mean={np.mean(_work_probs):.4f} std={np.std(_work_probs):.4f} | "
                  f"cons: mean={np.mean(_cons_vals):.4f} std={np.std(_cons_vals):.4f} | "
                  f"labor_binary: {sum(_labor_bits)}/{n_agents} | "
                  f"demand≈{_demand_approx:.0f} supply≈{_supply_approx:.0f}")

            # 执行决策，更新环境状态
            obs, _rew, _done, _info = env.step(actions)
            # 检查是否有agent的财富变为负数
            for idx in range(n_agents):
                coin = float(env.get_agent(str(idx)).inventory.get("Coin", 0.0))
                if coin < -1e-6:
                    global_counters["negative_wealth_count"] += 1
            # 更新memory状态
            for idx, text, raw, vf, uf, env_act, meta in parsed_rows:
                update_memory_state(memory_states[idx], raw, env_act, int(t))
            # 计算这个月的action统计信息
            month_action_stats = {
                k: global_counters[k] - before[k] for k in global_counters
            }
            # 收集环境统计信息
            monthly_series.append(
                collect_env_stats_step(
                    env,
                    int(epi),
                    month_action_stats,
                    global_counters,
                    float(initial_price),
                )
            )

            # 每3个月，构建季度反思（ACL24 style: 用 dialog4ref_queue 生成）
            # ACL24 uses (env.world.timestep+1)%3==0, i.e. end of quarter
            if (t + 1) % 3 == 0:
                reflection_prompt = (
                    "Given the previous quarter's economic environment, reflect on the labor, "
                    "consumption, and financial markets, as well as their dynamics. "
                    "What conclusions have you drawn? Your answer must be less than 200 words!"
                )
                for idx in range(n_agents):
                    dialog4ref_queue[idx].append({"role": "user", "content": reflection_prompt})

                # Generate reflections using dialog4ref_queue as context (with reflection system prompt)
                _REFL_SYS = {
                    "role": "system",
                    "content": (
                        "You are an economic agent reflecting on recent economic conditions. "
                        "Give a concise analysis in under 200 words about labor, consumption, and financial markets."
                    ),
                }
                ref_dialogs = [[_REFL_SYS] + list(dialog4ref_queue[idx]) for idx in range(n_agents)]
                reflection_texts: List[str] = []
                agent_mb_refl = int(getattr(self.config, "rollout_agent_micro_batch", 16))
                for start in range(0, n_agents, agent_mb_refl):
                    end = min(start + agent_mb_refl, n_agents)
                    sub_dp = self._build_gen_dataproto_from_dialogs(ref_dialogs[start:end])
                    sub_out = self._generate_with_gpu_padding(sub_dp)
                    sub_texts = self.tokenizer.batch_decode(
                        sub_out.batch["responses"], skip_special_tokens=True
                    )
                    reflection_texts.extend([t.strip() for t in sub_texts])

                for idx in range(n_agents):
                    dialog4ref_queue[idx].append({"role": "assistant", "content": reflection_texts[idx]})
                    memory_states[idx]["quarter_reflection_summary"] = reflection_texts[idx]
        # 12个月跑完后，整理并返回三样东西：所有agent每个月的样本数据、宏观经济统计数据、以及环境的dense_log（用于计算reward）
        env_stats = {
            "initial_price": float(initial_price),
            "monthly_series": monthly_series,
            "format_fail_count": int(global_counters["format_fail_count"]),
            "invalid_action_count": int(global_counters["invalid_action_count"]),
            "fallback_action_count": int(global_counters["fallback_action_count"]),
            "out_of_range_count": int(global_counters["out_of_range_count"]),
            "negative_wealth_count": int(global_counters["negative_wealth_count"]),
        }
        return {
            "episode_samples": episode_samples,
            "env_stats": env_stats,
            "dense_log": env.dense_log,
        }
    # 计算replica的年度奖励
    def compute_replica_reward(
        self,
        rollout_out: Mapping[str, Any],
        baselines: Mapping[str, Optional[float]],
        year_index: int,
    ) -> Tuple[Dict[str, Any], float]:
        from dry_run_grpo_pipeline import build_annual_metrics
        from reward_builder import DEFAULT_HIST_STATS, compute_annual_reward
        # 获取环境统计信息
        env_stats = rollout_out["env_stats"]
        cur_rows = env_stats.get("monthly_series") or []
        # 构建年度指标
        metrics = build_annual_metrics(
            env_stats,
            rollout_out["episode_samples"],
            cur_rows=list(cur_rows),
            prev_rows=None,
            prev_year_mean_price=baselines.get("prev_year_mean_price"),
            prev_year_mean_wage=baselines.get("prev_year_mean_wage"),
            prev_year_nominal_gdp=baselines.get("prev_year_nominal_gdp"),
            prev_year_real_gdp=baselines.get("prev_year_real_gdp"),
        )
        metrics.pop("_debug_n_months", None)
        metrics.pop("_debug_n_agents", None)
        metrics.pop("_resolved_prev_year_mean_price", None)
        metrics.pop("_resolved_prev_year_mean_wage", None)
        reward_out = compute_annual_reward(
            metrics, dict(DEFAULT_HIST_STATS), year_idx=int(year_index)
        )
        self._last_reward_breakdown = reward_out.get("breakdown", {})
        self._last_reward_R_macro = float(reward_out.get("R_macro", 0.0))
        self._last_reward_R_micro = float(reward_out.get("R_micro", 0.0))
        return metrics, float(reward_out["annual_reward"])

    # 把年度奖励和优势附加到每个样本中
    def attach_replica_reward_and_advantage(
        self,
        samples: List[Dict[str, Any]],
        annual_reward: float,
        advantage: float,
    ) -> None:
        # 把年度奖励和优势附加到每个样本中
        for s in samples:
            s["annual_reward"] = float(annual_reward)
            s["advantage"] = float(advantage)
    # 把GRPO样本堆叠成DataProto（每条 row = 一个 agent-month；prompt/response 分块 pad）
    def _stack_grpo_samples_to_dataproto(
        self,
        grpo_rows: List[Dict[str, Any]],
        *,
        meta_info: Optional[Dict[str, Any]] = None,
    ) -> DataProto:
        if not grpo_rows:
            raise ValueError("econ: empty grpo_rows")

        pad_id = int(self.tokenizer.pad_token_id)
        max_obs = int(
            getattr(self.config, "rollout_prompt_length", self.config.max_obs_length)
        )
        max_act = int(self.config.max_response_length)

        rows: List[Dict[str, Any]] = []
        for row in sorted(
            grpo_rows,
            key=lambda r: (
                str(r["replica_uid"]),
                int(r["agent_id"]),
                int(r["month"]),
            ),
        ):
            obs = list(map(int, row["prompt_token_ids"]))[-max_obs:]
            act = list(map(int, row["response_token_ids"]))[:max_act]

            gk = row.get(
                "group_key",
                f"ep{row.get('episode_id', 0)}:state_group:{row.get('state_group_id', 'default')}",
            )
            rows.append(
                {
                    "obs": obs,
                    "act": act,
                    "annual_reward": float(row["annual_reward"]),
                    "group_key": str(gk),
                    "replica_uid": str(row["replica_uid"]),
                    "agent_id": int(row["agent_id"]),
                    "month": int(row["month"]),
                }
            )

        prompt_width = max(len(r["obs"]) for r in rows)
        response_width = max(len(r["act"]) for r in rows)

        prompt_rows, response_rows = [], []
        prompt_attn_rows, response_attn_rows = [], []
        prompt_pos_rows, response_pos_rows = [], []
        full_action_mask_rows = []

        for r in rows:
            obs = r["obs"]
            act = r["act"]

            p = len(obs)
            a = len(act)

            p_pad = prompt_width - p
            a_pad = response_width - a

            prompt_ids = [pad_id] * p_pad + obs
            prompt_attn = [0] * p_pad + [1] * p
            prompt_pos = [0] * p_pad + list(range(p))

            # Response uses RIGHT padding (no gap between prompt and response)
            response_ids = act + [pad_id] * a_pad
            response_attn = [1] * a + [0] * a_pad
            response_pos = list(range(p, p + a)) + [0] * a_pad

            full_action_mask = [0] * prompt_width + [1] * a + [0] * a_pad

            prompt_rows.append(prompt_ids)
            response_rows.append(response_ids)
            prompt_attn_rows.append(prompt_attn)
            response_attn_rows.append(response_attn)
            prompt_pos_rows.append(prompt_pos)
            response_pos_rows.append(response_pos)
            full_action_mask_rows.append(full_action_mask)

        prompts = torch.tensor(prompt_rows, dtype=torch.long)
        responses = torch.tensor(response_rows, dtype=torch.long)

        prompt_attention = torch.tensor(prompt_attn_rows, dtype=torch.long)
        response_attention = torch.tensor(response_attn_rows, dtype=torch.long)
        attention_mask = torch.cat([prompt_attention, response_attention], dim=-1)

        prompt_position = torch.tensor(prompt_pos_rows, dtype=torch.long)
        response_position = torch.tensor(response_pos_rows, dtype=torch.long)
        position_ids = torch.cat([prompt_position, response_position], dim=-1)

        input_ids = torch.cat([prompts, responses], dim=-1)
        action_mask = torch.tensor(full_action_mask_rows, dtype=torch.long)

        annual_reward = torch.tensor(
            [r["annual_reward"] for r in rows], dtype=torch.float32
        )

        group_uids = np.array([r["group_key"] for r in rows], dtype=object)
        replica_uids = np.array([r["replica_uid"] for r in rows], dtype=object)

        mi = {"temperature": getattr(self, "_rollout_temperature", 1.0)}
        if meta_info:
            mi.update(meta_info)

        return DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "action_mask": action_mask,
                "annual_reward": annual_reward,
            },
            non_tensors={
                "uid": group_uids,
                "group_key": group_uids,
                "replica_uid": replica_uids,
            },
            meta_info=mi,
        )

    @staticmethod
    # GRPO的组内归一化：把4个replica的reward减去均值再除以标准差。eps=1e-8防止标准差为0时除零报错
    def _compute_group_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
        arr = np.asarray(list(rewards), dtype=float)
        n = int(arr.size)
        if n == 0:
            return []
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr, ddof=0)) if n > 1 else 0.0
        denom = std_r + float(eps)
        return [float((float(x) - mean_r) / denom) for x in arr]

    # 运行一年的经济仿真循环
    # 记录temperature（训练时=1，验证时=0），取出non_tensor数据（里面有year_index等字段）
    def run_econ_year_loop(
        self,
        meta_batch: DataProto,
        temperature: float = 1.0,
    ) -> DataProto:
        self._rollout_temperature = float(temperature)
        nt = meta_batch.non_tensor_batch or {}

        def _nth(name: str, idx: int, default):
            arr = nt.get(name, None)
            if arr is None:
                return default
            if isinstance(arr, np.ndarray):
                return arr[idx] if idx < len(arr) else default
            if isinstance(arr, (list, tuple)):
                return arr[idx] if idx < len(arr) else default
            return arr

        batch_size = 1
        for name in ("year_index", "state_group_id", "episode_id"):
            arr = nt.get(name, None)
            if isinstance(arr, (np.ndarray, list, tuple)):
                batch_size = max(batch_size, len(arr))

        all_outputs: List[DataProto] = []
        all_payloads: List[Dict[str, Any]] = []

        for bidx in range(batch_size):
            year_index = int(_nth("year_index", bidx, self.config.default_year_index))
            state_group_id = str(_nth("state_group_id", bidx, "default"))
            episode_id = int(_nth("episode_id", bidx, bidx))

            baselines_map: Dict[str, Optional[float]] = {
                "prev_year_mean_price": None,
                "prev_year_mean_wage": None,
                "prev_year_nominal_gdp": None,
                "prev_year_real_gdp": None,
            }
            # 构建模板环境（一次），从state bank记录里读取上一年的基准经济数据（用于计算通胀、GDP增长等相对指标）
            K = int(self.config.replica_group_size)
            template_env, template_obs, template_mem, template_ip, rec = self._materialize_template(
                year_index
            )
            if rec:
                for k in baselines_map:
                    if rec.get(k) is not None:
                        baselines_map[k] = float(rec[k])

            # 对每个replica，深拷贝一份模板环境，确保各replica独立
            group_rewards: List[float] = []
            replicas_payload: List[Dict[str, Any]] = []
            all_grpo: List[Dict[str, Any]] = []

            # --- CRN (Common Random Numbers) for GRPO variance control ---
            # Pre-draw shared random tables BEFORE the replica loop.
            # All K replicas in this group will use:
            #   1. bernoulli_crn: same U[0,1] per (month, agent) for work Bernoulli
            #   2. env_rng_seed: same seed for env-internal RNG (price/wage shocks)
            # This ensures within-group reward differences are driven by policy
            # outputs (work_prob, consumption), not by random noise.
            n_months_cfg = int(self.config.num_months)
            n_agents_cfg = int(self.config.num_agents)
            crn_rng = np.random.RandomState(
                seed=(episode_id * 100000 + year_index * 1000 + 7)
            )
            bernoulli_crn = crn_rng.uniform(size=(n_months_cfg, n_agents_cfg))
            env_rng_seed = int(crn_rng.randint(0, 2**31))

            for replica_id in range(K):
                env_r, obs_r = _clone_env_and_obs(template_env, template_obs)
                mem_r = copy.deepcopy(template_mem)

                # 运行单个replica的一年仿真
                rollout_out = self.run_single_replica_year(
                    env_r,
                    obs_r,
                    mem_r,
                    template_ip,
                    episode_id=episode_id,
                    replica_id=replica_id,
                    bernoulli_crn=bernoulli_crn,
                    env_rng_seed=env_rng_seed,
                )
                # 计算replica的年度奖励
                metrics, R_r = self.compute_replica_reward(
                    rollout_out, baselines_map, year_index
                )
                # 把年度奖励附加到每个样本中
                group_rewards.append(R_r)
                replicas_payload.append(
                    {
                        "replica_id": replica_id,
                        "metrics_used": metrics,
                        "reward_breakdown": dict(getattr(self, '_last_reward_breakdown', {})),
                        "R_macro": getattr(self, '_last_reward_R_macro', 0.0),
                        "R_micro": getattr(self, '_last_reward_R_micro', 0.0),
                        "rollout_out": rollout_out,
                    }
                )

            advantages = self._compute_group_advantages(group_rewards)

            # DEBUG: print per-replica reward breakdown
            print(f"[econ-debug] === GRPO Group (year={year_index}) ===")
            print(f"[econ-debug] group_rewards: {[f'{r:.3f}' for r in group_rewards]}")
            print(f"[econ-debug] advantages:    {[f'{a:.3f}' for a in advantages]}")
            for i, rep in enumerate(replicas_payload):
                bd = rep.get("reward_breakdown", {})
                print(f"[econ-debug] replica {i}: R_macro={rep.get('R_macro',0):.3f} "
                      f"R_micro={rep.get('R_micro',0):.3f} | "
                      f"u_ctr={bd.get('u_center',0):.3f} u_grd={bd.get('u_guard',0):.3f} "
                      f"work={bd.get('work_guard',0):.3f} "
                      f"gdp_r={bd.get('real_gdp_center',0):.3f} gdp_rG={bd.get('real_gdp_guard',0):.3f} "
                      f"gdp_n={bd.get('nominal_gdp_center',0):.3f} gdp_nG={bd.get('nominal_gdp_guard',0):.3f} "
                      f"pi={bd.get('pi_center',0):.3f} piG={bd.get('pi_guard',0):.3f} "
                      f"wpi={bd.get('wage_pi_center',0):.3f} wpiG={bd.get('wage_pi_guard',0):.3f} "
                      f"fmt={bd.get('fmt',0):.3f} valid={bd.get('valid',0):.3f}")

            from grpo_exporter import export_grpo_samples

            for i, rep in enumerate(replicas_payload):
                rid = int(rep["replica_id"])
                samples = export_grpo_samples(
                    rep["rollout_out"],
                    self.tokenizer,
                    replica_id=rid,
                    episode_id=int(episode_id),
                    state_group_id=str(state_group_id),
                    align_rollout_chat_template=False,  # prompt_text is already chat-templated
                )
                self.attach_replica_reward_and_advantage(
                    samples, group_rewards[i], advantages[i]
                )
                all_grpo.extend(samples)

            final_dp = self._stack_grpo_samples_to_dataproto(all_grpo)

            # Compute mean reward breakdown across replicas for logging
            _bd_keys = set()
            for r in replicas_payload:
                _bd_keys.update(r.get("reward_breakdown", {}).keys())
            mean_breakdown = {}
            for k in _bd_keys:
                vals = [float(r.get("reward_breakdown", {}).get(k, 0.0)) for r in replicas_payload]
                mean_breakdown[k] = sum(vals) / max(len(vals), 1)
            mean_R_macro = sum(r.get("R_macro", 0.0) for r in replicas_payload) / max(len(replicas_payload), 1)
            mean_R_micro = sum(r.get("R_micro", 0.0) for r in replicas_payload) / max(len(replicas_payload), 1)

            payload = {
                "group_id": f"ep{episode_id}:state_group:{state_group_id}",
                "state_group_id": str(state_group_id),
                "year_index": int(year_index),
                "group_rewards": list(group_rewards),
                "group_advantages": advantages,
                "mean_reward_breakdown": mean_breakdown,
                "mean_R_macro": float(mean_R_macro),
                "mean_R_micro": float(mean_R_micro),
                "replicas": [
                    {
                        "replica_id": int(r["replica_id"]),
                        "annual_reward": float(group_rewards[j]),
                        "advantage": float(advantages[j]),
                        "reward_breakdown": r.get("reward_breakdown", {}),
                        "R_macro": r.get("R_macro", 0.0),
                        "R_micro": r.get("R_micro", 0.0),
                        "episode_samples": r["rollout_out"]["episode_samples"],
                        "env_stats": r["rollout_out"]["env_stats"],
                    }
                    for j, r in enumerate(replicas_payload)
                ],
            }
            final_dp.meta_info["econ_group_payload"] = payload

            all_outputs.append(final_dp)
            all_payloads.append(payload)

        return self._concat_dataprotos(all_outputs, all_payloads)

    @staticmethod
    def _concat_dataprotos(
        dps: List[DataProto],
        payloads: List[Dict[str, Any]],
    ) -> DataProto:
        if not dps:
            raise ValueError("econ: empty dataprotos")

        if len(dps) == 1:
            dp = dps[0]
            dp.meta_info["econ_group_payload"] = payloads[0]
            return dp

        batch_keys = list(dps[0].batch.keys())
        tensors = {
            k: torch.cat([dp.batch[k] for dp in dps], dim=0)
            for k in batch_keys
        }

        nt_keys = set()
        for dp in dps:
            nt_keys.update((dp.non_tensor_batch or {}).keys())

        non_tensors: Dict[str, np.ndarray] = {}
        for k in nt_keys:
            parts = []
            for dp in dps:
                nt = dp.non_tensor_batch or {}
                if k not in nt:
                    continue
                v = nt[k]
                parts.append(v if isinstance(v, np.ndarray) else np.asarray(v, dtype=object))
            non_tensors[k] = np.concatenate(parts, axis=0) if parts else np.array([], dtype=object)

        meta = dict(dps[0].meta_info or {})
        meta["econ_group_payload"] = payloads
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta)