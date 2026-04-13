#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 尝试用 vLLM 做 rollout（在 small_formal_v1 基础上只改 rollout 相关项）
#
# 与正在 tmux 里跑的 HF 实验并行时：
#   - 不要 attach 进旧 session、不要改旧进程即可「不动 tmux」。
#   - 另开 SSH / 新 tmux：`tmux new -s econ_vllm`（新 session 名随意）。
#   - 若只有一块 GPU：同时跑两份训练会抢显存/极慢，请等 HF 跑完再启动本脚本，
#     或确认机器有多卡并为两路设置不同的 CUDA_VISIBLE_DEVICES。
#
# 环境请先跑（只装包、不启训练）：bash setup_econ_train_vllm_venv.sh
# 默认独立 venv：export VLLM_TRAIN_VENV=/path/to/venv（默认 /workspace/venvs/econ_train_vllm）
#
# 前置条件（否则一启动就会挂）：
#   1) pip 安装的 vLLM 版本必须是 VERL 白名单之一：
#      0.3.1 / 0.4.2 / 0.5.4 / 0.6.3（见 verl/third_party/vllm/__init__.py）
#   2) Blackwell (sm_120)：torch 2.4+cu121 无法在 GPU 上跑；vllm 0.6.3 wheel 又与 torch 2.12/cu128
#      ABI 不兼容。setup 脚本在 Blackwell 上**不会**装 pip vllm；本训练脚本需待你自行源码编译
#      vLLM（与当前 torch 一致）后再跑，否则继续用 train_econ_grpo_small_formal_v1.sh（HF）。
#   3) 非 Blackwell：setup 脚本走 README 栈 torch2.4+cu121 + vllm==0.6.3 即可。
#
# 显存：单卡上同时驻留 FSDP actor 与 vLLM 引擎很紧；若 OOM 可略降 gpu_memory_utilization
# 或暂时回到 train_econ_grpo_small_formal_v1.sh（HF rollout）。
# =============================================================================

TRAIN_VENV="${VLLM_TRAIN_VENV:-/workspace/venvs/econ_train_vllm}"
TRAIN_PY="${TRAIN_VENV}/bin/python"
export RAY_WORKER_EXECUTABLE="${TRAIN_PY}"
export PYTHONPATH=/workspace/Search-R1:/workspace/QWEN2.5_42_7b_main:${PYTHONPATH:-}
unset RAY_ADDRESS

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL='/workspace/model/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='econagent-verl-grpo-small-formal-v1-vllm'

RUN_AGENTS=100
RUN_MONTHS=3
REPLICA_GROUP_SIZE=4
STUB_DATASET_SIZE=8
ECON_ROLL_MICRO=8

PYTHONUNBUFFERED=1 "${TRAIN_PY}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  ++econ.num_agents=$RUN_AGENTS \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=$ECON_ROLL_MICRO \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.load_format=dummy_dtensor \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.do_sample=true \
  actor_rollout_ref.rollout.micro_batch_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=$ECON_ROLL_MICRO \
  ++econ.rollout_agent_micro_batch=$ECON_ROLL_MICRO \
  ++econ.rollout_prompt_length=384 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.fsdp_config.grad_offload=true \
  actor_rollout_ref.rollout.n_agent=$RUN_AGENTS \
  data.train_files=/workspace/QWEN2.5_42_7b_main/data/train_stub.parquet \
  data.val_files=/workspace/QWEN2.5_42_7b_main/data/val_stub.parquet \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=512 \
  data.max_start_length=256 \
  data.max_response_length=64 \
  data.max_obs_length=256 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.logger="['console']" \
  trainer.project_name='EconAgent-GRPO' \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.total_epochs=1 \
  trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
  ++econ.enabled=true \
  ++econ.stub_dataset_size=$STUB_DATASET_SIZE \
  ++econ.project_root=/workspace/QWEN2.5_42_7b_main \
  ++econ.num_months=$RUN_MONTHS \
  ++econ.replica_group_size=$REPLICA_GROUP_SIZE \
  ++econ.default_year_index=2 \
  trainer.save_freq=9999 \
  trainer.test_freq=9999 \
  actor_rollout_ref.rollout.temperature=1 \
  2>&1 | tee "${EXPERIMENT_NAME}.log"
