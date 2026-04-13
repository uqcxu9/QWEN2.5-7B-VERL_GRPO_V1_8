#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 真实配置「小实验」：与 train_econ_grpo.sh 相同 Econ + state bank + 长上下文，
# 但月份/epoch/批量更保守，logger 用 console，便于先看能否跑通与指标是否异常。
# 不修改 train_econ_grpo.sh。
#
# 环境是否与 vLLM 就绪：
#   - 优先用你已装好 vLLM（VERL 白名单版本）的 venv，并 export VLLM_TRAIN_VENV=...
#   - 仅做依赖安装（不启训练）：bash setup_econ_train_vllm_venv.sh
#   - Blackwell：勿 pip install vllm==0.6.3（会换 torch 2.4 → sm_120 无内核）。请源码安装：
#       bash scripts/install_vllm063_source_blackwell.sh
#     验证：env -u PYTHONPATH "$VLLM_TRAIN_VENV/bin/python" -c "import verl.third_party.vllm as v; print(v.vllm_version)"
#
# vLLM 初始化 OOM 时：把 gpu_memory_utilization 从 0.40 降到 0.35 再试。
#
# 怎么跑（示例）：
#   cd /workspace/Search-R1
#   export VLLM_TRAIN_VENV=/workspace/venvs/blackwell_train   # 换成已装好 vLLM 的 venv
#   bash train_econ_grpo_vllm_small_real_exp.sh
# 上 W&B：先在**同一 venv**里 wandb login，再：
#   export ECON_SMALL_EXP_LOGGER=wandb
#   bash train_econ_grpo_vllm_small_real_exp.sh
# =============================================================================

TRAIN_VENV="${VLLM_TRAIN_VENV:-${BLACKWELL_TRAIN_VENV:-/workspace/venvs/blackwell_train}}"
TRAIN_PY="${TRAIN_VENV}/bin/python"
export RAY_WORKER_EXECUTABLE="${TRAIN_PY}"
export PYTHONPATH=/workspace/Search-R1:/workspace/QWEN2.5_42_7b_main:${PYTHONPATH:-}
unset RAY_ADDRESS

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 注意：vLLM v0.6.3 的 TORCH_SDPA backend 在 CUDA 上并不完整（会在 metadata builder 处抛 NotImplementedError）。
# 因此这里默认使用 FLASH_ATTN（更符合 vLLM 预期路径）。若你明确想尝试别的 backend，可在外部自行 export 覆盖。
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL='/workspace/model/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='econagent-verl-grpo-vllm-small-real-exp'

ECON_ROLL_MICRO=8

case "${ECON_SMALL_EXP_LOGGER:-console}" in
  wandb|WANDB) TRAINER_LOGGER="['wandb']" ;;
  *) TRAINER_LOGGER="['console']" ;;
esac

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
  ++econ.num_agents=100 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.load_format=dummy_dtensor \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.do_sample=true \
  actor_rollout_ref.rollout.micro_batch_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=$ECON_ROLL_MICRO \
  ++econ.rollout_agent_micro_batch=$ECON_ROLL_MICRO \
  ++econ.rollout_prompt_length=768 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.fsdp_config.grad_offload=true \
  actor_rollout_ref.rollout.n_agent=100 \
  data.train_files=/workspace/QWEN2.5_42_7b_main/data/train_stub.parquet \
  data.val_files=/workspace/QWEN2.5_42_7b_main/data/val_stub.parquet \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=4342 \
  data.max_start_length=768 \
  data.max_response_length=96 \
  data.max_obs_length=768 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name='EconAgent-GRPO' \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.total_epochs=1 \
  trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
  ++econ.enabled=true \
  ++econ.project_root=/workspace/QWEN2.5_42_7b_main \
  ++econ.state_bank_path=/workspace/QWEN2.5_42_7b_main/data/state_bank/state_bank.json \
  ++econ.num_months=3 \
  ++econ.replica_group_size=4 \
  ++econ.default_year_index=2 \
  trainer.save_freq=9999 \
  trainer.test_freq=2 \
  actor_rollout_ref.rollout.temperature=1 \
  2>&1 | tee "${EXPERIMENT_NAME}.log"
