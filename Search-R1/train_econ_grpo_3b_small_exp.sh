#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Qwen2.5-3B-Instruct 小实验：state bank + vLLM + wandb
#
# 与 train_econ_grpo_vllm_small_real_exp.sh 相同逻辑，改动：
#   - 模型改为 Qwen2.5-3B-Instruct（显存更低，迭代更快）
#   - gpu_memory_utilization 调高到 0.50（3B 参数量小）
#   - ppo_mini_batch_size / micro_batch_size 可适当放大
#   - venv 默认指向 /workspace/venvs/train_vllm063
#
# 跑法：
#   cd /workspace/Search-R1
#   export VLLM_TRAIN_VENV=/workspace/venvs/train_vllm063
#   bash train_econ_grpo_3b_small_exp.sh
#
# 用 W&B：先在 venv 里 wandb login，再：
#   export ECON_SMALL_EXP_LOGGER=wandb
#   bash train_econ_grpo_3b_small_exp.sh
# =============================================================================

TRAIN_VENV="${VLLM_TRAIN_VENV:-/workspace/venvs/train_vllm063}"
TRAIN_PY="${TRAIN_VENV}/bin/python"
export RAY_WORKER_EXECUTABLE="${TRAIN_PY}"
export PYTHONPATH=/workspace/Search-R1:/workspace/QWEN2.5_42_7b_main:${PYTHONPATH:-}
unset RAY_ADDRESS

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL='/workspace/model/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME='econagent-grpo-qwen2.5-3b-small-exp'

ECON_ROLL_MICRO=16

case "${ECON_SMALL_EXP_LOGGER:-wandb}" in
  console|CONSOLE) TRAINER_LOGGER="['console']" ;;
  *) TRAINER_LOGGER="['wandb']" ;;
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
  actor_rollout_ref.actor.ppo_micro_batch_size=2 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=$ECON_ROLL_MICRO \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.load_format=dummy_dtensor \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.do_sample=true \
  actor_rollout_ref.rollout.micro_batch_size=2 \
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
  trainer.test_freq=9999 \
  ++trainer.val_before_train=false \
  actor_rollout_ref.rollout.temperature=1 \
  2>&1 | tee "${EXPERIMENT_NAME}.log"
