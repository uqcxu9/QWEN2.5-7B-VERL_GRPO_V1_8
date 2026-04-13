#!/usr/bin/env bash
set -euo pipefail

# Small formal run v1：在 train_econ_grpo_smoke.sh 基础上略放大数据步数，便于小规模正式感跑法。
# - rollout.name=hf，显式 do_sample=true（消除 transformers 对 top_k=0 + greedy 的告警）
# - stub_dataset_size=8 → 每 epoch 约 8 个训练 step（train_batch_size=1）
# - total_epochs=1，100 agents × 3 months × replica_group_size=4，其余与 smoke 对齐
# 冒烟/更小步数仍用: bash train_econ_grpo_smoke.sh

TRAIN_VENV="${BLACKWELL_TRAIN_VENV:-/workspace/venvs/blackwell_train}"
TRAIN_PY="${TRAIN_VENV}/bin/python"
export RAY_WORKER_EXECUTABLE="${TRAIN_PY}"
export PYTHONPATH=/workspace/Search-R1:/workspace/QWEN2.5_42_7b_main:${PYTHONPATH:-}
unset RAY_ADDRESS

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL='/workspace/model/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='econagent-verl-grpo-small-formal-v1'

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
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.do_sample=true \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.top_p=1.0 \
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
