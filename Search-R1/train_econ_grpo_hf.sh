#!/usr/bin/env bash
set -euo pipefail

# 与 train_econ_grpo.sh 相同实验设定，但 rollout 使用 HF（vLLM 未就绪时用本脚本）。
# 先进入环境: source "$(dirname "$0")/env_blackwell.sh"
# 冒烟联调: bash train_econ_grpo_smoke.sh

TRAIN_VENV="${BLACKWELL_TRAIN_VENV:-/workspace/venvs/blackwell_train}"
TRAIN_PY="${TRAIN_VENV}/bin/python"
export RAY_WORKER_EXECUTABLE="${TRAIN_PY}"
export PYTHONPATH=/workspace/Search-R1:/workspace/QWEN2.5_42_7b_main:${PYTHONPATH:-}
unset RAY_ADDRESS

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL='/workspace/model/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='econagent-verl-grpo-qwen2.5-7b-it'

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
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.do_sample=true \
  actor_rollout_ref.rollout.micro_batch_size=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
  ++econ.rollout_agent_micro_batch=16 \
  ++econ.rollout_prompt_length=768 \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
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
  trainer.logger="['wandb']" \
  trainer.project_name='EconAgent-GRPO' \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.total_epochs=5 \
  trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
  ++econ.enabled=true \
  ++econ.project_root=/workspace/QWEN2.5_42_7b_main \
  ++econ.state_bank_path=/workspace/QWEN2.5_42_7b_main/data/state_bank/state_bank.json \
  ++econ.num_months=12 \
  ++econ.replica_group_size=4 \
  ++econ.default_year_index=2 \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  actor_rollout_ref.rollout.temperature=1 \
  2>&1 | tee "${EXPERIMENT_NAME}.log"
