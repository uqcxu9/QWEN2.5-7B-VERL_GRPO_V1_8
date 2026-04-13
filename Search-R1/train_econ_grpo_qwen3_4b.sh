#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Qwen3-4B-Instruct-2507 + VERL 0.7.1 + vLLM 0.12 + State Bank + GRPO
#
# 正式实验：12 个月完整年度, 4 replicas, 100 agents, 3 epochs
#
# 跑法：
#   cd /workspace/Search-R1
#   bash train_econ_grpo_qwen3_4b.sh
# =============================================================================

TRAIN_VENV="${VERL_TRAIN_VENV:-/workspace/venvs/verl_latest}"
TRAIN_PY="${TRAIN_VENV}/bin/python"
export RAY_WORKER_EXECUTABLE="${TRAIN_PY}"
export PYTHONPATH=/workspace/Search-R1:/workspace/QWEN2.5_42_7b_main:${PYTHONPATH:-}
unset RAY_ADDRESS

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_CUMEM=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_SLEEP_LEVEL=0
export CUDA_VISIBLE_DEVICES=0
export ECON_ENABLED=1
export BASE_MODEL='/workspace/model/Qwen3-4B-Instruct-2507'
export EXPERIMENT_NAME='econagent-grpo-qwen3-4b-v3-seqmean-lr1e6'

ECON_ROLL_MICRO=16

case "${ECON_LOGGER:-wandb}" in
  console|CONSOLE) TRAINER_LOGGER="['console']" ;;
  *) TRAINER_LOGGER="['wandb']" ;;
esac

PYTHONUNBUFFERED=1 "${TRAIN_PY}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  ++actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  '++actor_rollout_ref.actor.optim.override_optimizer_config={foreach: false}' \
  actor_rollout_ref.actor.ppo_mini_batch_size=1200 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=2 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.18 \
  actor_rollout_ref.rollout.max_model_len=1536 \
  actor_rollout_ref.rollout.enforce_eager=true \
  ++actor_rollout_ref.rollout.enable_sleep_mode=false \
  actor_rollout_ref.rollout.free_cache_engine=false \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.do_sample=true \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
  ++econ.enabled=true \
  ++econ.project_root=/workspace/QWEN2.5_42_7b_main \
  ++econ.state_bank_path=/workspace/QWEN2.5_42_7b_main/data/state_bank_4b/state_bank_train.json \
  ++econ.num_agents=100 \
  ++econ.num_months=12 \
  ++econ.replica_group_size=4 \
  ++econ.default_year_index=2 \
  ++econ.rollout_agent_micro_batch=$ECON_ROLL_MICRO \
  ++econ.rollout_prompt_length=1216 \
  data.train_files=/workspace/QWEN2.5_42_7b_main/data/train_stub.parquet \
  data.val_files=/workspace/QWEN2.5_42_7b_main/data/val_stub.parquet \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=1280 \
  ++data.max_start_length=1216 \
  data.max_response_length=64 \
  ++data.max_obs_length=1216 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name='EconAgent-GRPO' \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.total_epochs=4 \
  trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
  ++trainer.val_before_train=false \
  trainer.save_freq=20 \
  trainer.test_freq=9999 \
  2>&1 | tee "${EXPERIMENT_NAME}.log"
