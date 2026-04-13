# QWEN2.5-7B-VERL_GRPO_V1_8

Snapshot for starting 50-agent GRPO on a fresh instance.

## Contents
- `Search-R1/` — verl 0.7.1 fork with EconAgent GRPO integration, training scripts, thought.txt, EXPERIMENT_LOG.txt
- `QWEN2.5_42_7b_main/` — EconAgent env + reward_builder + rollout_collector + grpo_exporter; 100-agent state bank (reference, not for v4)
- `QWEN3-4B_2507_50_AGENTS/` — 50-agent baseline simulation (builds v4 state bank via `python simulate.py --num_agents 50 --episode_length 240`)

## v4 setup sequence
1. `cd QWEN3-4B_2507_50_AGENTS && python simulate.py --policy_model gpt --model_type qwen --num_agents 50 --episode_length 240`  (~3-6h)
2. Build 50-agent state bank (train/val/test split).
3. Point `Search-R1/train_econ_grpo_qwen3_4b.sh` to new state_bank path.
4. Launch training.

## Known issues
- `one_step_economy.py` in `QWEN3-4B_2507_50_AGENTS/ai_economist/` uses hardcoded `open('data/profiles.json')` — must `cd` into that directory before running simulate.py.
- `simulate_utils.py` has `enable_thinking=True` which is no-op on Qwen3-4B-Instruct-2507 (non-thinking model); harmless but wastes `max_tokens` budget.
