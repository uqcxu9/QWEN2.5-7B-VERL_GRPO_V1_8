#!/usr/bin/env python3
"""
State bank warm-up：用 complex 策略跑满 N 个月（默认 240），
在每一步 env.step 之后滚动更新 per-agent memory（与 qwen rollout 口径一致），
并累积 monthly_series；每年末写快照，路径相对 state_bank.json 所在目录，供 restore。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import ai_economist.foundation as foundation
import yaml

from build_state_bank import (
    build_state_bank_records,
    default_state_bank_out_path,
    save_state_bank,
)
from rollout_collector import collect_env_stats_step, default_memory_state
from simulate import capture_env_snapshot, complex_actions, update_memory_and_history

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    p = argparse.ArgumentParser(description="Complex-policy warm-up → env_stats + state_bank.json")
    p.add_argument("--num-agents", type=int, default=100)
    p.add_argument("--episode-length", type=int, default=240)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--h", type=float, default=1.0)
    p.add_argument("--max-price-inflation", type=float, default=0.1)
    p.add_argument("--max-wage-inflation", type=float, default=0.05)
    p.add_argument(
        "--rollout-dir",
        type=str,
        default=None,
        help="写入 env_stats.json 的目录（默认：<repo>/data/complex_state_bank_warmup/rollout）",
    )
    p.add_argument(
        "--state-bank-out",
        type=str,
        default=None,
        help="state_bank.json 路径（默认：rollout 上级目录下的 state_bank/state_bank.json）",
    )
    args = p.parse_args()

    rollout_dir = args.rollout_dir or os.path.join(
        _SCRIPT_DIR, "data", "complex_state_bank_warmup", "rollout"
    )
    env_stats_path = os.path.join(rollout_dir, "env_stats.json")
    bank_out = args.state_bank_out or default_state_bank_out_path(env_stats_path)
    bank_base_dir = os.path.dirname(os.path.abspath(bank_out))
    snapshots_dir = os.path.join(bank_base_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    per_year_assets: Dict[int, Dict[str, str]] = {}

    cfg_path = os.path.join(_SCRIPT_DIR, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        run_configuration = yaml.safe_load(f)
    env_config: Dict[str, Any] = dict(run_configuration["env"])
    env_config["n_agents"] = int(args.num_agents)
    env_config["episode_length"] = int(args.episode_length)
    env_config["components"][2]["SimpleConsumption"]["max_price_inflation"] = float(
        args.max_price_inflation
    )
    env_config["components"][2]["SimpleConsumption"]["max_wage_inflation"] = float(
        args.max_wage_inflation
    )

    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    initial_price = float(env.world.price[0])
    memory_states: List[Dict[str, Any]] = [
        default_memory_state() for _ in range(env.num_agents)
    ]
    global_counters = {
        "format_fail_count": 0,
        "invalid_action_count": 0,
        "fallback_action_count": 0,
        "out_of_range_count": 0,
        "negative_wealth_count": 0,
    }
    monthly_series: List[Dict[str, Any]] = []
    episode_length = int(args.episode_length)

    for epi in range(episode_length):
        t = int(env.world.timestep)
        before = {k: global_counters[k] for k in global_counters}
        actions = complex_actions(
            env, obs, beta=args.beta, gamma=args.gamma, h=args.h
        )
        obs, _, _, _ = env.step(actions)

        for idx in range(env.num_agents):
            raw_work = float(actions[str(idx)][0])
            raw_cons_idx = float(actions[str(idx)][1])
            parsed_raw = {
                "work": raw_work,
                "consumption": raw_cons_idx * 0.02,
            }
            env_action = [int(raw_work), float(raw_cons_idx)]
            update_memory_and_history(
                memory_states[idx], parsed_raw, env_action, t
            )

        for idx in range(env.num_agents):
            coin = float(env.get_agent(str(idx)).inventory.get("Coin", 0.0))
            if coin < -1e-6:
                global_counters["negative_wealth_count"] += 1

        month_action_stats = {
            k: global_counters[k] - before[k] for k in global_counters
        }
        monthly_series.append(
            collect_env_stats_step(
                env,
                int(epi),
                month_action_stats,
                global_counters,
                initial_price,
            )
        )

        if (epi + 1) % 12 == 0:
            year_done = (epi + 1) // 12
            y_next = year_done + 1
            if y_next >= 2:
                stem = f"y{y_next}"
                out_dir = os.path.join(snapshots_dir, stem)
                os.makedirs(out_dir, exist_ok=True)

                paths = capture_env_snapshot(
                    env,
                    obs,
                    memory_states,
                    initial_price,
                    out_dir=out_dir,
                    stem=stem,
                )
                per_year_assets[y_next] = {
                    k: os.path.join("snapshots", stem, v) for k, v in paths.items()
                }

    env_stats = {
        "initial_price": float(initial_price),
        "monthly_series": monthly_series,
        "format_fail_count": int(global_counters["format_fail_count"]),
        "invalid_action_count": int(global_counters["invalid_action_count"]),
        "fallback_action_count": int(global_counters["fallback_action_count"]),
        "out_of_range_count": int(global_counters["out_of_range_count"]),
        "negative_wealth_count": int(global_counters["negative_wealth_count"]),
    }

    os.makedirs(rollout_dir, exist_ok=True)
    with open(env_stats_path, "w", encoding="utf-8") as f:
        json.dump(env_stats, f, ensure_ascii=False, indent=2)
    print(f"Wrote {env_stats_path} ({len(monthly_series)} months)")

    records = build_state_bank_records(env_stats, per_year_assets=per_year_assets)
    save_state_bank(records, bank_out)
    print(f"Wrote {len(records)} state_bank records → {bank_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
