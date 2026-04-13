#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO group **preprocessing** only (thought.txt Stages 1–8, non-trainer).

For one sampled initial economy state ``s_y``:
  * fork ``K`` replicas (default 4) from the **same** ``s_y``;
  * each runs ``episode_length`` monthly EconAgent steps via ``run_annual_rollout``;
  * compute replica-level ``R_r`` with ``reward_builder``;
  * export month-level samples with ``grpo_exporter``;
  * ``A_r = (R_r - mean_R) / (std_R + eps)`` and broadcast ``(R_r, A_r)`` to all
    samples of replica ``r``.

Does **not** call VERL, PPO/GRPO loss, KL, or parameter updates.

Initial state ``s_y``
---------------------
**Preferred:** ``state_bank.json`` row for ``state_bank_year_index`` (default =
training ``year_index``) contains non-null ``snapshot_path``, ``obs_path``,
``memory_state_path``, ``history_state_path`` pointing to pickles under the same
directory as the JSON. Then ``simulate.restore_env_snapshot`` loads one shared
template; each replica deep-copies that template (RNG is **not** restored
globally so per-replica seeds still diversify trajectories).

**Fallback (V0):** one ``reset()`` on a fresh env, then ``deepcopy`` env+obs as
the shared template (macro baselines from the bank row still apply to rewards).

Returns a top-level ``group_batch`` dict (memory-oriented) with ``replicas``
entries containing ``replica_id``, ``annual_reward``, ``advantage``, ``samples``
for trainers;
optional disk artifacts via ``persist_artifacts``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from dry_run_grpo_pipeline import _FakeTokenizerForDryRun, build_annual_metrics
from grpo_exporter import export_grpo_samples, summarize_grpo_sample_lengths
from reward_builder import DEFAULT_HIST_STATS, compute_annual_reward


def compute_group_advantages(
    rewards: Sequence[float], eps: float = 1e-8
) -> List[float]:
    """
    GRPO group normalization (thought.txt): A_r = (R_r - mean_R) / (std_R + eps).

    Uses sample std (ddof=1) when len > 1; ``std_R + eps`` avoids division by zero.
    """
    arr = np.asarray(list(rewards), dtype=float)
    n = int(arr.size)
    if n == 0:
        return []
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr, ddof=0)) if n > 1 else 0.0
    denom = std_r + float(eps)
    return [float((float(r) - mean_r) / denom) for r in arr]


def clone_env_and_obs(env: Any, obs: Any) -> Tuple[Any, Any]:
    """
    Clone env + obs for a fresh replica. Prefer ``deepcopy``; fallback ``pickle``.
    """
    try:
        return copy.deepcopy(env), copy.deepcopy(obs)
    except Exception:
        blob = pickle.dumps((env, obs), protocol=pickle.HIGHEST_PROTOCOL)
        return pickle.loads(blob)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_qwen_rollout_env_config(
    num_agents: int,
    episode_length: int,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Match ``simulate.py`` ``qwen_rollout`` branch + episode bounds."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = config_path or os.path.join(script_dir, "config.yaml")
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


def _record_has_restore_paths(rec: Optional[Mapping[str, Any]]) -> bool:
    if not rec:
        return False
    for k in ("snapshot_path", "obs_path", "memory_state_path", "history_state_path"):
        p = rec.get(k)
        if p is None or (isinstance(p, str) and not p.strip()):
            return False
    return True


def _restore_paths_exist_on_disk(bank_dir: str, rec: Mapping[str, Any]) -> bool:
    if not _record_has_restore_paths(rec):
        return False
    for k in ("snapshot_path", "obs_path", "memory_state_path", "history_state_path"):
        rel = rec[k]
        path = rel if os.path.isabs(str(rel)) else os.path.join(bank_dir, str(rel))
        if not os.path.isfile(path):
            return False
    return True


def _resolve_macro_baselines(
    *,
    year_index: int,
    state_bank_path: Optional[str],
    state_bank_year_index: Optional[int],
    prev_year_mean_price: Optional[float],
    prev_year_mean_wage: Optional[float],
    prev_year_nominal_gdp: Optional[float],
    prev_year_real_gdp: Optional[float],
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    """
    Priority: explicit CLI kwargs → state bank row for ``state_bank_year_index``
    (default ``year_index``) → None.

    Returns (resolved_floats_for_metrics, baseline_source_debug_dict).
    """
    bank_y = int(state_bank_year_index) if state_bank_year_index is not None else int(year_index)
    bank_rec: Optional[Dict[str, Any]] = None
    bank_records: Optional[List[Dict[str, Any]]] = None
    if state_bank_path:
        from build_state_bank import get_state_bank_record, load_state_bank

        bank_records = load_state_bank(state_bank_path)
        bank_rec = get_state_bank_record(bank_records, bank_y)

    def _pick(cli: Optional[float], key: str) -> Optional[float]:
        if cli is not None:
            return float(cli)
        if bank_rec is not None and bank_rec.get(key) is not None:
            return float(bank_rec[key])
        return None

    resolved = {
        "prev_year_mean_price": _pick(prev_year_mean_price, "prev_year_mean_price"),
        "prev_year_mean_wage": _pick(prev_year_mean_wage, "prev_year_mean_wage"),
        "prev_year_nominal_gdp": _pick(prev_year_nominal_gdp, "prev_year_nominal_gdp"),
        "prev_year_real_gdp": _pick(prev_year_real_gdp, "prev_year_real_gdp"),
    }
    source = {
        "state_bank_path": os.path.abspath(state_bank_path) if state_bank_path else None,
        "state_bank_year_index": bank_y if state_bank_path else None,
        "year_index_training": int(year_index),
        "matched_state_bank_record": bank_rec is not None,
        "prev_year_mean_price": resolved["prev_year_mean_price"],
        "prev_year_mean_wage": resolved["prev_year_mean_wage"],
        "prev_year_nominal_gdp": resolved["prev_year_nominal_gdp"],
        "prev_year_real_gdp": resolved["prev_year_real_gdp"],
    }
    return resolved, source


def run_multi_replica_group(
    *,
    num_replicas: int = 4,
    num_agents: int = 100,
    episode_length: int = 12,
    year_index: int,
    base_save_path: str = ".",
    policy_model_save: Optional[str] = None,
    fake_tokenizer: bool = False,
    seed: Optional[int] = None,
    prev_year_mean_price: Optional[float] = None,
    prev_year_mean_wage: Optional[float] = None,
    prev_year_nominal_gdp: Optional[float] = None,
    prev_year_real_gdp: Optional[float] = None,
    state_bank_path: Optional[str] = None,
    state_bank_year_index: Optional[int] = None,
    episode_id: int = 0,
    io_log_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    persist_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Run ``num_replicas`` rollouts from one shared initial ``s_y``; compute rewards,
    GRPO advantages, and GRPO samples with broadcast scalars.

    Returns a dict with ``group_batch`` (trainer-oriented), ``group_rewards``,
    ``group_advantages``, and ``replicas`` (full per-replica diagnostics including
    ``samples`` with ``annual_reward`` and ``advantage``).
    """
    import ai_economist.foundation as foundation
    from simulate import run_annual_rollout

    if policy_model_save is None:
        policy_model_save = (
            f"multi_replica-{num_agents}a-{episode_length}m-y{year_index}"
        )

    env_config = build_qwen_rollout_env_config(
        num_agents, episode_length, config_path=config_path
    )

    bank_dir = ""
    bank_rec: Optional[Dict[str, Any]] = None
    if state_bank_path:
        from build_state_bank import get_state_bank_record, load_state_bank, state_bank_base_dir

        bank_y = int(state_bank_year_index) if state_bank_year_index is not None else int(year_index)
        _records = load_state_bank(state_bank_path)
        bank_rec = get_state_bank_record(_records, bank_y)
        bank_dir = state_bank_base_dir(state_bank_path)
    else:
        bank_y = int(year_index)

    restored_from_snapshot = False
    template_mem: Optional[List[Dict[str, Any]]] = None
    template_initial_price: Optional[float] = None
    template_env: Any = None
    template_obs: Any = None

    if state_bank_path and bank_rec and _restore_paths_exist_on_disk(bank_dir, bank_rec):
        try:
            from simulate import restore_env_snapshot

            env_s, obs_s, mem_s, ip = restore_env_snapshot(
                bank_dir, bank_rec, restore_rng=False
            )
            template_env, template_obs = clone_env_and_obs(env_s, obs_s)
            template_mem = copy.deepcopy(mem_s)
            template_initial_price = float(ip)
            restored_from_snapshot = True
        except Exception:
            restored_from_snapshot = False
            template_mem = None
            template_initial_price = None

    if not restored_from_snapshot:
        env0 = foundation.make_env_instance(**env_config)
        if seed is not None:
            try:
                env0.seed(int(seed))
            except Exception:
                np.random.seed(int(seed))
                random.seed(int(seed))
        obs0 = env0.reset()
        template_env, template_obs = clone_env_and_obs(env0, obs0)

    if fake_tokenizer:
        tokenizer: Any = _FakeTokenizerForDryRun()
    else:
        from simulate_utils import get_qwen_model

        _, tokenizer = get_qwen_model()

    hist_stats = dict(DEFAULT_HIST_STATS)
    baselines, baseline_source = _resolve_macro_baselines(
        year_index=int(year_index),
        state_bank_path=state_bank_path,
        state_bank_year_index=state_bank_year_index,
        prev_year_mean_price=prev_year_mean_price,
        prev_year_mean_wage=prev_year_mean_wage,
        prev_year_nominal_gdp=prev_year_nominal_gdp,
        prev_year_real_gdp=prev_year_real_gdp,
    )
    baseline_source = dict(baseline_source)
    baseline_source["restored_from_snapshot"] = bool(restored_from_snapshot)
    baseline_source["state_bank_restore_year_index"] = int(bank_y) if state_bank_path else None

    replica_entries: List[Dict[str, Any]] = []
    group_rewards: List[float] = []

    for replica_id in range(int(num_replicas)):
        if seed is not None:
            np.random.seed(int(seed) + replica_id * 10_007)
            random.seed(int(seed) + replica_id * 10_007)

        env_r, obs_r = clone_env_and_obs(template_env, template_obs)
        mem_arg = copy.deepcopy(template_mem) if template_mem is not None else None

        rollout_out = run_annual_rollout(
            env_r,
            obs_r,
            episode_length=episode_length,
            io_log_dir=io_log_dir,
            checkpoint_dir=checkpoint_dir,
            memory_states=mem_arg,
            initial_price_for_stats=template_initial_price,
        )

        env_stats = rollout_out["env_stats"]
        episode_samples = rollout_out["episode_samples"]
        cur_rows = env_stats.get("monthly_series") or []

        metrics = build_annual_metrics(
            env_stats,
            episode_samples,
            cur_rows=list(cur_rows),
            prev_rows=None,
            prev_year_mean_price=baselines["prev_year_mean_price"],
            prev_year_mean_wage=baselines["prev_year_mean_wage"],
            prev_year_nominal_gdp=baselines["prev_year_nominal_gdp"],
            prev_year_real_gdp=baselines["prev_year_real_gdp"],
        )
        metrics.pop("_debug_n_months", None)
        metrics.pop("_debug_n_agents", None)
        metrics.pop("_resolved_prev_year_mean_price", None)
        metrics.pop("_resolved_prev_year_mean_wage", None)

        reward_out = compute_annual_reward(
            metrics, hist_stats, year_idx=int(year_index)
        )
        R_r = float(reward_out["annual_reward"])
        group_rewards.append(R_r)

        samples = export_grpo_samples(
            rollout_out,
            tokenizer,
            replica_id=replica_id,
            episode_id=int(episode_id),
            state_group_id=f"y{int(year_index)}",
        )
        len_stats = summarize_grpo_sample_lengths(samples)

        replica_entries.append(
            {
                "replica_id": replica_id,
                "annual_reward": R_r,
                "advantage": None,
                "reward_breakdown": dict(reward_out["breakdown"]),
                "metrics_used": dict(metrics),
                "baseline_source": dict(baseline_source),
                "num_samples": len(samples),
                "samples": samples,
                "env_stats": env_stats,
                "grpo_length_stats": len_stats,
            }
        )

    advantages = compute_group_advantages(group_rewards)
    for i, rep in enumerate(replica_entries):
        adv = float(advantages[i]) if i < len(advantages) else 0.0
        rep["advantage"] = adv
        R_i = float(rep["annual_reward"])
        for s in rep["samples"]:
            s["annual_reward"] = R_i
            s["advantage"] = adv

    replicas_full = [
        {
            "replica_id": r["replica_id"],
            "annual_reward": r["annual_reward"],
            "advantage": r["advantage"],
            "reward_breakdown": r["reward_breakdown"],
            "metrics_used": r["metrics_used"],
            "num_samples": r["num_samples"],
            "samples": r["samples"],
        }
        for r in replica_entries
    ]
    group_batch: Dict[str, Any] = {
        "year_index": int(year_index),
        "group_rewards": list(group_rewards),
        "group_advantages": [float(a) for a in advantages],
        "replicas": [
            {
                "replica_id": r["replica_id"],
                "annual_reward": r["annual_reward"],
                "advantage": r["advantage"],
                "samples": r["samples"],
            }
            for r in replica_entries
        ],
    }

    out: Dict[str, Any] = {
        "group_batch": group_batch,
        "group_rewards": list(group_rewards),
        "group_advantages": [float(a) for a in advantages],
        "replicas": replicas_full,
        "policy_model_save": policy_model_save,
        "year_index": int(year_index),
        "baseline_source": baseline_source,
    }

    if persist_artifacts:
        out["artifact_paths"] = save_multi_replica_artifacts(
            base_save_path,
            policy_model_save,
            replica_entries,
            out,
            baseline_source=baseline_source,
        )
    else:
        out["artifact_paths"] = {}
    out["save_dir"] = os.path.join(
        os.path.abspath(base_save_path), "data", policy_model_save, "multi_replica"
    )
    return out


def run_multi_replica_group_live(
    *,
    actor_rollout_wg: Any,
    tokenizer: Any,
    state_bank_record: Mapping[str, Any],
    bank_base_dir: str,
    year_index: int,
    replica_group_size: int = 4,
    num_agents: int = 100,
    num_months: int = 12,
    episode_id: int = 0,
    state_group_id: Optional[str] = None,
    econ_project_root: Optional[str] = None,
    state_bank_path: Optional[str] = None,
    qwen_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    VERL-facing live rollout: same ``s_y`` (from ``state_bank_record`` + ``bank_base_dir``),
    ``replica_group_size`` replicas, full year of monthly batched generation via
    ``actor_rollout_wg``.

    Returns the Econ group payload (rewards, replicas metadata) plus the trainer
    ``DataProto`` under key ``dataproto`` (caller runs ``compute_log_prob`` / union).
    """
    import torch
    from verl import DataProto

    from search_r1.llm_agent.econ_generation_manager import (
        EconGenerationConfig,
        EconGenerationManager,
    )

    root = econ_project_root or os.path.abspath(os.path.dirname(__file__))
    sg = str(state_group_id or f"y{int(year_index)}")
    rec = dict(state_bank_record)
    cfg = EconGenerationConfig(
        num_months=int(num_months),
        num_agents=int(num_agents),
        replica_group_size=int(replica_group_size),
        econ_project_root=root,
        state_bank_path=state_bank_path,
        default_year_index=int(year_index),
        state_bank_year_index=int(rec.get("year_index", year_index)),
        qwen_config_path=qwen_config_path,
        inline_state_bank_record=rec,
        inline_bank_base_dir=str(bank_base_dir),
    )
    mgr = EconGenerationManager(tokenizer, actor_rollout_wg, cfg)
    L = 8
    gen_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.zeros(1, L, dtype=torch.long),
            "attention_mask": torch.ones(1, L, dtype=torch.long),
            "position_ids": torch.arange(L).unsqueeze(0).long(),
        },
        non_tensors={
            "year_index": np.array([int(year_index)], dtype=object),
            "state_group_id": np.array([sg], dtype=object),
            "episode_id": np.array([int(episode_id)], dtype=object),
        },
        meta_info={
            "temperature": 1.0,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": False,
        },
    )
    out_dp = mgr.run_econ_year_loop(
        meta_batch=gen_batch,
        temperature=float(gen_batch.meta_info.get("temperature", 1.0)),
    )
    payload = dict(out_dp.meta_info.get("econ_group_payload", {}))
    payload["dataproto"] = out_dp
    payload["state_group_id"] = sg
    return payload


def save_multi_replica_artifacts(
    base_save_path: str,
    policy_model_save: str,
    replica_entries: Sequence[Mapping[str, Any]],
    group_payload: Mapping[str, Any],
    baseline_source: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    root = os.path.join(
        os.path.abspath(base_save_path), "data", policy_model_save, "multi_replica"
    )
    os.makedirs(root, exist_ok=True)
    paths: Dict[str, str] = {}

    summary_for_json = {
        "group_rewards": group_payload["group_rewards"],
        "group_advantages": group_payload["group_advantages"],
        "year_index": group_payload["year_index"],
        "policy_model_save": policy_model_save,
        "num_replicas": len(replica_entries),
        "replicas": [
            {
                "replica_id": r["replica_id"],
                "annual_reward": r["annual_reward"],
                "advantage": r["advantage"],
                "num_samples": r["num_samples"],
                "reward_breakdown": r["reward_breakdown"],
            }
            for r in replica_entries
        ],
    }
    if baseline_source is not None:
        summary_for_json["baseline_source"] = dict(baseline_source)
    p_sum = os.path.join(root, "group_summary.json")
    with open(p_sum, "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary_for_json), f, ensure_ascii=False, indent=2)
    paths["group_summary_json"] = p_sum

    for r in replica_entries:
        rid = int(r["replica_id"])
        pj = os.path.join(root, f"replica_{rid}_reward.json")
        with open(pj, "w", encoding="utf-8") as f:
            json.dump(
                _json_safe(
                    {
                        "replica_id": rid,
                        "annual_reward": r["annual_reward"],
                        "advantage": r["advantage"],
                        "reward_breakdown": r["reward_breakdown"],
                        "metrics_used": r["metrics_used"],
                        "baseline_source": r.get("baseline_source"),
                    }
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )
        paths[f"replica_{rid}_reward_json"] = pj

        pkl_path = os.path.join(root, f"replica_{rid}_samples.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(r["samples"], f, protocol=pickle.HIGHEST_PROTOCOL)
        paths[f"replica_{rid}_samples_pkl"] = pkl_path

        es = os.path.join(root, f"replica_{rid}_env_stats.json")
        with open(es, "w", encoding="utf-8") as f:
            json.dump(_json_safe(r["env_stats"]), f, ensure_ascii=False, indent=2)
        paths[f"replica_{rid}_env_stats_json"] = es

        gl = os.path.join(root, f"replica_{rid}_grpo_length_stats.json")
        with open(gl, "w", encoding="utf-8") as f:
            json.dump(_json_safe(r["grpo_length_stats"]), f, ensure_ascii=False, indent=2)
        paths[f"replica_{rid}_grpo_length_stats_json"] = gl

    return paths


def main() -> int:
    p = argparse.ArgumentParser(
        description="Multi-replica GRPO group preprocessing (no VERL trainer)."
    )
    p.add_argument("--num-replicas", type=int, default=4)
    p.add_argument("--num-agents", type=int, default=10)
    p.add_argument("--episode-length", type=int, default=12)
    p.add_argument("--year-index", type=int, default=4)
    p.add_argument("--save-base", type=str, default=".")
    p.add_argument("--policy-model-save", type=str, default=None)
    p.add_argument(
        "--fake-tokenizer",
        action="store_true",
        help=(
            "Use stub tokenizer only for grpo_exporter tokenization; "
            "run_annual_rollout still calls vLLM/Qwen for actions unless you change that separately."
        ),
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prev-year-mean-price", type=float, default=None)
    p.add_argument("--prev-year-mean-wage", type=float, default=None)
    p.add_argument("--prev-year-nominal-gdp", type=float, default=None)
    p.add_argument("--prev-year-real-gdp", type=float, default=None)
    p.add_argument(
        "--state-bank",
        type=str,
        default=None,
        help="V1 macro baseline JSON from build_state_bank.py",
    )
    p.add_argument(
        "--state-bank-year-index",
        type=int,
        default=None,
        help="Row year_index in state bank (default: same as --year-index)",
    )
    p.add_argument(
        "--no-persist-artifacts",
        action="store_true",
        help="Skip writing multi_replica JSON/PKL under data/ (in-memory group_batch only).",
    )
    args = p.parse_args()

    result = run_multi_replica_group(
        num_replicas=args.num_replicas,
        num_agents=args.num_agents,
        episode_length=args.episode_length,
        year_index=args.year_index,
        base_save_path=args.save_base,
        policy_model_save=args.policy_model_save,
        fake_tokenizer=args.fake_tokenizer,
        seed=args.seed,
        prev_year_mean_price=args.prev_year_mean_price,
        prev_year_mean_wage=args.prev_year_mean_wage,
        prev_year_nominal_gdp=args.prev_year_nominal_gdp,
        prev_year_real_gdp=args.prev_year_real_gdp,
        state_bank_path=args.state_bank,
        state_bank_year_index=args.state_bank_year_index,
        persist_artifacts=not args.no_persist_artifacts,
    )

    print("group_rewards:", result["group_rewards"])
    print("group_advantages:", result["group_advantages"])
    print("baseline_source:", json.dumps(_json_safe(result.get("baseline_source", {})), ensure_ascii=False))
    for rep in result["replicas"]:
        print(
            f"replica {rep['replica_id']}: num_samples={rep['num_samples']} "
            f"annual_reward={rep['annual_reward']}"
        )
        print(
            f"  reward_breakdown:",
            json.dumps(_json_safe(rep["reward_breakdown"]), ensure_ascii=False),
        )
    print("save_dir:", result["save_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
