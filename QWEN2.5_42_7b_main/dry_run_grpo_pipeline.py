from __future__ import annotations
import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Load rollout from disk
# ---------------------------------------------------------------------------


def load_rollout_from_dir(rollout_dir: str) -> Dict[str, Any]:
    """Load ``episode_samples.jsonl`` and ``env_stats.json`` into rollout dict."""
    rollout_dir = os.path.abspath(rollout_dir)
    stats_path = os.path.join(rollout_dir, "env_stats.json") # 宏观经济统计（每月失业率、GDP等）
    jsonl_path = os.path.join(rollout_dir, "episode_samples.jsonl") #每个 agent 每步的决策记录（每行一条）

    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"Missing env_stats.json: {stats_path}")
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Missing episode_samples.jsonl: {jsonl_path}")

    with open(stats_path, encoding="utf-8") as f:
        env_stats = json.load(f)

    episode_samples: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode_samples.append(json.loads(line))

    return {
        "episode_samples": episode_samples,
        "env_stats": env_stats,
    }


def infer_policy_model_save_from_rollout_dir(rollout_dir: str) -> str:
    """.../data/<policy_model_save>/rollout → <policy_model_save>."""
    rollout_dir = os.path.abspath(rollout_dir)
    parent = os.path.basename(os.path.dirname(rollout_dir))
    if parent == "data" or not parent:
        return "unknown_rollout"
    return parent

# 遍历所有样本，找最大的 agent_id，加 1 就是 agent 总数
def infer_num_agents(episode_samples: Sequence[Dict[str, Any]]) -> int:
    if not episode_samples:
        return 1
    return int(max(int(s.get("agent_id", 0)) for s in episode_samples)) + 1


# ---------------------------------------------------------------------------
# Annual metrics from env_stats (ACL24 / rollout_collector field names)
# ---------------------------------------------------------------------------

# 从 env_stats 提取月度数据
def _sorted_monthly_rows(env_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = env_stats.get("monthly_series") or []
    return sorted(rows, key=lambda r: int(r.get("month", 0)))


def _rows_by_month_index(rows: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(r["month"]): r for r in rows if "month" in r}


def select_annual_windows(
    env_stats: Dict[str, Any],
    window: str,
    window_start: Optional[int],
    window_length: int,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """
    Return (current_year_rows, previous_year_rows or None).

    ``window``:
      - ``last``: last ``window_length`` months in series
      - ``first``: first ``window_length`` months
      - ``manual``: months [window_start, window_start + window_length)
    """
    rows = _sorted_monthly_rows(env_stats)
    if not rows:
        return [], None

    n = len(rows)
    if window == "last":
        cur = rows[max(0, n - window_length) :]
    elif window == "first":
        cur = rows[: min(n, window_length)]
    elif window == "manual":
        if window_start is None:
            raise ValueError("--window-start required when --window manual")
        by_m = _rows_by_month_index(rows)
        cur = []
        for m in range(window_start, window_start + window_length):
            if m in by_m:
                cur.append(by_m[m])
    else:
        raise ValueError(f"unknown window mode: {window}")

    if len(cur) < window_length:
        pass  # still compute with shorter window

    # Previous calendar year: the ``window_length`` months immediately before ``cur[0]``
    prev: Optional[List[Dict[str, Any]]] = None
    if cur:
        m0 = int(cur[0]["month"])
        by_m = _rows_by_month_index(rows)
        prev_list: List[Dict[str, Any]] = []
        for m in range(m0 - window_length, m0):
            if m in by_m:
                prev_list.append(by_m[m])
        if len(prev_list) == window_length:
            prev = prev_list
        elif len(prev_list) > 0:
            prev = prev_list  # partial — still use for growth if sum > 0

    return cur, prev

# 从窗口的起始月份推断年份索引
def infer_year_index_from_window(cur_rows: Sequence[Dict[str, Any]]) -> int:
    """Map window start month (0-based env timestep) to 1-based training year index y."""
    if not cur_rows:
        return 1
    start_month = int(cur_rows[0]["month"])
    return start_month // 12 + 1

# 确保浮点数为正数
def _coerce_positive_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 1e-12:
        return None
    return v

# 解析前一年的平均价格
def resolve_prev_year_mean_price(
    env_stats: Dict[str, Any],
    prev_rows: Optional[Sequence[Dict[str, Any]]],
    cli_override: Optional[float],
) -> Optional[float]:
    """
    Prior-year **average** goods price for YoY inflation (state bank or long rollout).

    Precedence: ``cli_override`` → ``env_stats['prev_year_mean_price']`` →
    mean(``monthly_price``) over ``prev_rows`` when available.
    """
    v = _coerce_positive_float(cli_override)
    if v is not None:
        return v
    v = _coerce_positive_float(env_stats.get("prev_year_mean_price"))
    if v is not None:
        return v
    if prev_rows:
        pp = [
            float(r["monthly_price"])
            for r in prev_rows
            if r.get("monthly_price") is not None
        ]
        if pp:
            m = float(np.mean(pp))
            if m > 1e-12:
                return m
    return None

# 解析前一年的平均工资
def resolve_prev_year_mean_wage(
    env_stats: Dict[str, Any],
    prev_rows: Optional[Sequence[Dict[str, Any]]],
    cli_override: Optional[float],
) -> Optional[float]:
    """Same contract as price, using ``monthly_mean_wage`` / ``prev_year_mean_wage``."""
    v = _coerce_positive_float(cli_override)
    if v is not None:
        return v
    v = _coerce_positive_float(env_stats.get("prev_year_mean_wage"))
    if v is not None:
        return v
    if prev_rows:
        ww = [
            float(r["monthly_mean_wage"])
            for r in prev_rows
            if r.get("monthly_mean_wage") is not None
        ]
        if ww:
            m = float(np.mean(ww))
            if m > 1e-12:
                return m
    return None

# 构建年度指标
def build_annual_metrics(
    env_stats: Dict[str, Any],
    episode_samples: Sequence[Dict[str, Any]],
    *,
    window: str = "last",
    window_start: Optional[int] = None,
    window_length: int = 12,
    num_agents: Optional[int] = None,
    cur_rows: Optional[List[Dict[str, Any]]] = None,
    prev_rows: Optional[List[Dict[str, Any]]] = None,
    prev_year_mean_price: Optional[float] = None,
    prev_year_mean_wage: Optional[float] = None,
    prev_year_nominal_gdp: Optional[float] = None,
    prev_year_real_gdp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build ``reward_builder.compute_annual_reward`` metrics dict.

    Unemployment: mean monthly ``monthly_unemployment_rate`` (0–1) → **percent**.
    GDP growth: YoY on sums of ``monthly_*_gdp_proxy`` (matches thought.txt
    construction via S_t and prices in rollout_collector). If
    ``prev_year_nominal_gdp`` / ``prev_year_real_gdp`` are set (e.g. from V1 state
    bank), they override sums from ``prev_rows``.
    **Price / wage inflation (YoY, preferred):**
    ``(mean(current window monthly_price) / P_prev - 1) * 100`` where ``P_prev`` is
    ``prev_year_mean_price`` from the state bank, CLI override, or mean price in
    ``prev_rows``. Same for wages. **Fallback** (no baseline): intra-window
    (last/first month) % change — only for diagnostics when YoY is unavailable.

    Micro rates: sums of per-month failure counts / (num_agents * n_months).

    If ``cur_rows`` is provided (and typically ``prev_rows`` from the same
    ``select_annual_windows`` call), those slices are used; otherwise windows
    are selected from ``env_stats`` via ``window`` / ``window_start`` /
    ``window_length``.
    """
    n_agents = int(num_agents) if num_agents is not None else infer_num_agents(episode_samples)
    # 如果提供了当前窗口的行，则使用提供的行
    if cur_rows is not None:
        cur = cur_rows
        prev = prev_rows
    else:
        cur, prev = select_annual_windows(env_stats, window, window_start, window_length)
    if not cur:
        raise ValueError("empty monthly_series / window")
    # 计算当前窗口的月份数
    n_m = len(cur)
    u_fracs = [float(r.get("monthly_unemployment_rate", 0.0)) for r in cur]
    annual_unemployment = float(np.mean(u_fracs) * 100.0)
    # 计算当前窗口的就业率
    work_rates = [float(r.get("monthly_work_rate", 0.0)) for r in cur]
    max_monthly_work_rate = float(max(work_rates)) if work_rates else 0.0
    # 计算当前窗口的格式错误统计
    fmt_sum = sum(int(r.get("month_format_fail_count", 0)) for r in cur)
    # 计算当前窗口的无效动作统计
    inv_sum = sum(int(r.get("month_invalid_action_count", 0)) for r in cur)
    # 计算当前窗口的超出范围统计
    oor_sum = sum(int(r.get("month_out_of_range_count", 0)) for r in cur)
    # 计算当前窗口的负财富统计
    nw_sum = sum(int(r.get("month_negative_wealth_count", 0)) for r in cur)
    # 计算当前窗口的统计分母
    denom = float(max(1, n_agents * n_m))
    # 计算当前窗口的格式错误率
    format_fail_rate = fmt_sum / denom
    # 计算当前窗口的无效动作率
    parser_invalid_action_rate = inv_sum / denom
    out_of_range_rate = oor_sum / denom
    negative_wealth_rate = nw_sum / denom
    has_action_decomp = any(
        "month_out_of_range_count" in r or "month_negative_wealth_count" in r
        for r in cur
    )
    invalid_action_rate = None if has_action_decomp else parser_invalid_action_rate

    sum_nom_cur = float(sum(float(r.get("monthly_nominal_gdp_proxy", 0.0)) for r in cur))
    sum_real_cur = float(sum(float(r.get("monthly_real_gdp_proxy", 0.0)) for r in cur))

    annual_nominal_gdp_growth: Optional[float] = None
    annual_real_gdp_growth: Optional[float] = None
    # 如果提供了前一年的名义GDP，则使用提供的名义GDP
    if prev_year_nominal_gdp is not None and float(prev_year_nominal_gdp) > 1e-9:
        annual_nominal_gdp_growth = (
            (sum_nom_cur - float(prev_year_nominal_gdp)) / float(prev_year_nominal_gdp) * 100.0
        )
    # 如果提供了前一年的名义GDP，则使用提供的名义GDP
    elif prev and len(prev) > 0:
        # 计算前一年的名义GDP
        sum_nom_prev = float(sum(float(r.get("monthly_nominal_gdp_proxy", 0.0)) for r in prev))
        # 如果前一年的名义GDP大于1e-9，则计算名义GDP增长率
        if sum_nom_prev > 1e-9:
            # 计算名义GDP增长率
            annual_nominal_gdp_growth = (sum_nom_cur - sum_nom_prev) / sum_nom_prev * 100.0

    # 如果提供了前一年的实际GDP，则使用提供的实际GDP
    if prev_year_real_gdp is not None and float(prev_year_real_gdp) > 1e-9:
        # 计算实际GDP增长率
        annual_real_gdp_growth = (
            (sum_real_cur - float(prev_year_real_gdp)) / float(prev_year_real_gdp) * 100.0
        )
    # 如果提供了前一年的实际GDP，则使用提供的实际GDP    
    elif prev and len(prev) > 0:
        sum_real_prev = float(sum(float(r.get("monthly_real_gdp_proxy", 0.0)) for r in prev))
        if sum_real_prev > 1e-9:
            # 计算实际GDP增长率
            annual_real_gdp_growth = (sum_real_cur - sum_real_prev) / sum_real_prev * 100.0
    # 计算当前窗口的平均价格
    prices = [float(r.get("monthly_price", 0.0)) for r in cur if r.get("monthly_price") is not None]
    # 计算当前窗口的平均工资
    wages = [float(r.get("monthly_mean_wage", 0.0)) for r in cur if r.get("monthly_mean_wage") is not None]

    # 解析前一年的平均价格
    p_prev = resolve_prev_year_mean_price(env_stats, prev, prev_year_mean_price)
    # 解析前一年的平均工资
    w_prev = resolve_prev_year_mean_wage(env_stats, prev, prev_year_mean_wage)

    # 计算当前窗口的平均价格
    mean_p = float(np.mean(prices)) if prices else None
    # 计算当前窗口的平均工资
    mean_w = float(np.mean(wages)) if wages else None

    # 计算年度价格通胀
    annual_price_inflation: Optional[float] = None
    # 如果提供了前一年的平均价格，则使用前一年的平均价格
    if mean_p is not None and p_prev is not None:
        annual_price_inflation = (mean_p / p_prev - 1.0) * 100.0
    elif len(prices) >= 2 and prices[0] > 1e-12:
        annual_price_inflation = (prices[-1] / prices[0] - 1.0) * 100.0

    annual_wage_inflation: Optional[float] = None
    if mean_w is not None and w_prev is not None:
        annual_wage_inflation = (mean_w / w_prev - 1.0) * 100.0
    elif len(wages) >= 2 and wages[0] > 1e-12:
        annual_wage_inflation = (wages[-1] / wages[0] - 1.0) * 100.0

    return {
        "annual_unemployment": annual_unemployment,
        "annual_real_gdp_growth": annual_real_gdp_growth,
        "annual_nominal_gdp_growth": annual_nominal_gdp_growth,
        "annual_price_inflation": annual_price_inflation,
        "annual_wage_inflation": annual_wage_inflation,
        "max_monthly_work_rate": max_monthly_work_rate,
        "format_fail_rate": format_fail_rate,
        "invalid_action_rate": invalid_action_rate,
        "parser_invalid_action_rate": parser_invalid_action_rate,
        "out_of_range_rate": out_of_range_rate,
        "negative_wealth_rate": negative_wealth_rate,
        "budget_violation_rate": 0.0,
        "_debug_n_months": n_m,
        "_debug_n_agents": n_agents,
        "_resolved_prev_year_mean_price": p_prev,
        "_resolved_prev_year_mean_wage": w_prev,
    }


# ---------------------------------------------------------------------------
# Fake tokenizer (no vLLM)
# ---------------------------------------------------------------------------

# 假的tokenizer
class _FakeTokenizerForDryRun:
    """Matches grpo_exporter contract: apply_chat_template + encode."""

    eos_token_id = 999

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        return "<|user|>" + messages[0]["content"] + "<|assistant|>"

    def encode(self, text, add_special_tokens=False):
        step = max(1, len(text) // 40)
        return [10 + i for i in range(0, min(len(text), 120), step)]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# 安全的JSON序列化
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

# 运行实时rollout
def run_live_rollout(num_agents: int, episode_length: int) -> Tuple[Dict[str, Any], str]:
    """Run ``run_annual_rollout``; return (output, policy_model_save)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "config.yaml")
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

    import ai_economist.foundation as foundation
    from simulate import run_annual_rollout

    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    rollout_out = run_annual_rollout(
        env,
        obs,
        episode_length=episode_length,
        io_log_dir=None,
        checkpoint_dir=None,
    )
    policy_model_save = f"qwen_rollout-{num_agents}agents-{episode_length}months"
    return rollout_out, policy_model_save

# 主函数
def main() -> int:
    p = argparse.ArgumentParser(description="Dry-run GRPO preprocessing pipeline.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--rollout-dir",
        type=str,
        help="Path to .../data/<run>/rollout (contains env_stats.json, episode_samples.jsonl)",
    )
    g.add_argument(
        "--run-rollout",
        action="store_true",
        help="Run a minimal qwen_rollout via simulate.run_annual_rollout",
    )
    p.add_argument("--num-agents", type=int, default=10)
    p.add_argument("--episode-length", type=int, default=12)
    p.add_argument(
        "--save-base",
        type=str,
        default=".",
        help="Project-relative base; artifacts under save-base/data/<run>/dry_run/",
    )
    p.add_argument(
        "--year-index",
        type=int,
        default=None,
        help=(
            "Training year index y for reward gating; if omitted, infer from selected window "
            "(start_month // 12 + 1)"
        ),
    )
    p.add_argument(
        "--window",
        choices=("last", "first", "manual"),
        default="last",
        help="Which 12-month slice of monthly_series to use for annual metrics",
    )
    p.add_argument("--window-start", type=int, default=None, help="With --window manual: first month index")
    p.add_argument("--window-length", type=int, default=12, help="Months per annual window")
    p.add_argument("--replica-id", type=int, default=0)
    p.add_argument("--episode-id", type=int, default=0)
    p.add_argument(
        "--fake-tokenizer",
        action="store_true",
        help="Do not load vLLM; use stub tokenizer for exporter smoke test",
    )
    p.add_argument(
        "--prev-year-mean-price",
        type=float,
        default=None,
        help=(
            "Override prior-year mean goods price for YoY inflation (else "
            "env_stats['prev_year_mean_price'] or mean of prev window)"
        ),
    )
    p.add_argument(
        "--prev-year-mean-wage",
        type=float,
        default=None,
        help="Override prior-year mean monthly_mean_wage for YoY wage inflation",
    )
    p.add_argument("--prev-year-nominal-gdp", type=float, default=None)
    p.add_argument("--prev-year-real-gdp", type=float, default=None)
    args = p.parse_args()
    save_base = os.path.abspath(args.save_base)

    if args.run_rollout:
        print("Running live rollout (Qwen vLLM)...", flush=True)
        rollout_out, policy_model_save = run_live_rollout(
            args.num_agents, args.episode_length
        )
    else:
        rollout_out = load_rollout_from_dir(args.rollout_dir)
        policy_model_save = infer_policy_model_save_from_rollout_dir(args.rollout_dir)

    env_stats = rollout_out["env_stats"]
    episode_samples = rollout_out["episode_samples"]

    cur_rows, prev_rows = select_annual_windows(
        env_stats,
        args.window,
        args.window_start,
        args.window_length,
    )
    if not cur_rows:
        raise ValueError("empty monthly_series / window")

    year_idx = args.year_index
    if year_idx is None:
        year_idx = infer_year_index_from_window(cur_rows)

    metrics = build_annual_metrics(
        env_stats,
        episode_samples,
        window=args.window,
        window_start=args.window_start,
        window_length=args.window_length,
        num_agents=None,
        cur_rows=cur_rows,
        prev_rows=prev_rows,
        prev_year_mean_price=args.prev_year_mean_price,
        prev_year_mean_wage=args.prev_year_mean_wage,
        prev_year_nominal_gdp=args.prev_year_nominal_gdp,
        prev_year_real_gdp=args.prev_year_real_gdp,
    )
    # Drop debug keys before reward
    debug_n_m = metrics.pop("_debug_n_months", None)
    debug_n_a = metrics.pop("_debug_n_agents", None)
    baseline_price = metrics.pop("_resolved_prev_year_mean_price", None)
    baseline_wage = metrics.pop("_resolved_prev_year_mean_wage", None)

    from reward_builder import DEFAULT_HIST_STATS, compute_annual_reward

    hist_stats = dict(DEFAULT_HIST_STATS)
    reward_out = compute_annual_reward(metrics, hist_stats, year_idx=year_idx)
    breakdown = reward_out["breakdown"]

    if args.fake_tokenizer:
        tokenizer: Any = _FakeTokenizerForDryRun()
    else:
        from simulate_utils import get_qwen_model

        _, tokenizer = get_qwen_model()

    from grpo_exporter import export_grpo_samples, summarize_grpo_sample_lengths

    grpo_samples = export_grpo_samples(
        rollout_out,
        tokenizer,
        replica_id=args.replica_id,
        episode_id=args.episode_id,
        state_group_id="dryrun",
    )
    length_stats = summarize_grpo_sample_lengths(grpo_samples)

    dry_root = os.path.join(save_base, "data", policy_model_save, "dry_run")
    os.makedirs(dry_root, exist_ok=True)

    summary = {
        "annual_reward": reward_out["annual_reward"],
        "reward_breakdown": breakdown,
        "R_macro": reward_out.get("R_macro"),
        "R_micro": reward_out.get("R_micro"),
        "num_samples": len(grpo_samples),
        "length_stats": length_stats,
        "metrics_used": metrics,
        "year_index": year_idx,
        "window": args.window,
        "window_length": args.window_length,
        "policy_model_save": policy_model_save,
        "debug_n_months_in_window": debug_n_m,
        "debug_n_agents": debug_n_a,
        "inflation_baselines": {
            "prev_year_mean_price": baseline_price,
            "prev_year_mean_wage": baseline_wage,
        },
    }

    with open(os.path.join(dry_root, "dry_run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
    with open(os.path.join(dry_root, "reward_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(breakdown), f, ensure_ascii=False, indent=2)
    with open(os.path.join(dry_root, "grpo_length_stats.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(length_stats), f, ensure_ascii=False, indent=2)

    # Console output (required)
    print("--- dry_run_grpo_pipeline ---")
    print("annual_reward:", reward_out["annual_reward"])
    print("reward_breakdown:", json.dumps(_json_safe(breakdown), ensure_ascii=False))
    print("num_samples:", len(grpo_samples))
    it = length_stats.get("input_token_len", {})
    print("max_input_len:", it.get("max"))
    print("mean_input_len:", it.get("mean"))
    print("invalid_format_ratio:", length_stats.get("invalid_format_ratio"))
    print("fallback_ratio:", length_stats.get("fallback_ratio"))
    print("artifacts:", dry_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
