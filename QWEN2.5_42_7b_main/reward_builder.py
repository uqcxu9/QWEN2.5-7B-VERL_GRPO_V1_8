# -*- coding: utf-8 -*-
"""
Replica-level annual training reward for EconAgent GRPO.

Specification: ``thought.txt`` (EconAgent GRPO: Reward Design). This module
implements R_r = R_macro_r + R_micro_r with year gating; it does **not**
include Phillips/Okun, entropy, smoothness, or legacy Gate A logic.

Macro metrics (unemployment, GDP growth, inflations) should be computed
upstream with the same definitions as in the ACL24 / EconAgent environment
documentation (e.g. annual aggregates from monthly env trajectories, GDP
from S_t P_t with real GDP deflated by P_0 = env.world.price[0]).

**Training episode (12-month GRPO rollouts)**  
Each update uses a restored state ``s_y`` and only **twelve** monthly steps are
on-trajectory. YoY **price** (and wage) inflation should then be supplied as
**percent change** vs a stored prior-year baseline, e.g.  
``(mean(P_t over the 12 months) / prev_year_mean_price - 1) * 100``, where
``prev_year_mean_price`` is persisted in the state bank with ``s_y`` (same
price concept as ``env.world.price`` / ``monthly_price`` in rollout logs).
Wage inflation analogously uses a ``prev_year_mean_wage`` if available.
This module only consumes the resulting ``annual_price_inflation`` /
``annual_wage_inflation`` floats.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# US annual series (same source arrays as legacy reward.py) — used only to
# populate default historical moments for unemployment / CPI-style inflation /
# wage inflation. GDP growth defaults are separate scalars (see below).
# ---------------------------------------------------------------------------
_REAL_US_UNEMPLOYMENT = np.array(
    [
        4.675,
        5.783333333,
        5.575,
        5.083333333,
        4.608333333,
        4.616666666,
        5.8,
        9.283333333,
        9.608333333,
        8.933333333,
        8.075,
        7.358333333,
        6.158333333,
        5.275,
        4.875,
        4.358333333,
        3.891666667,
        3.666666667,
        8.091666667,
    ],
    dtype=float,
)
_REAL_US_INFLATION = np.array(
    [
        1.586031627,
        2.270094973,
        2.677236693,
        3.392746845,
        3.225944101,
        2.852672482,
        3.839100297,
        -0.355546266,
        1.640043442,
        3.156841569,
        2.069337265,
        1.464832656,
        1.622222977,
        0.118627136,
        1.261583206,
        2.130110004,
        2.442583297,
        1.812210075,
        1.233584396,
    ],
    dtype=float,
)
_REAL_US_WAGE_INFLATION = np.array(
    [
        2.912176,
        2.673797,
        2.105035,
        2.752391,
        3.873203,
        3.992632,
        3.777108,
        2.993819,
        2.396202,
        2.025195,
        1.474812,
        2.070218,
        2.301420,
        2.087801,
        2.441441,
        2.340697,
        2.982761,
        3.546125,
        5.012940,
    ],
    dtype=float,
)

# 真实 FRED 数据（2002-2020）
_GR = np.array([
    1.7, 2.8, 3.8, 3.5, 2.8, 2.0, 0.1, -2.6,
    2.7, 1.6, 2.3, 2.1, 2.5, 2.9, 1.8, 2.5, 3.0, 2.6, -2.1
], dtype=float)  # FRED A191RL1A225NBEA

_GN = np.array([
    3.3, 4.8, 6.6, 6.7, 6.0, 4.8, 2.0, -2.0,
    3.9, 3.7, 4.2, 3.9, 4.3, 3.9, 2.8, 4.3, 5.3, 4.3, -0.8
], dtype=float)  # FRED A191RP1A027NBEA

def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(x, ddof=0))
    return s if s > 1e-12 else 1e-6


DEFAULT_HIST_STATS: Dict[str, Any] = {
    "mu_u": float(np.mean(_REAL_US_UNEMPLOYMENT)),
    "sigma_u": _safe_std(_REAL_US_UNEMPLOYMENT),
    "u_min": 3.5,
    "u_max": 14.8,
    "mu_gr": float(np.mean(_GR)),
    "sigma_gr": _safe_std(_GR),
    "gr_min": float(np.min(_GR)),
    "gr_max": float(np.max(_GR)),
    "mu_gn": float(np.mean(_GN)),
    "sigma_gn": _safe_std(_GN),
    "gn_min": float(np.min(_GN)),
    "gn_max": float(np.max(_GN)),
    "mu_pi": float(np.mean(_REAL_US_INFLATION)),
    "sigma_pi": _safe_std(_REAL_US_INFLATION),
    "pi_min": float(np.min(_REAL_US_INFLATION)),
    "pi_max": float(np.max(_REAL_US_INFLATION)),
    "mu_piw": float(np.mean(_REAL_US_WAGE_INFLATION)),
    "sigma_piw": _safe_std(_REAL_US_WAGE_INFLATION),
    "piw_min": float(np.min(_REAL_US_WAGE_INFLATION)),
    "piw_max": float(np.max(_REAL_US_WAGE_INFLATION)),
    "c_max": 0.965,
    "tau_fmt": 0.001,
    "tau_valid": 0.001,
    # Spec names λ for GDP / inflation / work; unemployment λ not tabulated — default 1.0
    "lambda_u": 1.0,
    "lambda_gr": 0.25,
    "lambda_gn": 0.25,
    "lambda_pi": 0.5,
    "lambda_piw": 0.25,
    "lambda_work": 0.25,
}


def huber(x: float, delta: float = 1.0) -> float:
    """Huber H_delta(x): quadratic inside [|x|<=delta], linear outside."""
    ax = abs(float(x))
    d = float(delta)
    if ax <= d:
        return 0.5 * x * x
    return d * (ax - 0.5 * d)


def zscore(x: Optional[float], mu: float, sigma: float, eps: float = 1e-8) -> float:
    """(x - mu) / (sigma + eps). Non-finite or None -> 0.0 (no z-score signal)."""
    if x is None:
        return 0.0
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(xf):
        return 0.0
    return (xf - float(mu)) / (float(sigma) + float(eps))


def guardrail_penalty(
    x: Optional[float], low: float, high: float, lam: float
) -> float:
    """
    Stability guardrail matching thought.txt:
    -lam * [max(0, x - high) + max(0, low - x)]
    """
    if x is None:
        return 0.0
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(xf):
        return 0.0
    low = float(low)
    high = float(high)
    lam = float(lam)
    excess = max(0.0, xf - high) + max(0.0, low - xf)
    return -lam * excess


def compute_unemployment_terms(
    metrics: Dict[str, Any],
    hist_stats: Dict[str, Any],
    year_idx: int = 1,
) -> Dict[str, float]:
    """r^u,center and r^u,guard; active for year_idx >= 1 (default year_idx=1)."""
    if year_idx < 1:
        return {"u_center": 0.0, "u_guard": 0.0}

    u = metrics.get("annual_unemployment")
    if u is None or (isinstance(u, float) and not math.isfinite(u)):
        return {"u_center": 0.0, "u_guard": 0.0}

    u = float(u)
    z_u = zscore(u, hist_stats["mu_u"], hist_stats["sigma_u"])
    u_center = -1.0 * huber(z_u, 1.0)
    u_guard = guardrail_penalty(
        u, hist_stats["u_min"], hist_stats["u_max"], hist_stats["lambda_u"]
    )
    return {"u_center": float(u_center), "u_guard": float(u_guard)}


def compute_gdp_terms(
    metrics: Dict[str, Any], hist_stats: Dict[str, Any], year_idx: int
) -> Dict[str, float]:
    """Real / nominal GDP growth center (H_1 z-score) + guard; active for year_idx >= 2."""
    out = {
        "real_gdp_center": 0.0,
        "real_gdp_guard": 0.0,
        "nominal_gdp_center": 0.0,
        "nominal_gdp_guard": 0.0,
    }
    if year_idx < 2:
        return out

    gr = metrics.get("annual_real_gdp_growth")
    gn = metrics.get("annual_nominal_gdp_growth")

    if gr is not None and isinstance(gr, (int, float)) and math.isfinite(float(gr)):
        gr = float(gr)
        z = zscore(gr, hist_stats["mu_gr"], hist_stats["sigma_gr"])
        out["real_gdp_center"] = -1.0 * huber(z, 1.0)
        out["real_gdp_guard"] = guardrail_penalty(
            gr, hist_stats["gr_min"], hist_stats["gr_max"], hist_stats["lambda_gr"]
        )

    if gn is not None and isinstance(gn, (int, float)) and math.isfinite(float(gn)):
        gn = float(gn)
        z = zscore(gn, hist_stats["mu_gn"], hist_stats["sigma_gn"])
        out["nominal_gdp_center"] = -1.0 * huber(z, 1.0)
        out["nominal_gdp_guard"] = guardrail_penalty(
            gn, hist_stats["gn_min"], hist_stats["gn_max"], hist_stats["lambda_gn"]
        )

    return {k: float(v) for k, v in out.items()}


def compute_inflation_terms(
    metrics: Dict[str, Any], hist_stats: Dict[str, Any], year_idx: int
) -> Dict[str, float]:
    """Price and wage inflation center + guard; active for year_idx >= 4."""
    out = {
        "pi_center": 0.0,
        "pi_guard": 0.0,
        "wage_pi_center": 0.0,
        "wage_pi_guard": 0.0,
    }
    if year_idx < 4:
        return out

    pi = metrics.get("annual_price_inflation")
    if pi is not None and isinstance(pi, (int, float)) and math.isfinite(float(pi)):
        pi = float(pi)
        z = zscore(pi, hist_stats["mu_pi"], hist_stats["sigma_pi"])
        out["pi_center"] = -1.0 * huber(z, 1.0)
        out["pi_guard"] = guardrail_penalty(
            pi, hist_stats["pi_min"], hist_stats["pi_max"], hist_stats["lambda_pi"]
        )

    piw = metrics.get("annual_wage_inflation")
    if piw is not None and isinstance(piw, (int, float)) and math.isfinite(float(piw)):
        piw = float(piw)
        z = zscore(piw, hist_stats["mu_piw"], hist_stats["sigma_piw"])
        out["wage_pi_center"] = -1.0 * huber(z, 1.0)
        out["wage_pi_guard"] = guardrail_penalty(
            piw,
            hist_stats["piw_min"],
            hist_stats["piw_max"],
            hist_stats["lambda_piw"],
        )

    return {k: float(v) for k, v in out.items()}


def compute_work_guard(
    metrics: Dict[str, Any],
    hist_stats: Dict[str, Any],
    year_idx: int = 1,
) -> Dict[str, float]:
    """
    One-sided barrier on max monthly work rate c_r:
    r_work = -lambda_work * max(0, c_r - c_max). Active for year_idx >= 1.
    """
    if year_idx < 1:
        return {"work_guard": 0.0}

    c = metrics.get("max_monthly_work_rate")
    if c is None or (isinstance(c, float) and not math.isfinite(c)):
        return {"work_guard": 0.0}

    c = float(c)
    c_max = float(hist_stats["c_max"])
    lam = float(hist_stats["lambda_work"])
    work_guard = -lam * max(0.0, c - c_max)
    return {"work_guard": float(work_guard)}


def compute_micro_terms(
    metrics: Dict[str, Any],
    hist_stats: Dict[str, Any],
    year_idx: int = 1,
) -> Dict[str, float]:
    """
    r_fmt = -2 * H_1(p_fail / tau_fmt), r_valid = -1 * H_1(p_inv / tau_valid).
    Active for year_idx >= 1.
    """
    if year_idx < 1:
        return {"fmt": 0.0, "valid": 0.0}

    tau_fmt = float(hist_stats.get("tau_fmt", 1e-3))
    tau_valid = float(hist_stats.get("tau_valid", 1e-3))

    fmt_rate = float(metrics.get("format_fail_rate", 0.0) or 0.0)
    if not math.isfinite(fmt_rate):
        fmt_rate = 0.0

    if "invalid_action_rate" in metrics and metrics["invalid_action_rate"] is not None:
        invalid_rate = float(metrics["invalid_action_rate"])
    else:
        out_of_range_rate = float(metrics.get("out_of_range_rate", 0.0))
        negative_wealth_rate = float(metrics.get("negative_wealth_rate", 0.0))
        budget_violation_rate = float(metrics.get("budget_violation_rate", 0.0))
        parser_invalid_rate = float(metrics.get("parser_invalid_action_rate", 0.0))
        invalid_rate = min(
            1.0,
            out_of_range_rate
            + negative_wealth_rate
            + budget_violation_rate
            + parser_invalid_rate,
        )
    if not math.isfinite(invalid_rate):
        invalid_rate = 0.0

    r_fmt = -2.0 * huber(fmt_rate / max(tau_fmt, 1e-12), 1.0)
    r_valid = -1.0 * huber(invalid_rate / max(tau_valid, 1e-12), 1.0)
    return {
        "fmt": float(r_fmt),
        "valid": float(r_valid),
        "format_fail_rate": float(fmt_rate),
        "invalid_action_rate": float(invalid_rate),
    }


def compute_annual_reward(
    metrics: Dict[str, Any], hist_stats: Dict[str, Any], year_idx: int
) -> Dict[str, Any]:
    """
    Full replica annual reward R_r with breakdown. Year indices are 1-based
    training years (y >= 1, 2, 4 gates per thought.txt).
    """
    u = compute_unemployment_terms(metrics, hist_stats, year_idx)
    g = compute_gdp_terms(metrics, hist_stats, year_idx)
    inf = compute_inflation_terms(metrics, hist_stats, year_idx)
    w = compute_work_guard(metrics, hist_stats, year_idx)
    m = compute_micro_terms(metrics, hist_stats, year_idx)

    breakdown = {
        "u_center": u["u_center"],
        "u_guard": u["u_guard"],
        "work_guard": w["work_guard"],
        "real_gdp_center": g["real_gdp_center"],
        "real_gdp_guard": g["real_gdp_guard"],
        "nominal_gdp_center": g["nominal_gdp_center"],
        "nominal_gdp_guard": g["nominal_gdp_guard"],
        "pi_center": inf["pi_center"],
        "pi_guard": inf["pi_guard"],
        "wage_pi_center": inf["wage_pi_center"],
        "wage_pi_guard": inf["wage_pi_guard"],
        "fmt": m["fmt"],
        "valid": m["valid"],
    }

    r_macro = (
        breakdown["u_center"]
        + breakdown["u_guard"]
        + breakdown["work_guard"]
        + breakdown["real_gdp_center"]
        + breakdown["real_gdp_guard"]
        + breakdown["nominal_gdp_center"]
        + breakdown["nominal_gdp_guard"]
        + breakdown["pi_center"]
        + breakdown["pi_guard"]
        + breakdown["wage_pi_center"]
        + breakdown["wage_pi_guard"]
    )
    r_micro = breakdown["fmt"] + breakdown["valid"]
    annual_reward = r_macro + r_micro

    return {
        "annual_reward": float(annual_reward),
        "breakdown": breakdown,
        "R_macro": float(r_macro),
        "R_micro": float(r_micro),
    }


def _sanity_check() -> None:
    """Minimal numerical checks on Huber and gating."""
    assert abs(huber(0.0, 1.0)) < 1e-9
    assert abs(huber(1.0, 1.0) - 0.5) < 1e-9
    assert abs(huber(3.0, 1.0) - 2.5) < 1e-9

    hs = dict(DEFAULT_HIST_STATS)
    m_y1 = {
        "annual_unemployment": hs["mu_u"],
        "annual_real_gdp_growth": None,
        "annual_nominal_gdp_growth": None,
        "annual_price_inflation": None,
        "annual_wage_inflation": None,
        "max_monthly_work_rate": 0.5,
        "format_fail_rate": 0.0,
        "invalid_action_rate": 0.0,
    }
    o1 = compute_annual_reward(m_y1, hs, year_idx=1)
    assert o1["breakdown"]["real_gdp_center"] == 0.0 and o1["breakdown"]["pi_center"] == 0.0

    m_y4 = {
        "annual_unemployment": 5.0,
        "annual_real_gdp_growth": 2.0,
        "annual_nominal_gdp_growth": 3.0,
        "annual_price_inflation": 2.0,
        "annual_wage_inflation": 2.5,
        "max_monthly_work_rate": 0.97,
        "format_fail_rate": 0.0005,
        "invalid_action_rate": 0.0005,
    }
    o4 = compute_annual_reward(m_y4, hs, year_idx=4)
    assert math.isfinite(o4["annual_reward"])
    assert o4["breakdown"]["work_guard"] < 0.0  # 0.97 > 0.965


_sanity_check()


if __name__ == "__main__":
    metrics = {
        "annual_unemployment": 5.2,
        "annual_real_gdp_growth": 2.1,
        "annual_nominal_gdp_growth": 4.0,
        "annual_price_inflation": 2.2,
        "annual_wage_inflation": 3.0,
        "max_monthly_work_rate": 0.92,
        "format_fail_rate": 0.002,
        "invalid_action_rate": 0.001,
    }
    hist_stats = DEFAULT_HIST_STATS
    out = compute_annual_reward(metrics, hist_stats, year_idx=4)
    print(out)
