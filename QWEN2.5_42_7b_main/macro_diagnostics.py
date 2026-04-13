#!/usr/bin/env python3
"""
从 rollout 导出的 env_stats.json 读取 monthly_series，打印并绘制宏观诊断：

- 月度通胀率序列（由 monthly_price 环比）
- 三年（36 个月）滚动窗口内通胀振幅（窗口内 mom 通胀的 peak-to-trough）
- 失业率序列（monthly_unemployment_rate）
- 实际 GDP proxy 环比增长率（monthly_real_gdp_proxy）

用法:
  python macro_diagnostics.py path/to/env_stats.json
  python macro_diagnostics.py path/to/env_stats.json --out-dir ./figs/my_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _mom_inflation(prices: np.ndarray) -> np.ndarray:
    """月度环比通胀 π_t = (P_t - P_{t-1}) / P_{t-1}；长度 len(prices)-1，对应月份 1..T-1。"""
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 2:
        return np.array([])
    return np.diff(p) / (p[:-1] + 1e-12)


def _mom_growth(y: np.ndarray) -> np.ndarray:
    """环比增长率 (y_t - y_{t-1}) / (|y_{t-1}|+eps)；与价格通胀同索引对齐方式。"""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 2:
        return np.array([])
    return np.diff(y) / (np.abs(y[:-1]) + 1e-12)


def _rolling_ptp(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    滚动 peak-to-trough（振幅）：每个窗口内 max-min。
    返回 (values, end_indices)：第 k 个值对应原序列索引 end_indices[k]（窗口右端，含）。
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < window:
        if n == 0:
            return np.array([]), np.array([], dtype=int)
        return np.array([np.ptp(x)]), np.array([n - 1], dtype=int)
    vals = []
    ends = []
    for i in range(window - 1, n):
        seg = x[i - window + 1 : i + 1]
        vals.append(float(np.ptp(seg)))
        ends.append(i)
    return np.asarray(vals), np.asarray(ends, dtype=int)


def _summarize(name: str, x: np.ndarray, file=None) -> None:
    x = np.asarray(x, dtype=np.float64)
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        msg = f"{name}: (无有限值)"
        print(msg, file=file)
        return
    msg = (
        f"{name}: n={xf.size}  min={xf.min():.6g}  max={xf.max():.6g}  "
        f"mean={xf.mean():.6g}  std={xf.std():.6g}"
    )
    print(msg, file=file)


def run(env_stats_path: Path, out_dir: Path | None, show: bool) -> None:
    with open(env_stats_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("monthly_series") or []
    if not rows:
        print("monthly_series 为空", file=sys.stderr)
        sys.exit(1)

    months = np.array([int(r["month"]) for r in rows], dtype=int)
    prices = np.array([float(r["monthly_price"]) for r in rows], dtype=np.float64)
    unemp = np.array([float(r["monthly_unemployment_rate"]) for r in rows], dtype=np.float64)
    real_gdp = np.array([float(r["monthly_real_gdp_proxy"]) for r in rows], dtype=np.float64)

    infl = _mom_inflation(prices)
    # 与 infl 对齐的“月份”标签：第 i 个通胀对应从 month[i] 到 month[i+1] 的变化，这里用 months[1:] 作横轴
    month_for_infl = months[1:] if len(months) > 1 else np.array([])

    real_growth = _mom_growth(real_gdp)
    month_for_growth = months[1:] if len(months) > 1 else np.array([])

    # 36 个月价格 → 35 个相邻月通胀；滚动振幅取窗口长度 35
    WINDOW_INFL = 35
    roll_amp, roll_end_idx = _rolling_ptp(infl, WINDOW_INFL)
    # 滚动振幅对应到“通胀序列右端索引”，映射到日历月：infl[j] 介于 P[j] 与 P[j+1] 之间，右端月 index = months[j+1]
    if roll_end_idx.size and len(months) > 1:
        month_for_roll_amp = months[roll_end_idx + 1]
    else:
        month_for_roll_amp = np.array([])

    initial_p = float(data.get("initial_price", prices[0] if len(prices) else float("nan")))

    print("=== env_stats:", env_stats_path)
    print(f"initial_price (file): {initial_p:.6g}")
    print(f"月份数 T={len(months)} (month 索引 {months[0]}..{months[-1]})")
    print()

    print("--- 月度通胀率 π_m = (P_m - P_{m-1}) / P_{m-1} ---")
    _summarize("inflation_mom", infl)
    if infl.size:
        print(
            "前 5 个通胀:",
            np.array2string(infl[:5], precision=4, suppress_small=False),
        )
        print(
            "后 5 个通胀:",
            np.array2string(infl[-5:], precision=4, suppress_small=False),
        )
    print()

    print(f"--- 三年（36 个月）滚动通胀振幅：相邻 {WINDOW_INFL} 个月度通胀的 (max-min) ---")
    if roll_amp.size:
        _summarize("rolling_36m_inflation_amplitude", roll_amp)
        print(f"rolling 振幅 全局 max: {roll_amp.max():.6g}")
    else:
        print("(样本不足，无滚动窗口)")

    # 前 3 年 / 后 3 年：各取前 35 个与后 35 个通胀点（若足够长）
    if infl.size >= WINDOW_INFL:
        amp_first_3y = float(np.ptp(infl[:WINDOW_INFL]))
        print(f"前三年区间（首 {WINDOW_INFL} 个月度通胀）振幅: {amp_first_3y:.6g}")
    if infl.size >= WINDOW_INFL * 2:
        amp_last_3y = float(np.ptp(infl[-WINDOW_INFL:]))
        print(f"后三年区间（末 {WINDOW_INFL} 个月度通胀）振幅: {amp_last_3y:.6g}")
    print()

    print("--- 失业率 monthly_unemployment_rate ---")
    _summarize("unemployment_rate", unemp)
    print()

    print("--- 实际 GDP proxy 环比增长率 (monthly_real_gdp_proxy) ---")
    _summarize("real_gdp_proxy_mom_growth", real_growth)
    print()

    if plt is None:
        print("(未安装 matplotlib，跳过作图；可 pip install matplotlib 后重试)")
        return

    # 作图
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(f"Macro diagnostics\n{env_stats_path.name}", fontsize=11)

    ax0 = axes[0, 0]
    if month_for_infl.size:
        ax0.plot(month_for_infl, infl * 100.0, marker=".", markersize=3)
        ax0.axhline(0, color="gray", lw=0.8, ls="--")
        ax0.set_ylabel("mom inflation (%)")
        ax0.set_xlabel("month index (env)")
        ax0.set_title("Inflation (month-over-month, %)")

    ax1 = axes[0, 1]
    if month_for_roll_amp.size:
        ax1.plot(month_for_roll_amp, roll_amp * 100.0, color="darkorange", marker=".", markersize=3)
        ax1.set_ylabel("amplitude (ppt range)")
        ax1.set_xlabel("month index (window end)")
        ax1.set_title(f"3y rolling inflation amplitude\n(ptp over {WINDOW_INFL} mom rates)")

    ax2 = axes[1, 0]
    ax2.plot(months, unemp * 100.0, color="steelblue", marker=".", markersize=3)
    ax2.set_ylabel("rate (%)")
    ax2.set_xlabel("month index (env)")
    ax2.set_ylim(0, max(105, float(np.nanmax(unemp)) * 100 * 1.05))
    ax2.set_title("Unemployment rate")

    ax3 = axes[1, 1]
    if month_for_growth.size:
        ax3.plot(month_for_growth, real_growth * 100.0, color="seagreen", marker=".", markersize=3)
        ax3.axhline(0, color="gray", lw=0.8, ls="--")
        ax3.set_ylabel("mom growth (%)")
        ax3.set_xlabel("month index (env)")
        ax3.set_title("Real GDP proxy growth (m-o-m %)")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"macro_diag_{env_stats_path.stem}.png"
        fig.savefig(out_png, dpi=150)
        print(f"已保存图像: {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="env_stats.json 宏观诊断")
    ap.add_argument("env_stats", type=Path, help="rollout/env_stats.json 路径")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="图像输出目录；默认与 json 同级的 figs/macro_diag/",
    )
    ap.add_argument("--show", action="store_true", help="交互显示图像窗口")
    args = ap.parse_args()

    if not args.env_stats.is_file():
        print(f"文件不存在: {args.env_stats}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.env_stats.resolve().parent.parent / "figs" / "macro_diag"

    run(args.env_stats, out_dir, args.show)


if __name__ == "__main__":
    main()
