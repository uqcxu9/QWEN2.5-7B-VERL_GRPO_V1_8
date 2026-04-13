from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Keys for pickle-on-disk assets (relative paths preferred, resolved from the
# directory containing ``state_bank.json``).
ASSET_PATH_KEYS: Tuple[str, ...] = (
    "snapshot_path",
    "obs_path",
    "memory_state_path",
    "rng_state_path",
    "history_state_path",
)

# 恢复环境最少需要的四类路径（相对 state_bank.json 所在目录）
REQUIRED_RESTORE_PATH_KEYS: Tuple[str, ...] = (
    "snapshot_path",
    "obs_path",
    "memory_state_path",
    "history_state_path",
)

# 加载env_stats.json文件
def load_env_stats(env_stats_path: str) -> Dict[str, Any]:
    with open(env_stats_path, encoding="utf-8") as f:
        return json.load(f)

# 把monthly_series按年份分组
def group_monthly_series_by_year(
    monthly_series: Sequence[Dict[str, Any]],
    months_per_year: int = 12,
) -> List[Dict[str, Any]]:
    """
    Group ``monthly_series`` rows into full calendar years within the rollout.

    Each group has ``calendar_year_index`` (1-based: months 0..11 → year 1),
    ``months``, and ``rows``. Only **complete** ``months_per_year`` blocks are
    included (same rule as state bank records).
    """
    by_m = {int(r["month"]): r for r in monthly_series if "month" in r}
    if not by_m:
        return []
    # 获取最大的月份，作为分组的上限
    max_month = max(by_m.keys())
    # 创建一个列表来存储分组后的数据
    groups: List[Dict[str, Any]] = []
    # 初始化年份索引
    cal = 0
    # 遍历每个月份，直到最大的月份
    while cal * months_per_year <= max_month:
        # 计算当前年份的起始月份
        start = cal * months_per_year
        # 计算当前年份的月份列表
        months = list(range(start, start + months_per_year))
        # 如果当前年份的所有月份都在by_m中，则添加到groups中
        if all(m in by_m for m in months):
            groups.append(
                {
                    "calendar_year_index": cal + 1,
                    "months": months,
                    "rows": [by_m[m] for m in months],
                }
            )
        cal += 1
    return groups

# 把asset_paths合并到record中
def merge_asset_paths_into_record(
    record: MutableMapping[str, Any],
    assets: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Merge optional path strings into a state-bank row. Unknown keys ignored.
    Missing path keys are set to ``None`` so every row has a stable schema.
    """
    out = dict(record)
    for k in ASSET_PATH_KEYS:
        out.setdefault(k, None)
    if not assets:
        return out
    for k in ASSET_PATH_KEYS:
        if k in assets and assets[k] is not None:
            out[k] = assets[k]
    return out

# 构建state_bank记录
def build_state_bank_records(
    env_stats: Dict[str, Any],
    months_per_year: int = 12,
    per_year_assets: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    One record per training ``year_index`` y >= 2.

    Record ``year_index = y`` carries baselines from the **previous** calendar
    year in env timesteps: months ``(y-2)*12 .. (y-1)*12 - 1`` (e.g. y=5 → 36..47).
    Skips ``y`` if those 12 months are not all present in ``monthly_series``.

    ``per_year_assets[y]`` may provide pickle path strings for that row
    (``capture_env_snapshot`` / ``restore_env_snapshot`` in ``simulate.py``).
    """
    monthly_series = env_stats.get("monthly_series") or []
    by_m = {int(r["month"]): r for r in monthly_series if "month" in r}
    if not by_m:
        return []
    # 创建一个列表来存储state_bank记录
    records: List[Dict[str, Any]] = []
    y = 2
    while True:
        # 计算当前年份的月份列表
        months = list(range((y - 2) * months_per_year, (y - 1) * months_per_year))
        # 如果当前年份的任何一个月不在by_m中，则跳出循环
        if any(m not in by_m for m in months):
            break
        # 获取当前年份的月份数据
        rows = [by_m[m] for m in months]
        # 计算当前年份的平均价格
        prices = [float(r["monthly_price"]) for r in rows]
        # 计算当前年份的平均工资
        wages = [float(r["monthly_mean_wage"]) for r in rows]
        # 构建当前年份的state_bank记录
        row = {
            "year_index": y,
            "state_group_id": f"complex_y{y}",
            "months_prev_year": months,
            "prev_year_mean_price": sum(prices) / len(prices),
            "prev_year_mean_wage": sum(wages) / len(wages),
            # 计算当前年份的名义GDP
            "prev_year_nominal_gdp": sum(
                float(r.get("monthly_nominal_gdp_proxy", 0.0)) for r in rows
            ),
            # 计算当前年份的实际GDP
            "prev_year_real_gdp": sum(
                float(r.get("monthly_real_gdp_proxy", 0.0)) for r in rows
            ),
        }
        # 获取当前年份的资产路径
        assets = None
        # 如果per_year_assets不为空，则获取当前年份的资产路径
        if per_year_assets is not None:
            assets = per_year_assets.get(y) or per_year_assets.get(str(y))
        records.append(merge_asset_paths_into_record(row, assets))
        y += 1
    return records


def save_state_bank(state_bank_records: Sequence[Dict[str, Any]], out_path: str) -> str:
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(state_bank_records), f, ensure_ascii=False, indent=2)
    return out_path


def default_state_bank_out_path(env_stats_path: str) -> str:
    """``.../data/<run>/rollout/env_stats.json`` → ``.../data/<run>/state_bank/state_bank.json``."""
    rollout_dir = os.path.dirname(os.path.abspath(env_stats_path))
    run_dir = os.path.dirname(rollout_dir)
    return os.path.join(run_dir, "state_bank", "state_bank.json")


def load_state_bank(
    state_bank_path: str,
    *,
    strict_restore_files: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load ``state_bank.json`` list.

    When ``strict_restore_files`` is True (default), each row must have non-empty
    ``snapshot_path`` / ``obs_path`` / ``memory_state_path`` / ``history_state_path``
    and the resolved files must exist under ``state_bank.json``'s directory.
    ``rng_state_path`` is optional; if set, the file must exist.
    Set ``strict_restore_files=False`` for macro-only rows without pickle assets.
    """
    with open(state_bank_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("state_bank.json must be a JSON list")

    base_dir = state_bank_base_dir(state_bank_path)
    out = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"record {i} is not a dict")

        rec = merge_asset_paths_into_record(dict(rec), None)

        if "year_index" not in rec:
            raise ValueError(f"record {i} missing year_index")
        if int(rec["year_index"]) < 2:
            raise ValueError(f"record {i} has invalid year_index={rec['year_index']}")

        rec.setdefault("state_group_id", f"y{int(rec['year_index'])}")
        if not str(rec.get("state_group_id", "")).strip():
            raise ValueError(f"record {i} missing or empty state_group_id")

        if strict_restore_files:
            for k in REQUIRED_RESTORE_PATH_KEYS:
                v = rec.get(k)
                if v is None or (isinstance(v, str) and not str(v).strip()):
                    raise ValueError(
                        f"record {i} (year_index={rec.get('year_index')}) "
                        f"missing required restore field {k!r}"
                    )
                p = v if os.path.isabs(str(v)) else os.path.join(base_dir, str(v))
                if not os.path.isfile(p):
                    raise FileNotFoundError(
                        f"record {i} restore file missing for {k!r}: {p}"
                    )
            rngv = rec.get("rng_state_path")
            if rngv is not None and str(rngv).strip():
                rp = (
                    rngv
                    if os.path.isabs(str(rngv))
                    else os.path.join(base_dir, str(rngv))
                )
                if not os.path.isfile(rp):
                    raise FileNotFoundError(
                        f"record {i} rng_state_path set but file missing: {rp}"
                    )
        else:
            for k in ASSET_PATH_KEYS:
                v = rec.get(k)
                if v:
                    p = v if os.path.isabs(str(v)) else os.path.join(base_dir, str(v))
                    if not os.path.exists(p):
                        raise FileNotFoundError(
                            f"record {i} missing asset for {k}: {p}"
                        )

        out.append(rec)
    return out


def get_state_bank_record(
    state_bank_records: Sequence[Dict[str, Any]],
    year_index: int,
) -> Optional[Dict[str, Any]]:
    for rec in state_bank_records:
        if int(rec.get("year_index", -1)) == int(year_index):
            return merge_asset_paths_into_record(dict(rec), None)
    return None


def load_per_year_assets_json(path: str) -> Dict[int, Dict[str, Any]]:
    """JSON object keyed by ``year_index`` (int or string) → path dict."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("assets JSON must be an object keyed by year_index")
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in raw.items():
        yk = int(k)
        if isinstance(v, dict):
            out[yk] = dict(v)
        else:
            out[yk] = {}
    return out


def state_bank_base_dir(state_bank_path: str) -> str:
    """Directory containing ``state_bank.json`` (for resolving relative asset paths)."""
    return os.path.dirname(os.path.abspath(state_bank_path))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build state bank index (macro baselines + optional asset paths) from env_stats.json."
    )
    p.add_argument(
        "--env-stats",
        type=str,
        required=True,
        help="Path to rollout/env_stats.json",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: <run>/state_bank/state_bank.json)",
    )
    p.add_argument("--months-per-year", type=int, default=12)
    p.add_argument(
        "--assets-json",
        type=str,
        default=None,
        help=(
            "Optional JSON: { \"<year_index>\": { \"snapshot_path\": \"...\", ... } } "
            "merged into matching rows (paths relative to output state_bank.json dir)."
        ),
    )
    args = p.parse_args()

    env_stats = load_env_stats(args.env_stats)
    per_year = (
        load_per_year_assets_json(args.assets_json) if args.assets_json else None
    )
    records = build_state_bank_records(
        env_stats,
        months_per_year=args.months_per_year,
        per_year_assets=per_year,
    )
    out_path = args.out or default_state_bank_out_path(args.env_stats)
    save_state_bank(records, out_path)
    print(f"Wrote {len(records)} records to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
