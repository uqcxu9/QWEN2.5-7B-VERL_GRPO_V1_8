# “边跑仿真、边收集 RL 训练样本和环境统计”的采样器。
from __future__ import annotations

import json
import math
import os
import re
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

import simulate_utils as su

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

# 为每个agent初始化一个空的memory字典
def default_memory_state() -> Dict[str, Any]:
    return {
        "last_action_raw": None,       # 上个月LLM输出的原始{work, consumption}
        "last_action_env": None,       # 上个月转换后的env action [labor_bin, cons_idx]
        "recent_3m_summary": "",       # 最近3个月的行动摘要字符串
        "quarter_reflection_summary": "",  # 季度反思摘要
        "_recent_3m_deque": None,      # 内部用的deque，懒初始化
    }


def _ensure_deque(mem: Dict[str, Any]) -> Deque[str]:
    d = mem.get("_recent_3m_deque")
    if d is None:
        d = deque(maxlen=3)       # 最多存3个月，自动滚动
        mem["_recent_3m_deque"] = d
    return d


# ---------------------------------------------------------------------------
# Action parsing & env conversion
# ---------------------------------------------------------------------------

_ACTION_FALLBACK = {"work": 1.0, "consumption": 0.5}


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not m:
        return None
    frag = m.group(0)
    try:
        return json.loads(frag)
    except Exception:
        try:
            frag2 = frag.replace("'", '"')
            frag2 = re.sub(r",\s*}", "}", frag2)
            return json.loads(frag2)
        except Exception:
            return None


# 把最近3个月的统计信息打包成一个字符串，用于季度反思   
def build_compact_quarter_summary(
    quarter_rows: List[Dict[str, Any]],
    agent_recent_3m_summary: str,
) -> str:
    if not quarter_rows:
        return ""
    # 计算本季度平均工作率和平均失业率
    work = float(np.mean([r.get("monthly_work_rate", 0.0) for r in quarter_rows]))
    unemp = float(np.mean([r.get("monthly_unemployment_rate", 0.0) for r in quarter_rows]))
    # 计算本季度平均价格和平均工资
    p0 = float(quarter_rows[0].get("monthly_price", 0.0))
    p1 = float(quarter_rows[-1].get("monthly_price", 0.0))
    w0 = float(quarter_rows[0].get("monthly_mean_wage", 0.0))
    w1 = float(quarter_rows[-1].get("monthly_mean_wage", 0.0))

    price_trend = "up" if p1 > p0 + 1e-8 else ("down" if p1 < p0 - 1e-8 else "flat")
    wage_trend = "up" if w1 > w0 + 1e-8 else ("down" if w1 < w0 - 1e-8 else "flat")
    # 返回季度总结字符串，包含平均工作率、平均失业率、价格趋势、工资趋势，以及agent的最近3个月行动摘要
    return (
        f"quarter_macro: avg_work={work:.3f}, avg_unemp={unemp:.3f}, "
        f"price_trend={price_trend}, wage_trend={wage_trend}; "
        f"your_recent_actions={agent_recent_3m_summary or '(none)'}"
    )

# 解析LLM输出的JSON动作，检查格式、范围、合法性，并返回标准化后的action字典
def parse_action_json(
    text: str, stats: Optional[Dict[str, int]] = None
) -> Tuple[Dict[str, float], bool, bool]:
    if stats is None:
        stats = {}

    stats.setdefault("format_fail_count", 0)
    stats.setdefault("invalid_action_count", 0)
    stats.setdefault("fallback_action_count", 0)
    stats.setdefault("out_of_range_count", 0)
    stats.setdefault("negative_wealth_count", 0)

    obj = _safe_json_extract(text)
    valid_format = obj is not None
    used_fallback = False

    if not valid_format:
        stats["format_fail_count"] += 1
        stats["fallback_action_count"] += 1
        return dict(_ACTION_FALLBACK), False, True

    if "work" not in obj or "consumption" not in obj:
        stats["invalid_action_count"] += 1
        stats["fallback_action_count"] += 1
        return dict(_ACTION_FALLBACK), True, True

    try:
        w = float(obj["work"])
        c = float(obj["consumption"])
    except Exception:
        stats["invalid_action_count"] += 1
        stats["fallback_action_count"] += 1
        return dict(_ACTION_FALLBACK), True, True

    bad = (
        (not math.isfinite(w))
        or (not math.isfinite(c))
        or (w < 0.0)
        or (w > 1.0)
        or (c < 0.0)
        or (c > 1.0)
    )
    if bad:
        stats["out_of_range_count"] += 1
        stats["fallback_action_count"] += 1
        return dict(_ACTION_FALLBACK), True, True

    w = float(round(w / 0.02) * 0.02)
    c = float(round(c / 0.02) * 0.02)
    w = min(max(w, 0.0), 1.0)
    c = min(max(c, 0.0), 1.0)

    return {"work": w, "consumption": c}, True, False

# 判断这个agent上个月是否遭遇了商品短缺
def _compute_goods_shortage_agentwise(
    consumption: float, actions_log: List[Any], agent_idx: int
) -> bool:
    if (consumption <= 0) and len(actions_log) > 0:
        last_a = actions_log[-1]
        if isinstance(last_a, dict) and last_a.get(str(agent_idx)) is not None:
            ent = last_a[str(agent_idx)]
            sc = ent.get("SimpleConsumption", 0) if isinstance(ent, dict) else 0
            return sc > 0
    return False


def _compute_goods_shortage_acl24_literal(
    consumption: float, actions_log: List[Any]
) -> bool:
    """Matches ACL24 simulate.py condition exactly (for debug; often false on real dense_log shape)."""
    if len(actions_log) == 0:
        return False
    last = actions_log[-1]
    if not isinstance(last, dict):
        return False
    return (consumption <= 0) and (last.get("SimpleConsumption", 0) > 0)

# 判断价格趋势：上升、下降、持平
def _price_trend_label(env) -> str:
    t = env.world.timestep
    if t == 0:
        return "initial"
    prices = env.world.price
    if len(prices) < 2:
        return "same"
    p_now = float(prices[-1])
    p_prev = float(prices[-2])
    eps = 1e-12
    if p_now > p_prev + eps:
        return "up"
    if p_now < p_prev - eps:
        return "down"
    return "same"


def _labor_income_trend_label(
    env, agent_idx: int, job: str, current_skill: float
) -> Optional[str]:
    """Align with ACL24: compare current skill to dense_log states[-1] for same agent."""
    # 失业agent没有工资趋势，直接返回None
    if job == "Unemployment":
        return None
    # 从dense_log的历史状态里取上个月这个agent的skill值，和当前skill比较，判断工资涨跌趋势。这里比较skill而不是直接比较工资，因为工资=skill×最大劳动时间，两者单调相关
    states = env.dense_log.get("states") or []
    if not states:
        return None
    sid = str(agent_idx)
    prev = states[-1].get(sid)
    if not isinstance(prev, dict) or "skill" not in prev:
        return None
    prev_skill = float(prev["skill"])
    eps = 1e-12
    if current_skill > prev_skill + eps:
        return "up"
    if current_skill < prev_skill - eps:
        return "down"
    return "same"


# 在执行动作之前，抓取agent当前状态的快照，用于事后reward计算和debug
def build_sample_meta(env, obs: Dict[str, Any], agent_idx: int) -> Dict[str, Any]:
    this_agent = env.get_agent(str(agent_idx))
    skill = float(this_agent.state["skill"])
    max_l = float(env._components_dict["SimpleLabor"].num_labor_hours)
    consumption = float(this_agent.consumption["Coin"])
    actions_log = env.dense_log.get("actions") or []
    g_agent = _compute_goods_shortage_agentwise(consumption, actions_log, agent_idx)
    g_acl24 = _compute_goods_shortage_acl24_literal(consumption, actions_log)
    tp, ls, _ = _resolve_planner_tax_obs(env, obs, agent_idx)
    return {
        "savings": float(this_agent.inventory["Coin"]),
        "price": float(env.world.price[-1]),
        "interest_rate": float(env.world.interest_rate[-1]),
        "tax_paid": tp,
        "lump_sum": ls,
        "expected_income": float(skill * max_l),
        "goods_shortage_agentwise": bool(g_agent),
        "goods_shortage_acl24_literal": bool(g_acl24),
    }


def convert_raw_action_to_env_action(
    raw_action_dict,
    bernoulli_u: Optional[float] = None,
) -> List[float]:
    w = float(raw_action_dict.get("work", 0.0))
    c = float(raw_action_dict.get("consumption", 0.0))

    # 强制落到 0.02 网格，并裁剪到 [0, 1]
    w = float(np.clip(np.round(w / 0.02) * 0.02, 0.0, 1.0))
    c = float(np.clip(np.round(c / 0.02) * 0.02, 0.0, 1.0))

    # Bernoulli work decision: if bernoulli_u is provided (CRN mode),
    # use the pre-drawn random number shared across replicas;
    # otherwise fall back to independent sampling.
    u = bernoulli_u if bernoulli_u is not None else float(np.random.uniform())
    l = int(u <= w)
    c_idx = int(np.clip(np.round(c / 0.02), 0, 50))  # 0.00~1.00 -> 0~50

    return [l, c_idx]

# 从dense_log的action记录里提取劳动决策
def _labor_from_action_entry(v: Any) -> int:
    if isinstance(v, dict):
        return int(v.get("SimpleLabor", 0))
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        return int(v[0])
    try:
        return int(v)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def _resolve_planner_tax_obs(
    env: Any, obs: Dict[str, Any], agent_idx: int
) -> Tuple[float, float, Any]:
    """
    Read tax_paid, lump_sum, curr_rates for prompts / meta.

    When ``flatten_observations=True`` (common in saved snapshots), ``obs['p']['p{k}']``
    may be a numpy vector instead of a dict — string keys raise IndexError. Fall back
    to ``PeriodicBracketTax`` internal state in that case.
    """
    tax_paid: Optional[float] = None
    lump_sum: Optional[float] = None
    curr_rates: Any = None

    p_block = obs.get("p")
    pk = f"p{agent_idx}"
    if isinstance(p_block, dict):
        sub = p_block.get(pk)
        if isinstance(sub, dict):
            tax_paid = sub.get("PeriodicBracketTax-tax_paid")
            if tax_paid is None:
                tax_paid = sub.get("tax_paid")
            lump_sum = sub.get("PeriodicBracketTax-lump_sum")
            if lump_sum is None:
                lump_sum = sub.get("lump_sum")
        curr_rates = p_block.get("PeriodicBracketTax-curr_rates")

    tax_comp = env._components_dict.get("PeriodicBracketTax")
    if tax_paid is None or lump_sum is None:
        if tax_comp is not None:
            t = int(env.world.timestep)
            curr_tax: Dict[str, Any] = {}
            try:
                if t > 0 and tax_comp.taxes:
                    raw = tax_comp.taxes[t - 1]
                    curr_tax = raw if isinstance(raw, dict) else {}
            except (IndexError, KeyError, TypeError):
                curr_tax = {}
            entry = curr_tax.get(str(agent_idx), {})
            if not isinstance(entry, dict):
                entry = {}
            if tax_paid is None:
                tax_paid = float(entry.get("tax_paid", 0.0))
            if lump_sum is None:
                lump_sum = float(entry.get("lump_sum", 0.0))

    if curr_rates is None and tax_comp is not None:
        try:
            curr_rates = np.asarray(tax_comp._curr_rates_obs, dtype=np.float64)
        except Exception:
            try:
                curr_rates = np.asarray(tax_comp.curr_marginal_rates, dtype=np.float64)
            except Exception:
                curr_rates = np.array([], dtype=np.float64)

    return (
        float(0.0 if tax_paid is None else tax_paid),
        float(0.0 if lump_sum is None else lump_sum),
        curr_rates if curr_rates is not None else [],
    )


# 给LLM构建当月的观测prompt，内容包括
def build_monthly_observation(
    env,
    obs: Dict[str, Any],
    agent_idx: int,
    memory_state: Dict[str, Any],
    world_start_time,
    relativedelta,
) -> str:
    idx_s = str(agent_idx)
    this_agent = env.get_agent(idx_s)
    # 收集所有需要告诉agent的经济数据：技能、财富、消费、利率、价格、税收、转移支付、最大劳动时间、提供的工资
    skill = float(this_agent.state["skill"])
    wealth = float(this_agent.inventory["Coin"])
    consumption = float(this_agent.consumption["Coin"])
    interest_rate = float(env.world.interest_rate[-1])
    price = float(env.world.price[-1])
    tax_paid, lump_sum, curr_rates = _resolve_planner_tax_obs(env, obs, agent_idx)
    max_l = float(env._components_dict["SimpleLabor"].num_labor_hours)
    offered_wage = skill * max_l
    # 计算价格变化率
    if len(env.world.price) >= 2:
        price_prev = float(env.world.price[-2])
        price_change = (price - price_prev) / (price_prev + 1e-8)
    else:
        price_change = 0.0
    # 计算工资变化率
    labor_change = None
    states = env.dense_log.get("states") or []
    if states:
        prev = states[-1].get(str(agent_idx))
        if isinstance(prev, dict):
            prev_skill = float(prev.get("skill", skill))
            prev_wage = prev_skill * max_l
            labor_change = (offered_wage - prev_wage) / (prev_wage + 1e-8)

    current_time = world_start_time + relativedelta(months=int(env.world.timestep))
    month_label = current_time.strftime("%Y.%m")
    # 取agent的身份信息：名字、年龄、城市、当前职业、可接受的工作邀请
    endo = this_agent.endogenous
    name = endo["name"]
    age = endo["age"]
    city = endo["city"]
    job = endo["job"]
    offer = endo["offer"]

    job_unemp = job == "Unemployment"
    price_trend = "up" if price_change > 1e-8 else ("down" if price_change < -1e-8 else "flat")
    labor_trend = None
    if labor_change is not None:
        labor_trend = "up" if labor_change > 1e-8 else ("down" if labor_change < -1e-8 else "flat")

    real_savings = wealth / (price + 1e-8)
    # 判断这个agent上个月是否遭遇了商品短缺
    actions_log = env.dense_log.get("actions") or []
    goods_shortage_agentwise = _compute_goods_shortage_agentwise(
        consumption, actions_log, agent_idx
    )
    # 构建税率结构描述字符串（curr_rates 已由 _resolve_planner_tax_obs 处理 flatten）
    tax_line = (
        "tax_system=progressive_federal_with_equal_redistribution, "
        f"tax_brackets={list(map(float, su.brackets))}, "
        f"tax_rates={list(map(float, curr_rates))}"
    )
    # Note: last_action, recent_3m_summary, quarter_reflection are now provided
    # via dialog_queue multi-turn history, not embedded in the prompt.

    # 判断这个agent当前的就业状态
    job_status = "unemployed" if job_unemp else f"employed_as_{job}"

    # Decision context and format instructions are in the system prompt (ACL24 style).

    # 构建观测prompt的文本行
    lines = [
        f"[Month {month_label} | env_timestep={int(env.world.timestep)} | agent_id={agent_idx}]",
        f"identity: name={name}, age={age}, city={city}, job_status={job_status}, offer_if_unemp={offer}",
        f"last_month_income_coin={float(this_agent.income['Coin']):.2f}, last_month_consumption_coin={consumption:.2f}",
        f"tax_paid={tax_paid:.2f}, lump_sum_redistribution={lump_sum:.2f}",
        f"savings_coin={wealth:.2f}",
        f"offered_wage_if_work={offered_wage:.2f}, goods_price={price:.4f}, interest_rate={interest_rate*100:.4f}%",
        f"price_trend={price_trend}",
        f"price_change_ratio={price_change:.4f}",
    ]

    # 如果这个agent上个月失业，添加失业状态提示
    if job_unemp:
        lines.append("last_month_job_status=unemployed")
        lines.append("last_month_income_status=no_income")
        lines.append(f"offer_available={offer}")
    elif labor_trend is not None:
        lines.append(f"labor_income_trend={labor_trend}")

    if labor_change is not None:
        lines.append(f"income_change_ratio={labor_change:.4f}")

    lines.append(f"real_savings={real_savings:.4f}")
    lines.append(
        "last_month_consumption_constraint="
        + ("goods_shortage" if goods_shortage_agentwise else "none")
    )

    lines.append(f"goods_shortage_agentwise={int(goods_shortage_agentwise)}")
    # Note: last_month_action, recent_3m_summary, and quarter_reflection are
    # now provided via dialog_queue multi-turn history (ACL24 style), not
    # embedded in the user prompt. This avoids redundancy.

    # Tax structure (kept in user prompt as it may change with planner policy)
    lines.append(tax_line)
    # Format reminder (brief, main instructions are in system prompt)
    lines.append('Respond with JSON: {"work": 0.xx, "consumption": 0.xx}')
    return su.prettify_document(" ".join(lines))


# ---------------------------------------------------------------------------
# Memory update
# ---------------------------------------------------------------------------

# 把本月的原始action和env action存进memory，覆盖上月的记录
def update_memory_state(
    memory_state: Dict[str, Any],
    parsed_raw: Dict[str, float],
    env_action: List[float],
    month_index: int,
) -> None:
    memory_state["last_action_raw"] = dict(parsed_raw)
    memory_state["last_action_env"] = [int(env_action[0]), float(env_action[1])]
    one = f"m{month_index}:w={parsed_raw['work']:.2f},c={parsed_raw['consumption']:.2f}"
    dq = _ensure_deque(memory_state)
    dq.append(one)
    memory_state["recent_3m_summary"] = "; ".join(dq)

# 构建季度反思提示语
def build_quarterly_reflection_prompt(
    q_rows: List[Dict[str, Any]],
    recent_3m_summary: str,
) -> str:
    months = []
    for r in q_rows:
        months.append(
            f"(m={r['month']}, unemp={r['monthly_unemployment_rate']:.4f}, "
            f"work={r['monthly_work_rate']:.4f}, price={r['monthly_price']:.4f}, "
            f"wage={r['monthly_mean_wage']:.4f}, ir={r['monthly_interest_rate']:.4f})"
        )

    text = (
        "Reflect on the previous quarter in under 80 words. "
        "Focus on labor market, prices, wages, interest rate, and how these trends should affect next-quarter work and consumption decisions. "
        f"Recent personal actions summary: {recent_3m_summary}. "
        f"Quarter market summary: {' ; '.join(months)}. "
        "Return plain text only."
    )
    return su.prettify_document(text)

# 构建agent的季度反思对话提示语
def build_agent_quarter_reflection_dialog_prompt(
    env,
    obs: Dict[str, Any],
    agent_idx: int,
    memory_state: Dict[str, Any],
    world_start_time,
    relativedelta,
    pending_action: Optional[Dict[str, float]] = None,
) -> str:
    """Compact per-agent dialog prompt; not stored as training target."""
    this_agent = env.get_agent(str(agent_idx))
    name = this_agent.endogenous["name"]
    t = env.world.timestep
    ct = world_start_time + relativedelta(months=t)
    month_label = ct.strftime("%Y.%m")
    price = float(env.world.price[-1])
    ir = float(env.world.interest_rate[-1])
    wealth = float(this_agent.inventory["Coin"])
    recent = memory_state.get("recent_3m_summary") or ""
    prev_q = memory_state.get("quarter_reflection_summary") or ""
    pend = ""
    if pending_action:
        pend = (
            f" This month you chose work_prop={pending_action['work']:.2f}, "
            f"cons_prop={pending_action['consumption']:.2f}."
        )
    return (
        f"You are {name}. It is {month_label}. Price={price:.4f}, interest={ir*100:.2f}%, savings={wealth:.2f}. "
        f"Recent 3-month note: {recent}. Prior quarter summary: {prev_q}.{pend} "
        "In 1-2 short sentences (max 80 words), reflect on labor, consumption, and savings this quarter. "
        "Plain text only, no JSON."
    )


# ---------------------------------------------------------------------------
# Env statistics (per completed month)
# ---------------------------------------------------------------------------

def _end_of_month_agent_states(env: Any, states: List[Any], state_idx: int) -> Dict[str, Any]:
    """
    Dense log appends pre-step snapshots: ``states[t]`` is the state at the start of
    env timestep ``t``. Thus end-of-month ``m`` is ``states[m+1]`` when available.
    After the final ``env.step``, ``states[m+1]`` may not be appended yet, so we
    build a dense-log-compatible snapshot from live agent objects (not raw
    ``agent.state``, which may omit ``endogenous`` / ``income``).
    """
    if state_idx < len(states):
        return states[state_idx]

    snap: Dict[str, Any] = {}
    for i in range(env.num_agents):
        agent = env.get_agent(str(i))
        snap[str(i)] = {
            "skill": float(agent.state.get("skill", 0.0)),
            "income": {"Coin": float(agent.income.get("Coin", 0.0))},
            "endogenous": dict(agent.endogenous),
        }
    return snap


# 收集环境统计信息，返回的row字典包含：失业率、工作率、价格、利率、工资、GDP代理值，以及本月/累计的格式错误统计
def collect_env_stats_step(
    env,
    completed_month_idx: int,
    month_action_stats: Dict[str, int],
    cumulative_counters: Dict[str, int],
    initial_price: float,
) -> Dict[str, Any]:
    """
    Aggregate stats after env.step for the month ``completed_month_idx`` (0-based).

    ACL24 dense_log: ``actions[m]`` is this month's action; ``states[m]`` is the
    pre-step snapshot. Unemployment / income / skill / wage use the *end* of month
    ``m`` (``states[m+1]``, or a live dense-log-shaped snapshot after the last step).
    Labor supply
    and ``monthly_work_rate`` still use ``actions[m]``.
    """
    states = env.dense_log["states"]
    actions = env.dense_log["actions"]
    if completed_month_idx < 0 or completed_month_idx >= len(actions):
        return {"error": "missing_dense_log_index", "month": completed_month_idx}
    state_idx = completed_month_idx + 1
    if state_idx > len(states):
        return {"error": "missing_dense_log_index", "month": completed_month_idx}
    st = _end_of_month_agent_states(env, states, state_idx)
    act = actions[completed_month_idx]
    max_l = float(env._components_dict["SimpleLabor"].num_labor_hours)
    # 统计每个agent的就业状态、技能、工资、收入、劳动决策
    n_agents = env.num_agents
    employed_cnt = 0
    unemp_cnt = 0
    work_action_cnt = 0
    skills: List[float] = []
    wages_offered: List[float] = []
    incomes: List[float] = []

    # 计算总劳动供给和加权劳动供给
    supply_homogeneous = 0.0
    supply_skill_weighted = 0.0

    # 遍历每个agent，统计就业状态、技能、工资、收入、劳动决策
    for i in range(n_agents):
        sid = str(i)
        if sid not in st or not isinstance(st[sid], dict):
            continue
        # 取出当前agent的状态
        s = st[sid]
        sk = float(s.get("skill", 0.0))
        skills.append(sk)
        # 判断当前agent的就业状态
        job = (s.get("endogenous") or {}).get("job", "")
        if job == "Unemployment":
            unemp_cnt += 1
        else:
            employed_cnt += 1

        inc = float(s.get("income", {}).get("Coin", 0.0))
        incomes.append(inc)
        wages_offered.append(sk * max_l)

        lab = 0
        if sid in act:
            lab = _labor_from_action_entry(act[sid])

        if lab:
            work_action_cnt += 1

        # ACL24 口径：A = 1，不按 skill 加权
        supply_homogeneous += lab * max_l
        # 保留诊断字段，便于之后做对照分析
        supply_skill_weighted += lab * max_l * sk

    # 计算总劳动人口和失业率
    labor_force = employed_cnt + unemp_cnt
    unemp_rate = (unemp_cnt / labor_force) if labor_force else 0.0
    # 计算总工作人口和就业率
    work_rate = (work_action_cnt / n_agents) if n_agents else 0.0
    # 取出当前月份的价格和利率

    price_t = float(env.world.price[-1])
    ir_t = float(env.world.interest_rate[-1])
    # 计算名义GDP和实际GDP代理值
    monthly_nominal_gdp_proxy = supply_homogeneous * price_t
    monthly_real_gdp_proxy = supply_homogeneous * float(initial_price)

    # 构建返回的row字典
    row = {
        "month": int(completed_month_idx),
        "monthly_work_rate": float(work_rate),
        # 失业率
        "monthly_unemployment_rate": float(unemp_rate),
        "monthly_aggregate_labor": float(supply_homogeneous),
        "monthly_aggregate_effective_labor": float(supply_homogeneous),
        "monthly_skill_weighted_labor": float(supply_skill_weighted),
        "monthly_aggregate_income": float(sum(incomes)),
        "monthly_output_proxy": float(supply_homogeneous),
        "monthly_price": price_t,
        "monthly_interest_rate": ir_t,
        "monthly_mean_skill": float(np.mean(skills)) if skills else 0.0,
        "monthly_mean_wage": float(np.mean(wages_offered)) if wages_offered else 0.0,
        "monthly_wage_proxy": float(np.mean(wages_offered)) if wages_offered else 0.0,
        "monthly_nominal_gdp_proxy": float(monthly_nominal_gdp_proxy),
        "monthly_real_gdp_proxy": float(monthly_real_gdp_proxy),
        "month_format_fail_count": int(month_action_stats.get("format_fail_count", 0)),
        "month_invalid_action_count": int(month_action_stats.get("invalid_action_count", 0)),
        "month_fallback_action_count": int(month_action_stats.get("fallback_action_count", 0)),
        "month_out_of_range_count": int(month_action_stats.get("out_of_range_count", 0)),
        "month_negative_wealth_count": int(month_action_stats.get("negative_wealth_count", 0)),
        "cumulative_format_fail_count": int(cumulative_counters.get("format_fail_count", 0)),
        "cumulative_invalid_action_count": int(cumulative_counters.get("invalid_action_count", 0)),
        "cumulative_fallback_action_count": int(cumulative_counters.get("fallback_action_count", 0)),
        "cumulative_out_of_range_count": int(cumulative_counters.get("out_of_range_count", 0)),
        "cumulative_negative_wealth_count": int(cumulative_counters.get("negative_wealth_count", 0)),
    }
    return row


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def save_rollout_artifacts(
    base_save_path: str,
    policy_model_save: str,
    episode_samples: List[Dict[str, Any]],
    env_stats: Dict[str, Any],
    dense_log: Any,
) -> Dict[str, str]:
    """
    Writes under save_path/data/{policy_model_save}/rollout/
    """
    root = os.path.join(base_save_path, "data", policy_model_save, "rollout")
    os.makedirs(root, exist_ok=True)

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

    samples_path = os.path.join(root, "episode_samples.jsonl")
    with open(samples_path, "w", encoding="utf-8") as f:
        for row in episode_samples:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")

    stats_path = os.path.join(root, "env_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(env_stats), f, ensure_ascii=False, indent=2)

    import pickle as pkl

    dense_path = os.path.join(root, "dense_log.pkl")
    with open(dense_path, "wb") as f:
        pkl.dump(dense_log, f)

    return {
        "episode_samples_jsonl": samples_path,
        "env_stats_json": stats_path,
        "dense_log_pkl": dense_path,
    }
