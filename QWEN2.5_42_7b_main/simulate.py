from typing import Any, Dict, List, Mapping, Optional, Tuple
import argparse
import copy
import fire
import os
import random
import sys

import ai_economist.foundation as foundation
import numpy as np
import matplotlib.pyplot as plt
import yaml
from time import time
from collections import defaultdict
import re
from simulate_utils import *
import pickle as pkl
from itertools import product
from dateutil.relativedelta import relativedelta

from rollout_collector import (
    build_monthly_observation,
    build_quarterly_reflection_prompt,
    build_sample_meta,
    collect_env_stats_step,
    convert_raw_action_to_env_action,
    default_memory_state,
    parse_action_json,
    save_rollout_artifacts,
    update_memory_state,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_SCRIPT_DIR, "config.yaml"), "r") as f:
    run_configuration = yaml.safe_load(f)
env_config = run_configuration.get("env")
        
def gpt_actions(env, obs, dialog_queue, dialog4ref_queue, gpt_path, gpt_error, total_cost, model_type='gpt'):
    if not os.path.exists(gpt_path):
        os.makedirs(gpt_path)
    curr_rates = obs['p']['PeriodicBracketTax-curr_rates']
    current_time = world_start_time + relativedelta(months=env.world.timestep)
    current_time = current_time.strftime('%Y.%m')
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        skill = this_agent.state['skill']
        wealth = this_agent.inventory['Coin']
        consumption = this_agent.consumption['Coin']
        interest_rate = env.world.interest_rate[-1]
        price = env.world.price[-1]
        tax_paid = obs['p'][f'p{idx}']['PeriodicBracketTax-tax_paid']
        lump_sum = obs['p'][f'p{idx}']['PeriodicBracketTax-lump_sum']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        name = this_agent.endogenous['name']
        age = this_agent.endogenous['age']
        city = this_agent.endogenous['city']
        job = this_agent.endogenous['job']
        offer = this_agent.endogenous['offer']
        actions = env.dense_log['actions']
        states = env.dense_log['states']
        problem_prompt = f'''
                    You're {name}, a {age}-year-old individual living in {city}. As with all Americans, a portion of your monthly income is taxed by the federal government. This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings.
                    Now it's {current_time}.
                '''
        if job == 'Unemployment':
            job_prompt = f'''
                        In the previous month, you became unemployed and had no income. Now, you are invited to work as a(an) {offer} with monthly salary of ${skill*max_l:.2f}.
                    '''
        else:
            if skill >= states[-1][str(idx)]['skill']:
                job_prompt = f'''
                            In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is increased compared to the last month due to the inflation of labor market.
                        '''
            else:
                job_prompt = f'''
                            In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is decreased compared to the last month due to the deflation of labor market.
                        '''
        if (consumption <= 0) and (len(actions) > 0) and (actions[-1].get('SimpleConsumption', 0) > 0):
            consumption_prompt = f'''
                        Besides, you had no consumption due to shortage of goods.
                    '''
        else:
            consumption_prompt = f'''
                        Besides, your consumption was ${consumption:.2f}.
                    '''
        if env._components_dict['PeriodicBracketTax'].tax_model == 'us-federal-single-filer-2018-scaled':
            tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                            In this month, the government sets the brackets: {format_numbers(brackets)} and their corresponding rates: {format_numbers(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
        else:
            tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                            In this month, according to the optimal taxation theory, Saez Tax, the brackets are not changed: {format_numbers(brackets)} but the government has updated corresponding rates: {format_percentages(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
        if env.world.timestep == 0:
            price_prompt = f'''Meanwhile, in the consumption market, the average price of essential goods is now at ${price:.2f}.'''
        else:
            if price >= env.world.price[-2]:
                price_prompt = f'''Meanwhile, inflation has led to a price increase in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
            else:
                price_prompt = f'''Meanwhile, deflation has led to a price decrease in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
        job_prompt = prettify_document(job_prompt)
        obs_prompt = f'''
                        {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt}
                        Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%. 
                        With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price?
                        Please share your decisions in a JSON format. The format should have two keys: 'work' (a value between 0 and 1 with intervals of 0.02, indicating the willingness or propensity to work) and 'consumption' (a value between 0 and 1 with intervals of 0.02, indicating the proportion of all your savings and income you intend to spend on essential goods).
                    '''
        obs_prompt = prettify_document(obs_prompt)
        dialog_queue[idx].append({'role': 'user', 'content': obs_prompt})
        dialog4ref_queue[idx].append({'role': 'user', 'content': obs_prompt})
        
    def action_check(actions):
        if len(actions) != 2:
            return False
        else:
            return (actions[0] >= 0) & (actions[0] <= 1) & (actions[1] >= 0) & (actions[1] <= 1)
    if env.world.timestep%3 == 0 and env.world.timestep > 0:
        results, cost = get_multiple_completion([list(dialogs)[:2] + list(dialog4ref)[-3:-1] + list(dialogs)[-1:] for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue)], model_type=model_type)
        total_cost += cost
    else:
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog_queue], model_type=model_type)
        total_cost += cost
    actions = {}
    for idx in range(env.num_agents):
        content = results[idx]
        try:
            extracted_actions = list(eval(content).values())
            if not action_check(extracted_actions):
                extracted_actions = [1, 0.5]
                gpt_error += 1
        except:
            extracted_actions = [1, 0.5]
            gpt_error += 1
        extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
        extracted_actions[1] /= 0.02
        actions[str(idx)] = extracted_actions
        dialog_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
        dialog4ref_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
    actions['p'] = [0]
    for idx, agent_dialog in enumerate(dialog_queue):
        with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
            for dialog in list(agent_dialog)[-2:]:
                f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
        
    if (env.world.timestep+1)%3 == 0:
        reflection_prompt = '''Given the previous quarter's economic environment, reflect on the labor, consumption, and financial markets, as well as their dynamics. What conclusions have you drawn?
        Your answer must be less than 200 words!'''
        reflection_prompt = prettify_document(reflection_prompt)
        for idx in range(env.num_agents):
            # dialog_queue[idx].append({'role': 'user', 'content': reflection_prompt})
            dialog4ref_queue[idx].append({'role': 'user', 'content': reflection_prompt})
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog4ref_queue], temperature=0, max_tokens=200, model_type=model_type)
        total_cost += cost
        for idx in range(env.num_agents):
            content = results[idx]
            # dialog_queue[idx].append({'role': 'assistant', 'content': content})
            dialog4ref_queue[idx].append({'role': 'assistant', 'content': content})
        
        for idx, agent_dialog in enumerate(dialog4ref_queue):
             with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
                for dialog in list(agent_dialog)[-2:]:
                    f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
    return actions, gpt_error, total_cost


def generate_agent_actions_qwen(dialogs_list):
    """
    Month-level policy rollouts for GRPO: stochastic sampling (not greedy).
    sampling_params: temperature=1.0, max_tokens=50, top_p=1.0
    """
    return qwen_rollout_batch_generate(
        dialogs_list, temperature=1.0, max_tokens=96, top_p=1.0
    )


def generate_quarterly_reflection_qwen(dialogs_list):
    """Quarter-end reflection text; slightly longer cap than monthly JSON actions."""
    return qwen_rollout_batch_generate(
        dialogs_list, temperature=1.0, max_tokens=96, top_p=1.0
    )


def _agent_dialog_filename(env, agent_idx: int) -> str:
    """Same as ACL24 gpt_path: one file per agent, named by endogenous name."""
    name = env.get_agent(str(agent_idx)).endogenous["name"]
    return str(name).replace("/", "_").replace(os.sep, "_")


def _append_rollout_io_log(
    io_log_dir: str,
    env,
    agent_idx: int,
    user_text: str,
    assistant_text: str,
    section_title: str,
) -> None:
    """Append one user/assistant pair; format aligned with ACL24 simulate.py."""
    path = os.path.join(io_log_dir, _agent_dialog_filename(env, agent_idx))
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n======== {section_title} ========\n")
        f.write(f">>>>>>>>>user: {user_text}\n")
        f.write(f">>>>>>>>>assistant: {assistant_text}\n")


def build_monthly_prompt_from_state(
    env,
    obs: Dict[str, Any],
    agent_idx: int,
    memory_state: Dict[str, Any],
    world_start_time,
    relativedelta,
) -> str:
    """
    Single-turn monthly **user** string for VERL/GRPO (thought.txt).

    Uses only live ``env`` / ``obs`` plus **compressed** ``memory_state``
    (recent_3m_summary, last actions, etc.). Does **not** splice raw
    ``dialog_queue`` turns or quarter-reflection paragraphs into the prompt.
    """
    return build_monthly_observation(
        env, obs, agent_idx, memory_state, world_start_time, relativedelta
    )


def capture_env_snapshot(
    env,
    obs: Dict[str, Any],
    memory_states: List[Dict[str, Any]],
    initial_price: float,
    out_dir: str,
    stem: str = "s_y",
) -> Dict[str, str]:
    """
    Persist a restorable economy slice for state bank (thought.txt Stage 0).

    Writes separate pickle files under ``out_dir``; returns **relative** path
    keys for ``state_bank.json`` indexing. Large objects are never embedded in
    the JSON index.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{stem}_env.pkl"), "wb") as f:
        pkl.dump(env, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, f"{stem}_obs.pkl"), "wb") as f:
        pkl.dump(obs, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, f"{stem}_memory.pkl"), "wb") as f:
        pkl.dump(copy.deepcopy(memory_states), f, protocol=pkl.HIGHEST_PROTOCOL)
    rng_payload = {"numpy": np.random.get_state(), "python": random.getstate()}
    with open(os.path.join(out_dir, f"{stem}_rng.pkl"), "wb") as f:
        pkl.dump(rng_payload, f, protocol=pkl.HIGHEST_PROTOCOL)
    hist_payload: Dict[str, Any] = {"initial_price": float(initial_price)}
    with open(os.path.join(out_dir, f"{stem}_history.pkl"), "wb") as f:
        pkl.dump(hist_payload, f, protocol=pkl.HIGHEST_PROTOCOL)
    return {
        "snapshot_path": f"{stem}_env.pkl",
        "obs_path": f"{stem}_obs.pkl",
        "memory_state_path": f"{stem}_memory.pkl",
        "rng_state_path": f"{stem}_rng.pkl",
        "history_state_path": f"{stem}_history.pkl",
    }


def restore_env_snapshot(
    bank_base_dir: str,
    record: Mapping[str, Any],
    *,
    restore_rng: bool = True,
) -> Tuple[Any, Any, List[Dict[str, Any]], float]:
    """
    Load env, obs, per-agent memory, and ``initial_price`` from paths in
    ``record`` (relative to ``bank_base_dir``, usually the folder containing
    ``state_bank.json``).
    """
    def _p(key: str) -> str:
        rel = record.get(key)
        if not rel:
            raise KeyError(f"state bank record missing {key!r}")
        path = rel if os.path.isabs(rel) else os.path.join(bank_base_dir, rel)
        return path

    with open(_p("snapshot_path"), "rb") as f:
        env = pkl.load(f)
    with open(_p("obs_path"), "rb") as f:
        obs = pkl.load(f)
    with open(_p("memory_state_path"), "rb") as f:
        memory_states = pkl.load(f)
    with open(_p("history_state_path"), "rb") as f:
        hist = pkl.load(f)
    initial_price = float(hist["initial_price"])
    if restore_rng and record.get("rng_state_path"):
        with open(_p("rng_state_path"), "rb") as f:
            rng = pkl.load(f)
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return env, obs, memory_states, initial_price


def run_annual_rollout(
    env,
    obs,
    episode_length=None,
    io_log_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    memory_states: Optional[List[Dict[str, Any]]] = None,
    initial_price_for_stats: Optional[float] = None,
):
    if episode_length is None:
        episode_length = int(env.episode_length)

    if initial_price_for_stats is not None:
        initial_price = float(initial_price_for_stats)
    else:
        initial_price = float(env.world.price[0])
    if memory_states is not None:
        memory_states = copy.deepcopy(memory_states)
    else:
        memory_states = [default_memory_state() for _ in range(env.num_agents)]
    episode_samples = []
    global_counters = {
        "format_fail_count": 0,
        "invalid_action_count": 0,
        "fallback_action_count": 0,
        "out_of_range_count": 0,
        "negative_wealth_count": 0,
    }
    monthly_series = []

    if io_log_dir:
        os.makedirs(io_log_dir, exist_ok=True)

    for epi in range(episode_length):
        t = env.world.timestep
        dialogs = []
        for idx in range(env.num_agents):
            pt = build_monthly_prompt_from_state(
                env, obs, idx, memory_states[idx], world_start_time, relativedelta
            )
            dialogs.append([{"role": "user", "content": pt}])

        responses = generate_agent_actions_qwen(dialogs)

        before = {k: global_counters[k] for k in global_counters}
        actions = {"p": [0]}
        parsed_rows = []
        for idx, text in enumerate(responses):
            raw, valid_fmt, used_fb = parse_action_json(text, stats=global_counters)
            env_act = convert_raw_action_to_env_action(raw)
            actions[str(idx)] = env_act
            meta = build_sample_meta(env, obs, idx)
            parsed_rows.append((idx, text, raw, valid_fmt, used_fb, env_act, meta))

        for idx, text, raw, valid_fmt, used_fb, env_act, meta in parsed_rows:
            episode_samples.append(
                {
                    "agent_id": int(idx),
                    "month": int(t),
                    "prompt_text": dialogs[idx][0]["content"],
                    "response_text": text,
                    "parsed_action_raw": dict(raw),
                    "env_action": [int(env_act[0]), float(env_act[1])],
                    "valid_format": bool(valid_fmt),
                    "used_fallback": bool(used_fb),
                    "meta": meta,
                }
            )
            if io_log_dir:
                _append_rollout_io_log(
                    io_log_dir,
                    env,
                    idx,
                    dialogs[idx][0]["content"],
                    text,
                    section_title=(
                        f"monthly_action env_timestep={t} episode_index={epi}"
                    ),
                )

        obs, rew, done, info = env.step(actions)

        for idx in range(env.num_agents):
            coin = float(env.get_agent(str(idx)).inventory.get("Coin", 0.0))
            if coin < -1e-6:
                global_counters["negative_wealth_count"] += 1

        step_num = epi + 1
        if checkpoint_dir and (
            step_num % 6 == 0 or step_num == episode_length
        ):
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(
                os.path.join(checkpoint_dir, f"actions_{step_num}.pkl"), "wb"
            ) as f:
                pkl.dump(actions, f)
            with open(os.path.join(checkpoint_dir, f"obs_{step_num}.pkl"), "wb") as f:
                pkl.dump(obs, f)
            with open(os.path.join(checkpoint_dir, f"env_{step_num}.pkl"), "wb") as f:
                pkl.dump(env, f)
            with open(
                os.path.join(checkpoint_dir, f"dense_log_{step_num}.pkl"), "wb"
            ) as f:
                pkl.dump(env.dense_log, f)

        for idx, text, raw, valid_fmt, used_fb, env_act, meta in parsed_rows:
            update_memory_state(memory_states[idx], raw, env_act, int(t))

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

        if (epi + 1) % 3 == 0:
            q_rows = monthly_series[-3:]
            reflection_prompts = [
                build_quarterly_reflection_prompt(
                    q_rows,
                    memory_states[idx].get("recent_3m_summary", ""),
                )
                for idx in range(env.num_agents)
            ]
            reflection_texts: List[str] = []
            agent_mb = 16
            for start in range(0, env.num_agents, agent_mb):
                end = min(start + agent_mb, env.num_agents)
                sub_dialogs = [
                    [{"role": "user", "content": reflection_prompts[i]}]
                    for i in range(start, end)
                ]
                sub_out = generate_quarterly_reflection_qwen(sub_dialogs)
                reflection_texts.extend([t.strip() for t in sub_out])
            for idx in range(env.num_agents):
                memory_states[idx]["quarter_reflection_summary"] = reflection_texts[idx]

    env_stats = {
        "initial_price": float(initial_price),
        "monthly_series": monthly_series,
        "format_fail_count": int(global_counters["format_fail_count"]),
        "invalid_action_count": int(global_counters["invalid_action_count"]),
        "fallback_action_count": int(global_counters["fallback_action_count"]),
        "out_of_range_count": int(global_counters["out_of_range_count"]),
        "negative_wealth_count": int(global_counters["negative_wealth_count"]),
    }
    out = {
        "episode_samples": episode_samples,
        "env_stats": env_stats,
        "dense_log": env.dense_log,
    }
    if io_log_dir:
        out["io_log_dir"] = io_log_dir
    return out


def complex_actions(env, obs, beta=0.1, gamma=0.1, h=1):

    def consumption_len(price, wealth, curr_income, last_income, interest_rate):
        c = (price/(1e-8+wealth+curr_income))**beta
        c = min(max(c//0.02, 0), 50)
        return c
    def consumption_cats(price, wealth, curr_income, last_income, interest_rate):
        h1 = h / (1 + interest_rate)
        g = curr_income/(last_income+1e-8) - 1
        d = wealth/(last_income+1e-8) - h1
        c = 1 + (d - h1*g)/(1 + g + 1e-8)
        c = min(max(c*curr_income/(wealth+curr_income+1e-8)//0.02, 0), 50)
        return c
    def work_income_wealth(price, wealth, curr_income, last_income, expected_income, interest_rate):
        return int(np.random.uniform() < (curr_income/(wealth*(1 + interest_rate)+1e-8))**gamma)
    
    consumption_funs = [consumption_len, consumption_cats]
    work_funs = [work_income_wealth]

    actions = {}
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        price = env.world.price[-1]
        wealth = this_agent.inventory['Coin']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        max_income = max_l * this_agent.state['skill']
        last_income = this_agent.income['Coin']
        expected_income = max_l * this_agent.state['expected skill']
        interest_rate = env.world.interest_rate[-1]
        if 'consumption_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['consumption_fun_idx'] = np.random.choice(range(len(consumption_funs)))
        if 'work_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['work_fun_idx'] = np.random.choice(range(len(work_funs)))
        work_fun = work_funs[this_agent.endogenous['work_fun_idx']]
        l = work_fun(price, wealth, max_income, last_income, expected_income, interest_rate)
        curr_income = l * max_income
        consumption_fun = consumption_funs[this_agent.endogenous['consumption_fun_idx']]
        c = consumption_fun(price, wealth, curr_income, last_income, interest_rate)
        actions[str(idx)] = [l, c]
    actions['p'] = [0]
    return actions
    

def main(policy_model='gpt', num_agents=100, episode_length=240, dialog_len=3, beta=0.1, gamma=0.1, h=1, max_price_inflation=0.1, max_wage_inflation=0.05, model_type='gpt'):
    env_config['n_agents'] = num_agents
    env_config['episode_length'] = episode_length
    if policy_model == 'gpt':
        total_cost = 0
        env_config['flatten_masks'] = False
        env_config['flatten_observations'] = False
        env_config['components'][0]['SimpleLabor']['scale_obs'] = False
        env_config['components'][1]['PeriodicBracketTax']['scale_obs'] = False
        env_config['components'][3]['SimpleSaving']['scale_obs'] = False
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation
        
        gpt_error = 0
        from collections import deque
        dialog_queue = [deque(maxlen=dialog_len) for _ in range(env_config['n_agents'])]
        dialog4ref_queue = [deque(maxlen=7) for _ in range(env_config['n_agents'])]

    elif policy_model == 'qwen_rollout':
        env_config['flatten_masks'] = False
        env_config['flatten_observations'] = False
        env_config['components'][0]['SimpleLabor']['scale_obs'] = False
        env_config['components'][1]['PeriodicBracketTax']['scale_obs'] = False
        env_config['components'][3]['SimpleSaving']['scale_obs'] = False
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation

    elif policy_model == 'complex':
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation

    t = time()
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    actions = {}
    if policy_model == 'complex':
        policy_model_save = f'{policy_model}-{beta}-{gamma}-{h}-{max_price_inflation}-{max_wage_inflation}'
    elif policy_model == 'gpt':
        policy_model_save = f'{policy_model}-{dialog_len}-noperception-reflection-1'
    elif policy_model == 'qwen_rollout':
        policy_model_save = 'qwen_rollout'
    else:
        policy_model_save = str(policy_model)
    policy_model_save = f'{policy_model_save}-{num_agents}agents-{episode_length}months'
    if not os.path.exists(f'{save_path}data/{policy_model_save}'):
        os.makedirs(f'{save_path}data/{policy_model_save}')
    if not os.path.exists(f'{save_path}figs/{policy_model_save}'):
        os.makedirs(f'{save_path}figs/{policy_model_save}')

    if policy_model == 'qwen_rollout':
        io_log_dir = f'{save_path}data/{policy_model_save}/dialogs'
        os.makedirs(io_log_dir, exist_ok=True)
        ckpt_dir = f"{save_path}data/{policy_model_save}"
        rollout_out = run_annual_rollout(
            env,
            obs,
            episode_length=episode_length,
            io_log_dir=io_log_dir,
            checkpoint_dir=ckpt_dir,
        )
        artifact_paths = save_rollout_artifacts(
            save_path,
            policy_model_save,
            rollout_out['episode_samples'],
            rollout_out['env_stats'],
            rollout_out['dense_log'],
        )
        with open(f'{save_path}data/{policy_model_save}/dense_log.pkl', 'wb') as f:
            pkl.dump(rollout_out['dense_log'], f)
        with open(f'{save_path}data/{policy_model_save}/rollout/rollout_result.pkl', 'wb') as f:
            pkl.dump(
                {
                    'env_stats': rollout_out['env_stats'],
                    'artifact_paths': artifact_paths,
                    'n_samples': len(rollout_out['episode_samples']),
                    'io_log_dir': io_log_dir,
                    'checkpoint_dir': ckpt_dir,
                },
                f,
            )
        print(
            f'qwen_rollout done in {time()-t:.1f}s; io logs: {io_log_dir}; '
            f'artifacts: {artifact_paths}'
        )
        return rollout_out

    for epi in range(env.episode_length):
        if policy_model == 'gpt':
            actions, gpt_error, total_cost = gpt_actions(env, obs, dialog_queue, dialog4ref_queue, f'{save_path}data/{policy_model_save}/dialogs', gpt_error, total_cost, model_type=model_type)
        elif policy_model == 'complex':
            actions = complex_actions(env, obs, beta=beta, gamma=gamma, h=h)
        obs, rew, done, info = env.step(actions)
        if (epi+1) % 3 == 0:
            print(f'step {epi+1} done, cost {time()-t:.1f}s')
            if policy_model == 'gpt':
                print(f'#errors: {gpt_error}, cost ${total_cost:.1f} so far')
            t = time()
        if (epi+1) % 6 == 0 or epi+1 == env.episode_length:
            with open(f'{save_path}data/{policy_model_save}/actions_{epi+1}.pkl', 'wb') as f:
                pkl.dump(actions, f)
            with open(f'{save_path}data/{policy_model_save}/obs_{epi+1}.pkl', 'wb') as f:
                pkl.dump(obs, f)
            with open(f'{save_path}data/{policy_model_save}/env_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env, f)
            if policy_model == 'gpt':
                with open(f'{save_path}data/{policy_model_save}/dialog_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog_queue, f)
                with open(f'{save_path}data/{policy_model_save}/dialog4ref_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog4ref_queue, f)
            with open(f'{save_path}data/{policy_model_save}/dense_log_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env.dense_log, f)
                
    with open(f'{save_path}data/{policy_model_save}/dense_log.pkl', 'wb') as f:
        pkl.dump(env.dense_log, f)
        
    if policy_model == 'gpt':
        print(f'#gpt errors: {gpt_error}')


# --- Thin helpers for VERL / EconGenerationManager (orchestration stays outside this file) ---
def build_agent_month_prompt(
    env,
    obs: Dict[str, Any],
    agent_id: int,
    memory_state: Dict[str, Any],
) -> str:
    """Monthly user string for one agent (GRPO month-level sample)."""
    return build_monthly_prompt_from_state(
        env, obs, int(agent_id), memory_state, world_start_time, relativedelta
    )


def step_month_with_joint_action(env, joint_action: Dict[str, Any]):
    """One env month step; ``joint_action`` matches ``env.step`` contract."""
    return env.step(joint_action)


def update_memory_and_history(
    memory_state: Dict[str, Any],
    parsed_raw: Dict[str, float],
    env_action: List[float],
    month_index: int,
) -> None:
    update_memory_state(memory_state, parsed_raw, env_action, int(month_index))


if __name__ == "__main__":
    fire.Fire(main)