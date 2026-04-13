from typing import Any, Dict, List, Optional
import argparse
import copy
import fire
import json
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
    collect_env_stats_step,
    default_memory_state,
    update_memory_state,
)
from build_state_bank import (
    build_state_bank_records,
    default_state_bank_out_path,
    save_state_bank,
)

with open('config.yaml', "r") as f:
    run_configuration = yaml.safe_load(f)
env_config = run_configuration.get('env')


# ---- state_bank snapshot helpers (ACL24 骨架外的附加 save 钩子) ----

def capture_env_snapshot(
    env,
    obs: Dict[str, Any],
    memory_states: List[Dict[str, Any]],
    initial_price: float,
    out_dir: str,
    stem: str = "s_y",
) -> Dict[str, str]:
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
        
def gpt_actions(env, obs, dialog_queue, dialog4ref_queue, gpt_path, gpt_error, total_cost, model_type='gpt', global_counters=None, per_agent_log=None):
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
    if model_type == 'qwen':
        if env.world.timestep%3 == 0 and env.world.timestep > 0:
            json_results, full_results, cost = get_multiple_completion([list(dialogs)[:2] + list(dialog4ref)[-3:-1] + list(dialogs)[-1:] for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue)], model_type=model_type)
        else:
            json_results, full_results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog_queue], model_type=model_type)
        total_cost += cost
    else:
        if env.world.timestep%3 == 0 and env.world.timestep > 0:
            json_results, cost = get_multiple_completion([list(dialogs)[:2] + list(dialog4ref)[-3:-1] + list(dialogs)[-1:] for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue)], model_type=model_type)
        else:
            json_results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog_queue], model_type=model_type)
        full_results = json_results  # GPT mode: no thinking, full == json
        total_cost += cost
    actions = {}
    for idx in range(env.num_agents):
        content = json_results[idx]
        full_content = full_results[idx]
        parse_ok = True
        fallback_used = False
        try:
            parsed = eval(content)
            extracted_actions = [float(parsed['work']), float(parsed['consumption'])]
            if not action_check(extracted_actions):
                extracted_actions = [1, 0.5]
                gpt_error += 1
                fallback_used = True
                if global_counters is not None:
                    global_counters["invalid_action_count"] += 1
        except:
            extracted_actions = [1, 0.5]
            gpt_error += 1
            fallback_used = True
            parse_ok = False
            if global_counters is not None:
                global_counters["format_fail_count"] += 1
        if fallback_used and global_counters is not None:
            global_counters["fallback_action_count"] += 1
        raw_work = float(extracted_actions[0])
        raw_cons_frac = float(extracted_actions[1])
        extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
        extracted_actions[1] /= 0.02
        actions[str(idx)] = extracted_actions
        if per_agent_log is not None:
            per_agent_log.append({
                "parsed_raw": {"work": raw_work, "consumption": raw_cons_frac},
                "env_action": [int(extracted_actions[0]), float(extracted_actions[1])],
            })
        # dialog queue stores JSON only (for future LLM input context)
        dialog_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
        dialog4ref_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
        # dialog file stores full thinking + JSON (for human review)
        with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
            f.write(f'''>>>>>>>>>user: {list(dialog_queue[idx])[-2]['content']}\n''')
            f.write(f'''>>>>>>>>>assistant [thinking+answer]:\n{full_content}\n\n''')
    actions['p'] = [0]

    if (env.world.timestep+1)%3 == 0:
        reflection_prompt = '''Given the previous quarter's economic environment, reflect on the labor, consumption, and financial markets, as well as their dynamics. What conclusions have you drawn?
        Your answer must be less than 200 words!'''
        reflection_prompt = prettify_document(reflection_prompt)
        for idx in range(env.num_agents):
            dialog4ref_queue[idx].append({'role': 'user', 'content': reflection_prompt})
        if model_type == 'qwen':
            json_results, full_results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog4ref_queue], temperature=0, max_tokens=200, model_type=model_type)
        else:
            json_results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog4ref_queue], temperature=0, max_tokens=200, model_type=model_type)
            full_results = json_results
        total_cost += cost
        for idx in range(env.num_agents):
            content = json_results[idx]
            full_content = full_results[idx]
            dialog4ref_queue[idx].append({'role': 'assistant', 'content': content})
            with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
                f.write(f'''>>>>>>>>>reflection [thinking+answer]:\n{full_content}\n\n''')
    return actions, gpt_error, total_cost

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

    elif policy_model == 'complex':
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation

    t = time()
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    actions = {}
    if policy_model == 'complex':
        policy_model_save = f'{policy_model}-{beta}-{gamma}-{h}-{max_price_inflation}-{max_wage_inflation}'
    if policy_model == 'gpt':
        policy_model_save = f'{policy_model}-{dialog_len}-noperception-reflection-1'
    policy_model_save = f'{policy_model_save}-{num_agents}agents-{episode_length}months'
    if not os.path.exists(f'{save_path}data/{policy_model_save}'):
        os.makedirs(f'{save_path}data/{policy_model_save}')
    if not os.path.exists(f'{save_path}figs/{policy_model_save}'):
        os.makedirs(f'{save_path}figs/{policy_model_save}')

    # ---- state_bank bookkeeping (extra save hooks, does not alter ACL24 decision loop) ----
    run_dir = f'{save_path}data/{policy_model_save}'
    state_bank_base = os.path.join(run_dir, 'state_bank')
    snapshots_root = os.path.join(state_bank_base, 'snapshots')
    os.makedirs(snapshots_root, exist_ok=True)
    initial_price = float(env.world.price[0])
    memory_states: List[Dict[str, Any]] = [default_memory_state() for _ in range(env.num_agents)]
    global_counters: Dict[str, int] = {
        "format_fail_count": 0,
        "invalid_action_count": 0,
        "fallback_action_count": 0,
        "out_of_range_count": 0,
        "negative_wealth_count": 0,
    }
    monthly_series: List[Dict[str, Any]] = []
    per_year_assets: Dict[int, Dict[str, str]] = {}

    for epi in range(env.episode_length):
        before = {k: global_counters[k] for k in global_counters}
        t_step = int(env.world.timestep)
        per_agent_log: List[Dict[str, Any]] = []
        if policy_model == 'gpt':
            actions, gpt_error, total_cost = gpt_actions(
                env, obs, dialog_queue, dialog4ref_queue,
                f'{save_path}data/{policy_model_save}/dialogs',
                gpt_error, total_cost, model_type=model_type,
                global_counters=global_counters, per_agent_log=per_agent_log,
            )
        elif policy_model == 'complex':
            actions = complex_actions(env, obs, beta=beta, gamma=gamma, h=h)
            for idx in range(env.num_agents):
                raw_work = float(actions[str(idx)][0])
                raw_cons_idx = float(actions[str(idx)][1])
                per_agent_log.append({
                    "parsed_raw": {"work": raw_work, "consumption": raw_cons_idx * 0.02},
                    "env_action": [int(raw_work), float(raw_cons_idx)],
                })
        obs, rew, done, info = env.step(actions)

        for idx in range(env.num_agents):
            log_entry = per_agent_log[idx]
            update_memory_state(
                memory_states[idx],
                log_entry["parsed_raw"],
                log_entry["env_action"],
                t_step,
            )
        for idx in range(env.num_agents):
            coin = float(env.get_agent(str(idx)).inventory.get("Coin", 0.0))
            if coin < -1e-6:
                global_counters["negative_wealth_count"] += 1
        month_action_stats = {k: global_counters[k] - before[k] for k in global_counters}
        monthly_series.append(
            collect_env_stats_step(env, int(epi), month_action_stats, global_counters, initial_price)
        )

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

        # state_bank: snapshot at end of each year (after year >= 1 completed)
        if (epi + 1) % 12 == 0:
            year_done = (epi + 1) // 12
            y_next = year_done + 1
            if y_next >= 2:
                stem = f"y{y_next}"
                out_dir = os.path.join(snapshots_root, stem)
                paths = capture_env_snapshot(env, obs, memory_states, initial_price, out_dir=out_dir, stem=stem)
                per_year_assets[y_next] = {k: os.path.join("snapshots", stem, v) for k, v in paths.items()}

    with open(f'{save_path}data/{policy_model_save}/dense_log.pkl', 'wb') as f:
        pkl.dump(env.dense_log, f)

    # ---- finalize state_bank ----
    env_stats = {
        "initial_price": float(initial_price),
        "monthly_series": monthly_series,
        "format_fail_count": int(global_counters["format_fail_count"]),
        "invalid_action_count": int(global_counters["invalid_action_count"]),
        "fallback_action_count": int(global_counters["fallback_action_count"]),
        "out_of_range_count": int(global_counters["out_of_range_count"]),
        "negative_wealth_count": int(global_counters["negative_wealth_count"]),
    }
    rollout_dir = os.path.join(run_dir, 'rollout')
    os.makedirs(rollout_dir, exist_ok=True)
    env_stats_path = os.path.join(rollout_dir, 'env_stats.json')
    with open(env_stats_path, "w", encoding="utf-8") as f:
        json.dump(env_stats, f, ensure_ascii=False, indent=2)
    print(f'Wrote {env_stats_path} ({len(monthly_series)} months)')

    bank_out = os.path.join(state_bank_base, 'state_bank.json')
    records = build_state_bank_records(env_stats, per_year_assets=per_year_assets)
    save_state_bank(records, bank_out)
    print(f'Wrote {len(records)} state_bank records → {bank_out}')

    if policy_model == 'gpt':
        print(f'#gpt errors: {gpt_error}')

if __name__ == "__main__":
    fire.Fire(main)