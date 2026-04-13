import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
import re
import os
import multiprocessing
import scipy
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

save_path = './'

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103])*100/12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime
world_start_time = datetime.strptime('2001.01', '%Y.%m')

prompt_cost_1k, completion_cost_1k = 0.001, 0.002
QWEN_MODEL = None
QWEN_TOKENIZER = None

QWEN_SYSTEM_PROMPT = {
    'role': 'system',
    'content': (
        'You are an economic agent making monthly decisions. '
        'You must respond with ONLY a JSON object with exactly two keys: "work" and "consumption", both floats between 0 and 1 (in steps of 0.02).\n'
        'Example input: "...how is your willingness to work this month? ...how would you plan your expenditures..."\n'
        'Example output: {"work": 0.80, "consumption": 0.30}\n'
        'Another example output: {"work": 0.52, "consumption": 0.14}\n'
        'Do NOT output any explanation, only the JSON object.'
    )
}

QWEN_REFLECTION_SYSTEM_PROMPT = {
    'role': 'system',
    'content': (
        'You are an economic agent reflecting on recent economic conditions. '
        'Give a concise analysis in under 200 words about labor, consumption, and financial markets.'
    )
}

def get_qwen_model():

    global QWEN_MODEL, QWEN_TOKENIZER
    if QWEN_MODEL is None:
        print("Loading Qwen3-4B-Instruct-2507...")
        QWEN_MODEL = LLM(
            model="/workspace/model/Qwen3-4B-Instruct-2507",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=True
        )
        from transformers import AutoTokenizer
        QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
            "/workspace/model/Qwen3-4B-Instruct-2507",
            trust_remote_code=True
        )
        print(" Qwen model loaded!")
    return QWEN_MODEL, QWEN_TOKENIZER

def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def _is_reflection(dialogs):
    """Check if this is a reflection prompt (not a decision prompt)."""
    last_user = [d for d in dialogs if d['role'] == 'user']
    if last_user:
        return 'reflect on' in last_user[-1]['content'].lower()
    return False

def get_multiple_completion(dialogs, num_cpus=15, temperature=0, max_tokens=100, model_type='gpt'):
    if model_type == 'qwen':
        llm, tokenizer = get_qwen_model()

        prompts = []
        for dialog in dialogs:
            sys_prompt = QWEN_REFLECTION_SYSTEM_PROMPT if _is_reflection(dialog) else QWEN_SYSTEM_PROMPT
            dialog_with_sys = [sys_prompt] + list(dialog)
            prompt = tokenizer.apply_chat_template(
                dialog_with_sys,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )

        outputs = llm.generate(prompts, sampling_params)

        json_results = []
        full_results = []
        for output in outputs:
            full_text = output.outputs[0].text.strip()
            full_results.append(full_text)

            cleaned = full_text.replace('```json', '').replace('```', '')
            json_match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
            json_results.append(cleaned.strip())

        return json_results, full_results, 0.0
    else:
        from functools import partial
        get_completion_partial = partial(get_completion, temperature=temperature, max_tokens=max_tokens, model_type='gpt')
        with multiprocessing.Pool(processes=num_cpus) as pool:
            results = pool.map(get_completion_partial, dialogs)
        total_cost = sum([cost for _, cost in results])
        return [response for response, _ in results], total_cost


def get_completion(dialogs, temperature=0, max_tokens=100, model_type='gpt'):
    if model_type == 'qwen':
        # Single-dialog qwen inference (delegates to batch of 1)
        json_results, full_results, cost = get_multiple_completion([dialogs], temperature=temperature, max_tokens=max_tokens, model_type='qwen')
        return json_results[0], full_results[0], cost
    else:
        import openai
        openai.api_key = 'Your Key'
        import time

        max_retries = 20
        for i in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=dialogs,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                this_cost = prompt_tokens/1000*prompt_cost_1k + completion_tokens/1000*completion_cost_1k
                return response.choices[0].message["content"], this_cost
            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(6)
                else:
                    print(f"An error of type {type(e).__name__} occurred: {e}")
                    return "Error", 0.0

def format_numbers(numbers):
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'

def format_percentages(numbers):
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'