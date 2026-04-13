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
from transformers import AutoTokenizer

save_path = './'

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103])*100/12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime
world_start_time = datetime.strptime('2001.01', '%Y.%m')

prompt_cost_1k, completion_cost_1k = 0.001, 0.002
QWEN_MODEL = None
QWEN_TOKENIZER = None

def qwen_rollout_batch_generate(
    dialogs_list,
    temperature: float = 1.0,
    max_tokens: int = 50,
    top_p: float = 1.0,
):
    """
    Qwen vLLM batch generation for rollout (stochastic policy).
    Returns raw completion strings (no JSON stripping). Reuses get_qwen_model().
    """
    llm, tokenizer = get_qwen_model()
    prompts = []
    for dialog in dialogs_list:
        prompt = tokenizer.apply_chat_template(
            dialog,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def get_qwen_model():
    from vllm import LLM

    global QWEN_MODEL, QWEN_TOKENIZER
    if QWEN_MODEL is None:
        print("Loading Qwen2.5-7B-Instruct...")
        QWEN_MODEL = LLM(
            model="/workspace/model/Qwen2.5-7B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        from transformers import AutoTokenizer
        QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
            "/workspace/model/Qwen2.5-7B-Instruct",
            trust_remote_code=True
        )
        print(" Qwen model loaded!")
    return QWEN_MODEL, QWEN_TOKENIZER

def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def get_multiple_completion(dialogs, num_cpus=15, temperature=0, max_tokens=100, model_type='gpt'):
    if model_type == 'qwen':
        llm, tokenizer = get_qwen_model()
        
        prompts = []
        for dialog in dialogs:
            prompt = tokenizer.apply_chat_template(
                dialog,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )

        outputs = llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            text = text.replace('```json', '').replace('```', '')
            import re
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                text = json_match.group(0)
            results.append(text.strip())
        
        return results, 0.0
    else:
        from functools import partial
        get_completion_partial = partial(get_completion, temperature=temperature, max_tokens=max_tokens, model_type='gpt')
        with multiprocessing.Pool(processes=num_cpus) as pool:
            results = pool.map(get_completion_partial, dialogs)
        total_cost = sum([cost for _, cost in results])
        return [response for response, _ in results], total_cost


def get_completion(dialogs, temperature=0, max_tokens=100, model_type='gpt'):
    if model_type == 'qwen':
        llm, tokenizer = get_qwen_model()
        
        prompt = tokenizer.apply_chat_template(
            dialogs,
            tokenize=False,
            add_generation_prompt=True
        )

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )

        outputs = llm.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text.strip()
        
        text = text.replace('```json', '').replace('```', '')
        import re
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            text = json_match.group(0)
        
        return text.strip(), 0.0
    else:
        # GPT 代码不变
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