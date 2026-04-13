"""Count tokens for build_monthly_observation without loading vLLM."""
from __future__ import annotations

import re
import sys
import types
from datetime import datetime

import numpy as np

# Stub simulate_utils before rollout_collector import (avoids vllm)
_stub = types.ModuleType("simulate_utils")
_stub.brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103]) * 100 / 12)


def _prettify_document(document: str) -> str:
    return re.sub(r"\s+", " ", document).strip()


_stub.prettify_document = _prettify_document
sys.modules["simulate_utils"] = _stub

from dateutil.relativedelta import relativedelta  # noqa: E402

from rollout_collector import build_monthly_observation, default_memory_state  # noqa: E402


class _Labor:
    num_labor_hours = 1.0


class _World:
    def __init__(self, t: int):
        self.timestep = t
        self.interest_rate = [0.01, 0.02]
        self.price = [1.0, 1.02]


class _Agent:
    def __init__(self):
        self.state = {"skill": 0.5}
        self.inventory = {"Coin": 1200.0}
        self.consumption = {"Coin": 80.0}
        self.income = {"Coin": 100.0}
        self.endogenous = {
            "name": "TestAgent",
            "age": 30,
            "city": "TestCity",
            "job": "Unemployment",
            "offer": "Office",
        }


def _make_env(timestep: int):
    from types import SimpleNamespace

    agents = {"0": _Agent()}
    return SimpleNamespace(
        world=_World(timestep),
        _components_dict={"SimpleLabor": _Labor()},
        dense_log={
            "states": [{"0": {"skill": 0.48, "endogenous": {"job": "Unemployment"}}}],
            "actions": [{"0": {"SimpleLabor": 0}}],
        },
        get_agent=lambda sid: agents[sid],
    )


def _make_obs(agent_idx: int = 0):
    return {
        "p": {
            f"p{agent_idx}": {
                "PeriodicBracketTax-tax_paid": 12.5,
                "PeriodicBracketTax-lump_sum": 3.0,
            },
            "PeriodicBracketTax-curr_rates": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        }
    }


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    world_start = datetime.strptime("2001.01", "%Y.%m")
    mem = default_memory_state()
    mem["recent_3m_summary"] = "m0:w=0.50,c=0.30; m1:w=0.52,c=0.28"
    mem["quarter_reflection_summary"] = (
        "Labor slack persisted; prices rose slightly; keep consumption cautious next quarter."
    )

    for t in (0, 3):
        env = _make_env(t)
        obs = _make_obs(0)
        prompt_str = build_monthly_observation(
            env, obs, 0, mem, world_start, relativedelta
        )
        ids_raw = tokenizer.encode(prompt_str, add_special_tokens=False)
        dialog = [{"role": "user", "content": prompt_str}]
        full_text = tokenizer.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=True
        )
        ids_chat = tokenizer.encode(full_text, add_special_tokens=False)

        print(f"--- env.world.timestep={t} ---")
        print(f"prompt chars: {len(prompt_str)}")
        print(f"encode(prompt_str, add_special_tokens=False): {len(ids_raw)} tokens")
        print(f"encode(apply_chat_template user+gen_prompt, add_special_tokens=False): {len(ids_chat)} tokens")


if __name__ == "__main__":
    main()
