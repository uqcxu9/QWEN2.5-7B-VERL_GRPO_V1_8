from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np

# Tokenizer type: HuggingFace PreTrainedTokenizer from ``get_qwen_model()[1]`` —
# do **not** add a second AutoTokenizer initialization path in this module.


def build_response_only_training_fields(
    prompt_text: str,
    response_text: str,
    tokenizer: Any,
    *,
    align_rollout_chat_template: bool = True,
    append_eos_to_response: bool = False,
) -> Dict[str, Any]:
    """
    Build ``input_ids``, ``attention_mask``, and response-only ``loss_mask``.

    1. When ``align_rollout_chat_template`` is True (default), the prompt is the
       same string vLLM sees in ``simulate_utils.qwen_rollout_batch_generate``:
       ``apply_chat_template([{{"role":"user","content": prompt_text}}], ...,
       add_generation_prompt=True)``. This matches ACL24-style monthly **user**
       turn + generation header; only the assistant **completion** (``response_text``)
       is trained.
    2. ``prompt_text`` and ``response_text`` are encoded **separately**, then
       concatenated: ``input_ids = prompt_ids + response_ids``.
    3. ``attention_mask`` is all ones.
    4. ``loss_mask``: zeros on prompt span, ones on response span.
    5. ``kl_mask``: same as ``loss_mask`` so KL terms apply only to response
       tokens (prompt / environment context excluded), per Search-R1-style masking.

    Quarterly reflection / multi-turn history are **excluded** from this path
    because they are not part of ``response_text`` here — consistent with
    masking "only monthly assistant response tokens" for the monthly-action task.
    """
    if align_rollout_chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "tokenizer must support apply_chat_template when "
                "align_rollout_chat_template=True (use get_qwen_model()[1])."
            )
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_str = prompt_text

    def _ids(text: str) -> List[int]:
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(text, add_special_tokens=False)
        elif callable(tokenizer):
            out = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            if isinstance(out, Mapping):
                ids = out.get("input_ids")
            else:
                ids = out
        else:
            raise ValueError(
                "tokenizer must provide .encode() or be callable (use get_qwen_model()[1])."
            )
        if not isinstance(ids, list):
            ids = list(ids)
        return [int(t) for t in ids]

    prompt_ids = _ids(prompt_str)
    resp_ids = _ids(response_text if response_text is not None else "")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if append_eos_to_response and eos_id is not None:
        if not resp_ids or resp_ids[-1] != eos_id:
            resp_ids = resp_ids + [int(eos_id)]

    input_ids = prompt_ids + resp_ids
    attention_mask = [1] * len(input_ids)
    loss_mask = [0] * len(prompt_ids) + [1] * len(resp_ids)
    # Search-R1-style: exclude prompt/context from KL regularizer same as policy loss.
    kl_mask = list(loss_mask)

    return {
        "prompt_token_ids": prompt_ids,
        "response_token_ids": resp_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "kl_mask": kl_mask,
    }


def export_grpo_samples(
    annual_rollout_output: Mapping[str, Any],
    tokenizer: Any,
    replica_id: int = 0,
    episode_id: int = 0,
    *,
    state_group_id: str = "default",
    align_rollout_chat_template: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert ``run_annual_rollout`` return value to GRPO-ready month-level samples.

    One dict per (agent, month) in ``episode_samples``. Fields are filled for
    subsequent replica-level ``annual_reward`` / ``advantage`` broadcast (left
    ``None`` here).
    """
    samples_in = annual_rollout_output.get("episode_samples") or []
    out: List[Dict[str, Any]] = []

    for row in samples_in:
        if not isinstance(row, Mapping):
            continue
        prompt_text = str(row.get("prompt_text", "") or "")
        response_text = str(row.get("response_text", "") or "")

        fields = build_response_only_training_fields(
            prompt_text,
            response_text,
            tokenizer,
            align_rollout_chat_template=align_rollout_chat_template,
        )

        agent_id = int(row.get("agent_id", -1))
        month = int(row.get("month", -1))
        resp_ids = fields["response_token_ids"]
        n_resp = len(resp_ids)
        sg = str(state_group_id)
        epi = int(episode_id)
        rid = int(replica_id)
        group_key = f"ep{epi}:state_group:{sg}"
        replica_uid = f"ep{epi}:rep{rid}"
        sample_uid = f"ep{epi}:rep{rid}:a{agent_id}:m{month}"

        out.append(
            {
                "episode_id": epi,
                "replica_id": rid,
                "agent_id": agent_id,
                "month": month,
                "state_group_id": sg,
                "group_key": group_key,
                "replica_uid": replica_uid,
                "sample_uid": sample_uid,
                "response_token_count": int(n_resp),
                "prompt_text": prompt_text,
                "response_text": response_text,
                "prompt_token_ids": fields["prompt_token_ids"],
                "response_token_ids": fields["response_token_ids"],
                "input_ids": fields["input_ids"],
                "attention_mask": fields["attention_mask"],
                "loss_mask": fields["loss_mask"],
                "kl_mask": fields["kl_mask"],
                "valid_format": bool(row.get("valid_format", False)),
                "used_fallback": bool(row.get("used_fallback", False)),
                "parsed_action_raw": dict(row.get("parsed_action_raw") or {}),
                "env_action": list(row.get("env_action") or []),
                "meta": dict(row.get("meta") or {}),
                "annual_reward": None,
                "advantage": None,
            }
        )

    return out


def _percentile(a: np.ndarray, q: float) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))


def summarize_grpo_sample_lengths(grpo_samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Length and quality summary for exported samples.

    ``month_idx`` groups use the sample's ``month`` field (env timestep, same as
    ``rollout_collector`` / ACL24 monthly step index).
    """
    n = len(grpo_samples)
    if n == 0:
        empty_stats = {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
        return {
            "prompt_token_len": dict(empty_stats),
            "response_token_len": dict(empty_stats),
            "input_token_len": dict(empty_stats),
            "total_samples": 0,
            "invalid_format_ratio": 0.0,
            "fallback_ratio": 0.0,
            "by_month": {},
        }

    pl = np.array([len(s["prompt_token_ids"]) for s in grpo_samples], dtype=float)
    rl = np.array([len(s["response_token_ids"]) for s in grpo_samples], dtype=float)
    il = np.array([len(s["input_ids"]) for s in grpo_samples], dtype=float)

    def _stat(arr: np.ndarray) -> Dict[str, float]:
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "p50": _percentile(arr, 50),
            "p90": _percentile(arr, 90),
            "p95": _percentile(arr, 95),
        }

    invalid = sum(1 for s in grpo_samples if not s.get("valid_format", False))
    fallback = sum(1 for s in grpo_samples if s.get("used_fallback", False))

    by_month: Dict[int, Dict[str, Any]] = {}
    month_buckets: Dict[int, List[int]] = {}
    for i, s in enumerate(grpo_samples):
        m = int(s.get("month", -1))
        month_buckets.setdefault(m, []).append(i)

    for m, idxs in sorted(month_buckets.items()):
        pl_m = pl[idxs]
        rl_m = rl[idxs]
        il_m = il[idxs]
        by_month[m] = {
            "count": len(idxs),
            "prompt_mean": float(np.mean(pl_m)),
            "response_mean": float(np.mean(rl_m)),
            "input_max": float(np.max(il_m)),
        }

    return {
        "prompt_token_len": _stat(pl),
        "response_token_len": _stat(rl),
        "input_token_len": _stat(il),
        "total_samples": n,
        "invalid_format_ratio": invalid / n,
        "fallback_ratio": fallback / n,
        "by_month": by_month,
    }


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


def save_grpo_samples(
    base_save_path: str,
    policy_model_save: str,
    grpo_samples: Sequence[MutableMapping[str, Any]],
    stats_summary: Mapping[str, Any],
) -> Dict[str, str]:
    """
    Write under ``{base_save_path}/data/{policy_model_save}/rollout/``:

    * ``grpo_samples.jsonl``
    * ``grpo_samples.pkl``
    * ``grpo_length_stats.json``
    """
    root = os.path.join(base_save_path, "data", policy_model_save, "rollout")
    os.makedirs(root, exist_ok=True)

    jsonl_path = os.path.join(root, "grpo_samples.jsonl")
    pkl_path = os.path.join(root, "grpo_samples.pkl")
    stats_path = os.path.join(root, "grpo_length_stats.json")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in grpo_samples:
            f.write(json.dumps(_json_safe(dict(row)), ensure_ascii=False) + "\n")

    with open(pkl_path, "wb") as f:
        pickle.dump(list(grpo_samples), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(dict(stats_summary)), f, ensure_ascii=False, indent=2)

    return {
        "grpo_samples_jsonl": jsonl_path,
        "grpo_samples_pkl": pkl_path,
        "grpo_length_stats_json": stats_path,
    }


# ---------------------------------------------------------------------------
# Minimal demo (loads tokenizer via get_qwen_model — same as rollout)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    _fake_rollout = {
        "episode_samples": [
            {
                "agent_id": 0,
                "month": 0,
                "prompt_text": (
                    "[Month 2001.01 | env_timestep=0 | agent_id=0] "
                    "identity: name=Demo, age=30. Reply with JSON only: "
                    '{"work": 0.xx, "consumption": 0.xx}'
                ),
                "response_text": '{"work": 0.5, "consumption": 0.5}',
                "parsed_action_raw": {"work": 0.5, "consumption": 0.5},
                "env_action": [1, 25.0],
                "valid_format": True,
                "used_fallback": False,
                "meta": {"savings": 0.0, "price": 100.0},
            }
        ],
        "env_stats": {},
    }

    if os.environ.get("GRPO_EXPORTER_DEMO_FAKE"):
        class _DemoTok:
            eos_token_id = 999

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "<|user|>" + messages[0]["content"] + "<|assistant|>"

            def encode(self, text, add_special_tokens=False):
                return list(range(20, 20 + min(len(text), 6)))

        _tok = _DemoTok()
    else:
        from simulate_utils import get_qwen_model

        _, _tok = get_qwen_model()

    _grpo = export_grpo_samples(
        _fake_rollout, _tok, replica_id=0, episode_id=0, state_group_id="demo"
    )
    _stats = summarize_grpo_sample_lengths(_grpo)
    print("demo: 1 sample, input_len=", len(_grpo[0]["input_ids"]))
    print("demo: loss_mask sum (trainable tokens)=", sum(_grpo[0]["loss_mask"]))
    print("demo: stats keys=", list(_stats.keys()))
    print("demo: sample keys=", sorted(_grpo[0].keys()))
    if not os.environ.get("GRPO_EXPORTER_DEMO_FAKE"):
        print(
            "(Set GRPO_EXPORTER_DEMO_FAKE=1 to run demo without loading vLLM.)",
            file=sys.stderr,
        )
