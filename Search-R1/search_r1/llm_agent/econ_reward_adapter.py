from __future__ import annotations

from typing import Tuple

import torch
from verl import DataProto

# 把每条轨迹的 年度总 reward annual_reward，平均分配到这条轨迹里所有可训练的 action token 上
def build_replica_token_level_scores(batch: DataProto) -> torch.Tensor:
    responses       = batch.batch["responses"]
    response_length = responses.size(-1)
    attention_mask  = batch.batch["attention_mask"]

    # 选一个 mask，决定“哪些 token 可以拿 reward”,保证了 reward 不会错误地分配到中间那些 observation token 上
    if "action_mask" in batch.batch:
        score_mask = batch.batch["action_mask"][:, -response_length:].float()
    elif "loss_mask" in batch.batch:
        score_mask = batch.batch["loss_mask"][:, -response_length:].float()
    else:
        score_mask = attention_mask[:, -response_length:].float()
    # 把 yearly reward 均匀摊到所有 action token 上，生成 token-level score
    ar = batch.batch["annual_reward"].float()
    if ar.dim() == 2:
        ar = ar.squeeze(-1)

    denom     = score_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    per_token = (ar.unsqueeze(-1) / denom) * score_mask
    return per_token



def expand_advantage_to_response(batch: DataProto) -> Tuple[torch.Tensor, torch.Tensor]:
    responses       = batch.batch["responses"]
    response_length = responses.size(-1)

    # 改：同上，只广播到action token
    if "action_mask" in batch.batch:
        adv_mask = batch.batch["action_mask"][:, -response_length:].float()
    elif "loss_mask" in batch.batch:
        adv_mask = batch.batch["loss_mask"][:, -response_length:].float()
    else:
        adv_mask = batch.batch["attention_mask"][:, -response_length:].float()

    # After compute_advantage(adv_estimator='grpo'), the batch contains
    # 'advantages' as a (bs, response_length) tensor already expanded to
    # token level.  If called with pre-expanded advantages, just apply the
    # action mask; otherwise fall back to scalar broadcast.
    if "advantages" in batch.batch:
        adv_exp = batch.batch["advantages"].float()[:, -response_length:] * adv_mask
        return adv_exp, adv_exp.clone()

    # Legacy path: scalar advantage per sample (e.g. from manual GRPO norm).
    adv = batch.batch["annual_reward"].float()
    if adv.dim() == 2:
        adv = adv.squeeze(-1)

    adv_exp = adv.unsqueeze(-1) * adv_mask
    ret_exp = adv_exp.clone()
    return adv_exp, ret_exp
