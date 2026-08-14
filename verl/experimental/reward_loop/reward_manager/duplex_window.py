# Copyright 2026 Liquid AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Per-action wall-clock rewards for async duplex rollouts."""

from __future__ import annotations

from typing import Any

import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


def _field(window: Any, name: str, default=None):
    if isinstance(window, dict):
        return window.get(name, default)
    return getattr(window, name, default)


def score_duplex_windows(
    responses: torch.Tensor,
    action_response_pos: torch.Tensor,
    is_listen: torch.Tensor,
    frame_time: torch.Tensor,
    windows: list[Any] | None,
) -> tuple[torch.Tensor, int]:
    """Return token-level scores and the number of rewarded frame hits."""
    reward = torch.zeros_like(responses, dtype=torch.float32)
    valid = action_response_pos >= 0
    positions = action_response_pos[valid]
    listen = is_listen[valid].bool()
    times = frame_time[valid]
    hit_count = 0

    if windows is None:
        windows = []
    for window in windows:
        selected = (times >= float(_field(window, "t_start"))) & (times <= float(_field(window, "t_end")))
        only_on = _field(window, "only_on")
        if only_on == "listen":
            selected &= listen
        elif only_on == "speak":
            selected &= ~listen
        target = positions[selected]
        target = target[(target >= 0) & (target < reward.numel())]
        if target.numel():
            reward[target] += float(_field(window, "value"))
            hit_count += int(target.numel())

    return reward, hit_count


@register("duplex_window")
class DuplexWindowRewardManager(RewardManagerBase):
    """Map dataset reward windows directly onto LS action positions."""

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        # RewardLoopWorker supplies optional routing arguments shared by remote
        # reward managers; this local deterministic scorer does not use them.
        del kwargs
        super().__init__(config=config, tokenizer=tokenizer, compute_score=compute_score)

    async def run_single(self, data: DataProto) -> dict:
        if len(data) != 1:
            raise ValueError(f"DuplexWindowRewardManager expects one sample, got {len(data)}")
        item = data[0]
        for key in ("duplex_action_response_pos", "duplex_is_listen", "duplex_frame_time"):
            if key not in item.batch:
                raise KeyError(f"Duplex reward requires {key}")

        windows = item.non_tensor_batch.get("reward_windows")
        reward, hit_count = score_duplex_windows(
            responses=item.batch["responses"],
            action_response_pos=item.batch["duplex_action_response_pos"],
            is_listen=item.batch["duplex_is_listen"],
            frame_time=item.batch["duplex_frame_time"],
            windows=windows,
        )

        return {
            "reward_score": reward.cpu(),
            "reward_extra_info": {"duplex_hit_frames": hit_count, "duplex_windows": len(windows)},
        }

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[torch.Tensor]) -> torch.Tensor:
        expected_shape = tuple(data.batch["responses"].shape)
        result = torch.stack([torch.as_tensor(score) for score in scores]).to(
            device=data.batch["responses"].device, dtype=torch.float32
        )
        if tuple(result.shape) != expected_shape:
            raise RuntimeError(f"Duplex reward shape {tuple(result.shape)} != responses {expected_shape}")
        return result


__all__ = ["DuplexWindowRewardManager", "score_duplex_windows"]
