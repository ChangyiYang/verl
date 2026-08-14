# Copyright 2026 Liquid AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Token-level rewards for frame-synchronous duplex actions."""

from __future__ import annotations

from typing import Any

import torch


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
    """Map wall-clock reward windows directly onto LS action tokens."""
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


__all__ = ["score_duplex_windows"]
