# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A no-dependency example :class:`SandboxWarmupPlugin`.

Warms a fake in-memory "env" so the warm-up machinery can be exercised without a
real sandbox backend (used by the CPU test and as a template for real plugins).
A real Modal plugin would, in ``warm``, spawn the sandbox, set terminal state,
upload files (keyed off ``sample``), and return the live handle.
"""

from __future__ import annotations

import asyncio
from typing import Any

from omegaconf import DictConfig

from verl.experimental.sandbox_warmup.plugin import SandboxWarmupPlugin


class DummyEnv:
    """Fake env handle: records the rollout it was warmed for and whether it is open."""

    def __init__(self, warm_key: str, task_id: Any) -> None:
        self.warm_key = warm_key
        self.task_id = task_id
        self.alive = True


class DummyWarmupPlugin(SandboxWarmupPlugin):
    """Example plugin: 'warms' a :class:`DummyEnv` after a tiny simulated delay."""

    async def warm(self, warm_key: str, sample: dict, config: DictConfig) -> DummyEnv:  # noqa: ARG002
        await asyncio.sleep(0.01)  # stand in for sandbox create + setup latency
        task_id = None
        if isinstance(sample, dict):
            extra = sample.get("extra_info") or {}
            task_id = extra.get("task_id") if isinstance(extra, dict) else sample.get("index")
        return DummyEnv(warm_key, task_id)

    async def is_alive(self, handle: DummyEnv) -> bool:
        return bool(getattr(handle, "alive", False))

    async def release(self, handle: DummyEnv) -> None:
        if isinstance(handle, DummyEnv):
            handle.alive = False


__all__ = ["DummyEnv", "DummyWarmupPlugin"]
