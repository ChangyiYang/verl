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
"""User-facing plugin contract for per-rollout sandbox warm-up.

Implement this for your backend (e.g. Modal) and wire it in via
``actor_rollout_ref.rollout.agent.warmup_plugin_path``. The framework
(:class:`~verl.experimental.sandbox_warmup.registry.WarmupRegistry`) owns
scheduling/matching/liveness/lifecycle; the plugin owns *what an env is* and
*how to prepare it*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig


class SandboxWarmupPlugin(ABC):
    """Backend-specific per-rollout env warm-up.

    A plugin is instantiated once per :class:`WarmupRegistry` (i.e. once per rollout
    worker process). ``warm`` may be called concurrently for different rollouts, so
    implementations must be safe under concurrency.

    The returned *env handle* is opaque to the framework: the registry only stores
    it, hands it to the matching rollout, and passes it back to :meth:`release`.
    """

    @abstractmethod
    async def warm(self, warm_key: str, sample: dict, config: DictConfig) -> Any:
        """Create and prepare the env for one rollout, returning an opaque handle.

        Do the full per-rollout preparation here: create the sandbox, set terminal
        state, upload files, etc. ``sample`` is that rollout's dataset row (read
        ``task_id`` / ``task_path`` / prompt from it to drive task-specific setup);
        ``warm_key`` is the framework's unique handle for this rollout (use it only
        if you need to name/log the env). ``config`` is the full trainer config.

        Raising here is allowed; the registry surfaces the error to the matching
        rollout's :func:`acquire_warm_env` call (and still runs :meth:`release` on
        anything you managed to allocate, as long as you return it — otherwise clean
        up partial state yourself before raising).
        """
        raise NotImplementedError

    async def is_alive(self, handle: Any) -> bool:  # noqa: ARG002
        """Return whether a parked handle is still usable (default: always alive).

        A remote sandbox can die while parked (idle auto-stop, lifetime timeout).
        The registry calls this before handing a handle to its rollout and re-warms
        if it returns ``False``. Must not raise; a raise is treated as dead.
        """
        return True

    @abstractmethod
    async def release(self, handle: Any) -> None:
        """Tear down / release an env handle. Must be idempotent and swallow errors."""
        raise NotImplementedError


__all__ = ["SandboxWarmupPlugin"]
