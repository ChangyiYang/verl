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
"""Per-worker registry that warms rollout envs ahead of time and matches them back.

Keyed by a framework-minted per-rollout ``_warm_key``. The registry:

* schedules ``plugin.warm(...)`` in the background (bounded concurrency) at warm time;
* on :meth:`acquire`, returns the matching env — awaiting an in-flight warm, checking
  liveness (re-warming a dead one), or warming inline on a miss so a rollout never
  blocks forever;
* on :meth:`release` / :meth:`drain`, tears envs down without leaking.

The registry lives in the same process as both the warm-up trigger and the rollout
consumer (the rollout worker), so the in-memory ``_warm_key -> env`` map lines up.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from omegaconf import DictConfig

from verl.experimental.sandbox_warmup.plugin import SandboxWarmupPlugin

logger = logging.getLogger(__name__)

# Non-tensor batch column the framework stamps on each rollout row; the consumer
# reads it back to look up its own warmed env.
WARM_KEY = "_warm_key"


class WarmupRegistry:
    """Warm rollout envs ahead of consumption and match each back by ``warm_key``."""

    def __init__(self, plugin: SandboxWarmupPlugin, config: DictConfig, *, max_concurrency: int = 8) -> None:
        self._plugin = plugin
        self._config = config
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        # warm_key -> Task[handle]; a task both parks a ready handle and lets a
        # consumer that arrives early await the in-flight warm.
        self._tasks: dict[str, asyncio.Task] = {}
        # warm_key -> sample, kept for inline fallback / re-warm on a dead handle.
        self._samples: dict[str, dict] = {}

    def start_warm(self, warm_key: str, sample: dict) -> None:
        """Kick off a background warm for ``warm_key`` (idempotent per key)."""
        if warm_key in self._tasks:
            return
        self._samples[warm_key] = sample
        self._tasks[warm_key] = asyncio.ensure_future(self._warm_bounded(warm_key, sample))

    async def _warm_bounded(self, warm_key: str, sample: dict) -> Any:
        async with self._sem:
            return await self._plugin.warm(warm_key, sample, self._config)

    async def acquire(self, warm_key: str, sample: Optional[dict] = None) -> Any:
        """Return a live env for ``warm_key``, warming inline if it was never scheduled.

        Never hands back a dead env: a parked handle is liveness-checked and re-warmed
        if the backend reports it dead.
        """
        task = self._tasks.get(warm_key)
        if task is None:
            # Miss: never warmed (or already drained) — warm inline so the rollout runs.
            sample = sample if sample is not None else self._samples.get(warm_key, {})
            logger.warning("warm env %s not pre-warmed; warming inline", warm_key)
            return await self._warm_bounded(warm_key, sample)

        try:
            handle = await task  # ready result, or await the in-flight warm
        except Exception:
            logger.exception("pre-warm for %s failed; warming inline", warm_key)
            sample = sample if sample is not None else self._samples.get(warm_key, {})
            return await self._warm_bounded(warm_key, sample)

        if await self._is_alive(handle):
            return handle

        logger.warning("warm env %s died while parked; re-warming", warm_key)
        try:
            await self._plugin.release(handle)
        except Exception:
            logger.exception("release of dead env %s failed", warm_key)
        sample = sample if sample is not None else self._samples.get(warm_key, {})
        handle = await self._warm_bounded(warm_key, sample)
        # Replace the parked task so a duplicate acquire (should not happen for
        # per-rollout keys) still sees the live handle.
        self._tasks[warm_key] = _completed_task(handle)
        return handle

    async def _is_alive(self, handle: Any) -> bool:
        try:
            return bool(await self._plugin.is_alive(handle))
        except Exception:
            logger.warning("is_alive raised; treating env as dead", exc_info=True)
            return False

    async def release(self, warm_key: str) -> None:
        """Release the env for ``warm_key`` and forget it (safe to call once per rollout)."""
        task = self._tasks.pop(warm_key, None)
        self._samples.pop(warm_key, None)
        if task is None:
            return
        try:
            handle = await task
        except Exception:
            return  # warm failed; nothing to release
        try:
            await self._plugin.release(handle)
        except Exception:
            logger.exception("release failed for %s", warm_key)

    async def drain(self) -> None:
        """Release every remaining env (backstop for leaks at the end of a wave).

        Awaits any still-in-flight warm before releasing it, so a partially-created
        sandbox is not leaked. The normal path releases per-rollout; drain only
        catches envs that were warmed but never consumed.
        """
        for warm_key in list(self._tasks.keys()):
            await self.release(warm_key)


def _completed_task(value: Any) -> asyncio.Task:
    async def _identity() -> Any:
        return value

    return asyncio.ensure_future(_identity())


# ---------------------------------------------------------------------------
# Per-process current registry + consumer helpers
# ---------------------------------------------------------------------------
# One rollout worker == one process == one registry, so a module-global is enough
# for the consumer (the agent loop) to reach the registry without threading it
# through every call site.

_CURRENT: Optional[WarmupRegistry] = None


def set_current_registry(registry: Optional[WarmupRegistry]) -> None:
    global _CURRENT
    _CURRENT = registry


def current_registry() -> Optional[WarmupRegistry]:
    return _CURRENT


async def acquire_warm_env(warm_key: Optional[str], sample: Optional[dict] = None) -> Any:
    """Consumer-side one-liner: return this rollout's warmed env, or ``None`` if disabled.

    Call from inside an agent loop with the rollout's ``_warm_key`` (available in the
    run kwargs / sample). Returns ``None`` when warm-up is not configured, so callers
    can fall back to their own inline creation.
    """
    registry = current_registry()
    if registry is None or not warm_key:
        return None
    return await registry.acquire(warm_key, sample)


async def release_warm_env(warm_key: Optional[str]) -> None:
    """Consumer-side one-liner: release this rollout's warmed env (no-op if disabled)."""
    registry = current_registry()
    if registry is not None and warm_key:
        await registry.release(warm_key)


__all__ = [
    "WARM_KEY",
    "WarmupRegistry",
    "acquire_warm_env",
    "current_registry",
    "release_warm_env",
    "set_current_registry",
]
