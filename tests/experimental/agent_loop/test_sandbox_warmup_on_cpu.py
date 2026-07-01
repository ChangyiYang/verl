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
"""Unit tests for per-rollout sandbox warm-up (CPU-only, no GPU/Ray/backend)."""

import asyncio

import pytest

from verl.experimental.sandbox_warmup import (
    WARM_KEY,
    WarmupRegistry,
    acquire_warm_env,
    current_registry,
    release_warm_env,
    set_current_registry,
)
from verl.experimental.sandbox_warmup.example_dummy_plugin import DummyEnv, DummyWarmupPlugin


class FakePlugin:
    """Instrumented plugin: counts warm/release calls and can force a dead env."""

    def __init__(self, alive=True):
        self.warm_calls = 0
        self.released = []
        self._alive = alive

    async def warm(self, warm_key, sample, config):
        self.warm_calls += 1
        return DummyEnv(warm_key, (sample or {}).get("task_id"))

    async def is_alive(self, handle):
        return self._alive and getattr(handle, "alive", False)

    async def release(self, handle):
        self.released.append(handle.warm_key)
        handle.alive = False


def _reg(plugin, **kw):
    return WarmupRegistry(plugin, config=None, **kw)


def test_prewarm_then_acquire_returns_same_env():
    async def _run():
        p = FakePlugin()
        reg = _reg(p)
        reg.start_warm("k1", {"task_id": "t1"})
        env = await reg.acquire("k1")
        assert isinstance(env, DummyEnv) and env.warm_key == "k1" and env.task_id == "t1"
        assert p.warm_calls == 1  # warmed once, not re-warmed on acquire

    asyncio.run(_run())


def test_acquire_awaits_in_flight_warm():
    async def _run():
        p = FakePlugin()
        reg = _reg(p)
        reg.start_warm("k", {})
        # acquire immediately, before the background warm is awaited elsewhere
        env = await reg.acquire("k")
        assert env.alive
        assert p.warm_calls == 1

    asyncio.run(_run())


def test_acquire_miss_warms_inline():
    async def _run():
        p = FakePlugin()
        reg = _reg(p)
        env = await reg.acquire("never_warmed", {"task_id": "x"})  # no start_warm
        assert env.warm_key == "never_warmed"
        assert p.warm_calls == 1

    asyncio.run(_run())


def test_dead_parked_env_is_rewarmed():
    async def _run():
        p = FakePlugin(alive=False)  # is_alive -> False, so parked env looks dead
        reg = _reg(p)
        reg.start_warm("k", {})
        env = await reg.acquire("k")
        # warmed once for the park + once for the re-warm; dead one released
        assert p.warm_calls == 2
        assert env is not None

    asyncio.run(_run())


def test_release_closes_env_and_forgets_key():
    async def _run():
        p = FakePlugin()
        reg = _reg(p)
        reg.start_warm("k", {})
        await reg.acquire("k")
        await reg.release("k")
        assert p.released == ["k"]
        # after release, key is gone -> next acquire warms inline
        await reg.acquire("k", {})
        assert p.warm_calls == 2

    asyncio.run(_run())


def test_drain_releases_unconsumed_envs():
    async def _run():
        p = FakePlugin()
        reg = _reg(p)
        reg.start_warm("a", {})
        reg.start_warm("b", {})
        await reg.drain()
        assert sorted(p.released) == ["a", "b"]

    asyncio.run(_run())


def test_bounded_concurrency():
    """Semaphore caps simultaneous in-flight warms."""

    async def _run():
        max_seen = 0
        cur = 0
        gate = asyncio.Event()

        class SlowPlugin(FakePlugin):
            async def warm(self, warm_key, sample, config):
                nonlocal cur, max_seen
                cur += 1
                max_seen = max(max_seen, cur)
                await gate.wait()
                cur -= 1
                return DummyEnv(warm_key, None)

        p = SlowPlugin()
        reg = _reg(p, max_concurrency=2)
        for i in range(5):
            reg.start_warm(str(i), {})
        await asyncio.sleep(0.05)  # let warms pile up against the semaphore
        assert max_seen <= 2
        gate.set()
        await reg.drain()

    asyncio.run(_run())


def test_consumer_helpers_and_disabled_default():
    async def _run():
        # disabled: no current registry -> helpers are safe no-ops
        set_current_registry(None)
        assert current_registry() is None
        assert await acquire_warm_env("k") is None
        await release_warm_env("k")  # no-op

        # enabled: helpers route through the current registry
        p = FakePlugin()
        reg = _reg(p)
        set_current_registry(reg)
        reg.start_warm("k", {"task_id": "t"})
        env = await acquire_warm_env("k")
        assert env.warm_key == "k"
        await release_warm_env("k")
        assert p.released == ["k"]
        set_current_registry(None)

    asyncio.run(_run())


def test_warm_key_constant():
    assert WARM_KEY == "_warm_key"


def test_dummy_plugin_end_to_end():
    async def _run():
        reg = _reg(DummyWarmupPlugin())
        reg.start_warm("k", {"extra_info": {"task_id": "gsm8k_42"}})
        env = await reg.acquire("k")
        assert env.task_id == "gsm8k_42" and env.alive
        await reg.release("k")
        assert not env.alive

    asyncio.run(_run())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
