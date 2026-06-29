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
"""Unit tests for per-wave rollout resource warmup/cleanup hooks (CPU-only, no GPU/Ray)."""

import asyncio

import pytest

from verl.experimental.agent_loop.agent_loop import (
    _maybe_await,
    _resolve_rollout_resource_hook,
    _with_rollout_resource_hooks,
)

# Module-level functions so they are importable via fully-qualified name (mirrors real usage).


def sync_hook(batch, config):
    _CALLS.append(("sync", batch, config))


async def async_hook(batch, config):
    _CALLS.append(("async", batch, config))


_CALLS: list = []
NOT_CALLABLE = 123


def test_maybe_await_passthrough_and_await():
    async def _run():
        assert await _maybe_await(5) == 5

        async def coro():
            return 7

        assert await _maybe_await(coro()) == 7

    asyncio.run(_run())


def test_resolve_none_returns_none():
    assert _resolve_rollout_resource_hook(None, "warmup") is None
    assert _resolve_rollout_resource_hook("", "warmup") is None


def test_resolve_valid_path_returns_callable():
    fn = _resolve_rollout_resource_hook(f"{__name__}.sync_hook", "warmup")
    assert fn is sync_hook


def test_resolve_non_callable_raises():
    with pytest.raises(TypeError):
        _resolve_rollout_resource_hook(f"{__name__}.NOT_CALLABLE", "warmup")


def test_hooks_run_in_order_around_impl():
    """warmup -> impl -> cleanup, regardless of sync/async hooks."""
    events = []

    def warmup(batch, config):
        events.append(("warmup", batch, config))

    async def cleanup(batch, config):
        events.append(("cleanup", batch, config))

    async def impl():
        events.append("impl")
        return "result"

    out = asyncio.run(_with_rollout_resource_hooks(warmup, cleanup, "BATCH", "CFG", impl))
    assert out == "result"
    assert events == [("warmup", "BATCH", "CFG"), "impl", ("cleanup", "BATCH", "CFG")]


def test_no_hooks_is_noop():
    async def impl():
        return "ok"

    out = asyncio.run(_with_rollout_resource_hooks(None, None, "B", "C", impl))
    assert out == "ok"


def test_cleanup_runs_even_on_impl_error():
    cleaned = []

    def cleanup(batch, config):
        cleaned.append(True)

    async def impl():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(_with_rollout_resource_hooks(None, cleanup, "B", "C", impl))
    assert cleaned == [True]


def test_warmup_failure_skips_impl_but_runs_cleanup():
    """A partial/failed warmup must still trigger cleanup so partially-allocated resources
    (e.g. some spawned sandboxes) are released rather than leaked; impl is skipped."""
    ran = {"impl": False, "cleanup": False}

    def warmup(batch, config):
        raise RuntimeError("warmup failed")

    async def impl():
        ran["impl"] = True

    def cleanup(batch, config):
        ran["cleanup"] = True

    with pytest.raises(RuntimeError, match="warmup failed"):
        asyncio.run(_with_rollout_resource_hooks(warmup, cleanup, "B", "C", impl))
    # impl never ran (warmup raised first); cleanup still runs to release partial resources.
    assert ran == {"impl": False, "cleanup": True}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
