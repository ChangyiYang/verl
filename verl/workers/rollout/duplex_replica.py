# Copyright 2026 Liquid AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Ray replica adapter for the colocated official-HF duplex runtime.

Unlike vLLM/SGLang, the duplex engine already lives inside the hybrid
``ActorRolloutRefWorker``.  The replica therefore registers that worker's Ray
handle directly with verl's global load balancer instead of launching another
HTTP server or GPU process.
"""

from __future__ import annotations

from typing import Any

from verl.workers.rollout.replica import RolloutReplica


class DuplexReplica(RolloutReplica):
    """One single-GPU, single-session duplex rollout replica."""

    async def launch_servers(self):
        if self.world_size != 1:
            raise NotImplementedError(
                "The official-HF duplex rollout currently supports one GPU per replica; "
                f"got rollout world size {self.world_size}."
            )
        if len(self.workers) != 1:
            raise RuntimeError(f"Expected one hybrid worker, got {len(self.workers)}")

        worker = self.workers[0]
        self.servers = [worker]
        self._server_handle = worker
        # The load balancer treats addresses as opaque stable identifiers.
        self._server_address = f"duplex://replica-{self.replica_rank}"

    async def sleep(self):
        await self._server_handle.duplex_sleep.remote()

    async def wake_up(self):
        await self._server_handle.duplex_wake_up.remote()

    async def clear_kv_cache(self):
        await self._server_handle.duplex_clear_kv_cache.remote()

    async def release_kv_cache(self):
        await self._server_handle.duplex_clear_kv_cache.remote()

    async def resume_kv_cache(self):
        # The official runtime allocates KV lazily on the next request.
        return None

    async def abort_all_requests(self) -> dict[str, Any]:
        # A Ray actor serializes this single-session backend, so an update is
        # only issued after generation has drained.
        return {"aborted_count": 0, "request_ids": []}

    async def resume_generation(self):
        return None
