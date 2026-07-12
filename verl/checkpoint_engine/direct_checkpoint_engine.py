# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Direct-write checkpoint engines: trainer rank 0 broadcasts straight into the
SGLang TP worker processes over a persistent torch process group, instead of
staging buckets on CheckpointEngineWorkers.

Rollout-side data path per flush:
  rank0 zmq meta -> per-replica TP-leader CE worker posts ONE HTTP
  ``update_weights_from_distributed`` -> every SGLang TP worker allocates the
  flush tensors and receives the broadcast directly -> native in-place apply
  (``load_format="delta"`` uses SGLang's distributed delta receiver).

Because no CE-worker buckets exist, the manager's release_kv / finalize /
resume_kv choreography is skipped entirely for these backends.
"""
import json
import logging
import time

import torch
import torch.distributed as dist
import zmq

from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine.delta_checkpoint_engine import (
    DeltaCheckpointEngine,
    DeltaShardedCheckpointEngine,
)
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)

DIRECT_GROUP_NAME = "verl_direct_weights"


class _DirectWriteMixin:
    """Transport override: zmq carries metadata only; payload rides a persistent
    torch group whose members are trainer rank0 + every SGLang TP worker."""

    direct_write = True

    def _direct_state_init(self):
        self._direct_group = None          # rank0's handle
        self._direct_port = None
        self._direct_inited_engine = False # CE-worker side: HTTP init done

    # ---------------- sender (actor rank 0) ----------------

    def _direct_handshake_and_group(self):
        """First sync only: publish the torch-group rendezvous via zmq, then
        block joining it (SGLang TP workers join via the HTTP-triggered init)."""
        if self._direct_group is not None:
            return
        from sglang.srt.utils.common import init_custom_process_group

        self._direct_port, _ = get_free_port(self.ip)
        world = self.world_size  # 1 + number of rollout CE workers == 1 + total TP workers
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj({
            "direct_init": True,
            "master_address": self.ip,
            "master_port": self._direct_port,
            "world_size": world,
        })
        t0 = time.perf_counter()
        self._direct_group = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{self.ip}:{self._direct_port}",
            world_size=world,
            rank=0,
            group_name=DIRECT_GROUP_NAME,
        )
        logger.info("direct group up: world=%d in %.1fs", world, time.perf_counter() - t0)

    def _direct_send_sparse(self, flush, is_last: bool):
        pos_u8 = flush.positions_cpu.to("cuda", non_blocking=True).contiguous().view(torch.uint8)
        values = flush.values_gpu.contiguous()
        spec = {
            "encoding": self.encoding,
            "params": [vars(p) for p in flush.params],
            "checksum": int(flush.checksum),
        }
        meta = {
            "mode": "delta",
            "names": ["__positions__", "__values__"],
            "dtypes": ["uint8", str(values.dtype).replace("torch.", "")],
            "shapes": [[int(pos_u8.numel())], [int(values.numel())]],
            "delta": json.dumps(spec),
            "is_last": is_last,
        }
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)
        dist.broadcast(pos_u8, src=0, group=self._direct_group)
        dist.broadcast(values, src=0, group=self._direct_group)

    def _direct_send_dense(self, names, dtypes, shapes, tensors, is_last: bool):
        """Full tensors through SGLang's stock batched distributed path."""
        meta = {
            "mode": "dense",
            "names": names,
            "dtypes": dtypes,
            "shapes": shapes,
            "is_last": is_last,
        }
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)
        handles = [dist.broadcast(t, src=0, group=self._direct_group, async_op=True) for t in tensors]
        for h in handles:
            h.wait()

    def _direct_send_terminal(self):
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj({"mode": "terminal", "is_last": True})

    # ---------------- receiver (CheckpointEngineWorker) ----------------

    async def update_weights_via_server(self, server_adapter, global_steps: int | None = None) -> None:
        assert self.rank > 0, "Rank 0 should not receive weights."
        await server_adapter._init_server_adapter()
        engine = getattr(server_adapter, "_engine", None)
        is_leader = server_adapter._is_server_tp_leader()

        while True:
            self.socket.recv_string()
            meta = self.socket.recv_pyobj()

            if meta.get("direct_init"):
                if is_leader and not self._direct_inited_engine:
                    # The leader CE worker is TP0 of its replica and CE worker
                    # verl-ranks (1..N) map 1:1 onto rollout GPUs in order, so
                    # this worker's own rank IS the replica's rank offset in
                    # the direct group. (replica_rank is global across hybrid +
                    # standalone replicas and must not be used here.)
                    logger.warning(
                        "direct init: leader verl-rank=%s -> rank_offset=%s world=%s",
                        self.rank, self.rank, meta["world_size"],
                    )
                    await engine._make_async_request("init_weights_update_group", {
                        "master_address": meta["master_address"],
                        "master_port": meta["master_port"],
                        "rank_offset": self.rank,
                        "world_size": meta["world_size"],
                        "group_name": DIRECT_GROUP_NAME,
                        "backend": "nccl",
                    })
                    self._direct_inited_engine = True
                continue

            if meta.get("mode") == "terminal":
                break

            if is_leader:
                payload = {
                    "names": meta["names"],
                    "dtypes": meta["dtypes"],
                    "shapes": meta["shapes"],
                    "group_name": DIRECT_GROUP_NAME,
                    "flush_cache": bool(meta["is_last"]),
                }
                if meta["mode"] == "delta":
                    payload["load_format"] = "delta"
                    payload["delta"] = meta["delta"]
                await engine._make_async_request("update_weights_from_distributed", payload)

            if meta["is_last"]:
                break

        if engine is not None and is_leader and global_steps is not None:
            await server_adapter.server_actor.set_global_steps.remote(global_steps)

    def finalize(self):
        # Persistent transport: keep the zmq socket and the torch group alive.
        return


@CheckpointEngineRegistry.register("delta_sharded_direct")
class DeltaShardedDirectEngine(_DirectWriteMixin, DeltaShardedCheckpointEngine):
    """Sharded sparse delta over the direct-write transport."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._direct_state_init()

    # sender overrides: reuse all gather/diff logic; swap the publish primitives.
    def _publish_flush(self, flush, first, is_last):
        self._direct_handshake_and_group()
        self._direct_send_sparse(flush, is_last)

    def _publish_dense_flush(self, params, values, is_last):
        # dense seed: ship full tensors through the stock batched path
        self._direct_handshake_and_group()
        names, dtypes, shapes, tensors = [], [], [], []
        for p in params:
            names.append(p.name)
            dtypes.append(p.dtype)
            shapes.append(list(p.shape))
            tensors.append(values[p.val_start:p.val_end].view(p.shape).contiguous())
        self._direct_send_dense(names, dtypes, shapes, tensors, is_last)

    def _publish_terminal(self, first):
        self._direct_handshake_and_group()
        self._direct_send_terminal()


@CheckpointEngineRegistry.register("delta_direct")
class DeltaDirectEngine(_DirectWriteMixin, DeltaCheckpointEngine):
    """Rank-0 full-gather delta over the direct-write transport."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._direct_state_init()

    def _publish_flush(self, flush, first, is_last):
        self._direct_handshake_and_group()
        self._direct_send_sparse(flush, is_last)

    def _publish_dense_flush(self, params, values, is_last):
        self._direct_handshake_and_group()
        names, dtypes, shapes, tensors = [], [], [], []
        for p in params:
            names.append(p.name)
            dtypes.append(p.dtype)
            shapes.append(list(p.shape))
            tensors.append(values[p.val_start:p.val_end].view(p.shape).contiguous())
        self._direct_send_dense(names, dtypes, shapes, tensors, is_last)

    def _publish_terminal(self, first):
        self._direct_handshake_and_group()
        self._direct_send_terminal()


@CheckpointEngineRegistry.register("nccl_direct")
class NCCLDirectEngine(_DirectWriteMixin, DeltaCheckpointEngine):
    """Full-weight sync over the direct-write transport (miles-style): every
    sync broadcasts all parameters through SGLang's stock batched distributed
    receive. Reuses the delta engine's zmq plumbing; no diffing, no buckets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._direct_state_init()

    async def send_weights(self, weights, global_steps=None):
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."
        if not self.is_master:
            for _ in weights:  # participate in FSDP all-gathers
                pass
            return
        self._direct_handshake_and_group()
        names, dtypes, shapes, tensors, nbytes = [], [], [], [], 0
        n_flushes = 0

        def _flush(is_last):
            nonlocal names, dtypes, shapes, tensors, nbytes, n_flushes
            if not tensors and not is_last:
                return
            if tensors:
                self._direct_send_dense(names, dtypes, shapes, tensors, is_last)
                n_flushes += 1
            elif is_last:
                self._direct_send_terminal()
            names, dtypes, shapes, tensors, nbytes = [], [], [], [], 0

        for name, tensor in weights:
            t = tensor.detach()
            if t.is_floating_point():
                t = t.to(torch.bfloat16)
            t = t.to("cuda", non_blocking=True).contiguous()
            names.append(name)
            dtypes.append(str(t.dtype).replace("torch.", ""))
            shapes.append(list(t.shape))
            tensors.append(t)
            nbytes += t.numel() * t.element_size()
            if nbytes >= self.bucket_size:
                _flush(is_last=False)
        _flush(is_last=True)
        logger.info("nccl-direct send v=%s flushes=%d", global_steps, n_flushes)

    def receive_weights(self, global_steps: int | None = None):
        raise NotImplementedError("nccl_direct applies weights inside SGLang via update_weights_via_server")
