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
"""Delta weight-sync checkpoint engine (NCCL transport) for DISAGGREGATED rollout.

Puts the delta on the trainer->rollout wire: the trainer byte-diffs against a
pinned-CPU snapshot and broadcasts only the changed ``(position, value)`` pairs
over the same ``ray.util.collective`` NCCL group the full-weight
:class:`NCCLCheckpointEngine` uses (actor rank0 -> rollout CheckpointEngineWorkers).
Each rollout worker then hands its local copy of the sparse payload to its
colocated SGLang TP worker via same-GPU ``update_weights_from_tensor`` IPC, where
the verl-shipped :mod:`verl.workers.rollout.sglang_rollout.delta_loader` (registered through SGLang's
stock ``--custom-weight-loader`` hook — no SGLang fork or patch needed) decodes
and masked-applies it *in place* onto the live weights. No full-model mirror is
staged anywhere on the rollout side: receiver peak memory is one bucket plus one
decode chunk, independent of model size.

The first (seed) sync streams the backend's FULL HF export (``get_per_tensor_
param()``) over the values-only wire -- every backend already knows how to
assemble and convert its own full tensors, so resume works by construction and
the seed inherits Megatron/veomni assembly for free. After the seed the caller
primes the backend's pinned shard snapshots; every later sync ships the
backend-computed sparse HF delta.

Data ladder (sender side, steady) -- the names in this file anchor to these::

    backend HF delta ENTRY (slots, dtype, counts, hf_idx, hf_val, group)
    --_GatherQueue--> per-SLOT delta on rank 0 --_bucket_*--> _FlushPiece
    --_FlushBucket--> FLUSH (DeltaFlush) --_publish_flush--> wire

Wire encodings: ``indices`` (int32 positions + values; every steady sync) and
``values`` (values only; the seed). The values meta tag is ``"dense"`` for
protocol continuity with the receiver's delta_loader.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from unittest.mock import patch

import ray.util.collective as collective
import torch
import zmq

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

from .base import CheckpointEngineRegistry
from .delta_sync.encode import DeltaFlush, DeltaParam
from .delta_sync.encode import checksum as _checksum
from .delta_sync.sparse_gather import gather_slot_entries_to_rank0
from .nccl_checkpoint_engine import MasterMetadata, NCCLCheckpointEngine

logger = logging.getLogger(__name__)


def _prodshape(shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


@dataclass(slots=True)
class _FlushPiece:
    """One (possibly sliced) per-parameter piece buffered for a pending indices flush."""

    name: str
    dtype_str: str
    shape: list
    idx: torch.Tensor
    val: torch.Tensor


@dataclass(slots=True)
class _ValuesPiece:
    """One whole parameter's flat values buffered for the seed sync's values-only flush."""

    name: str
    dtype_str: str
    shape: list
    flat: torch.Tensor


class _FlushBucket:
    """One-flush-lookahead bucket pipeline, shared by the steady loop and both
    seed streams. Pieces accumulate until ``cap`` bytes; ``seal`` assembles them
    into the single pending flush, first emitting the previous pending with
    ``is_last=False`` (the lookahead: only the caller's finale knows which flush
    is last and emits it with ``is_last=True``). ``assemble`` and ``publish``
    carry the only real differences between the streams -- the wire format
    (indexed flush vs values-only flush) and the flush counters."""

    __slots__ = ("cap", "pieces", "nbytes", "pending", "_assemble", "_publish")

    def __init__(self, cap: int, assemble, publish):
        self.cap = int(cap)
        self.pieces: list = []
        self.nbytes = 0
        self.pending = None
        self._assemble = assemble
        self._publish = publish

    def add(self, piece, nbytes: int) -> None:
        self.pieces.append(piece)
        self.nbytes += int(nbytes)
        if self.nbytes >= self.cap:
            self.seal()

    def seal(self) -> None:
        if not self.pieces:
            return
        self.emit(is_last=False)
        self.pending = self._assemble(self.pieces)
        self.pieces, self.nbytes = [], 0

    def emit(self, is_last: bool) -> None:
        if self.pending is not None:
            self._publish(self.pending, is_last)
            self.pending = None


def _bucket_sliced(bkt: _FlushBucket, name: str, dtype_str: str, shape, aidx: torch.Tensor, aval: torch.Tensor) -> None:
    """Slice one param's (idx, val) delta into <= MAX_ENTRY_ELEMS pieces and bucket
    them (bounds the receiver-side decode transient; the masked apply is sequential,
    so splitting is transparent). Bucket bytes = actual wire bytes (int32 positions
    + values)."""
    max_elems = DeltaShardedCheckpointEngine.MAX_ENTRY_ELEMS
    for s in range(0, aidx.numel(), max_elems):
        e = min(s + max_elems, aidx.numel())
        bkt.add(
            _FlushPiece(name, dtype_str, list(shape), aidx[s:e], aval[s:e]),
            (e - s) * (4 + aval.element_size()),
        )


class _GatherQueue:
    """Per-gather-group batching of slot-keyed queue entries
    ``(slots, dtype_str, counts, idx, val)``. Entries carry FINAL-coordinate
    payloads (identity specs: one slot = the param itself; converter specs: the
    spec's hf_slots), so rank 0 never converts -- ``consume`` receives assembled
    per-slot pieces straight from the gather.

    One queue per ProcessGroup: separate queues stop pg alternation (dense fsdp
    group vs expert world group per layer) from shattering batches. The flush
    trigger is COUNT-ONLY: entry counts are identical on every rank while byte
    totals are not, so a count trigger is the only one that keeps the collective
    sequence identical across ranks (a per-rank byte trigger desyncs the gathers
    and deadlocks NCCL). Byte bounding happens INSIDE the batched gather via
    ``max_round_bytes``, decided from the all-gathered counts every rank sees."""

    __slots__ = ("batch_k", "max_round_bytes", "is_r0", "_consume", "_queues")

    def __init__(self, batch_k: int, max_round_bytes: int, is_r0: bool, consume):
        self.batch_k = max(int(batch_k), 1)
        self.max_round_bytes = int(max_round_bytes)
        self.is_r0 = is_r0
        self._consume = consume
        self._queues: dict[int, tuple] = {}  # id(pg) -> (pg, [entries])

    def put(self, pg, slots: list, dtype_str: str, counts: torch.Tensor, idx: torch.Tensor, val: torch.Tensor):
        # one queue per (group, value dtype): batches concatenate values, so a
        # batch must be dtype-homogeneous (fp8 codes / fp32 scales / bf16 mix
        # under quant mode). Entry order and dtypes are identical on every
        # rank, so the partition stays in lockstep.
        _pg, entries = self._queues.setdefault((id(pg), val.dtype), (pg, []))
        entries.append((slots, dtype_str, counts, idx, val))
        if len(entries) >= self.batch_k:
            self._flush(pg, entries)

    def flush_all(self) -> None:
        for pg, entries in self._queues.values():
            self._flush(pg, entries)

    def _flush(self, pg, entries: list) -> None:
        """One gather round for one group's queue."""
        if not entries:
            return
        batch = list(entries)
        entries.clear()
        if pg is None:
            # unsharded/replicated params: rank 0's local delta already is global
            if self.is_r0:
                for slots, dtype_str, counts, idx, val in batch:
                    off = 0
                    for (name, shape), c in zip(slots, counts.tolist(), strict=True):
                        self._consume(
                            name, dtype_str, tuple(shape), _prodshape(shape), idx[off : off + c], val[off : off + c]
                        )
                        off += c
            return
        dev = batch[0][3].device
        counts_concat = torch.cat([c for _, _, c, _, _ in batch]).to(dev)
        idx_concat = torch.cat([i for _, _, _, i, _ in batch])
        val_concat = torch.cat([v for _, _, _, _, v in batch])
        gathered = gather_slot_entries_to_rank0(
            idx_concat, val_concat, counts_concat, group=pg, max_round_bytes=self.max_round_bytes
        )
        if self.is_r0 and gathered is not None:
            slot_i = 0
            for slots, dtype_str, _counts, _i, _v in batch:
                for name, shape in slots:
                    aidx, aval = gathered[slot_i]
                    slot_i += 1
                    self._consume(name, dtype_str, tuple(shape), _prodshape(shape), aidx, aval)


@CheckpointEngineRegistry.register("delta_sharded")
class DeltaShardedCheckpointEngine(NCCLCheckpointEngine):
    """Sparse delta weight sync over NCCL, diffed on each rank's local shard.

    Reuses NCCLCheckpointEngine's group/zmq machinery but moves only changed
    positions+values: each actor rank keeps a pinned-CPU snapshot of only *its*
    shard, byte-diffs the shard, and only the changed ``(position, value)`` pairs
    are gathered to rank 0 and streamed to the rollout side -- no rank ever holds
    a full-model snapshot.

    ``send_weights`` takes the TRAINING ENGINE and drives the sync itself: the
    seed (first sync) streams the backend's full ``get_per_tensor_param()``
    export values-only and pins the diff base (``prime_delta_snapshots``); every
    steady sync consumes the backend's HF delta export
    ``get_per_tensor_param_delta_shard()`` — per-parameter FINAL-HF-coordinate
    entries ``(slots, dtype_str, counts, hf_idx, hf_val, gather_group)``. Naming,
    to-HF conversion, diff and snapshot all live on the backend side (see
    :mod:`verl.workers.engine.utils`); this engine only batches, gathers,
    buckets and ships, and so serves any backend that can produce HF deltas.
    """

    # Cap on changed elements per DeltaParam entry. The receiver-side decode
    # densifies per entry with an int64 index transient (8 B/element), so an
    # uncapped entry (e.g. a 7B model's whole embedding on the full seed, ~545M
    # elements) would spike several GiB at once. Oversized per-param deltas are
    # sliced into multiple entries (the masked apply is sequential, so splitting
    # is transparent); 64M elements bounds the transient to ~512 MiB.
    MAX_ENTRY_ELEMS = 64 << 20

    wire_format = "delta_flush"

    def prepare(self) -> MasterMetadata | None:
        # Delta broadcasts small per-flush buffers directly, so skip the parent's
        # 2 * bucket_size fixed buffers. Still hand back the master zmq endpoint
        # that build_topology() distributes to the rollout workers.
        return MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port) if self.is_master else None

    # ---- trainer side ----
    # ---- shared STREAMING wire ----
    # Broadcast each flush the moment it is produced and free it, instead of materializing every
    # flush up front. Peak device memory stays ~2 buckets (like NCCLCheckpointEngine's send/recv
    # buffers) rather than the whole model -- required for large models where the first (full-seed)
    # sync would otherwise hold the entire delta on rank 0. Wire: one zmq manifest + NCCL broadcast
    # per flush, with an ``is_last`` flag so the receiver loops until the stream ends. Each
    # CheckpointEngineWorker then hands its local copy of the sparse payload to its colocated
    # SGLang TP worker (same-GPU IPC), where the verl-shipped custom weight loader applies it
    # in place -- no full-model staging anywhere on the rollout side.
    def _publish_flush(self, flush: DeltaFlush, first: bool, is_last: bool) -> None:
        meta = {
            "is_full": first,
            "encoding": self.encoding,
            "is_last": is_last,
            "terminal_empty": False,
            "pos_numel": int(flush.positions_cpu.numel()),
            "val_numel": int(flush.values_gpu.numel()),
            "val_dtype": str(flush.values_gpu.dtype).replace("torch.", ""),
            "spec": {
                "encoding": self.encoding,
                "values_bytes": self.quantize_fp8,
                "params": [vars(p) for p in flush.params],
                "checksum": int(flush.checksum),
            },
        }
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)
        pos_u8 = flush.positions_cpu.to("cuda", non_blocking=True).contiguous().view(torch.uint8)
        val_u8 = flush.values_gpu.contiguous().view(torch.uint8)
        # Stage into cupy-owned buffers: ray's NCCL broadcast is enqueued on a separate
        # stream with no recordStream on its inputs, so broadcasting a zero-copy view of
        # these torch tensors (freed right after this call) would race with allocator reuse.
        pos_cp = cp.empty(pos_u8.numel(), dtype=cp.uint8)
        val_cp = cp.empty(val_u8.numel(), dtype=cp.uint8)
        pos_cp[:] = cp.asarray(pos_u8)
        val_cp[:] = cp.asarray(val_u8)
        collective.broadcast(pos_cp, src_rank=0, group_name=self.group_name)
        collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)

    def _publish_values_flush(
        self,
        params: list[DeltaParam],
        values: torch.Tensor,
        is_last: bool,
        verify: bool = False,
        values_bytes: bool = False,
    ) -> None:
        """Publish a values-only (full-coverage, positions-free) flush -- used by the first
        sync. The wire encoding tag stays ``"dense"`` -- it is protocol, shared with the
        receiver's delta_loader decode."""
        values = values.contiguous()
        empty_pos = torch.empty(0, dtype=torch.uint8, device=values.device)
        meta = {
            "is_full": True,
            "encoding": "dense",
            "is_last": is_last,
            "terminal_empty": False,
            "pos_numel": 0,
            "val_numel": int(values.numel()),
            "val_dtype": str(values.dtype).replace("torch.", ""),
            "spec": {
                "encoding": "dense",
                "verify": verify,
                "is_last": is_last,
                "values_bytes": values_bytes,
                "params": [vars(p) for p in params],
                "checksum": int(_checksum(empty_pos, values)),
            },
        }
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)
        val_u8 = values.view(torch.uint8)
        # cupy-owned staging: same lifetime rationale as _publish_flush.
        val_cp = cp.empty(val_u8.numel(), dtype=cp.uint8)
        val_cp[:] = cp.asarray(val_u8)
        collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)

    def _release_staging_pool(self, phase: str) -> None:
        """Return the cupy staging pool's blocks to CUDA and log the evidence:
        ``held`` is what the pool would have kept from the device without this
        release, and the device-free delta shows the memory actually coming
        back (warning level so the default worker log level records it)."""
        from verl.utils.device import get_torch_device

        pool = cp.get_default_memory_pool()
        held = pool.total_bytes()
        free_before, _ = get_torch_device().mem_get_info()
        pool.free_all_blocks()
        free_after, _ = get_torch_device().mem_get_info()
        logger.warning(
            "cupy staging pool after %s send: held %.2fGB; device free %.2f->%.2fGB on release",
            phase,
            held / (1 << 30),
            free_before / (1 << 30),
            free_after / (1 << 30),
        )

    def _publish_terminal(self, first: bool) -> None:
        """End-of-stream marker when zero flushes were produced (no broadcast, just a signal)."""
        meta = {"is_full": first, "encoding": self.encoding, "is_last": True, "terminal_empty": True}
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)

    # ---- rollout worker side ----
    def receive_weights(self, global_steps: int | None = None) -> Iterator[tuple[list[tuple[str, torch.Tensor]], bool]]:
        """Yield the sparse flushes for the server adapter to apply in place.

        Each item is ``(named_tensors, is_last)``: the sentinel-encoded flush
        (``__delta_spec__`` json bytes, optional ``__positions__``, ``__values__``)
        received into this worker's own GPU buffer -- one flush resident at a
        time, freed as soon as the consumer drops it. The sglang server adapter
        forwards each flush over same-GPU CUDA IPC to the verl-shipped
        ``delta_loader.apply_delta`` (registered through the custom-weight-loader
        hook), which decodes and masked-applies it against SGLang's live weights;
        no full-model mirror is staged anywhere.
        """
        assert self.rank > 0, "Rank 0 should not receive weights."
        applied = 0
        while True:
            self.socket.recv_string()
            meta = self.socket.recv_pyobj()
            if meta.get("terminal_empty"):
                break

            dense = meta.get("encoding") == "dense"
            val_dtype = getattr(torch, meta["val_dtype"])
            elem = torch.empty(0, dtype=val_dtype).element_size()
            val_u8 = torch.empty(meta["val_numel"] * elem, dtype=torch.uint8, device="cuda")
            if dense:
                pos = None
                collective.broadcast(val_u8, src_rank=0, group_name=self.group_name)
            else:
                pos = torch.empty(meta["pos_numel"], dtype=torch.uint8, device="cuda")
                collective.broadcast(pos, src_rank=0, group_name=self.group_name)
                collective.broadcast(val_u8, src_rank=0, group_name=self.group_name)
            val = val_u8.view(val_dtype)
            spec_bytes = json.dumps(meta["spec"]).encode()
            spec_t = torch.frombuffer(bytearray(spec_bytes), dtype=torch.uint8).to("cuda")
            named = [("__delta_spec__", spec_t), ("__values__", val)]
            if pos is not None:
                named.insert(1, ("__positions__", pos))
            is_last = bool(meta["is_last"])
            yield named, is_last
            applied += 1
            del pos, val_u8, val, spec_t
            if is_last:
                break
        logger.info("delta recv v=%s flushes=%d (yielded to server adapter)", global_steps, applied)

    def __init__(
        self,
        *args,
        encoding: str = "indices",
        batch_gather: int = 32,
        verify_every: int = 0,
        quantize_fp8: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert encoding == "indices", f"delta_sharded ships only the 'indices' position encoding; got {encoding!r}"
        self.encoding = encoding
        # every K-th steady sync appends a full state-verification sweep
        # (0 = off); see _verify_due / delta_loader._verify_dense.
        self.verify_every = int(verify_every)
        # fp8 rollout mode: quantize on the trainer and ship the rollout's
        # exact state (fp8 codes + blockwise scale_inv tensors). Currently
        # full-resync per sync (the quant-domain sparse steady path lands
        # next); the wire is already half the bf16 bytes per element.
        self.quantize_fp8 = bool(quantize_fp8)
        self._fp8_snaps: dict = {}
        self._shard_seeded = False
        # Gather the per-param sparse deltas in groups of this many parameters
        # (one count-matrix all_gather + two padded gathers per group instead of
        # three collectives per parameter).
        self.batch_gather = int(batch_gather)

    def _fp8_shard_stream(self, gen):
        """STEADY-side transform for fp8 mode: turn the backend's raw shard
        stream into the ROLLOUT-domain shard stream. Every quantizable 2D
        weight becomes two pseudo-params -- its fp8 code shard (same placement
        spec: codes are elementwise with the weight) and the fp32 scale grid
        (identical on every rank after the amax all_reduce, so a replicated
        spec lets rank 0 alone ship scale deltas). Everything else passes
        through in bf16. Downstream (snapshot prime, byte-diff, gather, wire)
        is the stock steady pipeline: rollout state = codes + scales, and both
        are just tensors to diff."""
        from verl.utils.fp8_sharded import sharded_scaled_fp8_blockwise
        from verl.workers.engine.spec import ShardSpec, derive_dtensor_placement

        helper = self._fp8_helper()
        block = helper.quant_config.get("weight_block_size", [128, 128])
        for name, flat, spec in gen:
            full_shape = tuple(int(x) for x in spec.full_shape)
            if len(full_shape) != 2 or not helper.should_quantize_param(name):
                yield name, flat, spec
                continue
            place, _contributes, pg = (
                (spec.place, spec.contributes, spec.gather_group)
                if spec.place is not None
                else derive_dtensor_placement(spec)
            )
            cols = full_shape[1]
            flat_off = place if isinstance(place, int) else place.flat_offset
            assert flat_off % cols == 0 and flat.numel() % cols == 0, (
                f"{name}: fp8 steady expects dim-0 row shards (offset {flat_off}, cols {cols})"
            )
            shard2d = flat.view(-1, cols)
            codes, descale = sharded_scaled_fp8_blockwise(
                shard2d, block, flat_off // cols, full_shape, group=pg
            )
            yield name, codes.reshape(-1), spec
            yield (
                name + "_scale_inv",
                descale.reshape(-1),
                ShardSpec(full_shape=tuple(descale.shape)),
            )

    def _mcore_fp8_entries(self, engine, prime_only: bool = False):
        """mcore quant-domain steady entries (or snapshot prime when
        ``prime_only``): per export-index record, run the comm-stubbed probe on
        the FULL local shard, NaN-aware-quantize each quantizable HF slot with
        the group-global scales, and byte-diff codes + scales against pinned
        snapshots. Emits up to three dtype-homogeneous entries per record
        (fp8 codes / fp32 scales / bf16 passthrough slots); scale grids are
        identical on every rank of the group after the all_reduce, so only the
        group's rank 0 contributes them."""
        import torch.distributed as dist

        from verl.checkpoint_engine.delta_sync.sparse_gather import shard_delta_indices
        from verl.utils.device import is_cuda_available
        from verl.utils.fp8_sharded import sharded_scaled_fp8_blockwise

        helper = self._fp8_helper()
        block = helper.quant_config.get("weight_block_size", [128, 128])
        index = engine._mcore_export_index()
        slot_cache = engine._delta_slot_cache

        def _snap_diff(key, flat):
            snap = self._fp8_snaps.get(key)
            if snap is None or snap.numel() != flat.numel():
                snap = torch.empty_like(flat, device="cpu", pin_memory=is_cuda_available)
                self._fp8_snaps[key] = snap
                snap.copy_(flat, non_blocking=True)
                return None  # first sight: primed, nothing to diff
            base = snap.to(flat.device, non_blocking=True)
            lidx, lval = shard_delta_indices(flat, base, 0)
            snap.copy_(flat, non_blocking=True)
            return lidx, lval

        for rec in index:
            outs = rec.probe.megatron_to_hf(rec.param.data.to(torch.bfloat16), rec.module)
            pg = rec.spec.gather_group
            group_rank = dist.get_rank(pg) if pg is not None else dist.get_rank()
            slots = slot_cache.get(rec.megatron_name)
            if slots is None:
                slots = [(n, tuple(int(x) for x in t.shape)) for n, t in outs.items()]
                slot_cache[rec.megatron_name] = slots
            buckets: dict = {}  # dtype -> (slot list, counts list, idx pieces, val pieces)

            def _emit(sname, sshape, flat, key, contributes=True):
                got = _snap_diff(key, flat)
                b = buckets.setdefault(flat.dtype, ([], [], [], []))
                b[0].append((sname, tuple(sshape)))
                if got is None or not contributes:
                    b[1].append(0)
                    return
                lidx, lval = got
                b[1].append(int(lidx.numel()))
                if lidx.numel():
                    b[2].append(lidx.to(torch.int32))
                    b[3].append(lval)

            for sname, _sshape in slots:
                t = outs[sname]
                if t.dim() == 2 and helper.should_quantize_param(sname):
                    codes, descale = sharded_scaled_fp8_blockwise(
                        t.to(torch.bfloat16), block, 0, tuple(t.shape), group=pg
                    )
                    _emit(sname, codes.shape, codes.reshape(-1), (rec.megatron_name, sname, "c"))
                    _emit(
                        sname + "_scale_inv",
                        descale.shape,
                        descale.reshape(-1),
                        (rec.megatron_name, sname, "s"),
                        contributes=(group_rank == 0),
                    )
                else:
                    flat = t.to(torch.bfloat16).reshape(-1)
                    _emit(sname, t.shape, flat, (rec.megatron_name, sname, "b"))

            if prime_only:
                continue
            for dtype, (slot_list, count_list, idx_pieces, val_pieces) in buckets.items():
                counts = torch.tensor(count_list, dtype=torch.int64)
                if idx_pieces:
                    hf_idx = torch.cat(idx_pieces)
                    hf_val = torch.cat(val_pieces)
                else:
                    hf_idx = torch.empty(0, dtype=torch.int32, device=rec.param.device)
                    hf_val = torch.empty(0, dtype=dtype, device=rec.param.device)
                yield (slot_list, str(dtype).replace("torch.", ""), counts, hf_idx, hf_val, pg)

    def _quantized_stream(self, weights):
        """Wrap the full HF export with rollout-scheme fp8 quantization: for
        every 2D weight the rollout side quantizes, yield ``(name, codes)`` +
        ``(name_scale_inv, descales)`` exactly as sglang's own quantizer
        helper names them; everything else passes through in bf16. Every rank
        walks the wrapper (the export is collective); the quantize kernel is
        cheap relative to the assembly it follows."""
        from verl.utils.fp8_sharded import sharded_scaled_fp8_blockwise

        helper = self._fp8_helper()
        block = helper.quant_config.get("weight_block_size", [128, 128])
        for name, t in weights:
            if t.dim() != 2 or not helper.should_quantize_param(name):
                yield name, t
                continue
            # SAME implementation as the steady shard quantizer (group=None ==
            # whole tensor): seed, steady and the verify re-export must agree
            # bit-for-bit, and fp32->fp8 tie rounding is implementation-
            # sensitive across kernels.
            codes, descale = sharded_scaled_fp8_blockwise(
                t.to(torch.bfloat16), block, 0, tuple(t.shape), group=None
            )
            yield name, codes
            yield name + "_scale_inv", descale

    def _verify_due(self) -> bool:
        """True on every K-th steady sync when constructed with ``verify_every=K``."""
        if self.verify_every <= 0:
            return False
        self._steady_count = getattr(self, "_steady_count", 0) + 1
        return self._steady_count % self.verify_every == 0

    def _fp8_helper(self):
        from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper, build_sglang_fp8_quant_config

        h = getattr(self, "_fp8_helper_inst", None)
        if h is None:
            h = SGLangFP8QuantizerHelper(build_sglang_fp8_quant_config())
            self._fp8_helper_inst = h
        return h

    def _assemble_flush(self, per_param: list[_FlushPiece]) -> DeltaFlush:
        """Build one DeltaFlush (indices encoding) from rank 0's gathered per-param deltas.

        ``per_param``: :class:`_FlushPiece` entries whose ``idx`` are within-parameter
        flat positions (== what the receiver decodes).

        Positions stay on the GPU end to end (int32 pieces -> one cat -> uint8 view);
        the wire broadcasts from the GPU anyway, and a host round-trip here
        (``.cpu().numpy().tobytes()`` + join) dominated the whole send at scale
        (~2.4s/sync at 7B steady state, ~83s on the full seed).
        """
        bytes_wire = self.quantize_fp8  # mixed dtypes (fp8 codes + fp32 scales + bf16)
        idx_pieces: list[torch.Tensor] = []
        val_pieces: list[torch.Tensor] = []
        params: list[DeltaParam] = []
        pos_off = val_off = 0
        for piece in per_param:
            nnz = int(piece.idx.numel())
            # positions ride the wire as int32 (pos_width=4); a parameter bigger than
            # 2^31 elements would silently wrap, so fail loud instead. DeltaParam
            # carries pos_width for a future 8-byte escalation if a model needs it.
            assert _prodshape(piece.shape) < (1 << 31), (
                f"{piece.name}: {_prodshape(piece.shape)} elements exceeds the int32 position encoding"
            )
            idx_pieces.append(piece.idx.to(torch.int32))
            val = piece.val.contiguous().view(torch.uint8) if bytes_wire else piece.val
            val_pieces.append(val)
            n_val = int(val.numel())  # elements, or bytes in bytes_wire mode
            params.append(
                DeltaParam(
                    name=piece.name,
                    dtype=piece.dtype_str,
                    shape=list(piece.shape),
                    pos_start=pos_off,
                    pos_end=pos_off + nnz * 4,
                    pos_width=4,
                    val_start=val_off,
                    val_end=val_off + n_val,
                )
            )
            pos_off += nnz * 4
            val_off += n_val

        values_gpu = torch.cat(val_pieces) if val_pieces else torch.empty(0, dtype=self.rollout_dtype, device="cuda")
        positions_u8 = (
            torch.cat(idx_pieces).contiguous().view(torch.uint8)
            if idx_pieces
            else torch.empty(0, dtype=torch.uint8, device=values_gpu.device)
        )
        cks = _checksum(positions_u8, values_gpu)
        return DeltaFlush(
            encoding=self.encoding, params=params, positions_cpu=positions_u8, values_gpu=values_gpu, checksum=cks
        )

    def _send_full_seed(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
        verify: bool = False,
        bytes_wire: bool = False,
        hold_last: bool = False,
    ) -> dict[str, float] | None:
        """First sync: stream the backend's FULL HF export over the values-only wire.

        ``weights`` is ``get_per_tensor_param()`` -- every backend already knows how
        to assemble and convert its own full tensors (FSDP all-gather, veomni expert
        restack, Megatron TP/PP fusion), so the seed inherits all of that for free
        and this engine only buckets and broadcasts. Every trainer rank iterates the
        generator (the per-tensor assembly is collective); rank 0 buckets. Resume
        works by construction: whatever the trainer restored is what ships."""
        is_r0 = self.is_master
        t0 = time.time()
        n_flushes = 0
        total_elems = 0
        wire_bytes = 0

        def _assemble_values(pieces: list[_ValuesPiece]):
            params = []
            val_off = 0
            for piece in pieces:
                n = int(piece.flat.numel())
                params.append(
                    DeltaParam(
                        name=piece.name,
                        dtype=piece.dtype_str,
                        shape=list(piece.shape),
                        pos_start=0,
                        pos_end=0,
                        pos_width=4,
                        val_start=val_off,
                        val_end=val_off + n,
                    )
                )
                val_off += n
            return params, torch.cat([piece.flat for piece in pieces])

        def _publish_values(pending, is_last: bool) -> None:
            nonlocal n_flushes, wire_bytes
            params, values = pending
            self._publish_values_flush(params, values, is_last=is_last, verify=verify, values_bytes=bytes_wire)
            n_flushes += 1
            wire_bytes += int(values.nbytes)

        bkt = _FlushBucket(self.bucket_size, _assemble_values, _publish_values)

        seen_names: set = set()
        for name, tensor in weights:
            # duplicate names in a full export mean two source params mapped to
            # the same HF tensor -- the receiver would apply whichever came last
            # and the delta diff base would silently disagree with the trainer
            # (observed: NemotronH A_log). Fail loud instead.
            assert name not in seen_names, f"full export yields duplicate HF tensor {name!r}"
            seen_names.add(name)
            tensor = tensor.detach()
            # fp8 codes and their fp32 scale_inv tensors ARE the rollout state:
            # never fold them into the bf16 wire dtype.
            keep_dtype = tensor.element_size() == 1 or name.endswith("_scale_inv")
            if tensor.is_floating_point() and tensor.dtype != self.rollout_dtype and not keep_dtype:
                tensor = tensor.to(self.rollout_dtype)
            if not is_r0:
                del tensor
                continue
            flat = tensor.contiguous().reshape(-1)
            total_elems += int(flat.numel())
            if bytes_wire:
                # mixed-dtype flushes: pack every piece as raw bytes; offsets
                # in the spec become BYTE offsets and the receiver reinterprets
                # per-param via ``values_bytes``.
                flat = flat.view(torch.uint8)
            bkt.add(_ValuesPiece(name, str(tensor.dtype).replace("torch.", ""), list(tensor.shape), flat), flat.nbytes)

        if not verify:
            self._shard_seeded = True
        if not is_r0:
            return
        bkt.seal()
        # ``hold_last`` keeps the receive session open for an appended verify
        # sweep (same contract as the steady path: only the sync's final flush
        # carries is_last).
        if bkt.pending is not None:
            bkt.emit(is_last=not hold_last)
        else:
            self._publish_terminal(not hold_last)
        # warning level on purpose: worker default log level swallows info, and the
        # one-off seed cost is the number people ask for when sizing a run.
        # the cupy staging pool does not return its blocks to CUDA on its own;
        # after streaming up to 2x bucket_size through it, give the memory back
        # so the trainer's next optimizer/forward pass can use it (raw cudaMalloc
        # OOMs on tight mcore shapes otherwise).
        self._release_staging_pool("seed")
        logger.warning(
            "delta-sharded FULL-%s v=%s done in %.1fs (flushes=%d elems=%d wire=%.1fGB)",
            "VERIFY" if verify else "SEED",
            global_steps,
            time.time() - t0,
            n_flushes,
            total_elems,
            wire_bytes / (1 << 30),
        )
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": 1.0,
            "checkpoint_engine/changed_elems": float(total_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(n_flushes),
        }

    async def send_weights(
        self,
        engine,
        global_steps: int | None = None,
    ) -> dict[str, float] | None:
        """Drive one weight sync from the TRAINING ENGINE (unlike the full-sync
        engines, which consume a weights iterator): the seed/steady phase choice
        and the snapshot prime are this engine's own state machine, so the worker
        stays delta-agnostic. The seed (first sync) streams the backend's full
        ``get_per_tensor_param()`` export over the values-only wire, then pins the
        diff base via ``engine.prime_delta_snapshots()``; every later sync consumes
        ``get_per_tensor_param_delta_shard()`` (backend-computed final-HF deltas).
        """
        # All actor ranks participate (gather-v is collective); only torch rank 0 broadcasts.
        # rank 0 accumulates the gathered per-param deltas into bucket_size-sized flushes and streams
        # each one as soon as it fills (then frees it), so peak memory is ~2 buckets rather than the
        # whole model.
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."
        if not self._shard_seeded:
            full, _ = engine.get_per_tensor_param(raw_master=self.quantize_fp8)
            if self.quantize_fp8:
                # fp8 rollout mode: the seed ships the rollout's EXACT state
                # (codes + scale_inv, trainer-quantized); snapshots pin the
                # SAME quantized shard stream, so the steady byte-diff runs
                # in the rollout's own domain.
                from verl.utils.device import is_cuda_available
                from verl.workers.engine.utils import prime_delta_snapshots

                metrics = self._send_full_seed(self._quantized_stream(full), global_steps, bytes_wire=True)
                if getattr(engine, "delta_shards_are_hf", False):
                    raw, _ = engine.get_per_tensor_param_shard()
                    prime_delta_snapshots(self._fp8_shard_stream(raw), self._fp8_snaps, pin=is_cuda_available)
                else:
                    for _ in self._mcore_fp8_entries(engine, prime_only=True):
                        pass
            else:
                metrics = self._send_full_seed(full, global_steps)
                # weights do not move during the sync, so the snapshots equal
                # exactly what the rollout just received.
                engine.prime_delta_snapshots()
            return metrics
        if self.quantize_fp8 and not getattr(engine, "delta_shards_are_hf", False):
            # mcore quant-domain sparse: the comm-stubbed probe turns the FULL
            # local shard into its HF view (real values at this rank's
            # positions, NaN placeholders elsewhere) -- the same transform the
            # bf16 steady already pays per sync. Quantize that view NaN-aware
            # (placeholders come out as the fp8 NaN byte, stable across syncs,
            # so the code byte-diff is zero there) and byte-diff codes and
            # scales against engine-held snapshots.
            weights = self._mcore_fp8_entries(engine)
        elif self.quantize_fp8:
            # quant-domain sparse steady: codes and scales are just tensors --
            # the stock diff/gather/wire pipeline runs on the transformed
            # shard stream against the engine-held quantized snapshots.
            import os

            from verl.workers.engine.utils import _hf_entry_identity, hf_delta_export

            prof = {"backend": 0.0, "quantize": 0.0, "diff": 0.0} if os.environ.get("VERL_DELTA_PROFILE") else None

            def _timed(gen, key):
                # cumulative per-phase wall time; device sync per item keeps the
                # attribution honest (profiling mode only).
                while True:
                    t0 = time.time()
                    try:
                        item = next(gen)
                    except StopIteration:
                        return
                    torch.cuda.synchronize()
                    prof[key] += time.time() - t0
                    yield item

            raw, _ = engine.get_per_tensor_param_shard()
            if prof is None:
                weights = hf_delta_export(self._fp8_shard_stream(raw), self._fp8_snaps, _hf_entry_identity)
            else:
                stream = _timed(self._fp8_shard_stream(_timed(iter(raw), "backend")), "quantize")
                weights = _timed(
                    iter(hf_delta_export(stream, self._fp8_snaps, _hf_entry_identity)), "diff"
                )
                self._fp8_prof = prof
        else:
            weights, _ = engine.get_per_tensor_param_delta_shard()
        is_r0 = self.is_master
        n_flushes = 0
        changed_elems = 0
        total_elems = 0
        wire_bytes = 0

        def _publish_steady(flush, is_last: bool) -> None:
            nonlocal n_flushes
            self._publish_flush(flush, first=False, is_last=is_last)
            n_flushes += 1

        bkt = _FlushBucket(self.bucket_size, self._assemble_flush, _publish_steady)

        batch_k = self.batch_gather

        def _bucket_slot_delta(
            name: str,
            dtype_str: str,
            full_shape: tuple,
            full_numel: int,
            aidx: torch.Tensor | None,
            aval: torch.Tensor | None,
        ) -> None:
            nonlocal total_elems, changed_elems, wire_bytes
            total_elems += int(full_numel)
            if aidx is None or aidx.numel() == 0:
                return
            changed_elems += int(aidx.numel())
            wire_bytes += int(aidx.numel()) * (4 + aval.element_size())
            _bucket_sliced(bkt, name, dtype_str, full_shape, aidx, aval)

        gq = _GatherQueue(batch_k, self.bucket_size, is_r0, _bucket_slot_delta)

        # ``weights`` is the BACKEND's HF delta stream (hf_delta_export): entries
        # already carry final HF coordinates -- naming, conversion, diff and
        # snapshot all happened on the backend side. This engine only batches,
        # gathers and ships.
        for slots, dtype_str, counts, hf_idx, hf_val, pg in weights:
            gq.put(pg, slots, dtype_str, counts, hf_idx, hf_val)
        gq.flush_all()

        # verify_every=K (engine kwarg) appends a full state-verification sweep to
        # every K-th steady sync, inside the SAME receive session: the steady
        # bucket keeps is_last unset and the sweep's final flush carries it. The
        # receiver bit-compares each copy_ destination before overwriting and
        # fails loud on any mismatch (see delta_loader._verify_dense).
        verify = self._verify_due()
        if is_r0:
            bkt.seal()  # seal the final partial bucket into the pending flush
            if bkt.pending is not None:
                bkt.emit(is_last=not verify)
            elif not verify:
                self._publish_terminal(False)
        if verify:
            # collective on every rank: the full export assembles per tensor.
            full, _ = engine.get_per_tensor_param(raw_master=self.quantize_fp8)
            if self.quantize_fp8:
                self._send_full_seed(self._quantized_stream(full), global_steps, verify=True, bytes_wire=True)
            else:
                self._send_full_seed(full, global_steps, verify=True)
        if not is_r0:
            return
        self._release_staging_pool("steady")  # return staging blocks to CUDA between syncs
        prof = getattr(self, "_fp8_prof", None)
        if prof is not None:
            quant = prof["quantize"] - prof["backend"]
            diff = prof["diff"] - prof["quantize"]
            logger.warning(
                "delta-fp8 profile v=%s backend_export=%.2fs quantize+amax=%.2fs diff+snap=%.2fs (rest=gather+wire)",
                global_steps,
                prof["backend"],
                quant,
                diff,
            )
            self._fp8_prof = None
        logger.info("delta-sharded send v=%s delta flushes=%d (streamed)", global_steps, n_flushes)
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": changed_elems / total_elems,
            "checkpoint_engine/changed_elems": float(changed_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(n_flushes),
        }
