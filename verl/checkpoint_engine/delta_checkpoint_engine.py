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

import contextlib
import json
import logging
import os
import time
import zlib
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
    # Gap bytes and their width, encoded ONCE where the piece is cut. Every
    # consumer (bucket accounting, the payload metric, the flush assembly) reads
    # these instead of re-encoding: each _gap_encode call ends in a device->host
    # read, and on rank 0's per-parameter path those reads serialise the CUDA
    # stream. Recomputing turned a 4x smaller wire into a slower sync.
    gaps: torch.Tensor = None
    pos_width: int = 4


@dataclass(slots=True)
class _ValuesPiece:
    """One whole parameter's flat values buffered for the seed sync's values-only flush."""

    name: str
    dtype_str: str
    shape: list
    flat: torch.Tensor




def _selfcheck_on():
    """Sender-side self-check: dump dir, or None when off (the default)."""
    return os.environ.get("VERL_DELTA_SELFCHECK_DIR") or None


def _selfcheck_sampled(key):
    """Deterministic fraction sample, keyed so a fusion group is kept whole.

    Same rule as the verify sweep, for the same reason: half a fused pair is a
    broken sample, not a smaller one -- that mistake is what took down the last
    verification run.
    """
    frac = float(os.environ.get("VERL_DELTA_SELFCHECK_FRACTION", "0.002"))
    return (zlib.crc32(key.encode()) & 0xFFFFFFFF) / 0x100000000 < frac


def _selfcheck_key(name):
    from verl.utils.fusion_groups import fusion_match

    hit = fusion_match(name)
    return f"{name[: -len(hit[1])]}{hit[0]}" if hit else name


def _selfcheck_record_piece(store, piece):
    """Record one ENCODED wire piece for a sampled parameter.

    Deliberately hooked after ``_slice_pieces`` rather than on the backend's
    entry stream: the gap encoding is part of what needs checking (the int16
    signed-limit bug lived exactly there, and it produced a perfectly plausible
    tensor), so the dump has to hold the bytes that actually ship, not the
    positions before they were encoded.
    """
    if store is None or not _selfcheck_sampled(_selfcheck_key(piece.name)):
        return
    store.setdefault(piece.name, []).append(
        {
            "dtype_str": piece.dtype_str,
            "shape": list(piece.shape),
            "pos_width": piece.pos_width,
            "gaps": piece.gaps.detach().cpu().clone(),
            "val": piece.val.detach().reshape(-1).cpu().clone(),
        }
    )


def _selfcheck_write(engine, spec_fn, quantize_fp8, step, pieces_store, out_dir, is_r0):
    """Dump this sync's sampled wire pieces AND the dense state they produce.

    The correctness question on DSv4 -- "does the delta actually describe the
    change?" -- is still open because both in-line attempts took the SGLang
    server down mid-sync. Nothing here touches the server: with two consecutive
    syncs' dumps, an offline script can check that dense[N] equals dense[N-1]
    with this sync's decoded delta applied, and that no byte outside the delta's
    positions moved. The second half is the one that catches a delta which is
    merely incomplete -- a dropped replica or half a fused pair passes the first
    check on its own.
    """
    # Is the dense export bit-DETERMINISTIC? The whole offline check assumes the
    # dense we dump here is the same artifact the delta was computed against, but
    # we obtain it by calling get_per_tensor_param a SECOND time, which re-runs
    # quantization. If absmax reduction order or boundary rounding varies, the two
    # differ -- and that would produce exactly what the first real run showed:
    # 0.309% of bytes off, concentrated on quantized MoE expert weights, a few KB
    # per tensor. So measure it before believing the 586 failures are a delta bug.
    # Needs no second training step, unlike the comparison it validates.
    if os.environ.get("VERL_DELTA_SELFCHECK_DETERMINISM") == "1":
        a, _ = engine.get_per_tensor_param(raw_master=quantize_fp8, quant_spec=spec_fn())
        first = {n: torch.hash_tensor(t.detach().contiguous().reshape(-1).view(torch.uint8)).reshape(()) for n, t in a}
        keys = sorted(first)
        h1 = torch.stack([first[k] for k in keys]).tolist() if keys else []
        b, _ = engine.get_per_tensor_param(raw_master=quantize_fp8, quant_spec=spec_fn())
        second = {n: torch.hash_tensor(t.detach().contiguous().reshape(-1).view(torch.uint8)).reshape(()) for n, t in b}
        h2 = torch.stack([second[k] for k in keys]).tolist() if keys else []
        diff = [k for k, x, y in zip(keys, h1, h2, strict=True) if x != y]
        logger.warning(
            "DELTA-SELFCHECK determinism: %d/%d tensors differ between two consecutive "
            "get_per_tensor_param calls (nonzero means the dense dump is NOT the artifact the "
            "delta was diffed against, so the offline mismatches are the tool's, not the delta's); "
            "first=%s",
            len(diff),
            len(keys),
            sorted(diff)[:10],
        )
    full, _ = engine.get_per_tensor_param(raw_master=quantize_fp8, quant_spec=spec_fn())
    dense = {}
    for name, tensor in full:  # drained in full on EVERY rank: the assembly is collective
        if is_r0 and _selfcheck_sampled(_selfcheck_key(name)):
            dense[name] = tensor.detach().reshape(-1).view(torch.uint8).cpu().clone()
    if not is_r0:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {"step": step, "dense": dense, "pieces": pieces_store},
            os.path.join(out_dir, f"selfcheck_step{step}.pt"),
        )
        logger.warning(
            "delta selfcheck: step=%s dense_params=%d wire_params=%d -> %s",
            step,
            len(dense),
            len(pieces_store),
            out_dir,
        )
    except OSError as e:
        logger.warning("delta selfcheck: could not write dump: %s", e)


def _verify_sample(gen):
    """Ship only a sampled share of the verification sweep.

    The sweep re-sends the WHOLE model densely and the receiver materialises a
    full-shape tensor per parameter to bit-compare before overwriting. On DSv4
    that took the SGLang server down mid-sync, so the one run that would have
    proved bit-correctness never produced a verdict -- leaving "it is fast"
    measured and "it is right" not.

    Sampling by name keeps the guarantee's shape (a real bit-compare against the
    trainer's own export, on real weights) at a fraction of the transfer. It is
    a spot check rather than a proof: a fault confined to unsampled parameters
    slips through, so treat a pass as evidence, not a certificate.

    The generator is consumed in full regardless -- the per-tensor assembly
    behind it is collective, so every rank must walk the same sequence. Only
    shipping is skipped. Selection hashes the name, so every rank picks the
    same set without sharing state.
    """
    frac = float(os.environ.get("VERL_DELTA_VERIFY_FRACTION", "1.0"))
    if frac >= 1.0:
        yield from gen
        return
    from verl.utils.fusion_groups import fusion_match

    kept = total = 0
    for name, tensor in gen:
        total += 1
        # Hash the FUSION GROUP, not the bare name. DSv4's loader rebuilds some
        # params by cat-ing two separately-named halves and asserts its cache is
        # empty on return, so the halves must travel together. Sampling by name
        # kept one half and dropped the other, and the sender's own
        # assert_drained caught it as "fusion groups never completed" -- the same
        # class as the five byte-level splitters this path already had to fix,
        # only this time the splitter was a sampling filter.
        hit = fusion_match(name)
        key = f"{name[: -len(hit[1])]}{hit[0]}" if hit else name
        if (zlib.crc32(key.encode()) & 0xFFFFFFFF) / 0x100000000 < frac:
            kept += 1
            yield name, tensor
    logger.warning(
        "delta verify: sampled %d/%d params (fraction=%.3f) -- a pass covers only what was sampled",
        kept,
        total,
        frac,
    )


class _Phase:
    """Wall-clock and count accumulator for one weight sync's send path.

    update_weights was a single number, and three separate explanations for it
    (fp8 amplification, per-parameter device reads, the backend's O(N) scan) were
    each refuted or corrected by measurement. Splitting it one layer per cluster
    run costs 70 minutes a layer, so this instruments every step of the path at
    once: whichever line owns the time, this run names it.

    ``sync=True`` brackets a span with device synchronisation, which is required
    for anything that only ENQUEUES GPU work (the broadcasts, the staging copies)
    -- otherwise the cost lands on whichever unrelated call happens to block
    next. It does remove some overlap, so the instrumented total runs slightly
    high; attribution is the point here, not the headline number.

    ``hot=True`` marks a span INSIDE the per-parameter loop, where that trade
    stops being acceptable: at 67,569 parameters a synced span is 135k
    ``cuda.synchronize()`` calls per sync, which does not merely inflate the
    number -- it serialises the device and destroys the overlap the profile is
    supposed to be measuring. Hot synced spans therefore only run under
    ``VERL_DELTA_PROFILE_SYNC=1``, and when they are off they record NOTHING
    rather than an unsynchronised number. A missing metric is honest; a number
    whose cost landed on an unrelated later call is worse than no number, because
    it reads exactly like a measurement.
    """

    __slots__ = ("t", "n")

    def __init__(self):
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}

    @contextlib.contextmanager
    def span(self, key: str, sync: bool = False, hot: bool = False):
        if hot and sync and os.environ.get("VERL_DELTA_PROFILE_SYNC", "0") != "1":
            self.n["hot_spans_skipped"] = self.n.get("hot_spans_skipped", 0) + 1
            yield
            return
        cuda = sync and torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            self.t[key] = self.t.get(key, 0.0) + (time.perf_counter() - t0)

    def bump(self, key: str, k: int = 1) -> None:
        self.n[key] = self.n.get(key, 0) + k

    def metrics(self) -> dict[str, float]:
        out = {f"checkpoint_engine/t_{k}_s": v for k, v in sorted(self.t.items())}
        out.update({f"checkpoint_engine/n_{k}": float(v) for k, v in sorted(self.n.items())})
        return out


def _gap_encode(idx: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sorted absolute positions -> gap bytes + the width used, per parameter.

    Stores ``idx[k] - idx[k-1] - 1`` with ``idx[-1] := -1``, so the first entry is
    the first absolute position and the receiver inverts with
    ``cumsum(gap + 1) - 1``. Same scheme as slime's ``encode_deltas``.

    Why it matters here: a measured DSv4 delta is 12-17% of elements and the wire
    was still 72% of a full sync, because each fp8 code (1 byte) carried an int32
    position (4 bytes) -- the position cost four times the data.

    Width is chosen PER PARAMETER, and 1 byte is on the table where slime offers
    only 2 or 4. That is not a guess: at the densities we measure, gaps are far
    smaller than at the ~2% slime's comment assumes. P(gap > 255) per position is
    3.4e-15 at 12.25% density and 2.5e-21 at 16.97%. The wider widths remain as
    fallbacks, so a sparser-than-expected parameter costs bytes, never
    correctness.

    Thresholds are the SIGNED limits of the carrier types, not the byte widths:
    uint8 is unsigned so it holds 0xFF, but the 2-byte carrier is torch.int16,
    which stops at 0x7FFF. Using 0xFFFF there wrapped gaps in [0x8000, 0xFFFF]
    to negative, and the receiver's ``cumsum`` then produced negative positions
    that scattered out of bounds -- a CUDA device-side fault that kills the
    SGLang scheduler outright rather than raising. torch.uint16 exists but too
    few ops accept it to carry through view/cumsum safely, so the 32768..65535
    band pays 4 bytes instead of 2. Correctness over a byte.

    Returns uint8/int16/int32 gaps; each is viewed as raw bytes by the caller.
    """
    if idx.numel() == 0:
        return idx.to(torch.int32), 4
    prev = torch.cat([idx.new_full((1,), -1), idx[:-1]])
    # gap = idx - prev, NOT idx - prev - 1. The extra -1 packed a little tighter
    # but made a repeated position encode as -1, so every parameter had to be
    # deduplicated first -- and dedup is a boolean mask, whose output size is
    # data-dependent, so it forced a device->host sync PER PARAMETER inside the
    # send loop. Letting a duplicate ride as gap 0 removes that sync entirely and
    # restores index_copy_'s old last-writer-wins semantics for free. The cost is
    # one larger unit of max_gap, which almost never changes the width tier.
    gaps = idx - prev
    # Gap encoding REQUIRES strictly increasing positions; absolute int32 did not,
    # which is why nothing upstream ever had to guarantee it. Assert rather than
    # trust: a descending step makes a negative gap, and a negative gap in an
    # unsigned or narrow carrier wraps to a large positive one. Nothing then looks
    # wrong -- the sender's own range check passes (the raw positions are fine),
    # the receiver sees only non-negative gaps -- and the decoded positions
    # silently overshoot into an out-of-bounds scatter.
    # ONE sync for both bounds: two .item() calls are two stream stalls, and this
    # runs per parameter on rank 0.
    min_gap, max_gap = torch.stack([gaps.min(), gaps.max()]).tolist()
    assert min_gap >= 0, (
        f"gap encoding needs non-decreasing positions, got a step of {min_gap}; "
        "sort (idx, val) together before encoding"
    )
    if max_gap <= 0xFF:  # uint8 is unsigned
        return gaps.to(torch.uint8), 1
    if max_gap <= 0x7FFF:  # int16 is SIGNED -- 0xFFFF here silently wrapped
        return gaps.to(torch.int16), 2
    return gaps.to(torch.int32), 4


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

    def add_atomic(self, sized_pieces: list[tuple]) -> None:
        """Add several pieces that must not be split across flushes.

        The cap check happens BEFORE the group goes in (seal the current bucket
        first if it would overflow), so the boundary can only fall between
        groups -- never inside one. A group larger than ``cap`` becomes its own
        oversized flush, which is correct if not ideal; the fused DSv4 params
        this exists for are a few MiB.
        """
        if not sized_pieces:
            return
        total = sum(int(nb) for _, nb in sized_pieces)
        if self.pieces and self.nbytes + total > self.cap:
            self.seal()
        for piece, nbytes in sized_pieces:
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


# Membership table shared with the rollout-side splitters -- see
# verl.utils.fusion_groups for why this exists and which loader needs it.
from verl.utils.fusion_groups import FUSION_GROUPS as _FUSION_GROUPS  # noqa: E402
from verl.utils.fusion_groups import fusion_match as _fusion_match  # noqa: E402


class _FusionStager:
    """Hold the members of a fused destination param until the group is complete,
    then release them together so they ride one flush.

    Two things have to be true at the receiver for a fused param to survive a
    sparse sync, and this covers both:

    * **completeness** -- a member with no changed elements is released as an
      EMPTY entry instead of being dropped. The receiver densifies it to an
      all-NaN full-shape tensor, and since ``_masked_copy`` keeps the
      destination wherever the source is NaN, cat-ing that half into the fused
      param is a no-op for it. (Verified: ``_decode_one`` already returns pure
      NaN for a zero-length entry, both in the fp8 byte path and the float path,
      so the receiver needs no change.)
    * **co-location** -- see ``_FlushBucket.add_atomic``; being complete does not
      help if the two halves land in different ``load_weights`` calls.

    Params outside any group pass straight through.
    """

    __slots__ = ("_pending", "n_groups", "n_filled")

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], dict] = {}
        self.n_groups = 0  # groups released with at least one changed member
        self.n_filled = 0  # halves materialised as all-NaN because nothing changed

    _match = staticmethod(_fusion_match)

    def offer(self, name: str, dtype_str: str, shape, aidx, aval):
        """Return ``(entries, is_group)``, or ``None`` while a group is incomplete.

        A non-member yields itself with ``is_group=False`` -- the caller keeps
        dropping unchanged non-members, so this costs nothing for the ~99% of
        params that are not fused. A member yields ``None`` until its siblings
        arrive, then the whole group in declared order with ``is_group=True`` --
        or ``([], True)`` if no member of the group changed at all, since there
        is then nothing to send.
        """
        matched = self._match(name)
        if matched is None:
            return [(name, dtype_str, shape, aidx, aval)], False

        key, sfx = matched
        suffixes = next(s for k, s in _FUSION_GROUPS if k == key)
        slot = self._pending.setdefault((name[: -len(sfx)], key), {})
        assert sfx not in slot, f"duplicate fusion member {name!r} for group {key!r}"
        slot[sfx] = (name, dtype_str, shape, aidx, aval)
        if len(slot) < len(suffixes):
            return None

        self._pending.pop((name[: -len(sfx)], key))
        members = [slot[s] for s in suffixes]
        if all(e[3] is None or e[3].numel() == 0 for e in members):
            return [], True
        # Materialise the absent halves. Device/dtype come from a member that did
        # change, so the empties cat cleanly with the rest of the flush.
        donor = next(e for e in members if e[3] is not None and e[3].numel())
        dev = donor[3].device
        out = []
        for m_name, m_dtype, m_shape, m_idx, m_val in members:
            if m_idx is None or m_idx.numel() == 0:
                m_idx = torch.empty(0, dtype=torch.int32, device=dev)
                m_val = torch.empty(0, dtype=getattr(torch, m_dtype), device=dev)
                self.n_filled += 1
            out.append((m_name, m_dtype, m_shape, m_idx, m_val))
        self.n_groups += 1
        return out, True

    def offer_piece(self, name: str, piece, nbytes: int):
        """Seed-path variant: co-locate a group's members, nothing else.

        A full export contains every member by construction, so there is no
        absent half to materialise -- only the flush boundary matters. Returns
        ``(list_of_(piece, nbytes), is_group)`` or ``None`` while incomplete.
        """
        matched = self._match(name)
        if matched is None:
            return [(piece, nbytes)], False
        key, sfx = matched
        suffixes = next(s for k, s in _FUSION_GROUPS if k == key)
        slot = self._pending.setdefault((name[: -len(sfx)], key), {})
        assert sfx not in slot, f"duplicate fusion member {name!r} for group {key!r}"
        slot[sfx] = (piece, nbytes)
        if len(slot) < len(suffixes):
            return None
        self._pending.pop((name[: -len(sfx)], key))
        self.n_groups += 1
        return [slot[s] for s in suffixes], True

    def assert_drained(self) -> None:
        assert not self._pending, (
            f"fusion groups never completed: {sorted(self._pending)}. Every member listed in "
            f"_FUSION_GROUPS must appear in the export stream, including unchanged ones."
        )


def _slice_pieces(name: str, dtype_str: str, shape, aidx: torch.Tensor, aval: torch.Tensor) -> list[tuple]:
    """Slice one param's (idx, val) delta into <= MAX_ENTRY_ELEMS ``(piece, nbytes)``
    pairs (bounds the receiver-side decode transient; the masked apply is sequential,
    so splitting is transparent). Bucket bytes = actual wire bytes (int32 positions
    + values).

    An empty delta yields ONE empty piece rather than none: a zero-length range()
    would emit nothing, but fusion-group members with no changed elements must
    still reach the receiver so it can densify them to all-NaN (see _FusionStager).

    Bucket bytes must track the width the positions will ACTUALLY be encoded at.
    Charging the old flat 4 bytes while gaps cost 1 makes every bucket seal at
    roughly 40% of its size, so the same payload is split into ~2.5x the flushes
    -- pure per-flush overhead, and it lands directly on the number this encoding
    exists to improve. One max() per parameter buys the right figure; per-piece
    widths can only be narrower, so this is a safe upper bound.
    """
    if aidx.numel() == 0:
        return [(_FlushPiece(name, dtype_str, list(shape), aidx, aval, aidx.to(torch.int32), 4), 0)]
    max_elems = DeltaShardedCheckpointEngine.MAX_ENTRY_ELEMS
    out = []
    for s in range(0, aidx.numel(), max_elems):
        e = min(s + max_elems, aidx.numel())
        # Encode here, once. A piece's first gap is its absolute start, which can
        # exceed every interior gap, so a piece's width is not derivable from the
        # parameter's -- it has to be per piece. What we avoid is doing it TWICE
        # (once to size the bucket, again to assemble the flush).
        gaps, width = _gap_encode(aidx[s:e])
        out.append(
            (
                _FlushPiece(name, dtype_str, list(shape), aidx[s:e], aval[s:e], gaps, width),
                (e - s) * (width + aval.element_size()),
            )
        )
    return out


def _bucket_sliced(bkt: _FlushBucket, name: str, dtype_str: str, shape, aidx: torch.Tensor, aval: torch.Tensor) -> None:
    for piece, nbytes in _slice_pieces(name, dtype_str, shape, aidx, aval):
        bkt.add(piece, nbytes)


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

    __slots__ = ("batch_k", "max_round_bytes", "is_r0", "_consume", "_queues", "timers")

    def __init__(self, batch_k: int, max_round_bytes: int, is_r0: bool, consume, timers=None):
        self.batch_k = max(int(batch_k), 1)
        self.max_round_bytes = int(max_round_bytes)
        self.is_r0 = is_r0
        self._consume = consume
        self.timers = timers
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
        # Time the collective alone. ship_s is 79% of update_weights while the
        # wire is only 45 GiB (0.42 GiB/s), so the cost is structural, not
        # bandwidth -- but gather, rank-0 encoding and publish are folded together
        # and want completely different fixes. Measure before choosing one.
        _t = time.perf_counter()
        gathered = gather_slot_entries_to_rank0(
            idx_concat,
            val_concat,
            counts_concat,
            group=pg,
            max_round_bytes=self.max_round_bytes,
            stats=self.timers.get("imbalance") if self.timers is not None else None,
        )
        if self.timers is not None:
            self.timers["gather"] += time.perf_counter() - _t
            self.timers["rounds"] = self.timers.get("rounds", 0) + 1
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
                # sparse flushes carry the quant config too: the receiver's
                # handshake (incl. the seed-required sentinel guard) must be
                # reachable on the steady path, not only on the dense seed.
                "quant_config": getattr(self, "_fp8_quant_cfg", None),
                "params": [vars(p) for p in flush.params],
                "checksum": int(flush.checksum),
            },
        }
        ph = getattr(self, "_phase", None) or _Phase()
        # Collect the PREVIOUS flush before touching the allocator for this one.
        # Everything between that broadcast and this line ran overlapped with it.
        self._await_publish_inflight(ph)
        # The manifest pickles one dict per param in the flush -- thousands of
        # them -- so this is not obviously free either.
        with ph.span("pub_zmq"):
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(meta)
        pos_u8 = flush.positions_cpu.to("cuda", non_blocking=True).contiguous().view(torch.uint8)
        val_u8 = flush.values_gpu.contiguous().view(torch.uint8)
        # Stage into cupy-owned buffers: ray's NCCL broadcast is enqueued on a separate
        # stream with no recordStream on its inputs, so broadcasting a zero-copy view of
        # these torch tensors (freed right after this call) would race with allocator reuse.
        with ph.span("pub_stage", sync=True):
            pos_cp = cp.empty(pos_u8.numel(), dtype=cp.uint8)
            val_cp = cp.empty(val_u8.numel(), dtype=cp.uint8)
            pos_cp[:] = cp.asarray(pos_u8)
            val_cp[:] = cp.asarray(val_u8)
        # Two broadcasts per flush. With the overlap on they are ENQUEUED ONLY --
        # the wait happens in _await_publish_inflight, either before the next
        # flush stages or at the end of the sync -- and pos_cp/val_cp are parked
        # on self until then, because dropping them here would let the allocator
        # reuse the memory mid-broadcast.
        #
        # The kill switch is not ceremony: parking a flush's staging buffers
        # raises peak cupy staging from one flush to two, and the cupy pool does
        # not return blocks to CUDA on its own. If a run OOMs, this separates "the
        # overlap costs too much memory" from "something else broke" without
        # rebuilding the frozen tree and paying another 45 minutes to find out.
        overlap = os.environ.get("VERL_DELTA_PUBLISH_OVERLAP", "1") == "1"
        with ph.span("pub_bcast_enqueue", sync=not overlap):
            collective.broadcast(pos_cp, src_rank=0, group_name=self.group_name)
            collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)
        if overlap:
            self._pub_inflight = (pos_cp, val_cp)
        ph.bump("pub_bcast_calls", 2)
        ph.bump("pub_flushes")

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
                "quant_config": getattr(self, "_fp8_quant_cfg", None),
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

    def _await_publish_inflight(self, ph=None) -> None:
        """Block until the previous flush's broadcasts have landed, then let its
        staging buffers go.

        This is the whole of the publish/build overlap: ``_publish_flush`` used to
        synchronise immediately after enqueueing its two broadcasts, so 5.9-9.8 s
        of the sync was rank 0 sitting idle while 79 other ranks also sat idle.
        Deferring the wait to just before the NEXT flush stages lets rank 0 build
        that flush -- sort, encode, bucket -- while the previous one is still on
        the wire.

        Two things make the deferral safe rather than a race:
          - the cupy staging buffers are held on ``self`` until the wait, because
            ray's broadcast keeps no reference to them and the allocator would
            otherwise be free to hand that memory to the next flush mid-transfer.
            That is the same hazard the staging copy exists for, one step later.
          - the wait is a FULL device synchronise, not a torch event: ray enqueues
            the broadcast on its own stream, so an event recorded on torch's
            current stream would not cover it and would return early.

        ``t_pub_await_s`` is therefore the publish time that could NOT be hidden --
        the metric that says whether this worked, where ``t_pub_bcast_enqueue_s``
        alone would only report how long it takes to hand work to NCCL.
        """
        if getattr(self, "_pub_inflight", None) is None:
            return
        ph = ph or getattr(self, "_phase", None) or _Phase()
        with ph.span("pub_await"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        self._pub_inflight = None
        ph.bump("pub_awaits")

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
        gather_round_mbytes: int = 0,
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
        self._shard_seeded = False
        # Gather the per-param sparse deltas in groups of this many parameters
        # (one count-matrix all_gather + two padded gathers per group instead of
        # three collectives per parameter).
        self.batch_gather = int(batch_gather)
        # Bytes per gather round. Profiling put 82.7 s of a 129 s sync in the
        # gather -- 218 rounds at 380 ms each, moving only ~200 MB per round,
        # while the same 42.9 GiB broadcast in 5.4 s (0.52 vs 7.9 GiB/s). The
        # rounds were bounded by the ENTRY COUNT (batch_gather=32), never by the
        # byte budget, so each one was small and paid full collective latency.
        # 0 keeps the old behaviour (budget = the flush bucket size).
        self.gather_round_bytes = int(gather_round_mbytes) << 20 if gather_round_mbytes else 0

    def _verify_due(self) -> bool:
        """True on every K-th steady sync (``verify_every=K``), and ALWAYS on the
        first steady sync -- the runtime fuse must exist even when periodic
        verification is off (verify_every=0 keeps only that one mandatory sweep)."""
        self._steady_count = getattr(self, "_steady_count", 0) + 1
        if self._steady_count == 1:
            # The mandatory first-sync sweep is a correctness fuse, but it is
            # expensive in two ways that matter when the point of the run is a
            # NUMBER rather than a check:
            #   * it runs INSIDE the region trainer_separate_async times as
            #     ``update_weights``, so it inflates the reported figure by a
            #     full extra dense pass;
            #   * it calls load_weights once per unit, and SGLang logs its entire
            #     "Some weights are not initialized from checkpoints: {...}" set
            #     (~60 KB) on every one of them -- 58,904 such lines and 771 MB on
            #     the head node in run ddp16d, which then killed the job with
            #     "RaySystemError: Sent message larger than max (536878753 vs
            #     536870912)" from ray's log-forwarding thread.
            # So it stays on by default and is opt-out, never opt-in.
            if os.environ.get("VERL_DELTA_SKIP_FIRST_VERIFY") == "1":
                logger.warning(
                    "VERL_DELTA_SKIP_FIRST_VERIFY=1: skipping the mandatory first-sync verify sweep. "
                    "timing_s/update_weights now measures the sync alone; correctness is NOT checked."
                )
                return False
            return True
        if self.verify_every <= 0:
            return False
        return self._steady_count % self.verify_every == 0

    def _fp8_spec(self, engine=None):
        """Distill the serving engine's quant config into a rollout-agnostic
        :class:`~verl.utils.fp8_sharded.QuantSpec` for the backend."""
        from verl.utils.fp8_sharded import QuantSpec

        h = self._fp8_helper(engine)
        return QuantSpec(
            weight_block_size=tuple(h.quant_config.get("weight_block_size", [128, 128])),
            should_quantize=self._quant_predicate(h),
        )

    def _quant_predicate(self, helper):
        """Which params to fp8-quantize: from the checkpoint, or from name patterns.

        ``VERL_FP8_SELECT_FROM_CKPT=1`` reads the safetensors headers of the model
        being served and asks each tensor's real dtype, which is what vLLM does on
        its side of the wire (``module.weight.dtype``) and what the name allowlist
        can only approximate. Off by default so the two can be compared on the same
        model before either is made the rule.

        Falls back to the allowlist -- loudly -- if the checkpoint cannot answer,
        because "selected nothing" is precisely the failure this exists to prevent.
        """
        if os.environ.get("VERL_FP8_SELECT_FROM_CKPT") != "1":
            return helper.should_quantize_param
        from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp8_predicate

        path = getattr(getattr(self, "model_config", None), "local_path", None) or os.environ.get(
            "VERL_FP8_CKPT_PATH"
        )
        if not path:
            logger.warning(
                "VERL_FP8_SELECT_FROM_CKPT=1 but no model path is reachable here; "
                "falling back to the name allowlist. Set VERL_FP8_CKPT_PATH."
            )
            return helper.should_quantize_param
        pred = build_ckpt_fp8_predicate(path)
        if pred is None:
            logger.warning("fp8 selection: checkpoint at %s could not answer; using the name allowlist", path)
            return helper.should_quantize_param
        logger.warning("fp8 selection: using CHECKPOINT dtypes from %s (not the name allowlist)", path)
        return pred

    def _fp8_helper(self, engine=None):
        """Build the quantizer helper from the SAME inputs the rollout used --
        the model's hf_config (real ignored_layers / modules_to_not_convert)
        -- instead of guessing bare defaults, and guard the supported mode."""
        from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper, build_sglang_fp8_quant_config

        h = getattr(self, "_fp8_helper_inst", None)
        if h is None:
            hf_config = None
            if engine is not None:
                model_config = getattr(engine, "model_config", None)
                hf_config = getattr(model_config, "hf_config", None)
            if hf_config is None:
                logger.warning(
                    "fp8 delta: no hf_config available; quant config built from bare defaults "
                    "(ignored_layers/modules_to_not_convert from the checkpoint will be missed)"
                )
            cfg = build_sglang_fp8_quant_config(hf_config)
            # supported-mode guards: this engine ships blockwise fp8 with a
            # plain fp32 scale_inv grid, nothing else.
            assert cfg.get("quant_method") == "fp8", f"unsupported quant_method {cfg.get('quant_method')!r}"
            if cfg.get("weight_block_size") is None:
                raise NotImplementedError("per-tensor fp8 is not supported by the delta engine (blockwise only)")
            for bad in ("scale_fmt", "ue8m0", "deepgemm"):
                assert bad not in cfg, f"unsupported scale format flag {bad!r} in quant config: {cfg}"
            self._fp8_quant_cfg = cfg
            h = SGLangFP8QuantizerHelper(cfg)
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
            assert _prodshape(piece.shape) < (1 << 31), (
                f"{piece.name}: {_prodshape(piece.shape)} elements exceeds the int32 position encoding"
            )
            gaps, width = piece.gaps, piece.pos_width  # encoded in _slice_pieces
            # Widths are chosen per parameter, so two invariants that held while
            # every position was int32 have to be re-established by hand:
            #   1. pieces are cat-ed as raw bytes -- torch.cat on mixed dtypes
            #      type-promotes (a uint8 piece beside an int32 one silently
            #      becomes int32), which would rewrite the byte layout.
            #   2. pos_start is padded up to the width, because the receiver
            #      views the slice as int16/int32 and a storage offset that is
            #      not a multiple of the element size cannot be viewed.
            pad = (-pos_off) % width
            if pad:
                idx_pieces.append(torch.zeros(pad, dtype=torch.uint8, device=gaps.device))
                pos_off += pad
            idx_pieces.append(gaps.contiguous().view(torch.uint8))
            val = piece.val.contiguous().view(torch.uint8) if bytes_wire else piece.val
            val_pieces.append(val)
            n_val = int(val.numel())  # elements, or bytes in bytes_wire mode
            params.append(
                DeltaParam(
                    name=piece.name,
                    dtype=piece.dtype_str,
                    shape=list(piece.shape),
                    pos_start=pos_off,
                    pos_end=pos_off + nnz * width,
                    pos_width=width,
                    val_start=val_off,
                    val_end=val_off + n_val,
                )
            )
            pos_off += nnz * width
            val_off += n_val

        ph = getattr(self, "_phase", None) or _Phase()
        with ph.span("asm_cat", sync=True):
            values_gpu = torch.cat(val_pieces) if val_pieces else torch.empty(0, dtype=self.rollout_dtype, device="cuda")
            positions_u8 = (
                torch.cat(idx_pieces).contiguous()  # already uint8 bytes, per-piece
                if idx_pieces
                else torch.empty(0, dtype=torch.uint8, device=values_gpu.device)
            )
        # A reduction over the whole flush (~500 MB), once per flush. Cheap per
        # byte, but it is on the critical path and nothing had measured it.
        with ph.span("asm_checksum", sync=True):
            cks = _checksum(positions_u8, values_gpu)
        ph.bump("asm_params", len(params))
        return DeltaFlush(
            encoding=self.encoding, params=params, positions_cpu=positions_u8, values_gpu=values_gpu, checksum=cks
        )

    def _send_full_seed(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
        verify: bool = False,
        bytes_wire: bool = False,
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
        # The seed streams the FULL export, so every fused member is present --
        # but byte bucketing splits pairs here exactly like it does in the steady
        # path, and the seed is the FIRST sync, so without this the run dies
        # before the steady staging is ever exercised.
        stager = _FusionStager()

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
            offered = stager.offer_piece(
                name,
                _ValuesPiece(name, str(tensor.dtype).replace("torch.", ""), list(tensor.shape), flat),
                flat.nbytes,
            )
            if offered is None:
                continue
            released, is_group = offered
            if is_group:
                bkt.add_atomic(released)
            else:
                for piece, nbytes in released:
                    bkt.add(piece, nbytes)

        if not verify:
            self._shard_seeded = True
        if not is_r0:
            return
        stager.assert_drained()
        logger.info("seed fusion staging: groups=%d", stager.n_groups)
        bkt.seal()
                # sweep (same contract as the steady path: only the sync's final flush
        # carries is_last).
        if bkt.pending is not None:
            bkt.emit(is_last=True)
        else:
            self._publish_terminal(True)
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
        # Wall clock for the ENTIRE call. export_s + ship_s only cover the main
        # consume loop, and on contrib2 they summed to 41.0/44.7 s against a
        # timing_s/update_weights of 51.0/53.0 -- so 16-20% of the sync sat
        # outside every span we had. That residual has two completely different
        # causes with completely different fixes: work on this rank that no span
        # covers, or time that is not on this rank at all (ray dispatch across
        # 80 workers, or another rank straggling in the collective gather).
        # total_s discriminates them in one number: if it lands near
        # update_weights the residual is ours to find, and if it lands near
        # export+ship the residual is outside this function entirely. Guessing
        # between the two would have sent the next optimisation at the wrong half.
        t_total0 = time.perf_counter()
        if not self._shard_seeded:
            # the BACKEND produces the seed in the rollout's exact format (with a
            # quant spec: codes + scale_inv off the raw master; without: bf16),
            # ships it verbatim, then snapshots the SAME domain as the diff base
            # -- weights do not move during the sync, so the snapshots equal
            # exactly what the rollout just received.
            spec = self._fp8_spec(engine) if self.quantize_fp8 else None
            full, _ = engine.get_per_tensor_param(raw_master=spec is not None, quant_spec=spec)
            metrics = self._send_full_seed(full, global_steps, bytes_wire=spec is not None)
            engine.prime_delta_snapshots(quant_spec=spec)
            return metrics
        # the BACKEND owns delta production for every dtype: with a quant spec
        # it yields quant-domain entries (codes + scale grids diffed against
        # engine-held snapshots), without one the bf16 shard deltas.
        _t = time.perf_counter()
        _spec = self._fp8_spec(engine) if self.quantize_fp8 else None
        t_spec = time.perf_counter() - _t
        _t = time.perf_counter()
        weights, _ = engine.get_per_tensor_param_delta_shard(quant_spec=_spec)
        t_delta_open = time.perf_counter() - _t
        # t_delta_open is 14% of the sync and is essentially one call:
        # load_megatron_model_to_gpu, bringing params back from the CPU offload
        # that runs while the rollout has the GPU. Split it, because resize_
        # (re-allocating the shard every sync) and the pinned H2D copy have
        # different fixes. The generator itself is lazy, so nothing else here
        # can account for the time.
        try:
            from verl.utils.megatron_utils import take_load_times

            _lt = take_load_times()
        except ImportError:
            _lt = {}
        try:
            from verl.workers.engine.megatron.delta_export import take_export_times

            _xt = take_export_times()
        except ImportError:
            _xt = {}
        is_r0 = self.is_master
        n_flushes = 0
        changed_elems = 0
        total_elems = 0
        wire_bytes = 0

        def _publish_steady(flush, is_last: bool) -> None:
            nonlocal n_flushes
            _t = time.perf_counter()
            self._publish_flush(flush, first=False, is_last=is_last)
            ship_t["publish"] += time.perf_counter() - _t
            n_flushes += 1

        bkt = _FlushBucket(self.bucket_size, self._assemble_flush, _publish_steady)
        # gather / consume(rank0 encode) / publish are the three costs inside
        # ship_s. publish is nested inside consume (bkt.add triggers it), so the
        # encode share is consume minus publish.
        ship_t = {"gather": 0.0, "consume": 0.0, "publish": 0.0, "imbalance": {}}
        pending_hi: list[tuple] = []

        def _drain_hi() -> None:
            """One device read for a batch of range checks instead of one each."""
            if not pending_hi:
                return
            highs = torch.stack([h for *_, h in pending_hi]).tolist()
            for (nm, shp, numel, _), hi in zip(pending_hi, highs, strict=True):
                assert hi < numel, (
                    f"{nm}: delta position {hi} does not address the declared shape "
                    f"{shp} ({numel} elements); positions and shape come from different "
                    f"derivations and have diverged (ratio {hi / max(numel, 1):.2f})"
                )
            pending_hi.clear()
        ph = self._phase = _Phase()
        stager = _FusionStager()
        nonlocal_missed: list[str] = []
        # Names that actually made it onto the wire this sync. The verify sweep
        # reports 39-55 mismatched .weight tensors with a DIFFERENT set per rank,
        # and the leading hypothesis is that a delta for those shards was never
        # sent at all -- replica dedup (contributes) excluding a rank that owns
        # data would look exactly like this. Comparing this set against the
        # sweep's mismatch list settles it: if the mismatched params are absent
        # here, they were dropped on the sender, not corrupted on the wire.
        sent_names = set() if os.environ.get("VERL_DELTA_SENT_DUMP") else None
        # Sampled encoded wire pieces, when the sender-side self-check is on.
        # None (the default) makes the hook in _bucket_slot_delta a single
        # identity test per piece, so an off self-check costs nothing measurable.
        selfcheck_dir = _selfcheck_on()
        selfcheck_pieces = {} if selfcheck_dir else None

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
            _t_consume = time.perf_counter()
            total_elems += int(full_numel)
            # Assert the sender's own invariant: positions must address the shape
            # this entry declares. They are two different derivations -- the bf16
            # path translates shard-local indices to within-param coordinates
            # (_hf_entry_identity) and takes shape from spec.full_shape, while the
            # quant path takes both from _quant_group_meta -- so nothing forces
            # them to agree. When they disagree the receiver index_copy_s out of
            # bounds, which is a device-side fault 11 nodes away that names
            # nothing. Fail here instead, where the offending name and both
            # numbers are in hand. One max() per entry against a multi-GB wire.
            # Sort, unconditionally. The old code first asked "is this block
            # already ordered?" and skipped the sort if so -- but the measurement
            # says 28875 of 28875 parameters were unordered, because the gather
            # concatenates per-rank blocks and every seam steps backwards. The
            # branch was never taken, and asking cost a device->host read inside
            # a per-parameter loop: 28875 pipeline stalls to answer a question
            # whose answer is always yes.
            #
            # No dedup either: duplicates now encode as gap 0 (see _gap_encode),
            # so the boolean mask -- whose data-dependent size was a second
            # per-parameter sync -- is gone as well.
            if aidx is not None and aidx.numel() > 1:
                # Spanned because it was the biggest term in ship and nothing was
                # measuring it. encode_s is a SUBTRACTION (consume - publish), not
                # a phase, so this sort hid inside it: the sub-spans left +4.72 s
                # unattributed in one step and -0.00 in the next, which reads as
                # broken instrumentation. It is not -- p_bucket contains the
                # mid-loop publishes that seal() triggers, while encode_s
                # subtracts ALL publishes including the terminal flush. Add the
                # publish total back and both steps agree at 9.8-10.6 s.
                # Split on purpose: the two halves point at different fixes. The
                # input is ~3.75 already-ascending runs (one per contributing
                # rank), concatenated -- so if the ARGSORT dominates, a k-way
                # merge of the runs recovers most of it. If the PERMUTE
                # dominates, no smarter sort helps, because the data movement is
                # the cost, and the only way out is not needing a global order at
                # all (gap-encode per run). Optimising the wrong half is the
                # default outcome of measuring them together.
                with ph.span("p_sort", sync=True, hot=True):
                    order = torch.argsort(aidx, stable=True)
                with ph.span("p_permute", sync=True, hot=True):
                    aidx, aval = aidx[order], aval[order]
            # The range check moves off the per-parameter path: collect the max
            # position as a device scalar and read a whole batch of them at once.
            # Same guard, 300x fewer stalls.
            if aidx is not None and aidx.numel():
                pending_hi.append((name, tuple(full_shape), int(full_numel), aidx[-1]))
                if len(pending_hi) >= 256:
                    _drain_hi()
            # Members of a fused destination param are held back until the group
            # is whole, then emitted as one indivisible run of pieces. Everything
            # else falls through unchanged. Note the accounting below runs on the
            # RELEASED entries, so an empty half contributes 0 changed elements
            # and 0 wire bytes -- it costs one entry, not one tensor.
            ph.bump("p_params")
            with ph.span("p_stage"):
                offered = stager.offer(name, dtype_str, full_shape, aidx, aval)
            if offered is None:
                ship_t["consume"] += time.perf_counter() - _t_consume
                return
            released, is_group = offered
            # Same diagnostic as get_named_tensor_buckets: a name the DSv4 loader
            # will try to fuse but that our table does not match makes the grouping
            # silently do nothing, and the failure then surfaces far away as
            # "cache_wqkv_a_weight not empty" with no hint which name was missed.
            # The quantized steady path keys entries as "{megatron_name}::{c|s}",
            # so this is exactly where a naming surprise is likely.
            if not is_group and any(t in name for t in (".wq_a.", ".wkv.", ".wgate.")):
                nonlocal_missed.append(name)
                if len(nonlocal_missed) <= 8:
                    logger.warning("delta fusion table MISSED param %r (bucketed ungrouped)", name)
            sized: list[tuple] = []
            for e_name, e_dtype, e_shape, e_idx, e_val in released:
                if e_idx is None or (e_idx.numel() == 0 and not is_group):
                    continue  # unchanged and not fused -- drop it, as before
                changed_elems += int(e_idx.numel())
                if sent_names is not None:
                    sent_names.add(e_name)
                # payload_mbytes reads the per-piece byte counts _slice_pieces
                # already computed, rather than re-encoding to ask the width.
                # Hardcoding 4 here made the metric report 5 B/element after the
                # switch to 1-byte gaps -- 96.2 GiB where the wire carried 38.5,
                # which is exactly the figure this work is judged on. The flush
                # count is the cross-check: 78 flushes only makes sense at ~2 B.
                with ph.span("p_slice_encode", sync=True, hot=True):
                    pieces = _slice_pieces(e_name, e_dtype, e_shape, e_idx, e_val)
                ph.bump("p_pieces", len(pieces))
                for _piece, _ in pieces:
                    _selfcheck_record_piece(selfcheck_pieces, _piece)
                wire_bytes += sum(nb for _, nb in pieces)
                sized.extend(pieces)
            with ph.span("p_bucket"):
                if is_group:
                    bkt.add_atomic(sized)
                else:
                    for piece, nbytes in sized:
                        bkt.add(piece, nbytes)
            ship_t["consume"] += time.perf_counter() - _t_consume

        gq = _GatherQueue(
            batch_k, self.gather_round_bytes or self.bucket_size, is_r0, _bucket_slot_delta, timers=ship_t
        )

        # ``weights`` is the BACKEND's HF delta stream (hf_delta_export): entries
        # already carry final HF coordinates -- naming, conversion, diff and
        # snapshot all happened on the backend side. This engine only batches,
        # gathers and ships.
        # Split the wall clock where the two costs actually live, because
        # update_weights is currently one number and every explanation for it has
        # been a guess. Pulling the generator runs the BACKEND's work -- quantize
        # the whole model, diff every parameter against the snapshot, refresh it
        # -- which is O(total params) and indifferent to how much changed.
        # Everything else (gather, sort, encode, bucket, broadcast) scales with
        # the delta. Measured: at 12-17% density time tracked payload; at 7-8% it
        # stopped tracking payload entirely, so one of these two terms grew and
        # the folded number cannot say which.
        t_export = t_ship = 0.0
        _it = iter(weights)
        while True:
            _t = time.perf_counter()
            try:
                item = next(_it)
            except StopIteration:
                t_export += time.perf_counter() - _t
                break
            t_export += time.perf_counter() - _t
            _t = time.perf_counter()
            gq.put(item[5], item[0], item[1], item[2], item[3], item[4])
            t_ship += time.perf_counter() - _t
        _t = time.perf_counter()
        gq.flush_all()
        t_ship += time.perf_counter() - _t
        _drain_hi()
        # A half still parked here means its sibling never came through the export
        # stream, so the receiver would have been handed an unpairable member and
        # died inside sglang's loader with a far less informative message.
        stager.assert_drained()
        if nonlocal_missed:
            logger.warning(
                "delta fusion table MISSED %d param(s) the DSv4 loader will try to fuse; first 8: %s",
                len(nonlocal_missed),
                nonlocal_missed[:8],
            )
        # Log unconditionally, including the zero case: if the export ever renames
        # these params, every suffix stops matching and the staging silently
        # degrades to a no-op -- which looks exactly like "no fused params in this
        # model". The count is the only thing that tells the two apart. DSv4 should
        # report 4 groups x 43 layers.
        if is_r0:
            logger.info(
                "delta fusion staging: groups=%d nan_filled_halves=%d",
                stager.n_groups,
                stager.n_filled,
            )

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
            if self.quantize_fp8:
                full, _ = engine.get_per_tensor_param(raw_master=True, quant_spec=self._fp8_spec(engine))
                self._send_full_seed(_verify_sample(full), global_steps, verify=True, bytes_wire=True)
            else:
                full, _ = engine.get_per_tensor_param()
                self._send_full_seed(_verify_sample(full), global_steps, verify=True)
        # The last flush is still on the wire at this point: nothing after the
        # loop waits for it, so drain it before reporting or returning.
        self._await_publish_inflight(ph)
        # Sender-side self-check, off by default. Placed here on purpose: the
        # dense re-export is collective, so it has to run on every rank BEFORE
        # the rank-0 early return, and it has to run after the delta was
        # produced so the dense state it records is the one this sync's delta
        # is supposed to yield from the previous sync's.
        if selfcheck_dir:
            _selfcheck_write(
                engine,
                lambda: self._fp8_spec(engine) if self.quantize_fp8 else None,
                self.quantize_fp8,
                global_steps,
                selfcheck_pieces,
                selfcheck_dir,
                is_r0,
            )
        if not is_r0:
            return
        _t = time.perf_counter()
        self._release_staging_pool("steady")  # return staging blocks to CUDA between syncs
        t_release = time.perf_counter() - _t
        prof = getattr(engine, "_quant_profile", None)
        if prof is not None:
            logger.warning(
                "AMAX-PROFILE scale_flips=%d/%d blocks code_changed=%d/%d bytes",
                prof["sf"],
                prof["st"],
                prof["cc"],
                prof["ct"],
            )
            engine._quant_profile = None
        logger.info("delta-sharded send v=%s delta flushes=%d (streamed)", global_steps, n_flushes)
        if not total_elems:
            return None
        _imb = ship_t.get("imbalance", {})
        # Dump the raw per-round x per-rank matrix when asked. 218 rounds x 16
        # ranks is a few thousand integers -- small enough to keep whole, and an
        # aggregate cannot answer "is it always the same rank".
        _sd = os.environ.get("VERL_DELTA_SENT_DUMP")
        if _sd and sent_names is not None:
            try:
                os.makedirs(os.path.dirname(_sd) or ".", exist_ok=True)
                with open(_sd, "a") as fh:
                    fh.write(json.dumps({"step": global_steps, "sent": sorted(sent_names)}) + "\n")
            except OSError as e:
                logger.warning("could not write the sent-name dump: %s", e)
        _dump = os.environ.get("VERL_DELTA_GATHER_DUMP")
        if _dump and is_r0 and _imb.get("rows"):
            try:
                with open(_dump, "a") as fh:
                    fh.write(json.dumps({"step": global_steps, "rows": _imb["rows"]}) + "\n")
            except OSError as e:
                logger.warning("could not write gather dump to %s: %s", _dump, e)
        return {
            "checkpoint_engine/changed_ratio": changed_elems / total_elems,
            "checkpoint_engine/changed_elems": float(changed_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(n_flushes),
            # export = backend quantize + diff, O(total params), payload-blind.
            # ship  = gather + sort + encode + bucket + broadcast, scales with the delta.
            "checkpoint_engine/export_s": t_export,
            "checkpoint_engine/ship_s": t_ship,
            # total_s - (export_s + ship_s + these) is what remains unexplained
            # ON THIS RANK; total_s vs timing_s/update_weights is what remains
            # outside this function. Keep both, they answer different questions.
            "checkpoint_engine/total_s": time.perf_counter() - t_total0,
            "checkpoint_engine/t_spec_s": t_spec,
            "checkpoint_engine/t_delta_open_s": t_delta_open,
            # NOT additive with each other: copy_enqueue is enqueue cost for a
            # non_blocking copy, not transfer time.
            # export internals -- the biggest single term and, until now, opaque.
            **{
                f"checkpoint_engine/t_export_{k}_s": v
                for k, v in _xt.items()
                if k != "records"
            },
            "checkpoint_engine/n_export_records": float(_xt.get("records", 0)),
            "checkpoint_engine/t_load_resize_s": _lt.get("resize", 0.0),
            "checkpoint_engine/t_load_copy_enqueue_s": _lt.get("copy_enqueue", 0.0),
            "checkpoint_engine/n_load_buffers": float(_lt.get("buffers", 0)),
            "checkpoint_engine/load_gbytes": _lt.get("bytes", 0) / (1 << 30),
            "checkpoint_engine/t_release_pool_s": t_release,
            "checkpoint_engine/gather_s": ship_t["gather"],
            # NOT a phase: consume minus publish, where publish includes the
            # terminal flush that happens outside the consume region. Kept for
            # continuity with earlier runs; prefer the explicit spans below,
            # which now sum to the real thing.
            "checkpoint_engine/encode_s": ship_t["consume"] - ship_t["publish"],
            "checkpoint_engine/publish_s": ship_t["publish"],
            "checkpoint_engine/n_gather_rounds": float(ship_t.get("rounds", 0)),
            # Measured, not inferred: what each rank actually contributed.
            # waste = padded / sum; 1.0 means the ranks were balanced and padding
            # cost nothing, world means one rank held everything.
            "checkpoint_engine/gath_elems_useful": float(_imb.get("sum", 0)),
            "checkpoint_engine/gath_elems_padded": float(_imb.get("padded", 0)),
            "checkpoint_engine/gath_waste_x": (_imb.get("padded", 0) / max(_imb.get("sum", 0), 1)),
            "checkpoint_engine/gath_ranks_with_data_avg": (
                _imb.get("nonzero_ranks", 0) / max(_imb.get("rounds", 0), 1)
            ),
            **ph.metrics(),
        }
