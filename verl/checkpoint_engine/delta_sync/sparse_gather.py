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
"""Sharded delta: diff each rank's local FSDP shard, gather only the changes to rank 0.

The default delta path all-gathers the full parameter (``DTensor.full_tensor()``) and
byte-diffs it against a full-model pinned-CPU snapshot on rank 0. This module instead lets
each rank keep a pinned snapshot of only *its* shard, byte-diff the shard locally, and
gather just the changed ``(within-parameter position, value)`` pairs to rank 0 -- so the
all-gather volume drops to the sparsity ratio (~1-3%) and rank 0 no longer needs a
full-model snapshot. The gathered result is bit-identical to the full-tensor diff, so the
downstream encode + broadcast and the receiver are unchanged.

Scope: FSDP2 ``Shard(0)`` DTensors (the common case) + replicated / non-DTensor params.
Other shard dims are strided in the flattened layout and raise NotImplementedError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import os

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    pass

_DTYPE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def shard_delta_indices(
    local_new: torch.Tensor,
    local_snap: torch.Tensor,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte-diff a local shard against its snapshot; return (global_positions, values).

    Positions are int64 indices into the *full flattened parameter* (offset + local index).
    Dtype-agnostic, bytewise (view-as-int), no arithmetic -- matches ``bytewise_diff_mask``.
    """
    es = local_new.element_size()
    int_dtype = _DTYPE_INT.get(es)
    if int_dtype is None:
        raise ValueError(f"unsupported element size {es}")
    mask = local_new.view(int_dtype) != local_snap.view(int_dtype)
    local_idx = mask.nonzero(as_tuple=False).view(-1)
    values = local_new[local_idx]
    global_idx = local_idx.to(torch.int64) + offset
    return global_idx, values


def _gather_p2p(idx_concat, val_concat, totals, world, rank, dst, group, dev):
    """Variable-length gather to rank 0: each rank sends exactly what it holds.

    ``totals[r]`` comes from the counts matrix every rank already all-gathered,
    so senders and receivers agree on every length without another collective.
    Ranks with nothing to contribute post no ops at all -- the padded path had
    them send a full max_n of zeros.

    Returns ``(idx_list, val_list)`` on rank 0 (rank r's slot holds exactly
    ``totals[r]`` elements, so the caller's per-rank offset arithmetic is
    unchanged), and ``(None, None)`` elsewhere.
    """
    if rank == 0:
        idx_list = [
            idx_concat
            if r == 0
            else torch.empty(totals[r], dtype=idx_concat.dtype, device=dev)
            for r in range(world)
        ]
        val_list = [
            val_concat
            if r == 0
            else torch.empty(totals[r], dtype=val_concat.dtype, device=dev)
            for r in range(world)
        ]
        ops = []
        for r in range(1, world):
            if not totals[r]:
                continue
            peer = dist.get_global_rank(group, r) if group is not None else r
            ops.append(dist.P2POp(dist.irecv, idx_list[r], peer, group))
            ops.append(dist.P2POp(dist.irecv, val_list[r], peer, group))
        for w in dist.batch_isend_irecv(ops) if ops else []:
            w.wait()
        return idx_list, val_list

    if totals[rank]:
        ops = [
            dist.P2POp(dist.isend, idx_concat.contiguous(), dst, group),
            dist.P2POp(dist.isend, val_concat.contiguous(), dst, group),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
    return None, None


def gather_slot_entries_to_rank0(
    idx_concat: torch.Tensor,
    val_concat: torch.Tensor,
    counts: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    max_round_bytes: int | None = None,
    stats: dict | None = None,
) -> list | None:
    """Variable-length sparse gather, batched: one collective round for K parameters.

    Each rank passes its K per-parameter deltas concatenated (``idx_concat``,
    ``val_concat``) plus the per-parameter length vector ``counts`` ([K] int64).
    One all_gather exchanges the K x world count matrix; two padded gathers move
    the blobs. Rank 0 slices per (rank, param) and returns K ``(idx, val)`` pairs
    (None elsewhere) -- bit-identical to K individual gathers, ~K x fewer
    collectives and host syncs.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    dst = dist.get_global_rank(group, 0) if group is not None else 0
    dev = idx_concat.device
    k = int(counts.numel())

    counts_all = [torch.zeros_like(counts) for _ in range(world)]
    dist.all_gather(counts_all, counts.to(dev), group=group)
    counts_cpu = torch.stack(counts_all).cpu().tolist()  # one D2H sync instead of `world`

    if max_round_bytes is not None and k > 1:
        # Deterministic sub-rounds: every rank sees the same counts matrix, so all
        # ranks derive the SAME slot partition (no per-rank trigger asymmetry) and
        # each padded round's largest blob stays within the byte budget.
        per_elem = idx_concat.element_size() + val_concat.element_size()
        budget = max(int(max_round_bytes) // per_elem, 1)  # elements per rank per round
        cuts = [0]
        run = [0] * world
        for i in range(k):
            run = [run[r] + counts_cpu[r][i] for r in range(world)]
            if max(run) > budget and cuts[-1] != i:
                cuts.append(i)
                run = [counts_cpu[r][i] for r in range(world)]
        cuts.append(k)
        if len(cuts) > 2:
            out_all: list = []
            my_off = [0]
            for i in range(k):
                my_off.append(my_off[-1] + counts_cpu[rank][i])
            for lo, hi in zip(cuts[:-1], cuts[1:], strict=False):
                sub_counts = torch.tensor(counts_cpu[rank][lo:hi], dtype=torch.int64, device=dev)
                sub = gather_slot_entries_to_rank0(
                    idx_concat[my_off[lo] : my_off[hi]],
                    val_concat[my_off[lo] : my_off[hi]],
                    sub_counts,
                    group=group,
                    stats=stats,
                )
                if rank == 0:
                    out_all.extend(sub)
            return out_all if rank == 0 else None

    totals = [sum(c) for c in counts_cpu]
    max_n = max(totals) if totals else 0
    # Record the imbalance directly instead of inferring it from timing. The
    # padded gather's waste factor is max_n / mean(totals), so these are the
    # numbers that decide whether padding costs anything at all -- balanced
    # ranks pay nothing. counts_cpu is already here; this only reads it.
    if stats is not None and totals:
        stats["sum"] = stats.get("sum", 0) + sum(totals)
        stats["max"] = stats.get("max", 0) + max_n
        stats["padded"] = stats.get("padded", 0) + max_n * world
        stats["nonzero_ranks"] = stats.get("nonzero_ranks", 0) + sum(1 for x in totals if x)
        stats["rounds"] = stats.get("rounds", 0) + 1
        # Keep the raw per-round vector too, not just the aggregate. An average
        # cannot show whether one rank always carries the round or whether the
        # heavy rank moves around, and those want different fixes.
        stats.setdefault("rows", []).append(list(totals))
    if max_n == 0:
        if rank != 0:
            return None
        empty_i = torch.empty(0, dtype=idx_concat.dtype, device=dev)
        empty_v = torch.empty(0, dtype=val_concat.dtype, device=dev)
        return [(empty_i, empty_v) for _ in range(k)]

    # dist.gather requires every rank to send the SAME length, so the padded path
    # below sends max_n from each rank regardless of what it actually holds. The
    # profile showed what that costs: 42.9 GiB of real delta moved as ~686 GiB
    # (world=16) and took 82.7 s, while the same bytes broadcast in 5.4 s. Rank 0
    # also has to allocate 2 x world x max_n, which is what caps the round size at
    # ~200 MB and made "just use bigger rounds" OOM at 1 GiB.
    #
    # counts_cpu is already all-gathered, so both sides know every exact length:
    # point-to-point moves precisely the real bytes. Off by default because it
    # replaces a collective with matched sends/receives, and a mismatch there
    # deadlocks rather than errors -- the counts matrix is what makes the match
    # safe, and it is the same object the padded path already trusts.
    if os.environ.get("VERL_DELTA_GATHER_P2P") == "1":
        idx_list, val_list = _gather_p2p(idx_concat, val_concat, totals, world, rank, dst, group, dev)
        if rank != 0:
            return None
    else:
        idx_pad = torch.zeros(max_n, dtype=idx_concat.dtype, device=dev)
        val_pad = torch.zeros(max_n, dtype=val_concat.dtype, device=dev)
        n = int(idx_concat.numel())
        idx_pad[:n] = idx_concat
        val_pad[:n] = val_concat

        idx_list = [torch.zeros(max_n, dtype=idx_pad.dtype, device=dev) for _ in range(world)] if rank == 0 else None
        val_list = [torch.zeros(max_n, dtype=val_concat.dtype, device=dev) for _ in range(world)] if rank == 0 else None
        dist.gather(idx_pad, idx_list, dst=dst, group=group)
        dist.gather(val_pad, val_list, dst=dst, group=group)
        if rank != 0:
            return None

    # per-rank cumulative offsets into each blob, sliced per param then stitched across ranks
    offs = [[0] * (k + 1) for _ in range(world)]
    for r in range(world):
        for i in range(k):
            offs[r][i + 1] = offs[r][i] + counts_cpu[r][i]
    out = []
    for i in range(k):
        idx_pieces = [idx_list[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts_cpu[r][i]]
        val_pieces = [val_list[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts_cpu[r][i]]
        if idx_pieces:
            out.append((torch.cat(idx_pieces), torch.cat(val_pieces)))
        else:
            out.append(
                (torch.empty(0, dtype=idx_concat.dtype, device=dev), torch.empty(0, dtype=val_concat.dtype, device=dev))
            )
    return out



def dense_gather_group(
    flat: torch.Tensor,
    sizes_local: list[int],
    group: dist.ProcessGroup | None = None,
) -> list[torch.Tensor] | None:
    """Values-only gather of one group record for the seed-as-steady transport.

    Each rank passes its group flat (its owned slots' full pieces concatenated,
    zero-length pieces for slots owned elsewhere; non-contributing replicas
    pass all-zero ``sizes_local``) plus the per-slot length vector. Rank 0 of
    ``group`` returns the per-slot full tensors, stitched by ownership; None
    elsewhere.

    No index tensors exist at any point (the sparse path's ~9 B/element staging
    is what OOMed the 100%-coverage work point), and the transfer is SEQUENTIAL
    point-to-point per contributing rank -- peak extra memory on rank 0 is one
    sender's flat, independent of world size (a padded dist.gather would
    allocate world x max_n there, which is the same wall wearing values'
    clothes). The [world, n_slots] size matrix is all-gathered first and is the
    trust base that keeps the send/recv pairing deadlock-free -- the same
    contract _gather_p2p already relies on.
    """
    k = len(sizes_local)
    if group is None and not (dist.is_available() and dist.is_initialized()):
        # unsharded / single process: this rank's pieces are the record
        out, off = [], 0
        for n in sizes_local:
            out.append(flat[off : off + n])
            off += n
        return out
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    dst = dist.get_global_rank(group, 0) if group is not None else 0
    dev = flat.device
    sizes = torch.tensor(sizes_local, dtype=torch.int64, device=dev)
    sizes_all = torch.zeros(world, k, dtype=torch.int64, device=dev)
    # equal [k]-sized chunk per rank into the flat [world*k] output
    dist.all_gather_into_tensor(sizes_all.view(-1), sizes, group=group)
    sizes_cpu = sizes_all.cpu().tolist()
    my_total = int(sizes.sum())
    if rank != 0:
        if my_total:
            dist.send(flat[:my_total].contiguous(), dst=dst, group=group)
        return None
    # rank 0: receive each contributor's flat sequentially, stitch per slot
    per_rank_flat: dict[int, torch.Tensor] = {}
    if my_total:
        per_rank_flat[0] = flat[:my_total]
    for r in range(1, world):
        n = sum(sizes_cpu[r])
        if not n:
            continue
        buf = torch.empty(n, dtype=flat.dtype, device=dev)
        src_rank = dist.get_global_rank(group, r) if group is not None else r
        dist.recv(buf, src=src_rank, group=group)
        per_rank_flat[r] = buf
    out: list[torch.Tensor] = []
    for i in range(k):
        owners = [r for r in range(world) if sizes_cpu[r][i]]
        assert len(owners) == 1, (
            f"slot {i}: {len(owners)} contributors (expected exactly 1) -- ownership map and "
            "contributes flags disagree; refusing to ship an ambiguous seed"
        )
        r = owners[0]
        off = sum(sizes_cpu[r][:i])
        out.append(per_rank_flat[r][off : off + sizes_cpu[r][i]])
    return out
