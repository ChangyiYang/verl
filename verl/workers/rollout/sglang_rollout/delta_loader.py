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
"""In-process sparse delta apply for SGLang, loaded via ``--custom-weight-loader``.

SGLang's ``update_weights_from_tensor`` supports pluggable loaders: when the
request's ``load_format`` names an import path registered in
``--custom-weight-loader``, SGLang ``dynamic_import``s it **inside every TP
worker process** and calls ``loader(model, named_tensors)``. That is exactly
the hook a sparse delta needs: the delta payload is decoded and applied *in
place* onto SGLang's live weights (masked overwrite of only the changed
positions), so the receiver never stages a full-model mirror anywhere —
peak memory is one decode chunk, independent of model size — and no SGLang
fork or patch is required.

Wire contract (what the delta checkpoint engine sends per flush):

* ``__delta_spec__`` — uint8 tensor holding a JSON manifest
  ``{"encoding", "params": [DeltaParam-dict...], "checksum"}``.
* ``__positions__`` — uint8 blob of packed positions (per-param slices are
  byte offsets ``pos_start:pos_end``; ``indices`` packs little-endian int32
  absolute positions).
* ``__values__``  — the changed values in the flush's (uniform) dtype;
  per-param slices are element offsets ``val_start:val_end``.

Register at server launch (verl config)::

    +actor_rollout_ref.rollout.engine_kwargs.sglang.custom_weight_loader='["verl.workers.rollout.sglang_rollout.delta_loader.apply_delta"]'
"""

from __future__ import annotations

import bisect
import itertools
import json
import contextlib
import logging
import os
import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

# Cap on the densified tensors handed to one model.load_weights call, matching
# SGLang's own delta-apply chunking default.
CHUNK_BYTES = 512 << 20

# The membership table lives in verl.utils.fusion_groups so the three split
# points (this chunk loop, the delta engine's two streams, and
# get_named_tensor_buckets) cannot drift apart. This chunk loop is the LAST
# place that can split a group: the sender keeps one inside a flush, but a flush
# is re-cut here by CHUNK_BYTES.
from verl.utils.fusion_groups import fusion_key as _fusion_key  # noqa: E402


def _atomic_units(params: list[dict]) -> list[list[dict]]:
    """Partition params so that members of one fusion group stay in one unit.

    Order follows first appearance, so a stream that already keeps a group
    adjacent is left untouched; a group split apart by the sender still gets
    reunited here.
    """
    units: list[list[dict]] = []
    at: dict = {}
    for p in params:
        key = _fusion_key(p["name"])
        if key is not None and key in at:
            units[at[key]].append(p)
            continue
        if key is not None:
            at[key] = len(units)
        units.append([p])
    return units


def _load_in_chunks(model, params: list[dict], materialize) -> None:
    """Feed params to ``model.load_weights`` in <= CHUNK_BYTES chunks, cutting
    only BETWEEN fusion groups. ``materialize(p) -> Tensor`` builds one param."""
    chunk: list[tuple[str, torch.Tensor]] = []
    chunk_bytes = 0
    for unit in _atomic_units(params):
        built = [(p["name"], materialize(p)) for p in unit]
        unit_bytes = sum(t.numel() * t.element_size() for _, t in built)
        if chunk and chunk_bytes + unit_bytes > CHUNK_BYTES:
            model.load_weights(chunk)
            chunk, chunk_bytes = [], 0
        chunk.extend(built)
        chunk_bytes += unit_bytes
    if chunk:
        model.load_weights(chunk)

# Import path callers pass as both --custom-weight-loader and load_format.
LOADER_FQN = "verl.workers.rollout.sglang_rollout.delta_loader.apply_delta"


def _find_live_quant_config(model: torch.nn.Module):
    """Read the live quantization config straight from a quantized sglang layer.

    Reads ``module.quant_method.quant_config`` (NOT ``module.quant_config``:
    layers excluded via ignored_layers keep the config object but get an
    UnquantizedLinearMethod, so the quant_method is the authority on whether
    the layer is actually quantized)."""
    for name, module in model.named_modules():
        qm = getattr(module, "quant_method", None)
        qc = getattr(qm, "quant_config", None)
        if qc is None:
            continue
        block = getattr(qc, "weight_block_size", None)
        if block is None or not hasattr(module, "weight_scale_inv"):
            continue
        return {
            "layer_name": name,
            "quant_method": type(qc).__name__,
            "weight_block_size": [int(x) for x in block],
            "use_mxfp8": bool(getattr(qc, "use_mxfp8", False)),
            "module": module,
        }
    return None


def _check_quant_handshake(model: torch.nn.Module, spec: dict) -> None:
    """Two separate questions, answered loudly at the first flush:

    1. config handshake -- do trainer and rollout use the same quantization
       definition? Compared against the LIVE layer config
       (``quant_method.quant_config``), falling back to shape inference only
       when no live config is discoverable.
    2. state sanity -- do the created params match that definition (scale grid
       shape = ceil(weight shape / block)), and has the state been seeded
       (sparse deltas on sentinel scales are refused)?
    """
    cfg = spec.get("quant_config")
    if cfg is None or getattr(model, "_delta_quant_handshake_done", False):
        return
    # seed-required guard: sglang boots serialized-fp8 params with SENTINEL
    # scales (never loaded from a bf16 ckpt) -- serving or sparse-patching that
    # state would silently produce garbage. The first quantized payload must be
    # the full seed; a sparse delta arriving on sentinel scales fails loud.
    if spec.get("encoding") != "dense":
        sentinel = torch.finfo(torch.float32).min
        for name, param in model.named_parameters():
            if not name.endswith("weight_scale_inv") or not param.numel():
                continue
            assert float(param.data.flatten()[0]) != sentinel, (
                f"sparse fp8 delta arrived but {name} still holds the unloaded sentinel scale "
                "-- the rollout was never seeded (full seed sync must precede any steady delta)"
            )
            break
    want = cfg.get("weight_block_size")
    if want is not None:
        want = [int(x) for x in want]
        live = _find_live_quant_config(model)
        if live is not None:
            assert not live["use_mxfp8"], (
                f"quant handshake failed: rollout layer {live['layer_name']} runs mxfp8 "
                f"({live['quant_method']}), trainer ships plain blockwise fp8 ({cfg})"
            )
            assert live["weight_block_size"] == want, (
                f"quant handshake failed: rollout {live['quant_method']} block size "
                f"{live['weight_block_size']} vs trainer config {want}"
            )
            # state sanity: created params match the agreed definition
            module = live["module"]
            w = getattr(module, "weight", None)
            si = getattr(module, "weight_scale_inv", None)
            if w is not None and si is not None and w.dim() == 2 and si.dim() == 2:
                expect = [
                    (w.shape[0] + want[0] - 1) // want[0],
                    (w.shape[1] + want[1] - 1) // want[1],
                ]
                assert list(si.shape) == expect, (
                    f"state sanity failed on {live['layer_name']}: scale grid {list(si.shape)} "
                    f"vs expected {expect} for block {want}"
                )
        else:
            # no discoverable live config: fall back to shape inference
            logger.warning("quant handshake: no live quant_config found on any module; falling back to shape check")
            for name, param in model.named_parameters():
                if not name.endswith("weight_scale_inv") or param.dim() != 2:
                    continue
                base = dict(model.named_parameters()).get(name[: -len("_scale_inv")])
                if base is None or base.dim() != 2:
                    continue
                bm = (base.shape[0] + param.shape[0] - 1) // param.shape[0]
                bn = (base.shape[1] + param.shape[1] - 1) // param.shape[1]
                assert [bm, bn] == want, (
                    f"quant handshake failed on {name}: live block size ~[{bm}, {bn}] "
                    f"vs trainer config {want} (spec quant_config={cfg})"
                )
                break
    model._delta_quant_handshake_done = True


def apply_delta(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Decode one sparse delta flush and masked-apply it onto ``model`` in place."""
    from verl.checkpoint_engine.delta_sync.encode import checksum as _checksum

    tensors = dict(named_tensors)
    spec = json.loads(bytes(tensors["__delta_spec__"].cpu().numpy().tobytes()).decode())
    values = tensors["__values__"]
    positions = tensors.get("__positions__")
    if positions is None:  # values-only flush (the seed) carries no positions
        positions = torch.empty(0, dtype=torch.uint8, device=values.device)

    got = _checksum(positions, values)
    if got != int(spec["checksum"]):
        raise RuntimeError(
            f"delta checksum mismatch in sglang loader: got {got}, expected {spec['checksum']}; "
            "indicates corruption between sender encode and receiver apply"
        )

    _check_quant_handshake(model, spec)
    if spec["encoding"] == "dense":
        if spec.get("verify"):
            _verify_dense(model, spec["params"], values, bool(spec.get("is_last")), bool(spec.get("values_bytes")))
            return
        _apply_dense(model, spec["params"], values, bool(spec.get("values_bytes")))
        if spec.get("is_last") and hasattr(model, "post_load_weights"):
            model.post_load_weights()
        return

    encoding = spec["encoding"]
    values_bytes = bool(spec.get("values_bytes"))
    with _masked_copy(_param_storage_index(model)):
        _load_in_chunks(
            model,
            spec["params"],
            lambda p: _decode_one(encoding, positions, values, p, values_bytes),
        )

    # Derived tensors (fp8 scales after requant, MLA w_kc/w_vc, MoE biases)
    # recompute from the now-final weights. Outside the masked-copy patch on
    # purpose: their writes are wholesale transforms, not sparse overlays --
    # sglang's own update_weights_from_tensor path does not trigger this hook,
    # so the delta loader replicates the full-load semantics itself once per
    # sync (the engine marks the sync's final flush with ``is_last``).
    if spec.get("is_last") and hasattr(model, "post_load_weights"):
        model.post_load_weights()


def _apply_dense(
    model: torch.nn.Module, params: list[dict], values: torch.Tensor, values_bytes: bool = False
) -> None:
    """Apply a dense (full-coverage) flush: plain chunked load, no masking needed.

    ``values_bytes`` marks a mixed-dtype flush (fp8 codes + fp32 scales + bf16
    leftovers): offsets are BYTE offsets into a uint8 blob and each param is
    reinterpreted, not cast. The ``.clone()`` re-bases the slice to storage
    offset 0 so the dtype view never trips torch's alignment check.

    This is the path the SEED takes, and it is where the fused-param assert
    actually fired -- hence the same atomic chunking as the sparse path."""

    def _materialize(p: dict) -> torch.Tensor:
        dtype = getattr(torch, p["dtype"])
        if values_bytes:
            return values[p["val_start"] : p["val_end"]].clone().view(dtype).view(p["shape"])
        return values[p["val_start"] : p["val_end"]].to(dtype).view(p["shape"])

    _load_in_chunks(model, params, _materialize)


_VERIFY_STATS: dict = {"params": 0, "pieces": []}


class _DedupUninitWarning(logging.Filter):
    """Drop repeats of SGLang's "weights are not initialized" warning.

    That single record is 65 KB (it inlines the whole name set) and SGLang emits
    one per ``load_weights`` call. The seed calls load_weights once per flush --
    533 times -- so it costs ~35 MB and nobody noticed. The verify sweep calls it
    once per ATOMIC UNIT, ~24k-32k times, because it has to snapshot and compare
    around each one: ~2 GB of log text. Ray forwards worker logs to the head node
    with a 512 MB message cap, so the job dies with

        RaySystemError: Sent message larger than max (536878753 vs 536870912)

    which looks nothing like "the verification did something wrong". It is not the
    data, the transfer, or the memory: the seed pushes the same full model through
    the same _apply_dense path every run and is fine. Only the call granularity
    differs, and the logging is quadratic in it.

    The first occurrence is kept -- it is genuinely useful once -- and the rest are
    counted and reported, so nothing is silently lost.
    """

    NEEDLE = "are not initialized from checkpoints"

    def __init__(self):
        super().__init__()
        self.suppressed = 0
        self.seen = False

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if self.NEEDLE not in msg:
            return True
        if not self.seen:
            self.seen = True
            return True
        self.suppressed += 1
        return False


@contextlib.contextmanager
def _quiet_uninit_warning():
    """Install the dedup filter on the root logger for the duration of a sweep."""
    filt = _DedupUninitWarning()
    root = logging.getLogger()
    handlers = list(root.handlers) or []
    for h in handlers:
        h.addFilter(filt)
    root.addFilter(filt)
    try:
        yield filt
    finally:
        for h in handlers:
            h.removeFilter(filt)
        root.removeFilter(filt)
        if filt.suppressed:
            logger.warning(
                "delta verify: suppressed %d repeats of SGLang's 65 KB "
                "'not initialized from checkpoints' warning (~%.1f GB of log text). "
                "Kept the first one.",
                filt.suppressed,
                filt.suppressed * 66776 / (1 << 30),
            )


def _model_state_hashes(model) -> dict:
    """One hash per parameter/buffer, with a SINGLE device sync for all of them.

    Hashes rather than clones because a full copy of the server's shard held
    across the sweep is GBs, and an OOM costs a 45-minute run. We learn WHICH
    tensor changed but not how many of its elements did; the identity is the
    actionable half.

    Every tensor is normalised to contiguous 1-D bytes before hashing, because
    two separate runs died here on shape/dtype edges rather than on anything to
    do with the delta:
      * torch.hash_tensor is an xor-reduce with no fp8 kernel
        ("xor_sum_cuda not implemented for Float8_e4m3fn"), and this model is
        mostly fp8 -> bitcast to uint8;
      * a 0-dim scalar cannot be viewed as a different element size
        ("self.dim() cannot be 0 to view Float as Byte") -> reshape(-1) first.
    Byte-level hashing is also what we want semantically: the question is whether
    the BITS changed, and fp8/bf16 payloads contain NaN, where a value compare
    would silently answer "equal".

    Anything that still refuses to hash is RECORDED AND REPORTED, not skipped
    silently and not raised: a third dead run to discover a fourth edge case is a
    bad trade, but quietly dropping tensors would weaken the very check this is.
    """
    import itertools

    names, hs, unhashable = [], [], []
    for name, t in itertools.chain(model.named_parameters(), model.named_buffers()):
        if t is None or not t.numel():
            continue
        try:
            flat = t.detach().contiguous().reshape(-1).view(torch.uint8)
            hs.append(torch.hash_tensor(flat).reshape(()))
            names.append(name)
        except Exception as e:  # noqa: BLE001 - the point is to survive and report
            unhashable.append(f"{name}({type(t).__name__},{t.dtype},{tuple(t.shape)}): {e}")
    if unhashable:
        logger.warning(
            "DELTA-VERIFY: %d tensor(s) could not be hashed and are NOT covered by this "
            "sweep; first: %s",
            len(unhashable),
            unhashable[:5],
        )
    if not hs:
        return {}
    # One sync for all of them: per-tensor .item() would be ~100k device syncs,
    # the same mistake that made this sweep unrunnable in the first place.
    vals = torch.stack(hs).tolist()
    return dict(zip(names, vals, strict=True))


def _post_load(model) -> None:
    if hasattr(model, "post_load_weights"):
        model.post_load_weights()


def _verify_dense(
    model: torch.nn.Module, params: list[dict], values: torch.Tensor, is_last: bool, values_bytes: bool = False
) -> None:
    """State-equivalence sweep, phrased as an IDEMPOTENCE check: replaying the
    trainer's FULL current weights onto an in-sync server must be a no-op.

    The comparison is taken on the state the server actually SERVES with, which
    means after ``post_load_weights``. The previous version compared before it,
    and that was the bug: post_load_weights recomputes derived tensors ("fp8
    scales after requant", MLA w_kc/w_vc, MoE biases), so the sweep was checking
    the trainer's TRANSPORTED scale against the server's RECOMPUTED one. With
    weight_scale_inv and scale being reciprocals a bit-level difference is
    near-certain, and the first successful sweep duly reported 4.34M mismatched
    elements -- 8,481 tensors each fully wrong, every one of them a scale.

    Both sides of the comparison now run post_load_weights, so the check is
    symmetric. Note this makes it STRICTER, not looser: excluding the derived
    tensors would have made the failure disappear without learning anything, and
    that is the one fix this must not be.
    """
    st = _VERIFY_STATS
    if "baseline" not in st:
        # Finish the steady sync's semantics before snapshotting, or the baseline
        # would be a state the server never actually serves from.
        _post_load(model)
        st["baseline"] = _model_state_hashes(model)

    _quiet = _quiet_uninit_warning()
    _quiet.__enter__()
    try:
        for unit in _verify_batches(params):
            _apply_dense(model, unit, values, values_bytes)
    finally:
        _quiet.__exit__(None, None, None)

    st["params"] += len(params)
    if not is_last:
        return

    _post_load(model)
    after = _model_state_hashes(model)
    base = st.pop("baseline", {})
    n = st["params"]
    st["params"] = 0

    changed = [k for k in base if k in after and base[k] != after[k]]
    missing = sorted(set(base) - set(after))
    logger.warning(
        "DELTA-VERIFY sweep: params=%d tensors=%d changed=%d missing=%d first=%s",
        n,
        len(base),
        len(changed),
        len(missing),
        sorted(changed)[:12],
    )
    if changed or missing:
        by_suffix: dict = {}
        for k in changed:
            by_suffix[k.rsplit(".", 1)[-1]] = by_suffix.get(k.rsplit(".", 1)[-1], 0) + 1
        logger.warning(
            "DELTA-VERIFY distribution: by_suffix=%s",
            sorted(by_suffix.items(), key=lambda kv: -kv[1])[:15],
        )
        raise RuntimeError(
            f"delta state verification FAILED: {len(changed)} tensors differ between the "
            f"server's delta-accumulated state and the trainer's full export "
            f"(both compared after post_load_weights); first: {sorted(changed)[:12]}"
        )


def _verify_batches(params):
    """Yield the groups of params to verify together.

    Per ATOMIC UNIT by default: a unit is the smallest safe granularity, because
    handing _apply_dense half a fused destination makes SGLang's loader buffer it
    and assert at the end of load_weights. That is a LOWER bound, not the only
    option -- the snapshot hook keys destinations by (data_ptr, numel, dtype) and
    snapshots each once before its first write, so a larger batch verifies exactly
    the same set. Flush boundaries never split a fusion group either (the bucket's
    cap check runs before a group goes in), so the whole flush is a legal batch.

    VERL_DELTA_VERIFY_BATCH=1 does the whole flush in one call: ~533 load_weights
    calls per sync instead of ~24k-32k, which is a ~60x cut and removes the 65 KB
    per-call log flood at its source rather than filtering it.

    It costs two things, which is why it is not the default yet:
      * snapshot memory -- clones of every destination the flush writes, ~514 MB
        against one param's worth
      * attribution -- a mismatch is reported against the batch, not the parameter
    Turn it on only once a per-unit sweep has passed, so a regression is
    attributable to the batching rather than to the delta.
    """
    if os.environ.get("VERL_DELTA_VERIFY_BATCH") == "1":
        return [list(params)]
    return _atomic_units(params)


def _decode_one(
    encoding: str, positions: torch.Tensor, values: torch.Tensor, p: dict, values_bytes: bool = False
) -> torch.Tensor:
    """Densify one param's sparse delta into a full-shape NaN-masked tensor.

    ``indices`` positions are reinterpreted via an int32 view (8 B/element
    transient) rather than a per-byte int64 unpack (32 B/element), so even a
    full-seed flush of a large embedding stays within a few GiB.
    ``values_bytes`` marks a mixed-dtype flush: value offsets are BYTE offsets
    into a uint8 blob and each slice is reinterpreted (the clone re-bases to
    storage offset 0 for the dtype view).
    """
    numel = math.prod(p["shape"])
    dtype = getattr(torch, p["dtype"])
    vals = values[p["val_start"] : p["val_end"]]

    pos_b = positions[p["pos_start"] : p["pos_end"]]
    if encoding == "indices":
        # Positions ride the wire GAP-ENCODED at a per-parameter width: the sender
        # stores idx[k] - idx[k-1] - 1 (with idx[-1] := -1) and picks 1, 2 or 4
        # bytes from the largest gap in that parameter. A DSv4 delta is 12-17% of
        # elements, so gaps are small and an absolute int32 position cost four
        # times the fp8 code it pointed at.
        #
        # pos_width must be read, not assumed: this decoder used to hardcode
        # int32. Note the wire is NOT backward compatible -- a pre-gap sender
        # wrote absolute int32 positions, which this reads as gaps and decodes
        # to garbage without erroring. Sender and receiver ship together and the
        # payload is never persisted across versions, so the fix is to keep them
        # on one commit, not to sniff the format.
        width = int(p.get("pos_width", 4))
        view_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32}.get(width)
        if view_dtype is None:
            raise ValueError(f"unsupported pos_width={width} for {p['name']!r}")
        gaps = pos_b.view(view_dtype).to(torch.int64)
        # inverse of the gap encoding: gap is idx - prev with idx[-1] := -1, so a
        # repeated position rides as gap 0. Duplicates are legal on purpose --
        # deduplicating them on the sender needed a boolean mask per parameter,
        # and that mask's data-dependent size forced a device->host sync inside
        # the send loop. index_copy_ below applies them last-writer-wins, which
        # is exactly what the absolute-position wire did.
        idx = torch.cumsum(gaps, dim=0) - 1
        # Guard the decode, because the way this fails otherwise is a process
        # kill, not an exception: positions feed a scatter, so one negative or
        # oversized index is an out-of-bounds device write that takes the SGLang
        # scheduler down with a CUDA fault and surfaces upstream only as
        # "Server disconnected". A signed-carrier overflow in the sender did
        # exactly that. Two reductions per parameter, against a multi-GB wire.
        if gaps.numel():
            if int(gaps.min()) < 0:
                raise ValueError(
                    f"{p['name']!r}: negative gap in the position stream "
                    f"(pos_width={width}) -- sender/receiver disagree on the encoding"
                )
            n_elem = 1
            for d in p["shape"]:
                n_elem *= int(d)
            hi = int(idx[-1])  # gaps are non-negative, so this is the max
            if hi >= n_elem:
                raise ValueError(
                    f"{p['name']!r}: decoded position {hi} exceeds the parameter's "
                    f"{n_elem} elements (pos_width={width})"
                )
    else:
        raise ValueError(f"unsupported delta encoding: {encoding!r}")

    if dtype.itemsize == 1:
        # float8 codes: torch's float8 kernel coverage (index_copy, full-with-
        # nan) is spotty across builds; densify entirely in byte space -- the
        # NaN sentinel is the all-ones magnitude byte and positions map 1:1.
        flat_u8 = torch.full((numel,), 0x7F, dtype=torch.uint8, device=values.device)
        if vals.numel():
            flat_u8.index_copy_(0, idx, vals.clone().view(torch.uint8))
        return flat_u8.view(dtype).view(p["shape"])

    flat = torch.full((numel,), float("nan"), dtype=dtype, device=values.device)
    if vals.numel() == 0:
        return flat.view(p["shape"])
    if values_bytes:
        vals = vals.clone().view(dtype)
    flat.index_copy_(0, idx, vals if values_bytes else vals.to(dtype))
    return flat.view(p["shape"])


def _param_storage_index(model: torch.nn.Module) -> list[tuple[int, int]]:
    """Sorted, merged ``(start, end)`` byte intervals of every parameter's and
    persistent buffer's storage. The masked-copy patch consults this so its
    skip-NaN semantics apply ONLY to writes that land in model state: any other
    ``copy_`` a loader performs on the way (scratch buffers, repacking temps,
    quant workspaces) must keep vanilla semantics, NaNs and all. Rebuilt per
    flush -- a named_parameters walk, microseconds against a wire decode."""
    spans = []
    for _, t in itertools.chain(model.named_parameters(), model.named_buffers()):
        if t.device.type == "meta" or t.numel() == 0:
            continue
        base = t.untyped_storage().data_ptr()
        spans.append((base, base + t.untyped_storage().nbytes()))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _in_param_storage(index: list[tuple[int, int]], t: torch.Tensor) -> bool:
    ptr = t.data_ptr()
    i = bisect.bisect_right(index, (ptr, float("inf"))) - 1
    return i >= 0 and index[i][0] <= ptr < index[i][1]


@contextmanager
def _masked_copy(storage_index: list[tuple[int, int]]) -> Iterator[None]:
    """Temporarily make ``Tensor.copy_`` skip NaN positions in the source.

    SGLang's per-model ``load_weights`` ultimately lands on ``param.copy_(loaded)``
    (possibly on a narrowed TP slice). Under this context a NaN-masked source
    overwrites only the changed positions; fully dense sources (e.g. the first
    full-seed flush) take the original fast path untouched. The masked
    semantics are scoped to destinations inside ``storage_index`` (model
    params/buffers): any other copy a loader performs passes through vanilla.
    """
    orig_copy = torch.Tensor.copy_

    def masked_copy_(self: torch.Tensor, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Sync-free masked overwrite: boolean advanced indexing (and a
        # ``mask.all()`` early-out) would force a device->host sync per
        # parameter -- ruinous for MoE flushes carrying >10k per-expert
        # entries. ``torch.where`` keeps everything on-stream; a NaN-free
        # (dense) source degenerates to a plain copy.
        if (
            isinstance(src, torch.Tensor)
            and src.is_floating_point()
            and self.shape == src.shape
            and _in_param_storage(storage_index, self)
        ):
            cast = src.to(self.dtype)
            if cast.element_size() == 1:
                # float8 masked overlay entirely in byte space: NaN sentinel is
                # the all-ones magnitude byte (isnan/where on float8 are not
                # portable across torch builds; uint8 ops are).
                cu8 = cast.contiguous().view(torch.uint8)
                nan_mask = (cu8 & 0x7F) == 0x7F
                merged = torch.where(nan_mask, self.contiguous().view(torch.uint8), cu8)
                return orig_copy(self, merged.view(self.dtype))
            return orig_copy(self, torch.where(torch.isnan(cast), self, cast))
        return orig_copy(self, src, *args, **kwargs)

    torch.Tensor.copy_ = masked_copy_
    try:
        yield
    finally:
        torch.Tensor.copy_ = orig_copy
