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
"""Bit-identity tests for the SGLang custom-weight-loader delta apply.

Builds sparse ``indices``-encoding flushes exactly as the sharded engines
assemble them (int32 within-parameter flat positions viewed as bytes + a value
stream + checksum) and feeds each through :func:`delta_loader.apply_delta`
against a stand-in model whose ``load_weights`` mimics SGLang's
``param.copy_(loaded)`` semantics. Verifies the masked in-place apply: changed
positions land bit-exactly, and positions outside the delta are never touched.
"""

from __future__ import annotations

import json

import torch

from verl.checkpoint_engine.delta_sync import DeltaParam, checksum
from verl.workers.rollout.sglang_rollout.delta_loader import apply_delta


class _FakeModel:
    """Holds live params; load_weights lands on param.copy_(loaded), like SGLang."""

    def __init__(self, named: list[tuple[str, torch.Tensor]]):
        self.params = {n: t.clone() for n, t in named}

    def load_weights(self, chunk):
        for name, tensor in chunk:
            self.params[name].copy_(tensor)

    # the storage-scoped masked copy consults these, like any nn.Module
    def named_parameters(self):
        return iter(self.params.items())

    def named_buffers(self):
        return iter(())


def _make_named(dtype=torch.bfloat16) -> list[tuple[str, torch.Tensor]]:
    torch.manual_seed(0)
    return [
        ("layer.0.weight", torch.randn(64, 32, dtype=dtype)),
        ("layer.1.weight", torch.randn(32, 16, dtype=dtype)),
    ]


def _sparse_indices_flush(old_named, new_named):
    """Assemble one indices-encoding flush from a bytewise old/new diff --
    the same layout the sharded engines' ``_assemble_flush`` produces."""
    params, idx_pieces, val_pieces = [], [], []
    pos_off = val_off = 0
    for (name, old), (_, new) in zip(old_named, new_named, strict=True):
        fo, fn = old.reshape(-1), new.reshape(-1)
        changed = (fo.view(torch.uint8).view(fo.numel(), -1) != fn.view(torch.uint8).view(fn.numel(), -1)).any(dim=-1)
        idx = changed.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        nnz = int(idx.numel())
        idx_pieces.append(idx.to(torch.int32))
        val_pieces.append(fn[idx])
        params.append(
            DeltaParam(
                name=name,
                dtype=str(new.dtype).replace("torch.", ""),
                shape=list(new.shape),
                pos_start=pos_off,
                pos_end=pos_off + nnz * 4,
                pos_width=4,
                val_start=val_off,
                val_end=val_off + nnz,
            )
        )
        pos_off += nnz * 4
        val_off += nnz
    positions = torch.cat(idx_pieces).contiguous().view(torch.uint8)
    values = torch.cat(val_pieces)
    return params, positions, values


def _named_tensors(params, positions, values, encoding="indices", extra_spec=None):
    spec = {
        "encoding": encoding,
        "params": [vars(p) for p in params],
        "checksum": int(checksum(positions, values)),
    }
    if extra_spec:
        spec.update(extra_spec)
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)
    out = [("__delta_spec__", spec_t), ("__values__", values.clone())]
    if positions.numel():
        out.insert(1, ("__positions__", positions.clone()))
    return out


def test_masked_apply_bit_identical():
    named = _make_named()
    model = _FakeModel(named)

    new_named = []
    for name, t in named:
        new = t.clone()
        flat = new.view(-1)
        idx = torch.tensor([1, 17, 200, 511], dtype=torch.int64) % flat.numel()
        flat[idx] = flat[idx] + 0.5
        new_named.append((name, new))

    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named)))

    for name, expected in new_named:
        got = model.params[name]
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), f"{name} not bit-identical"


def test_untouched_positions_preserved():
    """Positions absent from the delta must keep the model's LIVE values (not the
    trainer snapshot's) -- proves the apply is masked, not a full overwrite."""
    named = _make_named()
    model = _FakeModel(named)
    # Poison one untouched position in the live model; a full overwrite would revert it.
    sentinel_name = named[0][0]
    model.params[sentinel_name].view(-1)[3] = 42.0

    new_named = []
    for name, t in named:
        new = t.clone()
        new.view(-1)[7] = new.view(-1)[7] + 1.0  # change only position 7
        new_named.append((name, new))

    apply_delta(model, _named_tensors(*_sparse_indices_flush(named, new_named)))

    live = model.params[sentinel_name].view(-1)
    assert live[3].item() == 42.0, "masked apply must not touch positions outside the delta"
    assert live[7] == new_named[0][1].view(-1)[7]


def test_checksum_mismatch_raises():
    import pytest

    named = _make_named()
    new_named = [(n, t + 0.5) for n, t in named]
    named_tensors = _named_tensors(*_sparse_indices_flush(named, new_named))
    named_tensors[2][1].view(torch.uint8)[0] ^= 0xFF  # corrupt one value byte
    with pytest.raises(RuntimeError, match="checksum"):
        apply_delta(_FakeModel(named), named_tensors)


def test_dense_flush_applies_full_tensors():
    """Dense (first-sync) flushes carry values only; the loader must apply them verbatim."""
    named = _make_named()
    model = _FakeModel([(n, torch.zeros_like(t)) for n, t in named])  # dummy init

    params, pieces, val_off = [], [], 0
    for name, t in named:
        flat = t.contiguous().view(-1)
        params.append(
            {
                "name": name,
                "dtype": str(t.dtype).replace("torch.", ""),
                "shape": list(t.shape),
                "pos_start": 0,
                "pos_end": 0,
                "pos_width": 4,
                "val_start": val_off,
                "val_end": val_off + flat.numel(),
            }
        )
        pieces.append(flat)
        val_off += flat.numel()
    values = torch.cat(pieces)

    spec = {"encoding": "dense", "params": params, "checksum": int(checksum(torch.empty(0, dtype=torch.uint8), values))}
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)
    apply_delta(model, [("__delta_spec__", spec_t), ("__values__", values)])

    for name, expected in named:
        assert torch.equal(model.params[name].view(torch.int16), expected.view(torch.int16)), name


def _dense_verify_flush(named, is_last=True, verify=True):
    params, pieces, val_off = [], [], 0
    for name, t in named:
        flat = t.contiguous().view(-1)
        params.append(
            {
                "name": name,
                "dtype": str(t.dtype).replace("torch.", ""),
                "shape": list(t.shape),
                "pos_start": 0,
                "pos_end": 0,
                "pos_width": 4,
                "val_start": val_off,
                "val_end": val_off + flat.numel(),
            }
        )
        pieces.append(flat)
        val_off += flat.numel()
    values = torch.cat(pieces)
    spec = {
        "encoding": "dense",
        "verify": verify,
        "is_last": is_last,
        "params": params,
        "checksum": int(checksum(torch.empty(0, dtype=torch.uint8), values)),
    }
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)
    return [("__delta_spec__", spec_t), ("__values__", values)]


def test_verify_sweep_passes_on_identical_state():
    """A verify flush against a bit-identical model reports zero mismatches."""
    named = _make_named()
    model = _FakeModel([(n, t.clone()) for n, t in named])
    apply_delta(model, _dense_verify_flush(named))  # must not raise


def test_verify_sweep_fails_loud_on_divergence():
    """A single flipped element in the server state must fail the sweep."""
    import pytest

    named = _make_named()
    diverged = [(n, t.clone()) for n, t in named]
    diverged[1][1].view(-1)[3] += 1.0
    model = _FakeModel(diverged)
    with pytest.raises(RuntimeError, match="verification FAILED"):
        apply_delta(model, _dense_verify_flush(named))


class _ScratchCopyModel(_FakeModel):
    """A loader that also copies the incoming (NaN-masked) tensor into a
    scratch buffer on the way -- the exact pattern quant loaders use for
    repacking. Scratch writes must keep VANILLA copy semantics (NaNs pass
    through); only the param write is masked."""

    def __init__(self, named):
        super().__init__(named)
        self.scratch = {n: torch.zeros_like(t) for n, t in named}

    def load_weights(self, chunk):
        for name, tensor in chunk:
            self.scratch[name].copy_(tensor)  # NOT model state: no masking
            self.params[name].copy_(tensor)


def test_masked_copy_scoped_to_param_storage():
    """NaN-masked overlay applies to param storages only: a scratch copy in the
    same load path receives the NaN sentinels verbatim."""
    named = _make_named()
    model = _ScratchCopyModel(named)
    new_named = [(n, t.clone()) for n, t in named]
    new_named[0][1][3, 5] = 42.0
    params, positions, values = _sparse_indices_flush(named, new_named)
    apply_delta(model, _named_tensors(params, positions, values))

    assert model.params["layer.0.weight"][3, 5] == 42.0
    untouched = ~torch.isnan(model.params["layer.0.weight"])
    assert untouched.all(), "param must never hold NaN after a masked apply"
    scr = model.scratch["layer.0.weight"]
    assert torch.isnan(scr).sum() == scr.numel() - 1, "scratch copy must receive the NaN mask verbatim"
    assert scr[3, 5] == 42.0


class _PostLoadModel(_FakeModel):
    """Records that post_load_weights ran, and that it ran with VANILLA copy_
    semantics (a NaN source must write through -- proving the masked patch is
    off by the time derived tensors recompute)."""

    def __init__(self, named):
        super().__init__(named)
        self.post_load_calls = 0
        self.probe = torch.zeros(4)

    def post_load_weights(self):
        self.post_load_calls += 1
        self.probe.copy_(torch.full((4,), float("nan")))


def test_post_load_weights_runs_unpatched_on_last_flush():
    named = _make_named()
    model = _PostLoadModel(named)
    new_named = [(n, t.clone()) for n, t in named]
    new_named[0][1][0, 0] = 7.0
    params, positions, values = _sparse_indices_flush(named, new_named)

    apply_delta(model, _named_tensors(params, positions, values))
    assert model.post_load_calls == 0, "non-final flush must not trigger post_load_weights"

    apply_delta(model, _named_tensors(params, positions, values, extra_spec={"is_last": True}))
    assert model.post_load_calls == 1
    assert torch.isnan(model.probe).all(), "post_load_weights must run with vanilla copy_ semantics"
