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
"""CPU tests for the Megatron converter-profile (``to_hf``) spec.

The central claim of the Megatron shard export is that the mcore->HF conversion
is a pure permutation, so diffing on local mcore shards, rebuilding NaN-sentinel
shards, and converting yields exactly the same HF-coordinate delta as converting
the full tensors first and diffing after. These tests exercise a fused
qkv-style ``to_hf`` closure the same way the engine's NaN rebuild path does --
no megatron or torch.distributed required.
"""

import torch

from verl.checkpoint_engine.delta_sync.sparse_gather import shard_delta_indices
from verl.workers.engine.spec import ShardSpec


def _qkv_like_to_hf(local_shape):
    """Mimic the Megatron export's closure: reshape flat shards, concat TP shards,
    split fused qkv into three HF tensors."""

    def to_hf(shard_list, _shape=tuple(local_shape)):
        shard_list = [sh.view(_shape) for sh in shard_list]
        merged = torch.cat(shard_list, dim=0) if len(shard_list) > 1 else shard_list[0]
        rows = merged.shape[0]
        assert rows % 4 == 0
        q, k, v = torch.split(merged, [rows // 2, rows // 4, rows // 4], dim=0)
        return [("model.q_proj.weight", q), ("model.k_proj.weight", k), ("model.v_proj.weight", v)]

    return to_hf


def _sparse_delta(new: torch.Tensor, old: torch.Tensor):
    """Reference: full-tensor diff in HF coordinates."""
    flat_new, flat_old = new.reshape(-1), old.reshape(-1)
    pos = (
        flat_new.view(torch.uint8).view(flat_new.shape[0], -1) != flat_old.view(torch.uint8).view(flat_old.shape[0], -1)
    ).any(dim=-1)
    idx = pos.nonzero(as_tuple=False).view(-1)
    return idx, flat_new[idx]


def _nan_rebuild(grouped, shard_numel, dtype, to_hf):
    """Replicate the engine's NaN rebuild: flat sentinel buffers per rank -> to_hf."""
    shard_list = []
    for gi, gv in grouped:
        buf = torch.full((shard_numel,), float("nan"), dtype=dtype)
        buf[gi] = gv
        shard_list.append(buf)
    return to_hf(shard_list)


def test_nan_rebuild_matches_full_convert_then_diff():
    torch.manual_seed(0)
    for tp in (1, 2, 4):
        full_old = torch.randn(16, 8, dtype=torch.bfloat16)
        full_new = full_old.clone()
        full_new[3, 2] += 1.0
        full_new[9, 5] -= 0.5
        full_new[15, 7] += 0.25

        old_shards = list(full_old.chunk(tp, dim=0))
        new_shards = list(full_new.chunk(tp, dim=0))
        to_hf = _qkv_like_to_hf(new_shards[0].shape)

        grouped = []
        for old_s, new_s in zip(old_shards, new_shards, strict=True):
            idx, val = shard_delta_indices(new_s.reshape(-1), old_s.reshape(-1), 0)
            grouped.append((idx, val))

        rebuilt = _nan_rebuild(grouped, new_shards[0].numel(), full_new.dtype, to_hf)
        ref_new = dict(to_hf([sh.reshape(-1) for sh in new_shards]))
        ref_old = dict(to_hf([sh.reshape(-1) for sh in old_shards]))

        seen = 0
        for hf_name, hf_tensor in rebuilt:
            fl = hf_tensor.reshape(-1)
            pos = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
            ref_idx, ref_val = _sparse_delta(ref_new[hf_name], ref_old[hf_name])
            assert torch.equal(pos, ref_idx), f"tp={tp} {hf_name}: positions diverge"
            assert torch.equal(fl[pos], ref_val), f"tp={tp} {hf_name}: values diverge"
            seen += int(pos.numel())
        assert seen == 3


def test_converter_spec_shape():
    """A converter spec keeps the declarative fields alongside to_hf."""
    to_hf = _qkv_like_to_hf((4, 8))
    spec = ShardSpec(full_shape=(16, 8), to_hf=to_hf)
    assert spec.mesh is None and spec.placements is None
    full = torch.randn(16, 8)
    out = dict(spec.to_hf([sh.reshape(-1) for sh in full.chunk(4, dim=0)]))
    assert set(out) == {"model.q_proj.weight", "model.k_proj.weight", "model.v_proj.weight"}
    parts = [out["model.q_proj.weight"], out["model.k_proj.weight"], out["model.v_proj.weight"]]
    assert torch.equal(torch.cat(parts), full)
