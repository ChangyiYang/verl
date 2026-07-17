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


def test_expert_virtual_stack_rebuild_matches_reference():
    """mcore EP: each rank holds one LOCAL expert tensor per name (same name, different
    global expert). The exporter describes it as one block of the virtual
    [ep_size, *full_etp_shape] tensor. Verify the block NaN rebuild -> per-global-expert
    conversion equals converting full tensors first and diffing after."""
    torch.manual_seed(4)
    ep, etp = 4, 2
    n_per_rank = 2  # local experts per ep rank -> 8 global experts
    rows, cols = 6, 4  # per-expert fc weight [rows, cols], ETP shards rows
    full_etp = (rows, cols)

    def convert(gname, tensor):
        # stand-in for weight_converter.convert_param: rename to HF per-expert
        gid = int(gname.split(".weight")[-1])
        return [(f"model.experts.{gid}.up.weight", tensor)]

    for local_id in range(n_per_rank):
        # reference: global tensors per ep rank for THIS local slot
        olds = [torch.randn(full_etp, dtype=torch.bfloat16) for _ in range(ep)]
        news = [t.clone() for t in olds]
        news[1][2, 3] += 1.0
        news[3][0, 0] -= 0.5

        # engine-style: rebuild virtual [ep, rows, cols] from per-(ep, etp) blocks
        virtual_nan = torch.full((ep, rows, cols), float("nan"), dtype=torch.bfloat16)
        for ep_rank in range(ep):
            for etp_rank in range(etp):
                r0 = etp_rank * (rows // etp)
                lo = olds[ep_rank][r0 : r0 + rows // etp].contiguous().reshape(-1)
                ln = news[ep_rank][r0 : r0 + rows // etp].contiguous().reshape(-1)
                changed = (lo.view(torch.uint8).view(lo.numel(), -1) != ln.view(torch.uint8).view(ln.numel(), -1)).any(
                    dim=-1
                )
                lidx = changed.nonzero(as_tuple=False).view(-1)
                buf = torch.full((lo.numel(),), float("nan"), dtype=torch.bfloat16)
                buf[lidx] = ln[lidx]
                virtual_nan[ep_rank, r0 : r0 + rows // etp] = buf.view(rows // etp, cols)

        seen = 0
        for k in range(ep):
            gname = f"decoder.layers.0.mlp.experts.linear_fc1.weight{n_per_rank * k + local_id}"
            for hf_name, hf_tensor in convert(gname, virtual_nan[k]):
                fl = hf_tensor.reshape(-1)
                pos = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
                rn, ro = news[k].reshape(-1), olds[k].reshape(-1)
                ref = (
                    (rn.view(torch.uint8).view(rn.numel(), -1) != ro.view(torch.uint8).view(ro.numel(), -1))
                    .any(dim=-1)
                    .nonzero(as_tuple=False)
                    .view(-1)
                )
                assert torch.equal(pos, ref), f"{hf_name}: positions diverge"
                assert torch.equal(fl[pos], rn[pos]), f"{hf_name}: values diverge"
                seen += int(pos.numel())
        assert seen == 2
