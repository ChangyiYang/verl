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
"""CPU unit tests for fused-param grouping across every weight-transfer splitter.

SGLang's DeepSeek-V4 loader rebuilds ``wqkv_a`` / ``compressor.wkv_gate`` /
``indexer.compressor.wkv_gate`` by ``torch.cat``-ing two separately-named tensors,
buffering the first arrival in a cache it creates inside ``load_weights`` and asserts
empty on return. Both halves must therefore reach the SAME ``load_weights`` call.

verl splits weights in FIVE independent places, and each one can break that on its own:

1. ``delta_checkpoint_engine`` steady stream  -- ``_FusionStager`` + ``add_atomic``
2. ``delta_checkpoint_engine`` seed stream    -- ``_FusionStager.offer_piece``
3. ``delta_loader`` chunking                  -- ``_load_in_chunks``
4. ``sglang_rollout.get_named_tensor_buckets`` -- the plain full-sync path, used by the
   NCCL engine and by ``separate_async``'s hybrid replicas
5. ``delta_loader._verify_dense`` -- the odd one out: it split per PARAM by design,
   not by bytes, so no bucket size could avoid it. ``_verify_due()`` always returns
   True on the first steady sync, so it fired on every run that got that far.

These tests pin all five, plus the shared membership table in
``verl.utils.fusion_groups``. Run: ``pytest tests/checkpoint_engine/test_fusion_groups.py``
"""

import asyncio

import pytest
import torch

from verl.utils.fusion_groups import FUSION_GROUPS, fusion_key, fusion_match, group_size

L = "model.layers.7.self_attn"


def idx(n):
    return torch.arange(n, dtype=torch.int32)


def val(n, dtype=torch.bfloat16):
    return torch.ones(n, dtype=dtype)


# --------------------------------------------------------------------------- table


def test_table_covers_every_family_in_both_attention_spellings():
    keys = [k for k, _ in FUSION_GROUPS]
    for base in ("wqkv_a", "wqkv_a_scale", "wqkv_a_scale_inv", "compressor_wkv_gate",
                 "indexer_compressor_wkv_gate"):
        assert base in keys, f"{base} missing"
        assert base + "@attn" in keys, f"{base}@attn missing"
    assert all(group_size(k) == 2 for k, _ in FUSION_GROUPS)


@pytest.mark.parametrize(
    "name",
    [
        f"{L}.wq_a.weight",
        f"{L}.wkv.weight",
        # a bare ".wkv.weight" suffix would also match these two, which is why the
        # table spells the attention block out
        f"{L}.compressor.wkv.weight",
        f"{L}.indexer.compressor.wkv.weight",
        # Megatron-Bridge's DSv4 mapping writes layers.N.attn.*, SGLang sees self_attn.*
        "layers.7.attn.wq_a.weight",
        "layers.7.attn.compressor.wgate.weight",
        # the fp8 scale is named .scale in the export, weight_scale_inv after a
        # bf16-master requantize -- both must resolve, and to DIFFERENT groups
        "layers.0.attn.wq_a.scale",
        f"{L}.wq_a.weight_scale_inv",
    ],
)
def test_every_member_matches_exactly_one_group(name):
    assert fusion_match(name) is not None  # asserts internally on a double match


def test_weight_and_scale_are_separate_groups():
    # SGLang keys its cache on the destination param name, so wqkv_a.weight and
    # wqkv_a.weight_scale_inv are distinct entries each needing its own pair.
    assert fusion_key("layers.0.attn.wq_a.scale") == fusion_key("layers.0.attn.wkv.scale")
    assert fusion_key("layers.0.attn.wq_a.weight") == fusion_key("layers.0.attn.wkv.weight")
    assert fusion_key("layers.0.attn.wq_a.scale") != fusion_key("layers.0.attn.wq_a.weight")


def test_non_member_returns_none():
    assert fusion_key("model.layers.7.mlp.gate_proj.weight") is None
    assert fusion_key("model.embed_tokens.weight") is None


# ------------------------------------------------------------------ sender: steady


def _stager():
    from verl.checkpoint_engine.delta_checkpoint_engine import _FusionStager

    return _FusionStager()


def test_steady_holds_a_member_until_its_sibling_arrives():
    st = _stager()
    assert st.offer(f"{L}.wq_a.weight", "bfloat16", (4096, 1024), idx(5), val(5)) is None
    entries, is_group = st.offer(f"{L}.wkv.weight", "bfloat16", (4096, 128), idx(3), val(3))
    assert is_group
    assert [e[0] for e in entries] == [f"{L}.wq_a.weight", f"{L}.wkv.weight"]
    st.assert_drained()


def test_steady_materialises_an_unchanged_half_as_an_empty_entry():
    # The receiver densifies a zero-length entry to an all-NaN full-shape tensor and
    # _masked_copy then keeps the destination, so cat-ing it in is a no-op for that half.
    st = _stager()
    st.offer(f"{L}.wq_a.weight", "bfloat16", (4096, 1024), idx(5), val(5))
    entries, _ = st.offer(f"{L}.wkv.weight", "bfloat16", (4096, 128), None, None)
    empty = next(e for e in entries if e[0].endswith(".wkv.weight"))
    assert empty[3] is not None and empty[3].numel() == 0
    assert empty[4].numel() == 0 and empty[4].dtype == torch.bfloat16
    assert st.n_filled == 1
    st.assert_drained()


def test_steady_sends_nothing_when_no_member_changed():
    st = _stager()
    st.offer(f"{L}.wq_a.weight", "bfloat16", (4096, 1024), None, None)
    entries, is_group = st.offer(f"{L}.wkv.weight", "bfloat16", (4096, 128), None, None)
    assert entries == [] and is_group
    st.assert_drained()


def test_steady_leaves_non_members_alone():
    # unchanged non-members must keep being dropped: an entry costs more than its bytes
    st = _stager()
    entries, is_group = st.offer("model.layers.7.mlp.gate_proj.weight", "bfloat16", (1, 1), idx(9), val(9))
    assert not is_group and len(entries) == 1
    st.assert_drained()


def test_incomplete_group_fails_loudly():
    st = _stager()
    st.offer(f"{L}.wq_a.weight", "bfloat16", (4096, 1024), idx(5), val(5))
    with pytest.raises(AssertionError, match="never completed"):
        st.assert_drained()


def test_duplicate_member_fails_loudly():
    st = _stager()
    st.offer(f"{L}.wq_a.weight", "bfloat16", (4096, 1024), idx(5), val(5))
    with pytest.raises(AssertionError, match="duplicate fusion member"):
        st.offer(f"{L}.wq_a.weight", "bfloat16", (4096, 1024), idx(5), val(5))


# -------------------------------------------------------------------- sender: seed


def test_seed_path_co_locates_without_nan_filling():
    # a full export contains every member by construction, so only co-location matters
    st = _stager()
    assert st.offer_piece(f"{L}.wq_a.weight", "PIECE_q", 40) is None
    released, is_group = st.offer_piece(f"{L}.wkv.weight", "PIECE_kv", 40)
    assert is_group and [p for p, _ in released] == ["PIECE_q", "PIECE_kv"]
    st.assert_drained()


# ------------------------------------------------------------------- flush bucket


def test_flush_boundary_never_falls_inside_a_group():
    from verl.checkpoint_engine.delta_checkpoint_engine import _FlushBucket, _slice_pieces

    flushes = []
    bkt = _FlushBucket(cap=100, assemble=lambda ps: [p.name for p in ps],
                       publish=lambda f, last: flushes.append(f))
    bkt.add(_slice_pieces("filler", "bfloat16", (10,), idx(8), val(8))[0][0], 90)
    grp = []
    for nm in (f"{L}.wq_a.weight", f"{L}.wkv.weight"):
        grp.extend(_slice_pieces(nm, "bfloat16", (10,), idx(4), val(4)))
    bkt.add_atomic(grp)
    bkt.seal()
    bkt.emit(is_last=True)
    assert any(f"{L}.wq_a.weight" in f and f"{L}.wkv.weight" in f for f in flushes)
    assert not any(f"{L}.wq_a.weight" in f and "filler" in f for f in flushes)


def test_empty_delta_still_produces_one_piece():
    # a zero-length range() would emit nothing, and the receiver would never see the half
    from verl.checkpoint_engine.delta_checkpoint_engine import _slice_pieces

    pieces = _slice_pieces("x", "bfloat16", (4096, 128), torch.empty(0, dtype=torch.int32), torch.empty(0))
    assert len(pieces) == 1 and pieces[0][1] == 0


# ---------------------------------------------------------------------- receiver


class _FakeModel:
    def __init__(self):
        self.calls = []

    def load_weights(self, chunk):
        self.calls.append([n for n, _ in chunk])


def test_receiver_chunk_boundary_never_falls_inside_a_group():
    from verl.workers.rollout.sglang_rollout.delta_loader import CHUNK_BYTES, _load_in_chunks

    def mk(name, nbytes):
        return {"name": name, "nbytes": nbytes}

    m = _FakeModel()
    _load_in_chunks(
        m,
        [mk("filler", CHUNK_BYTES - 16), mk(f"{L}.wq_a.weight", 64), mk(f"{L}.wkv.weight", 64)],
        lambda p: torch.empty(p["nbytes"], dtype=torch.uint8),
    )
    assert len(m.calls) == 2, "expected a real split, otherwise this proves nothing"
    assert any(f"{L}.wq_a.weight" in c and f"{L}.wkv.weight" in c for c in m.calls)


def test_receiver_reunites_members_that_arrive_apart():
    from verl.workers.rollout.sglang_rollout.delta_loader import _atomic_units

    units = _atomic_units(
        [{"name": f"{L}.wq_a.weight"}, {"name": "model.embed_tokens.weight"}, {"name": f"{L}.wkv.weight"}]
    )
    assert len(units) == 2
    assert {p["name"] for p in units[0]} == {f"{L}.wq_a.weight", f"{L}.wkv.weight"}


# ------------------------------------------------- plain full-sync bucketer (nccl)


def _run_buckets(items, cap):
    from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

    async def go():
        return [b async for b in get_named_tensor_buckets(iter(items), cap)]

    return asyncio.run(go())


def _t(nbytes):
    return torch.empty(nbytes, dtype=torch.uint8)


def test_full_sync_bucketer_keeps_a_group_together():
    out = _run_buckets(
        [("filler", _t(1024 - 16)), (f"{L}.wq_a.weight", _t(64)), (f"{L}.wkv.weight", _t(64))], 1024
    )
    names = [[n for n, _ in b] for b in out]
    assert len(out) == 2, "expected a real split, otherwise this proves nothing"
    assert any(f"{L}.wq_a.weight" in b and f"{L}.wkv.weight" in b for b in names)


def test_full_sync_bucketer_reunites_members_that_arrive_apart():
    out = _run_buckets(
        [(f"{L}.wq_a.weight", _t(64)), ("other", _t(64)), (f"{L}.wkv.weight", _t(64))], 1024
    )
    names = [[n for n, _ in b] for b in out]
    assert any(f"{L}.wq_a.weight" in b and f"{L}.wkv.weight" in b for b in names)


def test_full_sync_bucketer_leaves_non_members_alone():
    items = [(f"p{i}", _t(600)) for i in range(4)]
    out = _run_buckets(items, 1024)
    assert [n for b in out for n, _ in b] == [f"p{i}" for i in range(4)]
    assert len(out) == 4


def test_full_sync_bucketer_fails_loudly_on_an_incomplete_group():
    with pytest.raises(AssertionError, match="never completed"):
        _run_buckets([(f"{L}.wq_a.weight", _t(64)), ("x", _t(64))], 1024)


def test_verify_sweep_feeds_whole_units_not_single_params():
    """The verify sweep is the FIFTH splitter and the only structural one.

    It used to call _apply_dense(model, [p]) once per param, which hands the DSv4
    loader half of a fused destination. _verify_due() always returns True on the
    first steady sync, so this fired on every run that got that far -- regardless
    of any byte-size setting, which is what makes it different from the other four.
    """
    from verl.workers.rollout.sglang_rollout import delta_loader

    seen = []

    def fake_apply_dense(model, params, values, values_bytes=False):
        seen.append([p["name"] for p in params])

    orig = delta_loader._apply_dense
    delta_loader._apply_dense = fake_apply_dense
    try:
        delta_loader._verify_dense(
            _FakeModel(),
            [
                {"name": f"{L}.wq_a.weight"},
                {"name": "model.embed_tokens.weight"},
                {"name": f"{L}.wkv.weight"},
            ],
            torch.empty(0),
            is_last=False,
        )
    finally:
        delta_loader._apply_dense = orig

    assert any(f"{L}.wq_a.weight" in c and f"{L}.wkv.weight" in c for c in seen), (
        f"fused pair must be verified in one call, got {seen}"
    )
    assert ["model.embed_tokens.weight"] in seen, "non-members still go one at a time"
