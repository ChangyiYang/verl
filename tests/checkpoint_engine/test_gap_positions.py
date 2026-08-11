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
"""Gap-encoded delta positions: round-trip, width selection, and the byte win.

A measured DSv4 delta changed 12-17% of elements yet put 72% of a full sync on
the wire, because every 1-byte fp8 code carried a 4-byte absolute int32 position.
Gap encoding stores idx[k] - idx[k-1] - 1 instead and picks the width per
parameter, which at these densities is almost always a single byte.

The decoder previously hardcoded int32, so these tests also pin that it reads
pos_width -- a narrowed sender against an assuming receiver is the kind of
mismatch that produces plausible garbage rather than an error.
"""

import pytest
import torch

from verl.checkpoint_engine.delta_checkpoint_engine import _gap_encode


def _decode(gaps: torch.Tensor) -> torch.Tensor:
    """Receiver's inverse, mirroring delta_loader._decode_one."""
    return torch.cumsum(gaps.to(torch.int64), dim=0) - 1


@pytest.mark.parametrize(
    "idx",
    [
        [0],
        [5],
        [0, 1, 2, 3],
        [0, 7, 9, 300],
        [3, 4, 100, 100000],
        list(range(0, 1000, 7)),
    ],
)
def test_round_trip(idx):
    t = torch.tensor(idx, dtype=torch.int64)
    gaps, _ = _gap_encode(t)
    assert torch.equal(_decode(gaps), t)


def test_width_is_chosen_per_parameter_from_the_largest_gap():
    dense = torch.arange(0, 1000, 3, dtype=torch.int64)  # gaps of 2
    assert _gap_encode(dense)[1] == 1
    mid = torch.tensor([0, 1000], dtype=torch.int64)  # gap 999
    assert _gap_encode(mid)[1] == 2
    sparse = torch.tensor([0, 100_000], dtype=torch.int64)  # gap > 65535
    assert _gap_encode(sparse)[1] == 4


@pytest.mark.parametrize("gap", [0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x7FFFFFFF])
def test_every_tier_boundary_round_trips(gap):
    """One value per tier is not enough -- the bug this pins lived in the BAND
    between two tiers' natural limits. The 2-byte carrier is torch.int16, so it
    stops at 0x7FFF, not 0xFFFF; gaps in [0x8000, 0xFFFF] used to wrap negative,
    decode to negative positions, and scatter out of bounds on the GPU (a device
    fault that kills the process instead of raising). Boundaries, not samples."""
    # gap = idx - prev now (prev of the first entry is -1), so a step of `gap`
    # is produced by idx = [0, gap], not [0, gap + 1].
    idx = torch.tensor([0, gap], dtype=torch.int64)
    gaps, width = _gap_encode(idx)
    assert int(gaps.max()) == gap, f"meant to exercise a gap of {gap}, got {int(gaps.max())}"
    assert torch.equal(_decode(gaps), idx), f"gap {gap} at width {width} did not round-trip"


def test_no_tier_can_emit_a_negative_gap():
    """The failure mode was silent: encoding produced a valid-looking tensor.
    Positions are non-negative and strictly increasing, so gaps are >= 0 --
    a negative anywhere means the carrier wrapped."""
    for gap in (0xFF, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 1 << 20):
        gaps, width = _gap_encode(torch.tensor([0, gap], dtype=torch.int64))
        assert (gaps.to(torch.int64) >= 0).all(), f"gap {gap} wrapped negative at width {width}"


def test_widest_widths_still_round_trip():
    """The fallbacks are what make narrowing safe, so they must actually work."""
    for idx in ([0, 1000], [0, 100_000], [7, 70_000, 700_000]):
        t = torch.tensor(idx, dtype=torch.int64)
        gaps, w = _gap_encode(t)
        assert torch.equal(_decode(gaps), t), f"width {w} failed on {idx}"


def test_empty_is_still_encodable():
    gaps, width = _gap_encode(torch.empty(0, dtype=torch.int64))
    assert gaps.numel() == 0 and width == 4


def test_uint8_at_the_densities_we_measured():
    """12-17% density is the regime this exists for; it must land on 1 byte."""
    torch.manual_seed(0)
    for density in (0.1225, 0.1697):
        mask = torch.rand(2_000_000) < density
        idx = mask.nonzero(as_tuple=False).view(-1)
        gaps, width = _gap_encode(idx)
        assert width == 1, f"density {density} chose width {width}, expected 1"
        assert torch.equal(_decode(gaps), idx)


def test_bytes_per_element_drops_from_five_to_two():
    """The whole point: an fp8 code plus its position."""
    torch.manual_seed(0)
    idx = (torch.rand(2_000_000) < 0.1225).nonzero(as_tuple=False).view(-1)
    _, width = _gap_encode(idx)
    before = 4 + 1  # int32 position + fp8 code
    after = width + 1
    assert after == 2
    assert after / before == pytest.approx(0.4)


def test_unsorted_input_is_rejected_not_wrapped():
    """The bug that cost three cluster runs. gather_slot_entries_to_rank0
    concatenates each rank's ascending block back to back, so the result is
    ascending WITHIN blocks and descending at every seam. Absolute int32
    positions did not care -- each stood alone -- so nothing upstream ever had
    to guarantee order, and switching to a stateful encoding quietly inherited a
    requirement no one was meeting.

    It has to ASSERT, not wrap: a descending step is a negative gap, and a
    negative gap in a uint8/int16 carrier becomes a large positive one. Then the
    sender's range check passes (raw positions are in range), the receiver sees
    only non-negative gaps, and the positions just silently land out of bounds."""
    seam = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.int64)
    with pytest.raises(AssertionError, match="non-decreasing"):
        _gap_encode(seam)


def test_the_wrap_this_prevents_would_have_been_invisible():
    """Pin why the assert is the fix and a guard downstream is not: encode the
    seam by hand and observe that every symptom looks healthy."""
    idx = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.int64)
    prev = torch.cat([idx.new_full((1,), -1), idx[:-1]])
    gaps = idx - prev - 1
    assert int(gaps.max()) <= 0xFF  # width 1 would be chosen from max alone
    wrapped = gaps.to(torch.uint8)
    assert (wrapped.to(torch.int64) >= 0).all()  # receiver's negative check passes
    decoded = _decode(wrapped)
    assert int(decoded.max()) > int(idx.max())  # ...yet positions overshoot


def _sort_dedupe(idx, val):
    """Mirror of the sender's normalisation in _bucket_slot_delta."""
    order = torch.argsort(idx, stable=True)
    idx, val = idx[order], val[order]
    keep = torch.ones_like(idx, dtype=torch.bool)
    keep[:-1] = idx[1:] != idx[:-1]
    return idx[keep], val[keep]


def test_duplicate_positions_encode_as_gap_zero():
    """Ranks report the same position twice for a replicated param. The encoding
    now carries that as gap 0 rather than requiring the sender to deduplicate:
    the dedup was a boolean mask, and its data-dependent output size forced a
    device->host sync per parameter inside the send loop. index_copy_ applies
    duplicates last-writer-wins, which is what the absolute wire did anyway."""
    idx = torch.tensor([1, 1, 5, 5, 9], dtype=torch.int64)  # sorted, with repeats
    gaps, _ = _gap_encode(idx)
    assert int(gaps.min()) == 0, "a repeat must be legal, not negative"
    assert torch.equal(_decode(gaps), idx)


def test_normalised_seam_input_encodes():
    """End to end on the shape the gather actually produces: per-rank ascending
    blocks concatenated, with repeats across blocks."""
    idx = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.int64)
    val = torch.arange(9, dtype=torch.int64)
    order = torch.argsort(idx, stable=True)
    sidx, sval = idx[order], val[order]
    gaps, width = _gap_encode(sidx)
    assert torch.equal(_decode(gaps), sidx) and width == 1
    assert int(gaps.min()) == 0, "repeats across ranks ride as gap 0"


def test_gap_encode_reads_the_device_once():
    """Guards the fix, not the feature. Each device->host read stalls the CUDA
    stream, and rank 0 runs this per parameter -- ~8 reads/param turned a 4x
    smaller wire into a SLOWER sync (2.00 -> 0.42 GiB/s measured). Asking for
    min and max separately is two stalls; one stacked read is one.

    Counting .item()/.tolist() calls is crude but it is the property that
    regressed, and a plausible-looking `int(gaps.min())` edit would silently
    bring the cost back."""
    import torch as _t

    calls = []
    real_tolist, real_item = _t.Tensor.tolist, _t.Tensor.item

    def spy_tolist(self):
        calls.append("tolist")
        return real_tolist(self)

    def spy_item(self):
        calls.append("item")
        return real_item(self)

    _t.Tensor.tolist, _t.Tensor.item = spy_tolist, spy_item
    try:
        _gap_encode(torch.arange(0, 4096, 3, dtype=torch.int64))
    finally:
        _t.Tensor.tolist, _t.Tensor.item = real_tolist, real_item

    assert len(calls) == 1, f"expected one device read, got {len(calls)}: {calls}"


def test_slice_pieces_encodes_each_piece_once():
    """The width was being computed to size the bucket and then thrown away, so
    _assemble_flush encoded the same piece again. Pieces now carry their gaps."""
    from verl.checkpoint_engine.delta_checkpoint_engine import _slice_pieces

    idx = torch.arange(0, 3000, 3, dtype=torch.int64)
    val = torch.zeros(idx.numel(), dtype=torch.bfloat16)
    pieces = _slice_pieces("p", "bfloat16", [4096], idx, val)
    for piece, nbytes in pieces:
        assert piece.gaps is not None, "piece must carry its encoded gaps"
        assert nbytes == piece.idx.numel() * (piece.pos_width + val.element_size())
        assert torch.equal(_decode(piece.gaps), piece.idx), "carried gaps must decode to the piece"


def test_verify_sampling_is_deterministic_and_consumes_everything(monkeypatch):
    """The sweep's generator is backed by a collective per-tensor assembly, so
    every rank must walk all of it even when only a share is shipped. And the
    chosen set has to match across ranks without them talking, hence hashing the
    name rather than drawing from an RNG."""
    from verl.checkpoint_engine.delta_checkpoint_engine import _verify_sample

    names = [f"layers.{i}.w" for i in range(400)]
    seen = []

    def gen():
        for n in names:
            seen.append(n)
            yield n, torch.zeros(1)

    monkeypatch.setenv("VERL_DELTA_VERIFY_FRACTION", "0.1")
    kept = [n for n, _ in _verify_sample(gen())]

    assert seen == names, "the generator must be drained in full -- it is collective"
    assert 0 < len(kept) < len(names), f"expected a sample, got {len(kept)}/{len(names)}"
    seen.clear()
    assert [n for n, _ in _verify_sample(gen())] == kept, "selection must be reproducible"


def test_verify_sampling_defaults_to_everything(monkeypatch):
    """Absent the knob, behaviour is unchanged: a full sweep."""
    from verl.checkpoint_engine.delta_checkpoint_engine import _verify_sample

    monkeypatch.delenv("VERL_DELTA_VERIFY_FRACTION", raising=False)
    names = [f"p{i}" for i in range(50)]
    kept = [n for n, _ in _verify_sample((n, torch.zeros(1)) for n in names)]
    assert kept == names
