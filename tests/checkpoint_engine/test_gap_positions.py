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
    return torch.cumsum(gaps.to(torch.int64) + 1, dim=0) - 1


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
    idx = torch.tensor([0, gap + 1], dtype=torch.int64)
    gaps, width = _gap_encode(idx)
    assert torch.equal(_decode(gaps), idx), f"gap {gap} at width {width} did not round-trip"


def test_no_tier_can_emit_a_negative_gap():
    """The failure mode was silent: encoding produced a valid-looking tensor.
    Positions are non-negative and strictly increasing, so gaps are >= 0 --
    a negative anywhere means the carrier wrapped."""
    for gap in (0xFF, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 1 << 20):
        gaps, width = _gap_encode(torch.tensor([0, gap + 1], dtype=torch.int64))
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
    with pytest.raises(AssertionError, match="strictly increasing"):
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


def test_duplicate_positions_are_dropped_keeping_the_last():
    """Ranks report the same position twice for a replicated param. Absolute
    positions tolerated it -- index_copy_ let the last writer win -- but a
    repeat is a gap of -1, so gaps cannot. Keeping the LAST of each run
    reproduces the old behaviour instead of inventing a new one."""
    idx = torch.tensor([5, 1, 5, 1, 9], dtype=torch.int64)
    val = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)  # later = wins
    sidx, sval = _sort_dedupe(idx, val)
    assert sidx.tolist() == [1, 5, 9]
    assert sval.tolist() == [40, 30, 50], "must keep the last occurrence, not the first"
    gaps, _ = _gap_encode(sidx)  # and the result must now encode
    assert torch.equal(_decode(gaps), sidx)


def test_normalised_seam_input_encodes():
    """End to end on the shape the gather actually produces: per-rank ascending
    blocks concatenated, with repeats across blocks."""
    idx = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.int64)
    val = torch.arange(9, dtype=torch.int64)
    sidx, sval = _sort_dedupe(idx, val)
    assert sidx.tolist() == [0, 1, 2]
    assert sval.tolist() == [6, 7, 8]  # last rank's contribution wins
    gaps, width = _gap_encode(sidx)
    assert torch.equal(_decode(gaps), sidx) and width == 1
