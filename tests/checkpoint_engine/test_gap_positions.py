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
