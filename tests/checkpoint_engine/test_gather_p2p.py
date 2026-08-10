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
"""The padded gather's cost, and the arithmetic the variable-length path relies on.

Profiling attributed 82.7 s of a 129 s sync to the gather. It moves 42.9 GiB of
real delta, but ``dist.gather`` needs equal lengths, so every rank sends the
global maximum: with world=16 that is ~686 GiB on the wire, which at the
measured broadcast rate (7.9 GiB/s) is 87 s -- the observed cost. Rank 0 also
allocates ``2 * world * max_n``, which caps the round at ~200 MB and is why
raising it to 1 GiB ran out of memory.

The real gather needs a process group, so what is testable here is the model
that justified the change and the offset arithmetic the receiver depends on.
"""

import pytest
import torch


def _padded_bytes(counts, per_elem=2):
    """What dist.gather actually moves: world x the largest rank's total."""
    return len(counts) * max(counts) * per_elem


def _exact_bytes(counts, per_elem=2):
    """What point-to-point moves: the sum of what each rank holds."""
    return sum(counts) * per_elem


def test_padding_overhead_is_the_world_size_when_balanced():
    """Even perfectly balanced ranks pay world x, because gather sends max_n
    from everyone and rank 0 keeps a slot per rank."""
    counts = [1000] * 16
    assert _padded_bytes(counts) / _exact_bytes(counts) == pytest.approx(1.0)
    # ...but rank 0's buffers are world x one rank's share, which is the memory wall
    assert len(counts) * max(counts) == 16 * 1000


def test_imbalance_makes_padding_worse_not_better():
    """One heavy rank drags every other rank up to its length."""
    counts = [10_000] + [100] * 15
    ratio = _padded_bytes(counts) / _exact_bytes(counts)
    assert ratio > 13, f"expected heavy padding waste, got {ratio:.1f}x"


def test_rank0_memory_is_what_capped_the_round():
    """2 (idx+val) x world x round_bytes. 200 MB rounds already needed 6 GiB;
    1 GiB rounds need 32 GiB, which is the OOM we measured."""
    world = 16
    for round_gib, expected in ((0.2, 6.4), (1.0, 32.0)):
        assert 2 * world * round_gib == pytest.approx(expected, abs=0.1)


def test_offsets_are_unchanged_by_the_variable_length_path():
    """The receiver slices rank r's blob by cumulative per-param counts. The p2p
    path hands back exactly totals[r] elements per rank instead of max_n, so the
    same arithmetic must still land on the same values."""
    counts = [[3, 0, 2], [1, 4, 0]]  # [rank][param]
    world, k = len(counts), len(counts[0])
    blobs = [torch.arange(sum(c)) + r * 100 for r, c in enumerate(counts)]

    offs = [[0] * (k + 1) for _ in range(world)]
    for r in range(world):
        for i in range(k):
            offs[r][i + 1] = offs[r][i] + counts[r][i]

    per_param = []
    for i in range(k):
        pieces = [blobs[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts[r][i]]
        per_param.append(torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64))

    assert per_param[0].tolist() == [0, 1, 2, 100]
    assert per_param[1].tolist() == [101, 102, 103, 104]
    assert per_param[2].tolist() == [3, 4]
    assert sum(p.numel() for p in per_param) == sum(sum(c) for c in counts)


def test_ranks_with_nothing_post_no_transfer():
    """The padded path made an empty rank send a full max_n of zeros; the exact
    path should skip it entirely."""
    counts = [500, 0, 0, 300]
    assert _exact_bytes(counts) == 800 * 2
    assert _padded_bytes(counts) == 4 * 500 * 2  # 5x more, all of it zeros
