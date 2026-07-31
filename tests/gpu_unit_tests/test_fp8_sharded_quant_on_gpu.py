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
"""A1 harness: sharded two-phase blockwise FP8 == whole-tensor kernel, bitwise.

Splits a tensor into contiguous dim-0 row shards (including offsets that fall
mid-block and a tail partial block), computes each shard's partial absmax grid,
max-combines them (the single-process stand-in for ``all_reduce(MAX)``),
quantizes each shard with the global descales, and compares codes + descales
against ``scaled_fp8_blockwise`` on the whole tensor -- byte for byte.
"""

import pytest
import torch

from verl.utils.fp8_sharded import (
    local_blockwise_absmax,
    quantize_shard_with_descale,
)
from verl.utils.kernel.fp8_kernel import FP8_MAX, scaled_fp8_blockwise

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="triton fp8 kernel needs CUDA")

BLOCK = (128, 128)


def _sharded_reference(full: torch.Tensor, row_splits: list[int]):
    """Run the two-phase scheme with a manual MAX-combine across shards."""
    grids = []
    shards = []
    off = 0
    for rows in row_splits:
        shard = full[off : off + rows]
        grids.append(local_blockwise_absmax(shard, BLOCK, off, tuple(full.shape)))
        shards.append((shard, off))
        off += rows
    assert off == full.shape[0]
    global_grid = torch.stack(grids).amax(dim=0)  # == all_reduce(MAX)
    absmax = global_grid.clamp_(min=1e-10)
    descale = absmax / FP8_MAX
    codes = torch.cat([quantize_shard_with_descale(s, descale, BLOCK, o) for s, o in shards], dim=0)
    return codes, descale


@requires_cuda
@pytest.mark.parametrize(
    "shape,row_splits",
    [
        ((512, 384), [128, 128, 128, 128]),  # block-aligned shards
        ((512, 384), [200, 112, 200]),  # offsets fall mid-block
        ((300, 384), [77, 100, 123]),  # tail partial block rows AND mid-block cuts
        ((512, 200), [256, 256]),  # dim-1 tail partial blocks
    ],
)
def test_sharded_matches_whole_tensor_bitwise(shape, row_splits):
    torch.manual_seed(0)
    full = torch.randn(*shape, dtype=torch.bfloat16, device="cuda") * 3
    ref_codes, ref_descale = scaled_fp8_blockwise(full, list(BLOCK))
    codes, descale = _sharded_reference(full, row_splits)
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8)), "fp8 codes diverge"
    assert torch.equal(descale, ref_descale), "descales diverge"


@requires_cuda
def test_zero_block_eps_path():
    """An all-zero block exercises the kernel's eps floor; sharded must agree."""
    full = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    full[:128, :128] = 0
    ref_codes, ref_descale = scaled_fp8_blockwise(full, list(BLOCK))
    codes, descale = _sharded_reference(full, [100, 156])
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(descale, ref_descale)


@requires_cuda
def test_code_domain_diff_sparsity():
    """The quant-delta premise: sub-quantum bf16 drift leaves codes unchanged.
    Bump EVERY element by one bf16 ulp (bf16 has 8 mantissa bits, e4m3 has 3,
    so a last-place bf16 change stays inside one fp8 quantization step almost
    everywhere) and assert the bf16 delta is fully dense while the code-domain
    delta is (near-)empty."""
    torch.manual_seed(1)
    full = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    codes0, descale0 = _sharded_reference(full, [256])

    bumped = (full.view(torch.int16) + 1).view(torch.bfloat16)
    assert (bumped.view(torch.int16) != full.view(torch.int16)).all(), "bf16 delta should be fully dense"

    codes1, descale1 = _sharded_reference(bumped, [256])
    if torch.equal(descale1, descale0):
        changed = (codes1.view(torch.uint8) != codes0.view(torch.uint8)).float().mean()
        assert changed < 0.2, f"code-domain delta should be sparse, got {changed:.2%}"
