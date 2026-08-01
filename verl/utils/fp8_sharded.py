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
"""Sharded blockwise FP8 quantization: bitwise-identical to the whole-tensor
kernel, computed shard-locally with one collective.

The rollout side quantizes whole HF tensors (``scaled_fp8_blockwise``:
per-block absmax -> descale = absmax / FP8_MAX -> codes = clamp(x / descale)).
A trainer rank holds only a slice of the tensor, but the block grid -- and
therefore every scale -- is defined on the FULL tensor. The sharded scheme
splits the kernel at its natural seam:

1. every rank computes a PARTIAL absmax grid over its own rows, laid out on
   the GLOBAL block grid (zeros where the shard does not overlap a block);
2. one ``all_reduce(MAX)`` over the gather group turns partials into the
   global grid (tiny: 4 bytes per 128x128 block);
3. every rank quantizes its own rows locally with the global descales,
   replicating the kernel's exact fp32 op order (absmax/FP8_MAX, then
   1/descale, multiply, clamp, cast) so codes and descales match the
   whole-tensor kernel BIT FOR BIT.

Scope: dim-0 contiguous row shards (FSDP ``Shard(0)``; the mcore block case
rides the same helpers per touched block). Row offsets may fall mid-block --
partial overlaps contribute partial maxima, exactly like the kernel's padding
mask contributes zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from verl.utils.kernel.fp8_kernel import FP8_DTYPE, FP8_MAX, ceil_div

# matches the triton kernel's numerical-stability floor for block absmax
_ABSMAX_EPS = 1e-10


def local_blockwise_absmax(
    shard: torch.Tensor,
    weight_block_size: list[int] | tuple[int, int],
    row_offset: int,
    full_shape: tuple[int, int],
) -> torch.Tensor:
    """Partial per-block absmax of a dim-0 row shard, on the GLOBAL block grid.

    Returns a float32 ``(ceil(M/BM), ceil(N/BN))`` grid; blocks the shard does
    not overlap hold 0 (abs values are >= 0, so ``all_reduce(MAX)`` composes
    partials correctly).
    """
    bm, bn = int(weight_block_size[0]), int(weight_block_size[1])
    m_full, n_full = int(full_shape[0]), int(full_shape[1])
    rows, cols = shard.shape
    assert cols == n_full, f"row shard must span full dim-1: {cols} != {n_full}"
    n_br, n_bc = ceil_div(m_full, bm), ceil_div(n_full, bn)
    grid = torch.zeros(n_br, n_bc, dtype=torch.float32, device=shard.device)
    if rows == 0:
        return grid

    x = shard.to(torch.float32).abs()
    # NaN placeholders (mcore probe output: positions owned by OTHER ranks)
    # must not poison the partial max; zeros never win a legitimate max.
    x = torch.nan_to_num(x, nan=0.0)
    # pad dim-1 to the block grid once (zeros never win a max)
    pad_n = n_bc * bn - n_full
    if pad_n:
        x = torch.nn.functional.pad(x, (0, pad_n))
    first_block = row_offset // bm
    r = 0
    for br in range(first_block, ceil_div(row_offset + rows, bm)):
        take = min((br + 1) * bm - (row_offset + r), rows - r)
        seg = x[r : r + take]
        grid[br] = seg.view(take, n_bc, bn).amax(dim=(0, 2))
        r += take
    return grid


def quantize_shard_with_descale(
    shard: torch.Tensor,
    descale: torch.Tensor,
    weight_block_size: list[int] | tuple[int, int],
    row_offset: int,
) -> torch.Tensor:
    """Quantize a dim-0 row shard using GLOBAL per-block descales, replicating
    the kernel's fp32 op order (``s_inv = 1.0 / descale``; ``clamp(x * s_inv)``;
    cast) so the codes are bitwise-identical to the whole-tensor kernel."""
    bm, bn = int(weight_block_size[0]), int(weight_block_size[1])
    rows, cols = shard.shape
    n_bc = descale.shape[1]
    s_inv = 1.0 / descale  # matches the kernel's second fp32 division

    # NaN passes through multiply/clamp/cast untouched and lands as the fp8
    # NaN byte -- exactly the wire sentinel for "not this rank's position".
    x = shard.to(torch.float32)
    pad_n = n_bc * bn - cols
    if pad_n:
        x = torch.nn.functional.pad(x, (0, pad_n))
    # per-row block-row index -> expand descale rows to shard rows
    br_of_row = (torch.arange(row_offset, row_offset + rows, device=shard.device) // bm) - (row_offset // bm)
    first_block = row_offset // bm
    s_rows = s_inv[first_block + br_of_row]  # (rows, n_bc)
    x = x.view(rows, n_bc, bn)
    x = x * s_rows.unsqueeze(-1)
    x = x.clamp_(min=-FP8_MAX, max=FP8_MAX).to(FP8_DTYPE)
    x = x.view(rows, n_bc * bn)
    if pad_n:
        x = x[:, :cols].contiguous()
    return x


def sharded_scaled_fp8_blockwise(
    shard: torch.Tensor,
    weight_block_size: list[int] | tuple[int, int],
    row_offset: int,
    full_shape: tuple[int, int],
    group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-phase sharded quantization: partial absmax -> all_reduce(MAX) ->
    local quantize. Returns ``(codes, descale)`` where ``codes`` covers this
    rank's rows and ``descale`` is the full global grid (identical on every
    rank after the reduce)."""
    import torch.distributed as dist

    grid = local_blockwise_absmax(shard, weight_block_size, row_offset, full_shape)
    if group is not None:
        dist.all_reduce(grid, op=dist.ReduceOp.MAX, group=group)
    absmax = grid.clamp_(min=_ABSMAX_EPS)  # kernel: maximum(max|x|, eps)
    descale = absmax / FP8_MAX  # kernel: x_s = absmax / fp8_max
    codes = quantize_shard_with_descale(shard, descale, weight_block_size, row_offset)
    return codes, descale


@dataclass
class QuantSpec:
    """Rollout-format request handed to a backend's ``get_per_tensor_param``.

    Deliberately rollout-agnostic: the caller (checkpoint engine) distills the
    serving engine's quantization config into a block shape plus a per-param
    predicate; the backend only honors the spec and never sees who asked.
    """

    weight_block_size: tuple[int, int]
    should_quantize: object  # Callable[[str], bool]


def quantize_hf_stream(weights, spec: QuantSpec):
    """Wrap a full HF ``(name, tensor)`` export with blockwise fp8 quantization:
    for every 2D weight the spec selects, yield ``(name, codes)`` +
    ``(name_scale_inv, descales)``; everything else passes through in bf16.
    Whole-tensor path (``group=None``) -- bitwise-identical to the sharded
    steady quantizer, which matters because fp32->fp8 tie rounding is
    implementation-sensitive across kernels.
    """
    block = list(spec.weight_block_size)
    for name, t in weights:
        if t.dim() != 2 or not spec.should_quantize(name):
            yield name, t
            continue
        codes, descale = sharded_scaled_fp8_blockwise(t.to(torch.bfloat16), block, 0, tuple(t.shape), group=None)
        yield name, codes
        yield name + "_scale_inv", descale
