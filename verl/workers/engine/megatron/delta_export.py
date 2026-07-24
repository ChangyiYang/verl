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
"""Megatron-side delta export machinery, built on Megatron-Bridge param mappings.

The delta engine consumes final HF-coordinate entries; everything mcore-specific
lives here: enumerating parameters through :meth:`AutoBridge.get_conversion_tasks`,
declaring each parameter's geometry as a :class:`ShardSpec`, and probing the
bridge's own ``megatron_to_hf`` converters with NaN sentinels to translate a
shard-local delta into HF coordinates.

The probe (form B) never runs a collective: every mapping is (shallow-)copied and
its process groups are replaced with this rank's single-rank groups, so
``megatron_to_hf`` short-circuits its TP gather (``tp_size == 1``) and PP
broadcast (``pp_size == 1``) and degrades into a pure permutation transform.
Feeding it a full-logical-shape NaN buffer with the rank's own delta scattered
into its TP slice yields HF tensors whose non-NaN survivors are exactly this
rank's contributions in final HF coordinates.

Scope (asserted in the exporter): TP + EP, PP=1, VPP=1, no LoRA.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import torch

from verl.workers.engine.spec import BlockPlacement, ShardSpec, translate_flat_indices

logger = logging.getLogger(__name__)

_SINGLE_RANK_PGS: Optional[SimpleNamespace] = None


def single_rank_pg_collection() -> SimpleNamespace:
    """This rank's degenerate ProcessGroupCollection: every parallel dim is a
    single-rank group, so any mapping carrying it sees tp/pp/ep/etp size 1 and
    short-circuits its collectives. ``dist.new_group`` must be called by every
    rank in the same order, so ALL ranks build ALL single-rank groups once."""
    global _SINGLE_RANK_PGS
    if _SINGLE_RANK_PGS is None:
        import torch.distributed as dist

        own = None
        for r in range(dist.get_world_size()):
            g = dist.new_group([r])
            if r == dist.get_rank():
                own = g
        # field names are what set_process_groups_from_pg_collection reads:
        # pp -> pp_group, ep -> ep_group, tp -> _tp_group, expt_tp -> _etp_group
        _SINGLE_RANK_PGS = SimpleNamespace(pp=own, ep=own, tp=own, expt_tp=own)
    return _SINGLE_RANK_PGS


def make_probe(mapping):
    """Copy a Megatron-Bridge param mapping and install degenerate single-rank
    process groups on it AND on any nested mapping (composite mappings like
    QKVMapping delegate their TP gather to an inner ``_tp_mapping`` that does
    not receive the outer injection), turning ``megatron_to_hf`` into a pure
    transform with zero collectives."""
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    pgs = single_rank_pg_collection()
    probe = copy.copy(mapping)
    probe.set_process_groups_from_pg_collection(pgs)
    for attr, value in list(vars(probe).items()):
        if isinstance(value, MegatronParamMapping):
            inner = copy.copy(value)
            inner.set_process_groups_from_pg_collection(pgs)
            setattr(probe, attr, inner)
    return probe


@dataclass
class McoreParamExport:
    """One mcore parameter's export record: geometry + probe + module handle."""

    megatron_name: str
    param: torch.Tensor
    spec: ShardSpec
    probe: Any  # degenerate-PG mapping copy (form-B evaluator)
    module: Any  # module handle megatron_to_hf reads config from
    # dim-0 offset of this rank's slice inside the probe's input buffer, per dim
    buffer_offset: tuple


def _mapping_partition_dim(mapping, param) -> Optional[int]:
    """TP partition dim of ``param`` under ``mapping``: mcore convention keeps
    it on the parameter (``tensor_model_parallel``/``partition_dim``); mappings
    that gather (Column/Row/QKV/GatedMLP inner AutoMapping) follow it."""
    if getattr(param, "tensor_model_parallel", False):
        return int(getattr(param, "partition_dim", 0))
    return None


def build_export_index(bridge, megatron_model, hf_path: Optional[str] = None) -> list[McoreParamExport]:
    """Enumerate every local mcore parameter through the bridge's conversion
    tasks and precompute its geometry declaration + form-B probe.

    The index is built once (parameter sets are static) and reused by both the
    shard export and the delta entry hook. Order follows the bridge's task
    enumeration, identical on every rank (lockstep requirement).
    """
    from megatron.core import parallel_state as mpu

    assert mpu.get_pipeline_model_parallel_world_size() == 1, (
        "megatron delta_sharded supports TP+EP only for now (PP=1); the seed full "
        "sync supports PP already, the steady relay is roadmapped"
    )

    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = torch.distributed.get_world_size(group=tp_group)
    tp_rank = torch.distributed.get_rank(group=tp_group)
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank() if ep_size > 1 else 0
    etp_size = mpu.get_expert_tensor_parallel_world_size() if ep_size > 1 else 1
    etp_rank = mpu.get_expert_tensor_parallel_rank() if ep_size > 1 else 0
    etp_ep_group = mpu.get_expert_tensor_and_model_parallel_group() if ep_size > 1 else None

    # hf_path is only needed when the bridge was built from a bare config (tests);
    # the production engine builds it from_hf_pretrained and passes None.
    tasks = (
        bridge.get_conversion_tasks(megatron_model, hf_path=hf_path)
        if hf_path
        else bridge.get_conversion_tasks(megatron_model)
    )
    index: list[McoreParamExport] = []
    for task in tasks:
        mapping = task.mapping
        param = task.param_weight
        name = task.global_param_name
        if param is None:
            # parameter not owned by this rank (pp scope guard keeps this rare)
            continue
        module = task.megatron_module

        is_expert = bool(getattr(mapping, "is_expert", False))
        pdim = _mapping_partition_dim(mapping, param)
        local_shape = tuple(int(x) for x in param.shape)

        if is_expert and ep_size > 1:
            # one LOCAL expert tensor; the virtual logical tensor stacks the ep
            # dim in front: [ep_size, *etp_full_shape]. The probe consumes the
            # ETP-full single-expert shape (its ep gather is short-circuited and
            # handled by the engine gather over etp_ep_group instead).
            etp_full = list(local_shape)
            inner_off = [0] * len(local_shape)
            if etp_size > 1 and pdim is not None:
                etp_full[pdim] *= etp_size
                inner_off[pdim] = etp_rank * local_shape[pdim]
            block = BlockPlacement(
                (1, *local_shape),
                (ep_rank, *inner_off),
                (ep_size, *etp_full),
            )
            spec = ShardSpec(full_shape=(ep_size, *etp_full), place=block, gather_group=etp_ep_group)
            buffer_offset = (ep_rank, *inner_off)
        elif pdim is not None and tp_world > 1:
            full = list(local_shape)
            full[pdim] *= tp_world
            offset = [0] * len(local_shape)
            offset[pdim] = tp_rank * local_shape[pdim]
            block = BlockPlacement(tuple(local_shape), tuple(offset), tuple(full))
            spec = ShardSpec(full_shape=tuple(full), place=block, gather_group=tp_group)
            buffer_offset = tuple(offset)
        else:
            # replicated across TP: single-slot geometry, engine's pg=None path
            # (rank 0 consumes its own entry directly, replicas contribute later
            # via lockstep zero counts -- ReplicatedMapping itself dedups too).
            spec = ShardSpec(full_shape=local_shape)
            buffer_offset = (0,) * len(local_shape)

        index.append(
            McoreParamExport(
                megatron_name=name,
                param=param,
                spec=spec,
                probe=make_probe(mapping),
                module=module,
                buffer_offset=buffer_offset,
            )
        )
    return index


def mcore_hf_delta_entry(rec: McoreParamExport, place, lidx: torch.Tensor, lval: torch.Tensor, slot_cache: dict):
    """Form-B probe: translate one mcore param's shard-local delta into its
    final HF-coordinate entry ``(slots, dtype_str, counts, hf_idx, hf_val)``.

    Scatters the delta into a NaN buffer of the probe's input shape (the LOCAL
    mcore param shape for replicated/expert params, the TP-full shape for TP
    params -- other ranks' slices stay NaN), runs the degenerate-PG
    ``megatron_to_hf`` (pure permutation), and extracts each output slot's
    surviving positions. The slot list is cached after the first call (the
    converter's output names are deterministic, so every rank's cache agrees
    and the batched gather stays aligned)."""
    spec = rec.spec
    dtype_str = str(lval.dtype).replace("torch.", "")

    # The probe consumes the mcore-logical single-param shape. For expert params
    # the engine-facing logical tensor prepends the ep dim; strip it for the
    # probe buffer (dim 0 of the virtual tensor selects the expert, the probe
    # sees one expert's ETP-full tensor).
    is_expert_virtual = isinstance(place, BlockPlacement) and len(place.full_shape) == len(rec.param.shape) + 1
    probe_shape = tuple(spec.full_shape[1:]) if is_expert_virtual else tuple(spec.full_shape)

    buf = torch.full(probe_shape, float("nan"), dtype=lval.dtype, device=lval.device)
    if lidx.numel():
        g = translate_flat_indices(lidx, place)
        if is_expert_virtual:
            inner = 1
            for x in probe_shape:
                inner *= int(x)
            g = g - int(place.global_offset[0]) * inner  # drop the ep-dim offset
        buf.view(-1)[g] = lval

    outs = rec.probe.megatron_to_hf(buf, rec.module)

    key = rec.megatron_name
    slots = slot_cache.get(key)
    if slots is None:
        slots = [(n, tuple(int(x) for x in t.shape)) for n, t in outs.items()]
        slot_cache[key] = slots
    counts = torch.zeros(len(slots), dtype=torch.int64)
    idx_pieces: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    for s_i, (sname, _sshape) in enumerate(slots):
        fl = outs[sname].reshape(-1)
        p_ = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
        if p_.numel():
            counts[s_i] = p_.numel()
            idx_pieces.append(p_.to(torch.int32))
            val_pieces.append(fl[p_])
    if idx_pieces:
        hf_idx = torch.cat(idx_pieces)
        hf_val = torch.cat(val_pieces)
    else:
        hf_idx = torch.empty(0, dtype=torch.int32, device=lval.device)
        hf_val = torch.empty(0, dtype=lval.dtype, device=lval.device)
    return slots, dtype_str, counts, hf_idx, hf_val
