"""Phase 1 distributed golden test for delta option 2 (real dist gather-v + reconstruct + native bridge).

Runs the actual sender-side data path across REAL torch.distributed ranks (gloo/CPU, no GPU/cluster):
each rank holds its own mcore shard, byte-diffs it locally, and gathers only the sparse delta to rank0
via the real `gather_v_grouped_to_rank0`; rank0 rebuilds NaN-sentinel shards and runs the native mbridge
`_weight_merge_across_tp` + `_weight_to_hf_format`, then `~isnan` -> HF delta. Asserts bit-identical to
the reference (full shards -> merge -> convert -> byte-diff).

    conda activate astraflow && torchrun --nproc_per_node=2 _runs/phase1_dist_golden.py
    conda activate astraflow && torchrun --nproc_per_node=4 _runs/phase1_dist_golden.py
"""
import os
import torch
import torch.distributed as dist
from types import SimpleNamespace

import importlib.util
from mbridge.core.bridge import Bridge

# load the REAL sharded.py by path (avoid triggering verl package __init__, which needs deps absent here)
_spec = importlib.util.spec_from_file_location(
    "_sharded", os.path.join(os.path.dirname(__file__), "..", "verl", "checkpoint_engine", "delta_sync", "sharded.py")
)
_sharded = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sharded)
gather_v_grouped_to_rank0 = _sharded.gather_v_grouped_to_rank0
shard_delta_indices = _sharded.shard_delta_indices

DT = torch.bfloat16
INT = torch.int16
SENTINEL = float("nan")
H, N_HEADS, N_KV, HEAD_DIM, FFN = 16, 8, 2, 2, 20


def make_stub(tp):
    def name_map(name):
        if "linear_qkv." in name and "layer_norm" not in name:
            return ["q", "k", "v"]
        if "linear_fc1.weight" in name or "linear_fc1.bias" in name:
            return ["gate", "up"]
        return [name]
    return SimpleNamespace(
        mpu=SimpleNamespace(tp_size=tp, etp_size=1),
        hf_config=SimpleNamespace(num_key_value_heads=N_KV, hidden_size=H,
                                  num_attention_heads=N_HEADS, head_dim=HEAD_DIM),
        _weight_name_mapping_mcore_to_hf=name_map,
    )


def split_into_shards(name, g, tp, pdim):
    if "linear_qkv." in name:
        return list(g.chunk(tp, dim=0))
    if "linear_fc1.weight" in name:
        gate, up = g.chunk(2, dim=0)
        gs, us = gate.chunk(tp, dim=0), up.chunk(tp, dim=0)
        return [torch.cat([gs[r], us[r]], dim=0) for r in range(tp)]
    return list(g.chunk(tp, dim=pdim))


def ref_param(pdim):
    p = torch.empty(0); p.tensor_model_parallel = True; p.partition_dim = pdim
    return p


def hf_flat(stub, name, mcore_full):
    ns, ts = Bridge._weight_to_hf_format(stub, name, mcore_full)
    return {n: t.reshape(-1) for n, t in zip(ns, ts)}


def run_case(name, gshape, pdim, tp, rank, seed):
    stub = make_stub(tp)
    rp = ref_param(pdim)
    # deterministic global old/new (identical construction on every rank -- fixed seed, NOT hash())
    g = torch.Generator().manual_seed(seed)
    g_old = torch.randn(*gshape, dtype=DT, generator=g)
    g_new = g_old.clone()
    flatn = g_new.view(-1)
    g2 = torch.Generator().manual_seed(1234)
    pert = torch.randperm(flatn.numel(), generator=g2)[:17]
    flatn[pert] += torch.randn(17, dtype=DT, generator=g2) * 0.5 + 0.1

    shards_old = split_into_shards(name, g_old, tp, pdim)
    shards_new = split_into_shards(name, g_new, tp, pdim)

    # ---- real distributed sender path: local diff on THIS rank's shard, gather-v grouped to rank0 ----
    sn, so = shards_new[rank].reshape(-1), shards_old[rank].reshape(-1)
    idx, val = shard_delta_indices(sn, so, 0)                 # LOCAL coords
    grouped = gather_v_grouped_to_rank0(idx, val)             # real dist all_gather + gather

    if rank != 0:
        return True

    shard_shape = tuple(shards_new[0].shape)
    shard_list = []
    for r in range(tp):
        gi, gv = grouped[r]
        buf = torch.full(shard_shape, SENTINEL, dtype=DT)
        buf.view(-1)[gi] = gv
        shard_list.append(buf)
    merged = Bridge._weight_merge_across_tp(stub, name, shard_list, rp)
    ns, ts = Bridge._weight_to_hf_format(stub, name, merged)
    new = {}
    for hn, ht in zip(ns, ts):
        fl = ht.reshape(-1); m = ~torch.isnan(fl); i = m.nonzero(as_tuple=False).view(-1)
        new[hn] = (i, fl[i])

    # ---- reference: real shards -> native merge/convert -> byte-diff in HF ----
    hf_new = hf_flat(stub, name, Bridge._weight_merge_across_tp(stub, name, shards_new, rp))
    hf_old = hf_flat(stub, name, Bridge._weight_merge_across_tp(stub, name, shards_old, rp))
    ref = {}
    for hn in hf_new:
        m = hf_new[hn].view(INT) != hf_old[hn].view(INT)
        i = m.nonzero(as_tuple=False).view(-1)
        ref[hn] = (i, hf_new[hn][i])

    ok = True
    for hn in set(ref) | set(new):
        ri, rv = ref.get(hn, (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=DT)))
        ni, nv = new.get(hn, (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=DT)))
        rs, nss = torch.argsort(ri), torch.argsort(ni)
        iok = ri.numel() == ni.numel() and torch.equal(ri[rs], ni[nss])
        vok = iok and torch.equal(rv[rs].view(INT), nv[nss].view(INT))
        ok = ok and iok and vok
    print(f"  [{name.split('.')[-2]}:{name.split('.')[-1]} tp={tp}] nnz_hf={sum(v[0].numel() for v in new.values())} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    qkv_rows = N_KV * (HEAD_DIM * N_HEADS // N_KV + 2 * HEAD_DIM)
    cases = [
        ("decoder.layers.0.self_attention.linear_qkv.weight", (qkv_rows, H), 0),
        ("decoder.layers.0.mlp.linear_fc1.weight", (2 * FFN, H), 0),
        ("embedding.word_embeddings.weight", (32, H), 0),
        ("decoder.layers.0.mlp.linear_fc2.weight", (H, FFN), 1),
    ]
    if rank == 0:
        print(f"===== distributed golden, world(tp)={world} =====")
    all_ok = True
    for ci, (name, shape, pdim) in enumerate(cases):
        all_ok = run_case(name, shape, pdim, world, rank, seed=100 + ci) and all_ok
    if rank == 0:
        print("OVERALL:", "ALL PASS ✅" if all_ok else "FAIL ❌")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
