"""Feasibility spike (task #2): sharded-snapshot delta diff + gather-v vs full-gather delta.

Proves the core algorithm for "shard the pinned snapshot, diff on each rank's local
FSDP shard, gather only the changed elements to rank 0" is bit-identical to today's
"all-gather the full tensor then diff" path — and measures the gather-volume reduction.

Run (single node, N GPUs):
    torchrun --nproc_per_node=4 _runs/spike_sharded_delta.py
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor


def gather_v_to_rank0(idx: torch.Tensor, val: torch.Tensor, world: int, rank: int, dev):
    """Gather variable-length (idx, val) from all ranks to rank 0.

    Real impl would exchange nnz counts then do a NCCL gather-v; here we use
    all_gather_object for the correctness spike (logical volume measured separately).
    """
    payload = (idx.cpu(), val.cpu())  # bf16 val (uint16-view breaks pickling)
    bucket = [None] * world
    dist.all_gather_object(bucket, payload)
    if rank != 0:
        return None, None
    idxs = torch.cat([b[0] for b in bucket])
    vals = torch.cat([b[1] for b in bucket])
    return idxs, vals


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world,))

    # Same full tensors on every rank (same seed) -> consistent sharding.
    N = 8_000_000
    torch.manual_seed(0)
    full_old = torch.randn(N, dtype=torch.bfloat16, device=dev)
    full_new = full_old.clone()
    g = torch.Generator(device=dev).manual_seed(123)
    k = N // 100  # ~1% sparse change
    pert_idx = torch.randint(0, N, (k,), device=dev, generator=g)
    full_new[pert_idx] = full_new[pert_idx] + (
        torch.randn(k, dtype=torch.bfloat16, device=dev, generator=g) * 0.1
    )

    # Shard both like FSDP2 (flat Shard(0)).
    dt_old = distribute_tensor(full_old, mesh, [Shard(0)])
    dt_new = distribute_tensor(full_new, mesh, [Shard(0)])
    loc_old = dt_old.to_local().contiguous().view(-1)
    loc_new = dt_new.to_local().contiguous().view(-1)

    # ---- NEW path: per-shard diff + global offset + gather-v ----
    lens = [torch.zeros(1, dtype=torch.long, device=dev) for _ in range(world)]
    dist.all_gather(lens, torch.tensor([loc_old.numel()], device=dev))
    lens = [int(x.item()) for x in lens]
    offset = sum(lens[:rank])

    mask = loc_new.view(torch.uint16) != loc_old.view(torch.uint16)
    local_idx = mask.nonzero(as_tuple=False).view(-1)
    local_val = loc_new[local_idx]
    global_idx = local_idx.to(torch.int64) + offset
    local_nnz = int(local_idx.numel())

    tot_nnz = torch.tensor([local_nnz], device=dev)
    dist.all_reduce(tot_nnz)
    sh_idx, sh_val = gather_v_to_rank0(global_idx, local_val, world, rank, dev)

    # ---- BASELINE path: all-gather full, then diff ----
    full_old_r = dt_old.full_tensor().contiguous().view(-1)
    full_new_r = dt_new.full_tensor().contiguous().view(-1)
    base_mask = full_new_r.view(torch.uint16) != full_old_r.view(torch.uint16)
    base_idx = base_mask.nonzero(as_tuple=False).view(-1).to(torch.int64)
    base_val = full_new_r[base_idx].cpu()  # bf16

    if rank == 0:
        # sort both by index; compare index (int64) and value (raw uint16 bytes)
        so = torch.argsort(sh_idx)
        sh_idx_s, sh_val_s = sh_idx[so], sh_val[so]
        base_idx_c = base_idx.cpu()
        bo = torch.argsort(base_idx_c)
        base_idx_s, base_val_s = base_idx_c[bo], base_val[bo]

        idx_match = torch.equal(sh_idx_s, base_idx_s)
        val_match = torch.equal(sh_val_s.view(torch.uint16), base_val_s.view(torch.uint16))
        full_bytes = N * 2  # bf16 full-tensor all-gather materialized on rank0
        delta_bytes = int(tot_nnz.item()) * 2

        print("=" * 60)
        print(f"[spike] world={world}  N={N}  local_shard≈{lens[0]}")
        print(f"[spike] baseline full-diff nnz = {base_idx.numel()}")
        print(f"[spike] sharded  gathered nnz = {sh_idx.numel()}  (all-reduce nnz={int(tot_nnz.item())})")
        print(f"[spike] INDEX bit-identical : {idx_match}")
        print(f"[spike] VALUE bit-identical : {val_match}")
        print(f"[spike] RESULT: {'PASS ✅' if (idx_match and val_match) else 'FAIL ❌'}")
        print(f"[spike] gather volume: full all-gather {full_bytes/1e6:.1f} MB  ->  "
              f"sharded gather-v {delta_bytes/1e6:.3f} MB  ({full_bytes/max(delta_bytes,1):.0f}x less)")
        print("=" * 60)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
