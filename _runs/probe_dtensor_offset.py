import torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate

def main():
    dist.init_process_group("nccl")
    r, w = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r); dev = torch.device("cuda", r)

    # --- API availability ---
    if r == 0:
        try:
            from torch.distributed.tensor._utils import compute_local_shape_and_global_offset as f
            import inspect
            print("[api] compute_local_shape_and_global_offset:", inspect.signature(f))
        except Exception as e:
            print("[api] compute_local_shape_and_global_offset MISSING:", e)

    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset as clsgo

    # --- Case 1: 1D mesh, UNEVEN Shard(0) on a 2D tensor ---
    mesh = init_device_mesh("cuda", (w,))
    R, C = 7, 3                      # 7 rows not divisible by 2 -> uneven
    full = torch.arange(R*C, dtype=torch.float32, device=dev).reshape(R, C)
    dt = distribute_tensor(full, mesh, [Shard(0)])
    loc = dt.to_local()
    lshape, goff = clsgo(dt.shape, dt.device_mesh, dt.placements)
    # verify: the shard sits at rows goff[0] .. goff[0]+loc.shape[0] of the full tensor
    ok = torch.equal(loc, full[goff[0]:goff[0]+loc.shape[0], :]) if loc.numel() else True
    flat_start = goff[0]*C
    print(f"[uneven Shard(0)] rank{r}: local_shape={tuple(loc.shape)} global_offset={goff} "
          f"flat_start={flat_start} placements={dt.placements} verify_slice={ok}")

    # --- Case 2: attributes exposed on the DTensor param ---
    if r == 0:
        print("[attrs] placements:", dt.placements, "| device_mesh:", dt.device_mesh,
              "| has _spec:", hasattr(dt, "_spec"), "| _spec.tensor_meta:", dt._spec.tensor_meta.shape)

    # --- Case 3: Replicate placement (not sharded) ---
    dt_rep = distribute_tensor(full, mesh, [Replicate()])
    lshape2, goff2 = clsgo(dt_rep.shape, dt_rep.device_mesh, dt_rep.placements)
    if r == 0:
        print(f"[replicate] local_shape={lshape2} global_offset={goff2} (offset all-0 + full shape = 每rank整份)")

    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
