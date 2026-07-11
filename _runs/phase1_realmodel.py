"""Phase 1 REAL-model validation: run the option-2 sender path against an actual TP-sharded megatron model.

Builds a real (small) Qwen2 mcore model with TP via mbridge, so every param has its REAL megatron name,
`.tensor_model_parallel`/`.partition_dim` attrs, and REAL fused QKV / gate_up layout. Then, for each
TP-sharded / fused param, runs the option-2 sender path (local byte-diff on this rank's real shard ->
real dist gather_v_grouped -> rank0 rebuild NaN-sentinel shards -> the model's OWN bound bridge
_weight_merge_across_tp + _weight_to_hf_format -> ~isnan) and asserts bit-identical to the reference
(all-gather the real shards -> same bridge merge/convert -> byte-diff).

Standalone layernorm params are skipped: with the local (non-TE) spec used here they aren't fused into
linear_qkv, so mbridge 0.1.0 can't name-map them (they're the trivial replicated/identity case anyway,
already covered by phase0/1).

    srun --gres=gpu:2 ... torchrun --nproc_per_node=2 _runs/phase1_realmodel.py
"""
import os
import importlib.util
import torch
import torch.distributed as dist

# force local (non-TE) layer spec so the model builds without transformer_engine
import mbridge.core.llm_bridge as lb
_orig = lb.get_gpt_decoder_block_spec
def _patched(config, **kw):
    kw["use_transformer_engine"] = False
    return _orig(config, **kw)
_patched.__signature__ = __import__("inspect").signature(_orig)
lb.get_gpt_decoder_block_spec = _patched

from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from transformers import Qwen2Config
from mbridge import AutoBridge

# real gather primitive + diff, loaded by path (avoid verl package init)
_spec = importlib.util.spec_from_file_location(
    "_sharded", os.path.join(os.path.dirname(__file__), "..", "verl", "checkpoint_engine", "delta_sync", "sharded.py")
)
_sharded = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sharded)
gather_v_grouped_to_rank0 = _sharded.gather_v_grouped_to_rank0
shard_delta_indices = _sharded.shard_delta_indices

INT = torch.int16  # bf16 -> 2 bytes
SENTINEL = float("nan")


def main():
    lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr)
    dist.init_process_group("nccl")
    tp = dist.get_world_size()
    rank = dist.get_rank()
    mpu.initialize_model_parallel(tensor_model_parallel_size=tp)
    model_parallel_cuda_manual_seed(1234)

    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                      max_position_embeddings=128, torch_dtype="bfloat16")
    bridge = AutoBridge.from_config(cfg)
    for _c in ([bridge.config] if hasattr(bridge,'config') else []):
        try: _c.sequence_parallel=False
        except Exception: pass
    model = bridge.get_model(weight_path=None)[0]

    # collect params, cast to bf16, snapshot as "old", then perturb a few to make "new"
    params = {}
    for lname, p in model.named_parameters():
        gname = bridge._weight_name_mapping_mcore_local_to_global(model).get(lname, lname)
        params[gname] = p
    g = torch.Generator(device="cuda").manual_seed(100 + rank)

    all_ok = True
    n_tested = 0
    for gname, p in params.items():
        is_tp = bool(getattr(p, "tensor_model_parallel", False)) and tp > 1
        if not is_tp:
            continue  # replicated params (layernorms): trivial + not name-mapped under local spec
        pref = p  # ref param carries .partition_dim for the generic merge branch
        old = p.detach().to(torch.bfloat16).contiguous()
        new = old.clone()
        # perturb a few local elements
        k = max(1, new.numel() // 20)
        idxp = torch.randperm(new.numel(), device="cuda", generator=g)[:k]
        new.view(-1)[idxp] += (torch.randn(k, dtype=torch.bfloat16, device="cuda", generator=g) * 0.3 + 0.05)

        # ---- reference: all_gather real shards -> bridge merge/convert -> byte-diff ----
        def gather_shards(t):
            lst = [torch.empty_like(t) for _ in range(tp)]
            dist.all_gather(lst, t.contiguous())
            return lst
        try:
            on, ot = bridge._weight_to_hf_format(gname, bridge._weight_merge_across_tp(gname, gather_shards(old), pref))
            nn, nt = bridge._weight_to_hf_format(gname, bridge._weight_merge_across_tp(gname, gather_shards(new), pref))
        except NotImplementedError:
            continue  # bridge can't map this name under local spec -> skip
        ref = {}
        for n0, t0, n1, t1 in zip(on, ot, nn, nt):
            m = t1.reshape(-1).view(INT) != t0.reshape(-1).view(INT)
            i = m.nonzero(as_tuple=False).view(-1)
            ref[n1] = (i, t1.reshape(-1)[i])

        # ---- option-2 path: local diff -> gather_v_grouped -> rank0 rebuild -> native merge/convert ----
        sn, so = new.reshape(-1), old.reshape(-1)
        idx, val = shard_delta_indices(sn, so, 0)
        grouped = gather_v_grouped_to_rank0(idx, val)
        if rank == 0:
            shard_shape = tuple(old.shape)
            shard_list = []
            for r in range(tp):
                gi, gv = grouped[r]
                buf = torch.full(shard_shape, SENTINEL, dtype=torch.bfloat16, device="cuda")
                buf.view(-1)[gi] = gv
                shard_list.append(buf)
            merged = bridge._weight_merge_across_tp(gname, shard_list, pref)
            new_set = {}
            hns, hts = bridge._weight_to_hf_format(gname, merged)
            for hn, ht in zip(hns, hts):
                fl = ht.reshape(-1); mm = ~torch.isnan(fl); ii = mm.nonzero(as_tuple=False).view(-1)
                new_set[hn] = (ii, fl[ii])

            ok = True
            for hn in set(ref) | set(new_set):
                ri, rv = ref.get(hn, (torch.empty(0, device="cuda", dtype=torch.long), torch.empty(0, device="cuda", dtype=torch.bfloat16)))
                ni, nv = new_set.get(hn, (torch.empty(0, device="cuda", dtype=torch.long), torch.empty(0, device="cuda", dtype=torch.bfloat16)))
                rs, ns = torch.argsort(ri), torch.argsort(ni)
                iok = ri.numel() == ni.numel() and torch.equal(ri[rs], ni[ns])
                vok = iok and torch.equal(rv[rs].view(INT), nv[ns].view(INT))
                ok = ok and iok and vok
            n_tested += 1
            all_ok = all_ok and ok
            print(f"  [{gname}] hf_nnz={sum(v[0].numel() for v in new_set.values())} -> {'PASS' if ok else 'FAIL'}")

    if rank == 0:
        print(f"OVERALL ({n_tested} real TP params tested):", "ALL PASS ✅" if all_ok else "FAIL ❌")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
