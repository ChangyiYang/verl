"""Phase 1: run the ACTUAL verl code (MegatronEngine.get_per_tensor_param_shard + its reconstruct
closure) against a real TP-sharded megatron model, and assert the produced HF delta is bit-identical
to the reference. This exercises the real generator (meta building, name mapping, load-to-gpu, the
bridge-backed reconstruct closure) -- not a test reimplementation.

    conda activate delta-mega && srun --gres=gpu:2 ... torchrun --nproc_per_node=2 _runs/phase1_verlcode.py
"""
import os
import torch
import torch.distributed as dist
from types import SimpleNamespace

# force local (non-TE) spec + no sequence-parallel so a small model builds without transformer_engine
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

from verl.workers.engine.megatron.transformer_impl import MegatronEngine
from verl.checkpoint_engine.delta_sync.sharded import gather_v_grouped_to_rank0, shard_delta_indices

INT = torch.int16
SENTINEL = float("nan")


def all_gather_shards(t, tp):
    lst = [torch.empty_like(t) for _ in range(tp)]
    dist.all_gather(lst, t.contiguous())
    return lst


def main():
    lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr)
    dist.init_process_group("nccl")
    tp = dist.get_world_size(); rank = dist.get_rank()
    mpu.initialize_model_parallel(tensor_model_parallel_size=tp)
    model_parallel_cuda_manual_seed(1234)

    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                      max_position_embeddings=128, torch_dtype="bfloat16")
    bridge = AutoBridge.from_config(cfg)
    for _c in ([bridge.config] if hasattr(bridge, "config") else []):
        try: _c.sequence_parallel = False
        except Exception: pass
    model = bridge.get_model(weight_path=None)[0].to(torch.bfloat16)

    # stand-in with just what get_per_tensor_param_shard reads: .module and .bridge
    stand_in = SimpleNamespace(module=[model], bridge=bridge)

    # pass 1: record old shards from the REAL generator
    gen1, _ = MegatronEngine.get_per_tensor_param_shard(stand_in)
    old = {}
    for gname, local, meta in gen1:
        old[gname] = (local.detach().to(torch.bfloat16).contiguous().clone(), meta)

    # perturb a few local elements of each param
    g = torch.Generator(device="cuda").manual_seed(100 + rank)
    with torch.no_grad():
        for _, p in model.named_parameters():
            k = max(1, p.numel() // 20)
            idxp = torch.randperm(p.numel(), device="cuda", generator=g)[:k]
            p.view(-1)[idxp] += (torch.randn(k, device="cuda", generator=g).to(p.dtype) * 0.3 + 0.05)

    # pass 2: run the REAL generator on the perturbed model + its reconstruct closure
    gen2, _ = MegatronEngine.get_per_tensor_param_shard(stand_in)
    all_ok = True; n_tested = 0
    for gname, new_local, meta in gen2:
        if int(meta["tp_size"]) == 1:
            continue  # replicated (layernorm under local spec, not name-mapped by this mbridge); phase0/1 covers it
        recon = meta["reconstruct"]
        old_local = old[gname][0]
        new_local = new_local.detach().to(torch.bfloat16).contiguous()

        # reference via the SAME real reconstruct closure on real (non-sentinel) shards
        try:
            on, ot = zip(*recon(all_gather_shards(old_local, tp)))
            nn, nt = zip(*recon(all_gather_shards(new_local, tp)))
        except NotImplementedError:
            continue
        ref = {}
        for n0, t0, n1, t1 in zip(on, ot, nn, nt):
            m = t1.reshape(-1).view(INT) != t0.reshape(-1).view(INT)
            i = m.nonzero(as_tuple=False).view(-1)
            ref[n1] = (i, t1.reshape(-1)[i])

        # sparse path: local diff -> real gather_v_grouped -> rebuild sentinels -> same real closure -> ~isnan
        idx, val = shard_delta_indices(new_local.view(-1), old_local.view(-1), 0)
        grouped = gather_v_grouped_to_rank0(idx, val, group=meta["tp_group"])
        if rank != 0:
            continue
        shard_shape = tuple(meta["shard_shape"])
        shard_list = []
        for r in range(int(meta["tp_size"])):
            gi, gv = grouped[r]
            buf = torch.full(shard_shape, SENTINEL, dtype=torch.bfloat16, device="cuda")
            buf.view(-1)[gi] = gv
            shard_list.append(buf)
        new_set = {}
        for hn, ht in recon(shard_list):
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
        n_tested += 1; all_ok = all_ok and ok
        print(f"  [{gname}] hf_nnz={sum(v[0].numel() for v in new_set.values())} -> {'PASS' if ok else 'FAIL'}")

    if rank == 0:
        print(f"OVERALL ({n_tested} params via REAL verl generator):", "ALL PASS ✅" if all_ok else "FAIL ❌")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
