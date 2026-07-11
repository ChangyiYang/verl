"""Phase 0 de-risk for delta option 2 (sparse mcore delta -> rank0 reconstruct -> native merge/convert).

Core claim to validate, OFFLINE (single process, simulate TP), CPU bf16, using the REAL mbridge
methods `_weight_merge_across_tp` + `_weight_to_hf_format`:

  reconstruct per-rank NaN-sentinel shards from a LOCAL byte-diff, run them through the native
  merge+convert, then `~isnan` -> (hf_name, pos, val)
      ==  (bit-identical)  ==
  run the REAL shards through merge+convert to full HF tensors, then byte-diff vs the converted snapshot.

If this holds for QKV(GQA) / gate_up / plain-col / plain-row, the whole option-2 design is sound and
everything downstream is plumbing.

    conda activate astraflow && python _runs/phase0_bitident.py
"""
import torch
from types import SimpleNamespace

from mbridge.core.bridge import Bridge

torch.manual_seed(0)
DT = torch.bfloat16
SENTINEL = torch.tensor(float("nan"), dtype=DT)  # NaN sentinel for "unchanged"
INT = torch.int16  # bf16 is 2 bytes -> view as int16 for a bytewise compare


# ---- tiny model config (GQA) ----
H = 16          # hidden
N_HEADS = 8
N_KV = 2
HEAD_DIM = 2    # note H // N_HEADS = 2, consistent
FFN = 20        # per gate/up rows


def make_stub(tp_size):
    """Minimal `self` so we can call Bridge's unbound layout methods faithfully."""
    def name_map(name):
        if "linear_qkv." in name and "layer_norm" not in name:
            return ["q", "k", "v"]
        if "linear_fc1.weight" in name or "linear_fc1.bias" in name:
            return ["gate", "up"]
        return [name]

    return SimpleNamespace(
        mpu=SimpleNamespace(tp_size=tp_size, etp_size=1),
        hf_config=SimpleNamespace(
            num_key_value_heads=N_KV, hidden_size=H,
            num_attention_heads=N_HEADS, head_dim=HEAD_DIM,
        ),
        _weight_name_mapping_mcore_to_hf=name_map,
    )


def merge(stub, name, shard_list, ref):
    return Bridge._weight_merge_across_tp(stub, name, shard_list, ref)


def to_hf(stub, name, merged):
    return Bridge._weight_to_hf_format(stub, name, merged)


def split_into_shards(name, global_w, tp_size, partition_dim):
    """Split a global mcore tensor into the per-rank shards that merge() would reassemble."""
    if "linear_qkv." in name:                       # merge = cat(dim=0)
        return list(global_w.chunk(tp_size, dim=0))
    if "linear_fc1.weight" in name:                 # merge = per-rank [gate_r; up_r]
        gate, up = global_w.chunk(2, dim=0)         # global = [all_gate; all_up]
        gates = gate.chunk(tp_size, dim=0)
        ups = up.chunk(tp_size, dim=0)
        return [torch.cat([gates[r], ups[r]], dim=0) for r in range(tp_size)]
    return list(global_w.chunk(tp_size, dim=partition_dim))  # generic cat(dim=partition_dim)


def ref_param(partition_dim):
    p = torch.empty(0)
    p.tensor_model_parallel = True
    p.partition_dim = partition_dim
    return p


def hf_set(stub, name, mcore_full):
    """Reference: convert a full mcore tensor to HF and return {hf_name: flat tensor}."""
    names, tensors = to_hf(stub, name, mcore_full)
    return {n: t.reshape(-1) for n, t in zip(names, tensors)}


def run_case(name, global_shape, partition_dim, tp_size=2, n_changed=17):
    stub = make_stub(tp_size)

    # snapshot (old) and current (new = old with n_changed random elements perturbed)
    g_old = torch.randn(*global_shape, dtype=DT)
    g_new = g_old.clone()
    flat = g_new.view(-1)
    pert = torch.randperm(flat.numel())[:n_changed]
    flat[pert] += (torch.randn(n_changed, dtype=DT) * 0.5 + 0.1)
    # guard: sentinel must not collide with a real changed value
    assert not torch.isnan(flat[pert]).any()

    shards_old = split_into_shards(name, g_old, tp_size, partition_dim)
    shards_new = split_into_shards(name, g_new, tp_size, partition_dim)
    rp = ref_param(partition_dim)

    # ---------- REFERENCE path: real shards -> merge -> convert -> byte-diff in HF space ----------
    hf_old = hf_set(stub, name, merge(stub, name, shards_old, rp))
    hf_new = hf_set(stub, name, merge(stub, name, shards_new, rp))
    ref = {}
    for hn in hf_new:
        mask = hf_new[hn].view(INT) != hf_old[hn].view(INT)
        idx = mask.nonzero(as_tuple=False).view(-1)
        ref[hn] = (idx, hf_new[hn][idx])

    # ---------- NEW path: local diff -> reconstruct NaN-sentinel shards -> native merge/convert -> ~isnan
    sentinels = []
    for r in range(tp_size):
        sn, so = shards_new[r].reshape(-1), shards_old[r].reshape(-1)
        chg = (sn.view(INT) != so.view(INT)).nonzero(as_tuple=False).view(-1)  # LOCAL coords
        buf = torch.full_like(shards_new[r], SENTINEL).reshape(-1)
        buf[chg] = sn[chg]                                   # scatter real new values, rest NaN
        sentinels.append(buf.view_as(shards_new[r]))
    merged_sent = merge(stub, name, sentinels, rp)
    names_s, tensors_s = to_hf(stub, name, merged_sent)
    new = {}
    for hn, ht in zip(names_s, tensors_s):
        flat = ht.reshape(-1)
        mask = ~torch.isnan(flat)
        idx = mask.nonzero(as_tuple=False).view(-1)
        new[hn] = (idx, flat[idx])

    # ---------- compare (bit-identical, order-insensitive) ----------
    ok = True
    details = []
    keys = set(ref) | set(new)
    for hn in keys:
        ri, rv = ref.get(hn, (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=DT)))
        ni, nv = new.get(hn, (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=DT)))
        rs = torch.argsort(ri); ns = torch.argsort(ni)
        idx_ok = ri.numel() == ni.numel() and torch.equal(ri[rs], ni[ns])
        val_ok = idx_ok and torch.equal(rv[rs].view(INT), nv[ns].view(INT))
        ok = ok and idx_ok and val_ok
        details.append(f"    {hn}: nnz ref={ri.numel()} new={ni.numel()} idx={idx_ok} val={val_ok}")
    print(f"[{name.split('.')[-2]}:{name.split('.')[-1]} shape={tuple(global_shape)} pdim={partition_dim} tp={tp_size}] "
          f"-> {'PASS' if ok else 'FAIL'}")
    for d in details:
        print(d)
    return ok


def main():
    qkv_rows = N_KV * (HEAD_DIM * N_HEADS // N_KV + 2 * HEAD_DIM)  # = q + k + v rows
    cases = [
        # (name, global_shape, partition_dim)
        ("decoder.layers.0.self_attention.linear_qkv.weight", (qkv_rows, H), 0),   # fused QKV (GQA)
        ("decoder.layers.0.mlp.linear_fc1.weight",            (2 * FFN, H), 0),    # fused gate_up
        ("embedding.word_embeddings.weight",                  (30, H), 0),         # plain column-parallel
        ("decoder.layers.0.mlp.linear_fc2.weight",            (H, FFN), 1),        # plain row-parallel
    ]
    all_ok = True
    for tp in (2, 4):
        print(f"===== TP={tp} =====")
        for name, shape, pdim in cases:
            # skip shapes not divisible by tp for a clean even split
            if name.endswith("linear_fc2.weight"):
                if shape[1] % tp:
                    continue
            elif "linear_fc1" in name:
                if (shape[0] // 2) % tp:
                    continue
            elif shape[0] % tp:
                continue
            all_ok = run_case(name, shape, pdim, tp_size=tp) and all_ok
    print("=" * 50)
    print("OVERALL:", "ALL PASS ✅" if all_ok else "FAIL ❌")
    print("=" * 50)


if __name__ == "__main__":
    main()
