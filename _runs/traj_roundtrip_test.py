"""Trajectory-level delta-sync correctness check against the CURRENT PR code.

Real sglang Engine + the PR's custom-weight-loader path (update_weights_from_tensor
with load_format = verl loader), no mocks:

  1. Engine loads Qwen2.5-0.5B; greedy-generate BASELINE texts.
  2. Trainer side seeds DeltaState from the same HF weights (no flush emitted).
  3. Perturb a few weights mildly -> delta flushes -> apply into the live engine.
     Generate MID texts: must be coherent (no garbage), may differ from baseline.
  4. Revert weights to the originals -> delta flushes -> apply.
     Generate FINAL texts: must be BYTE-IDENTICAL to baseline (greedy) -- proves the
     sparse apply lands bit-exactly on the live TP worker weights.
"""
import json
import os

import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "The capital of France is",
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Answer:",
    "def fibonacci(n):",
]


def log(*a):
    print("[TRAJ]", *a, flush=True)


def gen_all(engine):
    outs = []
    for p in PROMPTS:
        o = engine.generate(p, {"temperature": 0.0, "max_new_tokens": 32})
        outs.append(o["text"] if isinstance(o, dict) else o[0]["text"])
    return outs


def flush_to_named(flush):
    spec = {
        "encoding": flush.encoding,
        "params": [vars(p) for p in flush.params],
        "checksum": int(flush.checksum),
    }
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8).cuda()
    return [
        ("__delta_spec__", spec_t),
        ("__positions__", flush.positions_cpu.cuda()),
        ("__values__", flush.values_gpu.cuda()),
    ]


def apply_flushes(engine, flushes, tag):
    total = 0
    for i, fl in enumerate(flushes):
        engine.update_weights_from_tensor(
            named_tensors=flush_to_named(fl),
            load_format="verl.checkpoint_engine.delta_sync.sglang_loader.apply_delta",
            flush_cache=(i == len(flushes) - 1),
        )
        total += int(fl.values_gpu.numel())
    log(f"{tag}: applied {len(flushes)} flushes, nnz={total}")


def main():
    import sglang as sgl
    from transformers import AutoModelForCausalLM

    from verl.checkpoint_engine.delta_sync import DeltaState, iter_delta_flushes
    from verl.checkpoint_engine.delta_sync.sglang_loader import LOADER_FQN

    log("launching sglang engine ...")
    engine = sgl.Engine(
        model_path=MODEL, dtype="bfloat16", mem_fraction_static=0.6,
        disable_cuda_graph=True, attention_backend="flashinfer", log_level="warning",
        custom_weight_loader=[LOADER_FQN],
    )
    base = gen_all(engine)
    for t in base:
        log("BASE:", repr(t))

    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    orig = {n: p.detach().clone() for n, p in hf.named_parameters()}
    named = [(n, p.detach()) for n, p in hf.named_parameters()]

    state = DeltaState()
    seed = list(iter_delta_flushes((x for x in named), state, encoding="indices", bucket_bytes=1 << 26))
    assert seed == [], f"seed emitted {len(seed)} flushes, expected 0"
    log("state seeded,", len(named), "tensors")

    # Mild, output-affecting perturbation on a few tensors (like one RL step, but larger).
    torch.manual_seed(0)
    with torch.no_grad():
        for tname in ["model.layers.10.mlp.down_proj.weight",
                      "model.layers.20.self_attn.o_proj.weight",
                      "model.embed_tokens.weight"]:
            t = dict(hf.named_parameters())[tname]
            mask = torch.rand_like(t, dtype=torch.float32) < 0.02  # ~2% of elements
            t[mask] += torch.randn_like(t)[mask] * 0.02

    flushes = list(iter_delta_flushes((x for x in named), state, encoding="indices", bucket_bytes=1 << 26))
    apply_flushes(engine, flushes, "PERTURB")
    mid = gen_all(engine)
    for t in mid:
        log("MID: ", repr(t))

    # Revert to the original weights -> engine must return to baseline bit-exactly.
    with torch.no_grad():
        for n, p in hf.named_parameters():
            p.copy_(orig[n])
    flushes = list(iter_delta_flushes((x for x in named), state, encoding="indices", bucket_bytes=1 << 26))
    apply_flushes(engine, flushes, "REVERT")
    fin = gen_all(engine)
    for t in fin:
        log("FINAL:", repr(t))

    ok = fin == base
    garbled = any(len(t.strip()) == 0 for t in mid)
    print("RESULT roundtrip_identical =", ok)
    print("RESULT mid_state_nonempty =", not garbled)
    engine.shutdown()
    assert ok, "FINAL generations differ from BASE -- delta apply is NOT bit-exact!"


if __name__ == "__main__":
    main()
