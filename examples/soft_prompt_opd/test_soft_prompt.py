#!/usr/bin/env python3
"""Prove the soft prompt is the ONLY thing that trains.

The claim being tested is the premise of the whole program -- the backbone is
frozen -- and it is a claim that fails silently. A run with a broken mask trains
the entire embedding matrix, converges nicely, and reports a perfectly healthy
loss curve. `actor/grad_norm` being smaller than a full-model run is suggestive,
not proof. So: put ordinary token ids in the batch alongside the soft ids and
assert that their rows receive exactly zero gradient.

Runs on CPU in seconds on a randomly-initialised tiny Qwen2 -- the mask logic
does not depend on model size, and depending on a downloaded checkpoint would
make this something nobody runs.

    PYTHONPATH=. python3 examples/soft_prompt_opd/test_soft_prompt.py
"""

import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from verl.utils.soft_prompt import install_soft_prompt, register_grad_mask, reserved_vocab_ids

VOCAB, REGISTERED, N_SOFT = 64, 60, 4


def main() -> int:
    cfg = AutoConfig.for_model(
        "qwen2", vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)

    soft_ids = reserved_vocab_ids(VOCAB, range(REGISTERED), N_SOFT)
    assert soft_ids == [60, 61, 62, 63], soft_ids

    before = model.get_input_embeddings().weight.detach().clone()
    ids = install_soft_prompt(model, soft_ids, generator=torch.Generator().manual_seed(0))
    register_grad_mask(model, ids)

    failures = []

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable != ["model.embed_tokens.weight"]:
        failures.append(f"trainable params {trainable}, expected only the input embedding")

    # Ordinary ids (5, 9, 12, 30) are in the batch on purpose: without the mask
    # their rows would collect gradient too, and that is the failure to catch.
    x = torch.tensor([[60, 61, 62, 63, 5, 9, 12, 30]])
    model(input_ids=x, labels=x).loss.backward()

    grad = model.get_input_embeddings().weight.grad
    nonzero = (grad.abs().sum(dim=-1) > 0).nonzero().flatten().tolist()
    if nonzero != soft_ids:
        failures.append(f"rows with gradient {nonzero}, expected {soft_ids}")

    after = model.get_input_embeddings().weight.detach()
    changed = (after != before).any(dim=-1).nonzero().flatten().tolist()
    if not set(changed) <= set(soft_ids):
        failures.append(f"init touched rows {changed}, outside {soft_ids}")

    # Init copies real vocabulary rows rather than sampling noise, so every soft
    # row must equal some existing row of the original matrix.
    for row in soft_ids:
        if not (before == after[row]).all(dim=-1).any():
            failures.append(f"row {row} does not match any original vocabulary row")

    for f in failures:
        print("FAIL:", f)
    print("PASS: soft prompt is the only thing that trains" if not failures else "FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
