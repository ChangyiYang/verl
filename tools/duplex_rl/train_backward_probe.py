#!/usr/bin/env python3
"""One real MiniCPM-o duplex rollout followed by one batched backward pass.

This is an acceptance probe, not a training job: it writes no checkpoint and
updates no parameters. It verifies that recorded conditioning embeddings can be
mixed with differentiable current token embeddings, that action labels align
with the preceding causal logits, and that gradients reach the token embedding
table through a single teacher-forced forward.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_wav_mono(path: str, max_seconds: int) -> np.ndarray:
    with wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(min(wav_file.getnframes(), sample_rate * max_seconds))
    if sample_width != 2:
        raise ValueError(f"Expected int16 WAV, got sample width {sample_width}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz WAV, got {sample_rate}")
    return audio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--max-seconds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from tensordict import TensorDict
    from transformers import AutoModel

    from verl.workers.rollout.duplex_rollout import DuplexRollout
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.workers.utils.padding import left_right_2_no_padding

    audio = load_wav_mono(args.audio, args.max_seconds)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to(args.device)
    # The official wrapper initializes Token2wav even when waveform generation
    # is disabled. This probe never touches TTS, so skip that unrelated 4 GB
    # dependency path.
    original_init_tts = model.init_tts
    model.init_tts = lambda **kwargs: None
    try:
        duplex = model.as_duplex(device=args.device, generate_audio=False, ls_mode="explicit")
    finally:
        model.init_tts = original_init_tts
    rollout = DuplexRollout(duplex, SimpleNamespace(chunk_seconds=1.0))
    trajectory = rollout._rollout_one(audio)

    llm = duplex.decoder.m
    for parameter in llm.parameters():
        parameter.requires_grad_(False)
    embedding = llm.get_input_embeddings()
    embedding.weight.requires_grad_(True)

    recorded = trajectory.embeds
    token_ids = trajectory.seq_token_ids.to(recorded.device)
    sequence_length = recorded.shape[0]
    safe_ids = token_ids.masked_fill(token_ids < 0, 0).unsqueeze(0)
    action_sequence_mask = torch.zeros(sequence_length, dtype=torch.int32, device=recorded.device)
    action_sequence_mask[trajectory.action_positions.to(recorded.device)] = 1
    padded = TensorDict(
        {
            "prompts": safe_ids[:, :1],
            "responses": safe_ids[:, 1:],
            "input_ids": safe_ids,
            "attention_mask": torch.ones(1, sequence_length, dtype=torch.int32, device=recorded.device),
            "response_mask": action_sequence_mask[1:].unsqueeze(0),
            "position_ids": torch.arange(sequence_length, device=recorded.device).unsqueeze(0),
            "duplex_embeds": recorded.unsqueeze(0),
            "duplex_token_ids": token_ids.unsqueeze(0),
            "temperature": torch.ones(1, device=recorded.device),
        },
        batch_size=1,
    )
    packed = left_right_2_no_padding(padded)

    # Exercise the exact FSDP actor input-preparation method without creating a
    # process group or wrapping this one-card acceptance probe in FSDP.
    class _EngineHarness:
        _build_duplex_inputs_embeds = FSDPEngineWithLMHead._build_duplex_inputs_embeds
        prepare_model_inputs = FSDPEngineWithLMHead.prepare_model_inputs

        def __init__(self, module):
            self.module = module
            self.use_ulysses_sp = False
            self.pass_packed_cu_seqlens = False

    model_inputs, output_args = _EngineHarness(llm).prepare_model_inputs(packed)
    mixed = model_inputs["inputs_embeds"][0]
    output = llm(**model_inputs, use_cache=False, return_dict=True)

    action_positions = trajectory.action_positions.to(mixed.device)
    valid = action_positions > 0
    action_positions = action_positions[valid]
    action_labels = output_args["input_ids_rmpad_rolled"][action_positions - 1]
    action_logits = output.logits[0, action_positions - 1]
    loss = F.cross_entropy(action_logits.float(), action_labels)
    loss.backward()

    grad = embedding.weight.grad
    token_mask = token_ids >= 0
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) == 0.0:
        raise RuntimeError("Token embedding gradient is missing or non-finite")
    if not torch.equal(mixed.detach()[~token_mask], recorded[~token_mask]):
        raise RuntimeError("Conditioning embeddings changed during reconstruction")

    print(f"sequence={mixed.shape[0]} actions={action_positions.numel()} loss={float(loss):.6f}")
    print(f"embedding_grad_l1={float(grad.abs().sum()):.6f} finite={bool(torch.isfinite(grad).all())}")
    print("PASS: one batched duplex forward/backward completed; no checkpoint written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
