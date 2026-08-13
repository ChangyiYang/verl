#!/usr/bin/env python3
"""Terminal duplex-RL acceptance case: real audio through one GRPO update.

Two trajectories share the same real PCM and first-action state. Their first LS
action is counterfactually forced to LISTEN vs SPEAK, producing a deterministic
G=2 group. WindowRewardManager assigns -1/+1, the actor recomputes old logprobs
from mixed inputs_embeds, and one LoRA PPO/GRPO step must increase P(SPEAK).

No checkpoint is created.
"""

from __future__ import annotations

import argparse
import wave
from types import SimpleNamespace

import numpy as np


def load_wav_mono(path: str, max_seconds: int) -> np.ndarray:
    with wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        raw = wav_file.readframes(min(wav_file.getnframes(), sample_rate * max_seconds))
    if sample_rate != 16000 or width != 2:
        raise ValueError(f"Expected 16 kHz int16 WAV, got sr={sample_rate}, width={width}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio.reshape(-1, channels).mean(1) if channels > 1 else audio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--max-seconds", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from tensordict import TensorDict
    from transformers import AutoModel

    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.workers.reward_manager.window import RewardWindow, WindowRewardManager
    from verl.workers.rollout.duplex_rollout import DuplexRollout
    from verl.workers.utils.padding import left_right_2_no_padding

    audio = load_wav_mono(args.audio, args.max_seconds)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to("cuda")
    original_init_tts = model.init_tts
    model.init_tts = lambda **kwargs: None
    try:
        duplex = model.as_duplex(device="cuda", generate_audio=False, ls_mode="explicit")
    finally:
        model.init_tts = original_init_tts
    rollout = DuplexRollout(duplex, SimpleNamespace(chunk_seconds=1.0))
    listen_id, speak_id = rollout.listen_id, rollout.speak_id

    def forced_trajectory(action_id: int):
        decoder = duplex.decoder
        original_decode = decoder.decode
        state = {"first": True}

        def force_first(logits, *positional, **kwargs):
            if state["first"]:
                state["first"] = False
                return torch.tensor([action_id], dtype=torch.long, device=logits.device)
            return original_decode(logits, *positional, **kwargs)

        decoder.decode = force_first
        try:
            return rollout._rollout_one(audio)
        finally:
            decoder.decode = original_decode

    trajectories = [forced_trajectory(listen_id), forced_trajectory(speak_id)]
    batch_size = len(trajectories)
    max_length = max(int(item.embeds.shape[0]) for item in trajectories)
    hidden_size = int(trajectories[0].embeds.shape[-1])
    device, dtype = trajectories[0].embeds.device, trajectories[0].embeds.dtype

    embeds = torch.zeros(batch_size, max_length, hidden_size, device=device, dtype=dtype)
    token_ids = torch.full((batch_size, max_length), -1, device=device, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, device=device, dtype=torch.int32)
    position_ids = torch.zeros(batch_size, max_length, device=device, dtype=torch.long)
    action_pos = torch.full((batch_size, 1), -1, device=device, dtype=torch.long)
    action_response_pos = torch.full_like(action_pos, -1)
    is_listen = torch.zeros(batch_size, 1, device=device, dtype=torch.int32)
    frame_time = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)

    for batch_index, trajectory in enumerate(trajectories):
        length = int(trajectory.embeds.shape[0])
        first_action = int(trajectory.action_positions[0])
        embeds[batch_index, :length] = trajectory.embeds
        token_ids[batch_index, :length] = trajectory.seq_token_ids.to(device)
        attention_mask[batch_index, :length] = 1
        position_ids[batch_index, :length] = torch.arange(length, device=device)
        action_pos[batch_index, 0] = first_action
        action_response_pos[batch_index, 0] = first_action - 1
        is_listen[batch_index, 0] = int(trajectory.frames[0].is_listen)

    # A counterfactual group must share the exact state at which the action is
    # chosen. This also guards the per-trajectory audio KV-cache reset in
    # DuplexRollout: a leaked cache makes these prefixes diverge sharply.
    if int(action_pos[0, 0]) != int(action_pos[1, 0]):
        raise RuntimeError(f"First action positions differ: {action_pos[:, 0].tolist()}")
    branch_position = int(action_pos[0, 0])
    prefix_gap = float((embeds[0, :branch_position].float() - embeds[1, :branch_position].float()).abs().max())
    if prefix_gap > 1e-3:
        raise RuntimeError(f"Counterfactual prefixes differ before LS branch: {prefix_gap}")
    if not torch.equal(token_ids[0, :branch_position], token_ids[1, :branch_position]):
        raise RuntimeError("Counterfactual prefix token ids differ before LS branch")

    safe_ids = token_ids.masked_fill(token_ids < 0, 0)
    response_mask = torch.zeros(batch_size, max_length - 1, device=device, dtype=torch.int32)
    response_mask.scatter_(1, action_response_pos, 1)
    batch = TensorDict(
        {
            "prompts": safe_ids[:, :1],
            "responses": safe_ids[:, 1:],
            "input_ids": safe_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "duplex_embeds": embeds,
            "duplex_token_ids": token_ids,
            "duplex_action_pos": action_pos,
            "duplex_action_response_pos": action_response_pos,
            "duplex_is_listen": is_listen,
            "duplex_frame_time": frame_time,
        },
        batch_size=batch_size,
    )
    windows = [
        [
            RewardWindow(0.0, 0.0, 1.0, "trigger", only_on="speak"),
            RewardWindow(0.0, 0.0, -1.0, "miss", only_on="listen"),
        ]
        for _ in range(batch_size)
    ]
    proto = DataProto(batch=batch, non_tensor_batch={"reward_windows": np.asarray(windows, dtype=object)})
    rewards = WindowRewardManager()(proto)
    group_rewards = rewards.gather(1, action_response_pos).squeeze(1)
    advantages = (group_rewards - group_rewards.mean()) / group_rewards.std(unbiased=False).clamp_min(1e-6)
    if not torch.equal(group_rewards.cpu(), torch.tensor([-1.0, 1.0])):
        raise RuntimeError(f"Unexpected window rewards: {group_rewards.tolist()}")

    actor = get_peft_model(
        duplex.decoder.m,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    actor.train()
    trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)

    packed = left_right_2_no_padding(batch.clone())
    packed["temperature"] = torch.ones(batch_size, device=device)
    tu.assign_non_tensor_data(packed, "use_remove_padding", False)

    class _EngineHarness:
        _build_duplex_inputs_embeds = FSDPEngineWithLMHead._build_duplex_inputs_embeds
        prepare_model_inputs = FSDPEngineWithLMHead.prepare_model_inputs

        def __init__(self, module):
            self.module = module
            self.use_ulysses_sp = False
            self.pass_packed_cu_seqlens = False

    engine = _EngineHarness(actor)

    def action_log_probs():
        model_inputs, _ = engine.prepare_model_inputs(packed)
        output = actor(**model_inputs, use_cache=False, return_dict=True)
        rows = torch.stack(
            [output.logits[index, int(action_pos[index, 0]) - 1] for index in range(batch_size)]
        ).float()
        pair = F.log_softmax(rows[:, [listen_id, speak_id]], dim=-1)
        chosen_index = torch.tensor([0, 1], device=device)
        return pair, pair.gather(1, chosen_index[:, None]).squeeze(1)

    with torch.no_grad():
        before_pair, old_log_probs = action_log_probs()
    initial_state_gap = float((before_pair[0] - before_pair[1]).abs().max())
    if initial_state_gap > 1e-3:
        raise RuntimeError(f"Counterfactual first-action states differ: {initial_state_gap}")

    optimizer.zero_grad(set_to_none=True)
    _, current_log_probs = action_log_probs()
    ratios = torch.exp(current_log_probs - old_log_probs)
    unclipped = ratios * advantages
    clipped = ratios.clamp(0.8, 1.2) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    loss.backward()
    grad_norm = torch.sqrt(
        sum(parameter.grad.float().square().sum() for parameter in trainable if parameter.grad is not None)
    )
    if not torch.isfinite(grad_norm) or float(grad_norm) == 0.0:
        raise RuntimeError(f"Invalid LoRA gradient norm: {float(grad_norm)}")
    before_parameters = [parameter.detach().clone() for parameter in trainable]
    optimizer.step()
    parameters_changed = any(
        not torch.equal(before, after.detach()) for before, after in zip(before_parameters, trainable, strict=True)
    )

    with torch.no_grad():
        after_pair, _ = action_log_probs()
    delta_speak = float((after_pair[:, 1] - before_pair[:, 1]).mean())
    delta_listen = float((after_pair[:, 0] - before_pair[:, 0]).mean())

    checks = {
        "step0_ratio_is_one": bool(torch.allclose(ratios.detach(), torch.ones_like(ratios))),
        "finite_loss": bool(torch.isfinite(loss)),
        "trainable_params_changed": parameters_changed,
        "speak_logprob_increased": delta_speak > 0.0,
        "listen_logprob_decreased": delta_listen < 0.0,
    }
    print(f"rewards={group_rewards.tolist()} advantages={advantages.tolist()}")
    print(f"step0_ratios={ratios.detach().tolist()} loss={float(loss):.6f} grad_norm={float(grad_norm):.6f}")
    print(f"delta_logp_speak={delta_speak:+.6f} delta_logp_listen={delta_listen:+.6f}")
    print(f"checks={checks}")
    if not all(checks.values()):
        raise RuntimeError(f"E2E acceptance failed: {checks}")
    print("PASS: PCM -> duplex rollout -> window reward -> GRPO update -> correct policy direction")
    print("No checkpoint written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
