# Copyright 2026 Liquid AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Agent-loop manager for frame-synchronous MiniCPM-o duplex trajectories."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.duplex_window import score_duplex_windows
from verl.utils.duplex_prompt import extract_duplex_audio_and_system_prompt
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.duplex_rollout import pack_duplex_payloads


def _load_mono_16khz(audio: Any) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        waveform, sample_rate = audio, 16000
    elif isinstance(audio, torch.Tensor):
        waveform, sample_rate = audio.detach().cpu().numpy(), 16000
    elif isinstance(audio, dict) and "array" in audio:
        waveform = np.asarray(audio["array"])
        sample_rate = int(audio.get("sampling_rate", audio.get("sample_rate", 16000)))
    elif isinstance(audio, (str, os.PathLike)):
        import soundfile as sf

        waveform, sample_rate = sf.read(os.fspath(audio), dtype="float32", always_2d=False)
    else:
        raise TypeError(f"Unsupported duplex audio payload: {type(audio).__name__}")

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise ValueError(f"Duplex audio must be mono or channel-last, got shape {waveform.shape}")
    if sample_rate != 16000:
        from scipy.signal import resample_poly

        from math import gcd

        divisor = gcd(sample_rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, sample_rate // divisor).astype(np.float32)
    return waveform


class DuplexAgentLoopManager:
    """Direct-Ray async manager that preserves mixed embedding trajectories."""

    def __init__(self, config, llm_client, teacher_client=None, reward_loop_worker_handles=None):
        del teacher_client, reward_loop_worker_handles
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.llm_client = llm_client

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    async def _generate_one(self, messages, priority: int, batch_index: int):
        audio, system_prompt = extract_duplex_audio_and_system_prompt(messages)
        if audio is None:
            raise ValueError("Duplex prompt contains no audio item")
        waveform = _load_mono_16khz(audio)
        request_id = f"duplex-{priority}-{uuid4().hex}"
        start = time.perf_counter()
        extra_kwargs = {}
        if self.rollout_config.duplex.get("counterfactual_first_actions", False):
            extra_kwargs["force_first_action"] = "listen" if batch_index % 2 == 0 else "speak"
        output = await self.llm_client.generate(
            request_id=request_id,
            prompt_ids=[],
            sampling_params={
                "temperature": self.rollout_config.temperature,
                "top_k": self.rollout_config.top_k,
                "top_p": self.rollout_config.top_p,
            },
            audio_data=[waveform],
            system_prompt=system_prompt,
            **extra_kwargs,
        )
        return output, time.perf_counter() - start

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        raw_prompts = prompts.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise KeyError("DuplexAgentLoopManager requires non_tensor_batch['raw_prompt']")
        priorities = prompts.non_tensor_batch.get("priority", np.arange(len(raw_prompts)))
        results = await asyncio.gather(
            *(
                self._generate_one(messages, int(priority), batch_index)
                for batch_index, (messages, priority) in enumerate(zip(raw_prompts, priorities))
            )
        )
        outputs, durations = zip(*results, strict=True)
        payloads = [output.extra_fields["duplex_trajectory"] for output in outputs]
        versions = np.asarray([int(output.extra_fields["global_steps"]) for output in outputs], dtype=np.int64)
        if int(versions.min()) != int(versions.max()):
            raise RuntimeError(f"Duplex rollout batch mixed policy versions: {versions.tolist()}")
        trainer_step = prompts.meta_info.get("global_steps")
        if trainer_step is not None:
            expected_version = int(trainer_step) - 1
            if int(versions[0]) != expected_version:
                raise RuntimeError(
                    f"Stale duplex rollout policy: trainer_step={trainer_step}, "
                    f"expected_version={expected_version}, got={int(versions[0])}"
                )
            print(
                f"[duplex] trainer_step={int(trainer_step)} rollout_policy_version={int(versions[0])} "
                f"batch={len(outputs)}"
            )

        packed = pack_duplex_payloads(payloads)
        reward_windows = prompts.non_tensor_batch.get("reward_windows")
        if reward_windows is None:
            raise KeyError("DuplexAgentLoopManager requires non_tensor_batch['reward_windows']")
        scores = []
        reward_hits = []
        for batch_index, windows in enumerate(reward_windows):
            score, hit_count = score_duplex_windows(
                responses=packed[batch_index]["responses"],
                action_response_pos=packed[batch_index]["duplex_action_response_pos"],
                is_listen=packed[batch_index]["duplex_is_listen"],
                frame_time=packed[batch_index]["duplex_frame_time"],
                windows=windows,
            )
            scores.append(score)
            reward_hits.append(hit_count)
        packed["rm_scores"] = torch.stack(scores)

        non_tensor_batch = {
            # ray_trainer expects this key while collecting optional image stats.
            "multi_modal_inputs": np.asarray([{} for _ in outputs], dtype=object),
            "global_steps": versions,
            "min_global_steps": versions.copy(),
            "max_global_steps": versions.copy(),
            "duplex_reward_hits": np.asarray(reward_hits, dtype=np.int64),
        }
        timing = {
            "agent_loop/generate_sequences/min": float(min(durations)),
            "agent_loop/generate_sequences/max": float(max(durations)),
            "agent_loop/generate_sequences/mean": float(np.mean(durations)),
        }
        return DataProto(
            batch=packed,
            non_tensor_batch=non_tensor_batch,
            meta_info={"timing": timing, "duplex": True, "reward_extra_keys": ["duplex_reward_hits"]},
        )


__all__ = ["DuplexAgentLoopManager", "_load_mono_16khz"]
