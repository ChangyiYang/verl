import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.utils.duplex_prompt import extract_duplex_audio_and_system_prompt
from verl.utils.duplex_reward import score_duplex_windows
from verl.workers.rollout.duplex_rollout import DuplexRollout, DuplexTrajectory, pack_duplex_payloads


class _Resettable:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def reset_streaming(self):
        self.reset_count += 1


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2, bias=False)
        self.register_buffer("scale", torch.ones(1))
        self.audio_past_key_values = object()
        self.reset_count = 0

    def reset_session(self, reset_token2wav_cache=True):
        assert reset_token2wav_cache
        self.reset_count += 1


def _rollout_for_update():
    rollout = object.__new__(DuplexRollout)
    rollout.module = SimpleNamespace(model=_TinyModel(), decoder=_Resettable(), processor=_Resettable())
    rollout.global_steps = 0
    rollout.replica_rank = 0
    rollout._update_in_progress = False
    rollout._weights_valid = True
    rollout._released = False
    return rollout


def _payload(length, hidden, actions):
    return {
        "embeds": torch.randn(length, hidden),
        "seq_token_ids": torch.tensor([-1] + list(range(1, length))),
        "response_mask": torch.ones(length, dtype=torch.int32),
        "action_positions": torch.tensor(actions, dtype=torch.long),
        "frames": [
            {"is_listen": index % 2 == 0, "t_start": float(index)} for index in range(len(actions))
        ],
    }


def test_duplex_prompt_extracts_one_system_message_and_audio():
    audio = np.zeros(160, dtype=np.float32)
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "First instruction."},
                {"type": "text", "text": "Second instruction."},
            ],
        },
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]

    extracted_audio, system_prompt = extract_duplex_audio_and_system_prompt(messages)

    assert extracted_audio is audio
    assert system_prompt == "First instruction.\nSecond instruction."

    extracted_audio, system_prompt = extract_duplex_audio_and_system_prompt(
        [
            {"role": "system", "content": "Single instruction."},
            {"role": "user", "content": [{"type": "audio", "audio": audio}]},
        ]
    )
    assert extracted_audio is audio
    assert system_prompt == "Single instruction."


def test_duplex_prompt_rejects_multiple_system_messages():
    messages = [
        {"role": "system", "content": "First instruction."},
        {"role": "system", "content": "Second instruction."},
        {"role": "user", "content": [{"type": "audio", "audio": np.zeros(160)}]},
    ]

    with pytest.raises(ValueError, match="at most one system message"):
        extract_duplex_audio_and_system_prompt(messages)


def test_pack_duplex_payloads_uses_response_coordinates():
    batch = pack_duplex_payloads([_payload(5, 3, [2]), _payload(4, 3, [1, 3])])

    assert batch["duplex_action_pos"].tolist() == [[2, -1], [1, 3]]
    assert batch["duplex_action_response_pos"].tolist() == [[1, -1], [0, 2]]
    assert batch["response_mask"].tolist() == [[0, 1, 0, 0], [1, 0, 1, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]


def test_duplex_full_weight_update_commits_version_and_resets_cache():
    rollout = _rollout_for_update()
    target = rollout.module.model.state_dict()
    weights = [(f"_fsdp_wrapped_module.{name}", torch.full_like(value, 7)) for name, value in target.items()]

    asyncio.run(rollout.update_weights(iter(weights), global_steps=3))

    assert rollout.global_steps == 3
    assert rollout._weights_valid
    assert all(torch.equal(value, torch.full_like(value, 7)) for value in rollout.module.model.state_dict().values())
    assert rollout.module.model.reset_count == 1
    assert rollout.module.decoder.reset_count == 1
    assert rollout.module.processor.reset_count == 1


def test_duplex_weight_update_failure_stays_fail_closed():
    rollout = _rollout_for_update()
    name, value = next(iter(rollout.module.model.state_dict().items()))

    with pytest.raises(RuntimeError, match="Missing"):
        asyncio.run(rollout.update_weights(iter([(name, value.clone())]), global_steps=1))

    assert not rollout._weights_valid
    assert rollout.global_steps == 0


def test_duplex_weight_update_rejects_shape_mismatch():
    rollout = _rollout_for_update()
    name, value = next(
        (name, value) for name, value in rollout.module.model.state_dict().items() if value.numel() > 1
    )

    with pytest.raises(RuntimeError, match="Shape mismatch"):
        asyncio.run(rollout.update_weights(iter([(name, value.reshape(-1)[:1])]), global_steps=1))

    assert not rollout._weights_valid


def test_duplex_generate_threads_normalized_sampling_params():
    rollout = object.__new__(DuplexRollout)
    rollout.listen_id = 7
    rollout.speak_id = 8
    rollout.replica_rank = 0
    rollout.global_steps = 0
    rollout._update_in_progress = False
    rollout._weights_valid = True
    rollout._released = False
    rollout._last_audit_payload = None
    captured = {}

    def fake_rollout_one(wav, system_prompt, force_first_action_id=None, sampling_params=None):
        captured.update(sampling_params)
        return DuplexTrajectory(
            frames=[],
            token_ids=[],
            embeds=torch.empty(0, 3),
            seq_token_ids=torch.empty(0, dtype=torch.long),
            response_mask=torch.empty(0, dtype=torch.int32),
            action_positions=torch.empty(0, dtype=torch.long),
        )

    rollout._rollout_one = fake_rollout_one
    asyncio.run(
        rollout.generate(
            request_id="request",
            prompt_ids=[],
            sampling_params={"temperature": 0.0, "top_k": -1, "top_p": 1.0},
            audio_data=[np.zeros(16000, dtype=np.float32)],
        )
    )

    assert captured == {"decode_mode": "greedy", "temperature": 1.0, "top_k": 0, "top_p": 1.0}


def test_duplex_sampling_params_fail_closed_instead_of_being_ignored():
    with pytest.raises(ValueError, match="Unsupported duplex sampling params"):
        DuplexRollout._normalize_sampling_params(
            {"temperature": 1.0, "top_k": -1, "top_p": 1.0, "repetition_penalty": 1.1}
        )


def test_async_duplex_window_reward_stays_on_action_positions():
    reward, hit_count = score_duplex_windows(
        responses=torch.zeros(4, dtype=torch.long),
        action_response_pos=torch.tensor([0, 2]),
        is_listen=torch.tensor([1, 0], dtype=torch.int32),
        frame_time=torch.tensor([0.0, 1.0]),
        windows=[
            {"t_start": 0.0, "t_end": 0.0, "value": -1.0, "only_on": "listen"},
            {"t_start": 1.0, "t_end": 1.0, "value": 1.0, "only_on": "speak"},
        ],
    )

    assert reward.tolist() == [-1.0, 0.0, 1.0, 0.0]
    assert hit_count == 2
