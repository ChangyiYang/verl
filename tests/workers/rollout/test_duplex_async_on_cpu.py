import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.duplex_window import DuplexWindowRewardManager
from verl.workers.rollout.duplex_rollout import DuplexRollout, pack_duplex_payloads


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


def test_async_duplex_window_reward_stays_on_action_positions():
    batch = TensorDict(
        {
            "responses": torch.zeros(1, 4, dtype=torch.long),
            "duplex_action_response_pos": torch.tensor([[0, 2]]),
            "duplex_is_listen": torch.tensor([[1, 0]], dtype=torch.int32),
            "duplex_frame_time": torch.tensor([[0.0, 1.0]]),
        },
        batch_size=1,
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            "reward_windows": np.asarray(
                [[{"t_start": 0.0, "t_end": 0.0, "value": -1.0, "only_on": "listen"},
                  {"t_start": 1.0, "t_end": 1.0, "value": 1.0, "only_on": "speak"}]],
                dtype=object,
            )
        },
    )
    manager = object.__new__(DuplexWindowRewardManager)

    output = asyncio.run(manager.run_single(data))
    rewards = DuplexWindowRewardManager.assemble_rm_scores(data, [output["reward_score"]])

    assert rewards.tolist() == [[-1.0, 0.0, 1.0, 0.0]]
