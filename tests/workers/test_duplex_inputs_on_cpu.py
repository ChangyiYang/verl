import torch
from tensordict import TensorDict

from verl.workers.utils.padding import left_right_2_no_padding, rebuild_duplex_inputs_embeds


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embedding


def test_duplex_padding_uses_input_id_jagged_layout():
    batch_size, seq_len, hidden = 2, 5, 4
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.int32)
    data = TensorDict(
        {
            "prompts": torch.zeros(batch_size, 1, dtype=torch.long),
            "responses": torch.zeros(batch_size, seq_len - 1, dtype=torch.long),
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "attention_mask": attention_mask,
            "response_mask": torch.ones(batch_size, seq_len - 1, dtype=torch.int32),
            "position_ids": torch.arange(seq_len).repeat(batch_size, 1),
            "duplex_embeds": torch.randn(batch_size, seq_len, hidden),
            "duplex_token_ids": torch.tensor([[-1, 3, -1, 4, 5], [-1, 6, 7, -1, -1]]),
        },
        batch_size=batch_size,
    )

    packed = left_right_2_no_padding(data)

    assert packed["duplex_embeds"].is_nested
    assert packed["duplex_token_ids"].is_nested
    assert packed["duplex_embeds"].offsets().tolist() == [0, 5, 8]
    assert packed["duplex_token_ids"].values().tolist() == [-1, 3, -1, 4, 5, -1, 6, 7]


def test_rebuild_duplex_embeddings_keeps_audio_and_backprops_tokens():
    module = _TinyLM()
    recorded = torch.randn(5, 4)
    token_ids = torch.tensor([-1, 3, -1, 4, -1])

    mixed = rebuild_duplex_inputs_embeds(recorded, token_ids, module.embedding)

    conditioning = token_ids < 0
    assert torch.equal(mixed[conditioning], recorded[conditioning])
    expected = module.embedding(token_ids[~conditioning])
    assert torch.equal(mixed[~conditioning], expected)

    mixed.sum().backward()
    grad = module.embedding.weight.grad
    assert grad is not None
    assert grad[3].abs().sum() > 0
    assert grad[4].abs().sum() > 0
    assert grad[0].abs().sum() == 0
