# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Which params get fp8-quantized before being shipped to the rollout.

``should_quantize_param`` is an allowlist of Llama-style names with "do not
quantize" as the default, so a model whose linears are named differently ships
BF16 into fp8 destinations and SGLang rejects it:

    ValueError: Downcasting not allowed:
      target.dtype=torch.float8_e4m3fn, loaded_weight.dtype=torch.bfloat16
    name='model.layers.0.self_attn.wq_b.weight'

That is what DeepSeek-V4 did (run ddp16g). It stayed hidden until a delta was
non-empty, because a param with no changed elements is never written at all.

The expectations below are read off the DeepSeek-V4-Flash-FP8 checkpoint
headers, not inferred from the code. The trap worth remembering: ``wo_a`` is
BF16 while ``wo_b`` is FP8 -- same prefix, opposite dtype.
"""

import pytest

from verl.utils.fp8_utils import FP8QuantizerHelper


@pytest.fixture
def helper():
    return FP8QuantizerHelper.__new__(FP8QuantizerHelper)


# (name, must_quantize) -- DSv4 names in both the SGLang spelling (self_attn)
# and the Megatron-Bridge spelling (attn)
DSV4 = [
    ("model.layers.0.self_attn.wq_a.weight", True),
    ("model.layers.0.self_attn.wq_b.weight", True),
    ("model.layers.0.self_attn.wkv.weight", True),
    ("model.layers.0.self_attn.wo_b.weight", True),
    ("layers.0.ffn.experts.7.w1.weight", True),
    ("layers.0.ffn.experts.7.w2.weight", True),
    ("layers.0.ffn.shared_experts.w3.weight", True),
    # BF16 in the checkpoint -- quantizing these would corrupt them
    ("model.layers.0.self_attn.wo_a.weight", False),
    ("model.layers.0.self_attn.q_norm.weight", False),
    ("model.layers.0.self_attn.kv_norm.weight", False),
    ("layers.0.attn_norm.weight", False),
    ("layers.0.ffn_norm.weight", False),
    ("layers.0.ffn.gate.weight", False),
    ("model.layers.0.self_attn.compressor.wkv.weight", False),
    ("model.embed_tokens.weight", False),
    ("lm_head.weight", False),
]


@pytest.mark.parametrize("name,expected", DSV4)
def test_dsv4_selection_matches_the_checkpoint(helper, name, expected):
    assert helper.should_quantize_param(name) is expected


def test_wo_a_and_wo_b_are_decided_independently(helper):
    """The one that is easy to get wrong with a ".wo_" pattern."""
    assert helper.should_quantize_param("model.layers.0.self_attn.wo_a.weight") is False
    assert helper.should_quantize_param("model.layers.0.self_attn.wo_b.weight") is True


LLAMA = [
    ("model.layers.0.self_attn.q_proj.weight", True),
    ("model.layers.0.self_attn.o_proj.weight", True),
    ("model.layers.0.mlp.down_proj.weight", True),
    ("model.layers.0.input_layernorm.weight", False),
    ("model.layers.0.mlp.gate.weight", False),
    ("model.layers.0.self_attn.q_proj.bias", False),
]


@pytest.mark.parametrize("name,expected", LLAMA)
def test_llama_selection_unchanged(helper, name, expected):
    """The DSv4 additions must not disturb the models this already served."""
    assert helper.should_quantize_param(name) is expected
