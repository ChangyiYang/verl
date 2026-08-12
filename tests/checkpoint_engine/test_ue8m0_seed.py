"""The seed must reproduce a ue8m0 checkpoint's bytes, not merely approximate them.

The ladder pinned the failure (B5, 2026-08-12): under the plain amax/FP8_MAX
scale every quantized pair (fp8 codes AND scale_inv) differed from the disk
state, while every bf16 pass-through matched. The mechanism is the round trip:
a power-of-two scale shifts exponents losslessly, an arbitrary real rewrites
mantissas. These tests pin the dialect switch end to end at the pure-function
level: formula, stream output, config plumbing, and the round-trip invariant
that is the whole point.
"""

import torch

from verl.utils.fp8_sharded import FP8_MAX, QuantSpec, quantize_hf_stream, ue8m0_descale
from verl.utils.sglang.sglang_fp8_utils import build_sglang_fp8_quant_config


def _spec(scale_fmt=None):
    return QuantSpec(
        weight_block_size=(128, 128),
        should_quantize=lambda n: n.endswith(".weight"),
        scale_fmt=scale_fmt,
    )


def _stream(spec, t):
    return dict(quantize_hf_stream(iter([("x.weight", t)]), spec))


def test_ue8m0_descale_is_power_of_two_and_matches_the_converter():
    amax = torch.rand(64, 64) * 100 + 1e-6
    d = ue8m0_descale(amax)
    log = torch.log2(d)
    assert torch.equal(log, log.round()), "ue8m0 descale must be a power of two"
    # the DSv4 nccl converter's literal expression (sglang_fp8_utils.py) --
    # the two paths must be byte-identical or the ladder's cross-path
    # comparison can never close
    ref = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-10) / FP8_MAX)))
    assert torch.equal(d, ref)


def test_stream_emits_ue8m0_scales_when_asked():
    t = torch.randn(256, 256, dtype=torch.bfloat16)
    out = _stream(_spec("ue8m0"), t)
    log = torch.log2(out["x.weight_scale_inv"])
    assert torch.equal(log, log.round())


def test_stream_default_formula_is_unchanged():
    """Non-ue8m0 checkpoints (every model validated before DSv4) keep their bytes."""
    t = torch.randn(256, 256, dtype=torch.bfloat16)
    legacy = _stream(QuantSpec(weight_block_size=(128, 128), should_quantize=lambda n: True), t)
    explicit = _stream(_spec(None), t)
    assert torch.equal(
        legacy["x.weight_scale_inv"], explicit["x.weight_scale_inv"]
    ) and torch.equal(
        legacy["x.weight"].view(torch.uint8), explicit["x.weight"].view(torch.uint8)
    )


def test_ue8m0_round_trip_is_bit_exact():
    """quantize(dequant(codes, scales)) must reproduce codes AND scales bitwise.

    This is the property that makes seed == disk possible at all: the trainer
    holds dequant(ckpt) in bf16 and the seed re-quantizes it. Deliberately NOT
    asserted for the plain formula -- it does not hold there, which is the bug.
    """
    spec = _spec("ue8m0")
    t = torch.randn(256, 256, dtype=torch.bfloat16)
    first = _stream(spec, t)
    codes, descale = first["x.weight"], first["x.weight_scale_inv"]
    # dequantize per 128x128 block, as the trainer-side bridge does
    up = codes.to(torch.float32).reshape(2, 128, 2, 128) * descale.reshape(2, 1, 2, 1)
    second = _stream(spec, up.reshape(256, 256).to(torch.bfloat16))
    assert torch.equal(second["x.weight_scale_inv"], descale), "scales must survive the round trip"
    assert torch.equal(
        second["x.weight"].view(torch.uint8), codes.view(torch.uint8)
    ), "fp8 codes must survive the round trip"


def test_build_config_preserves_scale_fmt():
    cfg = build_sglang_fp8_quant_config({"quantization_config": {"weight_block_size": [128, 128], "scale_fmt": "ue8m0"}})
    assert cfg.get("scale_fmt") == "ue8m0", f"scale_fmt dropped again: {cfg}"
    cfg2 = build_sglang_fp8_quant_config({"quantization_config": {"weight_block_size": [128, 128]}})
    assert "scale_fmt" not in cfg2
