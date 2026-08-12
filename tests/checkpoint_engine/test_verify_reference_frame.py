"""The verify sweep must compare the state the server actually serves with.

The first successful full sweep reported 4.34M mismatched elements: 8,481
tensors each fully wrong, every one of them an fp8 scale. The cause was the
reference frame, not the delta. ``post_load_weights`` recomputes derived tensors
("fp8 scales after requant", MLA w_kc/w_vc, MoE biases), the normal apply path
calls it and the verify path returned before it -- so the sweep was comparing the
trainer's TRANSPORTED scale against the server's RECOMPUTED one, and
weight_scale_inv/scale being reciprocals makes that differ almost by definition.

These tests pin the fix and, more importantly, pin that the check can still FAIL.
Making the mismatch disappear by excluding derived tensors would have been the
easy fix and the wrong one.
"""

import pytest
import torch

from verl.workers.rollout.sglang_rollout.delta_loader import _VERIFY_STATS, _model_state_hashes


class FakeModel(torch.nn.Module):
    """Model whose post_load_weights recomputes a derived tensor, like sglang's.

    Carries an fp8 buffer on purpose. The first version of these tests used only
    float32 and passed, while the real run died with
    ``NotImplementedError: "xor_sum_cuda" not implemented for 'Float8_e4m3fn'`` --
    the model is mostly fp8, which is the entire point of this project. A fixture
    whose dtypes do not match the system under test is not a test.
    """

    def __init__(self, recompute=True, drift=False):
        super().__init__()
        self.w = torch.nn.Parameter(torch.arange(16, dtype=torch.float32), requires_grad=False)
        self.register_buffer("w_scale_inv", torch.zeros(4))
        self.register_buffer("codes", torch.arange(8, dtype=torch.uint8).view(torch.float8_e4m3fn))
        self.recompute = recompute
        self.drift = drift
        self._calls = 0

    def post_load_weights(self):
        self._calls += 1
        if not self.recompute:
            return
        # derived from the current weights -- deterministic unless drift is set
        base = self.w.detach().reshape(4, 4).abs().amax(dim=1)
        self.w_scale_inv.copy_(base + (self._calls if self.drift else 0))


@pytest.fixture(autouse=True)
def _clean_stats():
    _VERIFY_STATS.pop("baseline", None)
    _VERIFY_STATS["params"] = 0
    yield
    _VERIFY_STATS.pop("baseline", None)
    _VERIFY_STATS["params"] = 0


def test_hashes_cover_parameters_and_buffers():
    """A check that skipped buffers would miss the scales entirely."""
    m = FakeModel()
    h = _model_state_hashes(m)
    assert "w" in h and "w_scale_inv" in h


def test_hashes_change_when_a_tensor_changes():
    """If the hash cannot notice a change, the whole sweep is decorative."""
    m = FakeModel()
    before = _model_state_hashes(m)
    with torch.no_grad():
        m.w[3] += 1.0
    after = _model_state_hashes(m)
    assert before["w"] != after["w"]


def test_hashes_are_stable_for_an_unchanged_tensor():
    m = FakeModel()
    assert _model_state_hashes(m)["w"] == _model_state_hashes(m)["w"]


def test_deterministic_recompute_is_not_flagged():
    """The actual fix: post_load_weights on BOTH sides, so a derived tensor that
    recomputes to the same value is identical, not a mismatch."""
    m = FakeModel(recompute=True)
    m.post_load_weights()
    before = _model_state_hashes(m)
    m.post_load_weights()  # replay: same weights -> same derived value
    after = _model_state_hashes(m)
    assert before == after, "a deterministic recompute must not read as a mismatch"


def test_a_drifting_derived_tensor_is_still_caught():
    """Guard against 'fixing' this by ignoring derived tensors: if the recompute
    is NOT idempotent, the sweep must still notice."""
    m = FakeModel(recompute=True, drift=True)
    m.post_load_weights()
    before = _model_state_hashes(m)
    m.post_load_weights()
    after = _model_state_hashes(m)
    changed = [k for k in before if before[k] != after[k]]
    assert "w_scale_inv" in changed, "a non-idempotent derived tensor must still be caught"


def test_a_real_weight_change_is_caught():
    m = FakeModel()
    before = _model_state_hashes(m)
    with torch.no_grad():
        m.w[0] = 999.0
    after = _model_state_hashes(m)
    assert [k for k in before if before[k] != after[k]] == ["w"]


def test_fp8_tensors_can_be_hashed():
    """The regression that cost a 45-minute run: hash_tensor has no fp8 kernel."""
    m = FakeModel()
    h = _model_state_hashes(m)
    assert "codes" in h, "an fp8 buffer must be hashable, not skipped"


def test_fp8_change_is_detected():
    m = FakeModel()
    before = _model_state_hashes(m)
    with torch.no_grad():
        m.codes.view(torch.uint8)[2] = 77
    after = _model_state_hashes(m)
    assert before["codes"] != after["codes"]


def test_non_contiguous_tensor_is_hashable():
    """A transposed / sliced view would raise on .view(uint8) without the guard."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("t", torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t())

    assert "t" in _model_state_hashes(M())


# --- the shape/dtype zoo -------------------------------------------------------
# Two runs died here, on fp8 and then on a 0-dim scalar. Guessing which awkward
# cases exist has now failed twice, so this fixture carries every combination the
# real model can present and asserts the hash survives ALL of them.


class ZooModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scalar", torch.tensor(3.0))                       # 0-dim
        self.register_buffer("scalar_int", torch.tensor(7, dtype=torch.int64))  # 0-dim int
        self.register_buffer("fp8", torch.arange(8, dtype=torch.uint8).view(torch.float8_e4m3fn))
        self.register_buffer("bf16", torch.arange(6, dtype=torch.bfloat16))
        self.register_buffer("u8", torch.arange(5, dtype=torch.uint8))
        self.register_buffer("transposed", torch.arange(12, dtype=torch.float32).reshape(3, 4).t())
        self.register_buffer("empty", torch.zeros(0))
        self.register_buffer("nan", torch.tensor([float("nan"), 1.0]))
        self.w = torch.nn.Parameter(torch.zeros(4), requires_grad=False)


def test_every_shape_and_dtype_hashes():
    h = _model_state_hashes(ZooModel())
    for k in ("scalar", "scalar_int", "fp8", "bf16", "u8", "transposed", "nan", "w"):
        assert k in h, f"{k} was not hashed -- it would be invisible to the sweep"
    assert "empty" not in h, "empty tensors carry no state and are skipped by design"


def test_zero_dim_change_is_detected():
    """The exact shape that killed run 3."""
    m = ZooModel()
    before = _model_state_hashes(m)
    with torch.no_grad():
        m.scalar.fill_(4.0)
    assert _model_state_hashes(m)["scalar"] != before["scalar"]


def test_nan_payload_change_is_detected():
    """Byte comparison, not value comparison: nan != nan must not mask a change."""
    m = ZooModel()
    before = _model_state_hashes(m)
    with torch.no_grad():
        m.nan[1] = 2.0
    assert _model_state_hashes(m)["nan"] != before["nan"]


def test_an_unhashable_tensor_is_reported_not_fatal(caplog, monkeypatch):
    """A fourth edge case must not cost another 45-minute run -- but it must be
    reported, because silently dropping tensors would weaken the sweep."""
    import verl.workers.rollout.sglang_rollout.delta_loader as dl

    real = torch.hash_tensor

    def boom(t):
        if t.numel() == 5:  # the u8 buffer
            raise RuntimeError("synthetic hashing failure")
        return real(t)

    monkeypatch.setattr(dl.torch, "hash_tensor", boom)
    with caplog.at_level("WARNING"):
        h = dl._model_state_hashes(ZooModel())
    assert "u8" not in h
    assert "bf16" in h, "one bad tensor must not abort the rest"
    assert any("could not be hashed" in r.getMessage() for r in caplog.records)
