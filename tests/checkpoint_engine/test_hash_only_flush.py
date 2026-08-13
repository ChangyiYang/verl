"""A hash_only update must hash and change nothing.

The ladder needs to hash SGLang's weights at four points, but the model lives in
SGLang's TP workers and the custom loader is the only entry point handed the model
object -- and that entry point is reached only by the delta path, so the nccl full
sync would be invisible. Sending a zero-payload update through the loader runs the
probe without touching any sync path, because a verification tool must not modify
what it verifies.

These tests pin the two properties that make it safe: it writes nothing, and it
does not need a checksum it has no payload for.
"""

import json

import pytest
import torch

from verl.workers.rollout.sglang_rollout.delta_loader import apply_delta


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)
        self.register_buffer("codes", torch.arange(4, dtype=torch.uint8).view(torch.float8_e4m3fn))
        self.register_buffer("weight_scale_inv", torch.full((2, 2), 0.5))


def _spec_tensor(spec: dict) -> torch.Tensor:
    return torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8).clone()


def _hash_only_update(stage="0"):
    return [("__delta_spec__", _spec_tensor({"hash_only": True, "stage": stage}))]


def test_hash_only_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_DELTA_LADDER_DIR", str(tmp_path))
    m = TinyModel()
    before_w = m.w.detach().clone()
    before_c = m.codes.detach().view(torch.uint8).clone()
    apply_delta(m, _hash_only_update())
    assert torch.equal(m.w.detach(), before_w), "hash_only must not touch parameters"
    assert torch.equal(m.codes.detach().view(torch.uint8), before_c), "hash_only must not touch buffers"


def test_hash_only_writes_a_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_DELTA_LADDER_DIR", str(tmp_path))
    monkeypatch.setenv("VERL_DELTA_LADDER_TAG", "nccl")
    apply_delta(TinyModel(), _hash_only_update(stage="1"))
    files = list(tmp_path.glob("ladder_nccl_1*_rank*.json"))
    assert len(files) == 1, f"expected one snapshot, got {[f.name for f in files]}"
    d = json.loads(files[0].read_text())
    h = d["hashes"]
    assert "w" in h and "codes" in h, "must hash parameters AND buffers"
    assert isinstance(d.get("raw_scales"), dict), "small scale grids must be dumped raw beside the hashes"
    import base64

    import numpy as np

    rs = d["raw_scales"]["weight_scale_inv"]
    vals = np.frombuffer(base64.b64decode(rs["b64"]), dtype=np.float32)
    assert vals.tolist() == [0.5] * 4, "raw dump must carry the exact bytes"


def test_hash_only_needs_no_checksum(tmp_path, monkeypatch):
    """It carries no positions or values, so demanding a checksum would mean
    fabricating a payload just to satisfy the check."""
    monkeypatch.setenv("VERL_DELTA_LADDER_DIR", str(tmp_path))
    apply_delta(TinyModel(), _hash_only_update())  # no "checksum" key at all


def test_hash_only_is_off_without_the_env(tmp_path, monkeypatch):
    """No dir configured -> no files, and still no mutation."""
    monkeypatch.delenv("VERL_DELTA_LADDER_DIR", raising=False)
    m = TinyModel()
    before = m.w.detach().clone()
    apply_delta(m, _hash_only_update())
    assert torch.equal(m.w.detach(), before)
    assert not list(tmp_path.glob("*.json"))


def test_probe_batch_round_trips_through_the_loader(tmp_path, monkeypatch):
    """The batch the rollout worker actually POSTS must be the one the loader
    answers: build it with ladder_probe_batch (the sender's only source) and feed
    it to apply_delta unmodified. Pins the sender/receiver spec contract without
    importing sglang (the sender module needs libcuda, this test env has none)."""
    from verl.workers.rollout.sglang_rollout.delta_loader import ladder_probe_batch

    monkeypatch.setenv("VERL_DELTA_LADDER_DIR", str(tmp_path))
    monkeypatch.setenv("VERL_DELTA_LADDER_TAG", "nccl")
    m = TinyModel()
    before = m.w.detach().clone()
    apply_delta(m, ladder_probe_batch("sync3"))
    assert torch.equal(m.w.detach(), before)
    files = list(tmp_path.glob("ladder_nccl_sync3_*_rank*.json"))
    assert len(files) == 1, f"stage must survive the trip into the filename: {[f.name for f in files]}"


def test_a_normal_flush_is_unaffected(tmp_path, monkeypatch):
    """The branch must not swallow real updates -- it keys on hash_only only."""
    monkeypatch.setenv("VERL_DELTA_LADDER_DIR", str(tmp_path))
    with pytest.raises(Exception):
        # a real spec with no payload tensors should fail somewhere downstream,
        # proving we did not silently take the hash_only exit
        apply_delta(TinyModel(), [("__delta_spec__", _spec_tensor({"encoding": "dense", "checksum": 0}))])
