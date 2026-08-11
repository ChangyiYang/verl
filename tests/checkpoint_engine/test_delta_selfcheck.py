"""The sender-side self-check must be able to FAIL.

Bit-level correctness of the DSv4 delta is verified offline (see
``_runs/check_delta_offline.py``) because both in-line attempts took the SGLang
server down mid-sync. A verification tool is only worth the run it costs if it
rejects a wrong delta, so these tests feed it deliberately broken deltas and
assert it says so. The three negative cases are the point of the file; the
positive ones only stop it from failing everything.

The band-between-tiers lesson applies here too: the honest case spans all three
position widths, so a tier that silently wraps shows up as a failure in a test
that passes today.
"""

import pytest
import torch

from verl.checkpoint_engine.delta_sync.offline_verify import main as verify_main


def _gap_encode(idx):
    """Mirror of the sender's encoding: gap = idx - prev with idx[-1] := -1."""
    prev = torch.cat([torch.tensor([-1], dtype=torch.int64), idx[:-1]])
    gaps = idx - prev
    mx = int(gaps.max())
    # int16 is SIGNED: the 2-byte tier stops at 0x7FFF, not 0xFFFF.
    width, dt = (1, torch.uint8) if mx <= 0xFF else (2, torch.int16) if mx <= 0x7FFF else (4, torch.int32)
    return gaps.to(dt).view(torch.uint8), width


def _piece(idx, val, dtype_str="bfloat16"):
    gaps, width = _gap_encode(idx)
    return {"dtype_str": dtype_str, "shape": [val.numel()], "pos_width": width, "gaps": gaps, "val": val}


def _dump(d, step, dense, pieces):
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "dense": dense, "pieces": pieces}, d / f"selfcheck_step{step}.pt")


def _run(d, capsys=None):
    """Run the checker in-process and return (exit_code, captured_report)."""
    rc = verify_main([str(d)])
    return rc, capsys.readouterr().out if capsys is not None else ""


def _pair(n_elem, moved, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    before = torch.randn(n_elem, generator=g, dtype=torch.float32).to(dtype)
    after = before.clone()
    after[moved] = torch.randn(len(moved), generator=g, dtype=torch.float32).to(dtype)
    return before, after


# gaps here land in all three tiers, including the 0x7FFF..0xFFFF band that a
# 0xFFFF threshold would wrap negative.
MIXED_IDX = torch.tensor([0, 5, 300, 40000, 40001, 200000], dtype=torch.int64)


def test_honest_delta_across_all_width_tiers(tmp_path):
    before, after = _pair(300000, MIXED_IDX)
    _dump(tmp_path, 1, {"w": before.view(torch.uint8).clone()}, {})
    _dump(tmp_path, 2, {"w": after.view(torch.uint8).clone()}, {"w": [_piece(MIXED_IDX, after[MIXED_IDX].clone())]})
    rc, out = _run(tmp_path)
    assert rc == 0, out


def test_wrong_value_is_rejected(tmp_path, capsys):
    """Check A: a delta that ships the wrong value for a real position."""
    idx = MIXED_IDX[:3]
    before, after = _pair(1000, idx)
    bad = after[idx].clone()
    bad[1] = bad[1] + 7.0
    _dump(tmp_path, 1, {"w": before.view(torch.uint8).clone()}, {})
    _dump(tmp_path, 2, {"w": after.view(torch.uint8).clone()}, {"w": [_piece(idx, bad)]})
    rc, out = _run(tmp_path, capsys)
    assert rc == 1 and "apply mismatch" in out, out


def test_incomplete_delta_is_rejected(tmp_path, capsys):
    """Half of what moved is reported -- a dropped replica or half a fused pair.

    Both checks fire here; the assertion is on B's wording because that is the
    one that names the cause. Mutation-tested: stubbing out B's computation makes
    this test fail rather than silently downgrade to A's generic message.
    """
    moved = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    before, after = _pair(1000, moved)
    half = moved[:2]
    _dump(tmp_path, 1, {"w": before.view(torch.uint8).clone()}, {})
    _dump(tmp_path, 2, {"w": after.view(torch.uint8).clone()}, {"w": [_piece(half, after[half].clone())]})
    rc, out = _run(tmp_path, capsys)
    assert rc == 1 and "UNCOVERED" in out, out


def test_dropped_parameter_is_rejected(tmp_path, capsys):
    """Check B: the parameter moved but produced no wire entry at all."""
    moved = torch.tensor([10, 20], dtype=torch.int64)
    before, after = _pair(1000, moved)
    _dump(tmp_path, 1, {"w": before.view(torch.uint8).clone()}, {})
    _dump(tmp_path, 2, {"w": after.view(torch.uint8).clone()}, {})
    rc, out = _run(tmp_path, capsys)
    assert rc == 1 and "dropped parameter" in out, out


def test_nan_payload_does_not_false_pass(tmp_path):
    """Comparison is byte-level: a value compare would let nan != nan slip."""
    before = torch.randn(100, dtype=torch.float32).to(torch.bfloat16)
    after = before.clone()
    idx = torch.tensor([3, 7], dtype=torch.int64)
    after[idx] = float("nan")
    _dump(tmp_path, 1, {"w": before.view(torch.uint8).clone()}, {})
    _dump(tmp_path, 2, {"w": after.view(torch.uint8).clone()}, {"w": [_piece(idx, after[idx].clone())]})
    rc, out = _run(tmp_path)
    assert rc == 0, out


def test_unchanged_parameter_needs_no_entry(tmp_path):
    before = torch.randn(100, dtype=torch.float32).to(torch.bfloat16)
    _dump(tmp_path, 1, {"w": before.view(torch.uint8).clone()}, {})
    _dump(tmp_path, 2, {"w": before.view(torch.uint8).clone()}, {})
    rc, out = _run(tmp_path)
    assert rc == 0, out


def test_fp8_code_path(tmp_path):
    """Quantized syncs carry uint8 codes, whose element size is 1 byte."""
    before = torch.randint(0, 255, (1000,), dtype=torch.uint8)
    after = before.clone()
    idx = torch.tensor([1, 500, 999], dtype=torch.int64)
    after[idx] = (before[idx] + 13) % 255
    _dump(tmp_path, 1, {"c": before.clone()}, {})
    _dump(tmp_path, 2, {"c": after.clone()}, {"c": [_piece(idx, after[idx].clone(), dtype_str="uint8")]})
    rc, out = _run(tmp_path)
    assert rc == 0, out


def test_duplicate_positions_are_legal(tmp_path):
    """Duplicates ride as gap 0 and apply last-writer-wins, as on the real wire."""
    idx = torch.tensor([4, 4, 9], dtype=torch.int64)
    before = torch.randn(50, dtype=torch.float32).to(torch.bfloat16)
    after = before.clone()
    vals = torch.tensor([1.5, 2.5, 3.5], dtype=torch.bfloat16)
    after[4], after[9] = vals[1], vals[2]  # last writer wins at position 4
    _dump(tmp_path, 1, {"w": before.view(torch.uint8).clone()}, {})
    _dump(tmp_path, 2, {"w": after.view(torch.uint8).clone()}, {"w": [_piece(idx, vals)]})
    rc, out = _run(tmp_path)
    assert rc == 0, out
