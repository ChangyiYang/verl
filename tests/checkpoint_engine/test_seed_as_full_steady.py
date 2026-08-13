"""The seed as a full steady sync: one producer for first and every later sync.

hf_delta_export(full=True) must (a) emit EVERY position without requiring a
prior snapshot, (b) prime the snapshot inline so the NEXT (normal) call diffs
against exactly what the seed shipped -- an unchanged shard then yields an
empty delta. This is the property that lets the delta engine drop the bridge's
stream export (and both of its hidden re-quantizers) from the seed path.
"""

import torch

from verl.workers.engine.spec import ShardSpec
from verl.workers.engine.utils import hf_delta_export


def _entry(name, spec, place, lidx, lval):
    return (name, place, lidx, lval)


def _spec():
    return ShardSpec(full_shape=(8,), place=0, gather_group=None, contributes=True)


def test_full_mode_emits_everything_and_primes():
    snaps = {}
    shard = torch.arange(8, dtype=torch.float32)
    out = list(hf_delta_export(iter([("p", shard, _spec())]), snaps, _entry, full=True))
    (name, place, lidx, lval, pg) = out[0]
    assert torch.equal(lidx, torch.arange(8)), "full mode must emit every position"
    assert torch.equal(lval, shard)
    assert "p" in snaps and torch.equal(snaps["p"], shard), "snapshot must be primed inline"


def test_steady_after_full_seed_is_empty_when_unchanged():
    snaps = {}
    shard = torch.arange(8, dtype=torch.float32)
    list(hf_delta_export(iter([("p", shard, _spec())]), snaps, _entry, full=True))
    out = list(hf_delta_export(iter([("p", shard.clone(), _spec())]), snaps, _entry))
    (_, _, lidx, lval, _) = out[0]
    assert lidx.numel() == 0 and lval.numel() == 0, "unchanged shard after seed must diff to empty"


def test_steady_after_full_seed_sees_only_the_change():
    snaps = {}
    shard = torch.arange(8, dtype=torch.float32)
    list(hf_delta_export(iter([("p", shard, _spec())]), snaps, _entry, full=True))
    moved = shard.clone()
    moved[3] = 100.0
    out = list(hf_delta_export(iter([("p", moved, _spec())]), snaps, _entry))
    (_, _, lidx, lval, _) = out[0]
    assert lidx.tolist() == [3] and lval.tolist() == [100.0]


def test_steady_without_seed_still_fails_loud():
    import pytest

    with pytest.raises(AssertionError, match="seed"):
        list(hf_delta_export(iter([("p", torch.ones(4), _spec())]), {}, _entry))


def test_dense_gather_group_single_process_slices_by_slot():
    from verl.checkpoint_engine.delta_sync.sparse_gather import dense_gather_group

    flat = torch.arange(10, dtype=torch.float32)
    pieces = dense_gather_group(flat, [4, 0, 6], group=None)
    assert [p.numel() for p in pieces] == [4, 0, 6]
    assert torch.equal(pieces[0], flat[:4]) and torch.equal(pieces[2], flat[4:])
