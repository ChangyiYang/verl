"""The verify sweep's log filter must actually stop the flood.

Dense verification never failed on the data. The seed pushes the same full model
through the same _apply_dense path every run and completes in ~280 s. The verify
sweep differs only in CALL GRANULARITY: it snapshots and compares around each
atomic unit, so it calls load_weights ~24k-32k times instead of the seed's 533.
SGLang logs a 65 KB "not initialized from checkpoints" record per call, so the
sweep produces ~2 GB of log text and ray's log forwarder kills the job at its
512 MB cap with RaySystemError -- which reads nothing like a verification fault.
"""

import logging

import pytest

from verl.workers.rollout.sglang_rollout.delta_loader import (
    _DedupUninitWarning,
    _quiet_uninit_warning,
)

NEEDLE = "Some weights are not initialized from checkpoints: {'a', 'b'}"


def test_first_occurrence_is_kept():
    """The warning is useful once -- suppressing all of it would hide a real signal."""
    f = _DedupUninitWarning()
    rec = logging.LogRecord("x", logging.WARNING, __file__, 1, NEEDLE, None, None)
    assert f.filter(rec) is True
    assert f.suppressed == 0


def test_repeats_are_dropped_and_counted():
    f = _DedupUninitWarning()
    recs = [logging.LogRecord("x", logging.WARNING, __file__, 1, NEEDLE, None, None) for _ in range(1000)]
    kept = [r for r in recs if f.filter(r)]
    assert len(kept) == 1, "only the first should survive"
    assert f.suppressed == 999


def test_unrelated_records_pass_through():
    """A filter that eats everything would hide the failure we are looking for."""
    f = _DedupUninitWarning()
    for msg in ("delta checksum mismatch", "DELTA-VERIFY sweep: mismatch_elems=17", "anything else"):
        rec = logging.LogRecord("x", logging.WARNING, __file__, 1, msg, None, None)
        assert f.filter(rec) is True, msg
    assert f.suppressed == 0


def test_context_manager_removes_the_filter_afterwards():
    """Leaving it installed would silence the warning for the rest of the run."""
    root = logging.getLogger()
    before = len(root.filters)
    with _quiet_uninit_warning():
        assert len(root.filters) == before + 1
    assert len(root.filters) == before


def test_filter_is_removed_even_on_exception():
    root = logging.getLogger()
    before = len(root.filters)
    with pytest.raises(ValueError):
        with _quiet_uninit_warning():
            raise ValueError("boom")
    assert len(root.filters) == before


def test_the_flood_is_actually_bounded(caplog):
    """End to end: 30k emits of the 65 KB record must not produce 30k records."""
    log = logging.getLogger("sglang.fake_loader")
    with caplog.at_level(logging.WARNING):
        with _quiet_uninit_warning():
            for _ in range(30_000):
                log.warning(NEEDLE)
    hits = [r for r in caplog.records if "are not initialized from checkpoints" in r.getMessage()]
    assert len(hits) <= 1, f"flood not bounded: {len(hits)} records survived"
