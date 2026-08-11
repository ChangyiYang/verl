"""The overlap must not drop a flush's staging buffers while its broadcast is live."""
import os, sys, types
import pytest

# Exercise the await/park protocol on a stand-in, since the real engine needs cupy+NCCL.
class _Ph:
    def __init__(self): self.t={}; self.n={}
    def span(self, key, sync=False, hot=False):
        import contextlib
        @contextlib.contextmanager
        def _c():
            self.t[key] = self.t.get(key, 0.0) + 1.0
            yield
        return _c()
    def bump(self, key, k=1): self.n[key] = self.n.get(key,0)+k

class Fake:
    """Mirrors the park/await protocol exactly."""
    def __init__(self, overlap=True):
        self.overlap=overlap; self._pub_inflight=None
        self.synced=0; self.freed=[]; self.live=[]
    def _await_publish_inflight(self, ph=None):
        if getattr(self, "_pub_inflight", None) is None: return
        self.synced += 1
        self.freed.append(self._pub_inflight)
        self.live.remove(self._pub_inflight)
        self._pub_inflight=None
    def publish(self, tag):
        self._await_publish_inflight()
        buf = f"buf{tag}"
        self.live.append(buf)          # staged
        # broadcast enqueued here
        if self.overlap: self._pub_inflight = buf
        else: self.synced += 1; self.freed.append(buf); self.live.remove(buf)

def test_at_most_one_flush_in_flight():
    f = Fake()
    for i in range(5):
        f.publish(i)
        assert len(f.live) <= 1, "more than one flush parked -> unbounded staging growth"

def test_buffer_outlives_its_broadcast():
    """The parked buffer must still be live until the NEXT publish awaits it."""
    f = Fake()
    f.publish(0)
    assert f.live == ["buf0"] and f.synced == 0, "buf0 freed before its broadcast was awaited"
    f.publish(1)
    assert f.freed == ["buf0"] and f.live == ["buf1"]

def test_terminal_await_drains_the_last_flush():
    f = Fake()
    f.publish(0); f.publish(1)
    f._await_publish_inflight()
    assert f.live == [], "last flush never drained -> receiver waits forever / buffer races"
    assert f.synced == 2

def test_await_is_idempotent():
    f = Fake(); f.publish(0)
    f._await_publish_inflight(); n = f.synced
    f._await_publish_inflight()
    assert f.synced == n, "double await must be a no-op"

def test_kill_switch_restores_synchronous_behaviour():
    f = Fake(overlap=False)
    for i in range(3):
        f.publish(i)
        assert f.live == [], "with overlap off nothing may stay in flight"
    assert f.synced == 3
