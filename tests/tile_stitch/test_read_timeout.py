"""Bounded tile-read path in ``biahub.tile_stitch._core``: a stalled read is
abandoned + retried on a fresh thread, a persistent stall fails loudly instead of
wedging, and the prefetch reader's ``get()`` falls back to ``None`` on a deadline."""

from __future__ import annotations

import threading
import time

import pytest

from biahub.tile_stitch import _core
from biahub.tile_stitch._core import PrefetchReader, _read_with_retry


def test_read_with_retry_recovers_from_transient_stall():
    """First attempt stalls past the timeout; the retry on a fresh thread succeeds."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(0.5)  # > timeout -> abandoned
            return "late"
        return "ok"

    assert _read_with_retry(fn, timeout=0.15, retries=2) == "ok"
    assert calls["n"] >= 2  # retried after the stall


def test_read_with_retry_raises_on_persistent_stall():
    """A read that never returns raises TimeoutError (loud) instead of hanging."""
    release = threading.Event()

    def fn():
        release.wait(timeout=30)  # blocks until released
        return "never"

    try:
        with pytest.raises(TimeoutError):
            _read_with_retry(fn, timeout=0.1, retries=1)
    finally:
        release.set()  # let the abandoned pool threads exit cleanly


def test_read_with_retry_propagates_real_errors_without_retry():
    """A read that raises (a real error, not a stall) propagates immediately."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _read_with_retry(fn, timeout=5, retries=3)
    assert calls["n"] == 1  # not retried


def test_prefetch_get_times_out_to_none(monkeypatch):
    """If the background read wedges, get() returns None (sync fallback) instead of
    waiting forever."""
    monkeypatch.setattr(_core, "_GET_TIMEOUT_S", 0.2)
    block = threading.Event()

    def hang(tid):
        block.wait(timeout=30)  # loader never produces the tile in time
        return tid

    r = PrefetchReader(hang, order=[7], depth=1)
    try:
        assert r.get(7) is None  # backstop deadline -> None
    finally:
        block.set()
        r.stop()
