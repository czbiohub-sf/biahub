"""Guarded NVTX instrumentation for Nsight Systems (nsys) profiling.

Zero-cost when the optional ``nvtx`` package isn't installed (no-op context
managers / counters), and near-zero-cost when it is installed but no profiler is
attached — ``nvtx.get_domain`` returns a disabled domain whose methods no-op.

Usage::

    from biahub.tile_stitch import _nvtx
    with _nvtx.stage("recon_fft", "green"):
        ...
    _nvtx.counter("bytes_h2d", unit="bytes").sample(nbytes)
"""

from __future__ import annotations

import contextlib

try:
    import nvtx as _nvtx  # optional; install with `uv pip install nvtx` for profiling

    _DOMAIN = _nvtx.get_domain("tile_stitch")
except Exception:  # not installed -> everything no-ops
    _nvtx = None
    _DOMAIN = None

_COUNTERS: dict = {}


def stage(message: str, color: str = "blue"):
    """Context manager for a named NVTX range on the ``tile_stitch`` domain."""
    if _nvtx is None:
        return contextlib.nullcontext()
    return _nvtx.annotate(message=message, color=color, domain="tile_stitch")


class _NoopCounter:
    def sample(self, *_a, **_k):
        pass

    def batch_submit(self, *_a, **_k):
        pass


def counter(name: str, unit: str | None = None, integer: bool = True):
    """Return a cached NVTX counter (time-series lane in nsys), or a no-op."""
    c = _COUNTERS.get(name)
    if c is not None:
        return c
    if _DOMAIN is None:
        c = _NoopCounter()
    else:
        try:
            sem = _nvtx.CounterSemantics(unit=unit) if unit else None
            c = _DOMAIN.get_counter(name, int if integer else float, semantics=sem)
        except Exception:
            c = _NoopCounter()
    _COUNTERS[name] = c
    return c


def mark(message: str, color: str = "red"):
    """Record an instantaneous event on the ``tile_stitch`` domain."""
    if _nvtx is not None:
        _nvtx.mark(message=message, color=color, domain="tile_stitch")
