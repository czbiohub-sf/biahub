"""Optional NVTX ranges and counters for tile-stitch profiling."""


import contextlib

try:
    import nvtx as _nvtx

    _DOMAIN = _nvtx.get_domain("tile_stitch")
except Exception:
    _nvtx = None
    _DOMAIN = None

_COUNTERS: dict = {}


def stage(message: str, color: str = "blue"):
    """Create an NVTX range context.

    Parameters
    ----------
    message : str
        Range label.
    color : str, optional
        NVTX display color.

    Returns
    -------
    contextlib.AbstractContextManager
        NVTX annotation, or a no-op context when profiling is unavailable.
    """
    if _nvtx is None:
        return contextlib.nullcontext()
    return _nvtx.annotate(message=message, color=color, domain="tile_stitch")


class _NoopCounter:
    def sample(self, *_a, **_k):
        pass

    def batch_submit(self, *_a, **_k):
        pass


def counter(name: str, unit: str | None = None, integer: bool = True):
    """Return a cached NVTX counter or no-op replacement.

    Parameters
    ----------
    name : str
        Counter name.
    unit : str or None, optional
        Counter unit passed to NVTX semantics.
    integer : bool, optional
        Use an integer counter when ``True``; otherwise use floating point.

    Returns
    -------
    object
        Counter exposing ``sample`` and ``batch_submit``.
    """
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
    """Record an instantaneous NVTX event.

    Parameters
    ----------
    message : str
        Event label.
    color : str, optional
        NVTX display color.
    """
    if _nvtx is not None:
        _nvtx.mark(message=message, color=color, domain="tile_stitch")
