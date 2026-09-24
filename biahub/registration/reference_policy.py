"""Reference policies: what to compare a moving frame against.

A `ReferencePolicy` only answers "what does `mov` at time `t` register against" -- it
never decides how to start the optimizer (see `seed_policy.py`) or which method computes
the transform (see `estimators.py`). Registration (cross-channel) and stabilization
(same channel over time) are the same operation under this abstraction, differing only
in which `ReferencePolicy` is used.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from numpy.typing import ArrayLike


@runtime_checkable
class ReferencePolicy(Protocol):
    """Returns the reference array for timepoint `t`, given the full moving series."""

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike: ...


class CrossChannel:
    """Registration: a fixed external reference series (a different channel)."""

    def __init__(self, ref: ArrayLike):
        self.ref = ref

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike:
        return np.asarray(self.ref)[t]


class FixedFrame:
    """Stabilization `t_reference: "first"`: every t compares against one fixed frame."""

    def __init__(self, t_ref: int = 0):
        self.t_ref = t_ref

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike:
        return np.asarray(mov)[self.t_ref]


class PreviousFrame:
    """Stabilization `t_reference: "previous"`: t compares against t-1 (t=0 against itself)."""

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike:
        mov = np.asarray(mov)
        return mov[max(t - 1, 0)]
