"""Policies: what a moving frame is compared against, and where the optimizer starts.

A `ReferencePolicy` answers "what does `mov` at time `t` register against": another
channel (registration) or the channel's own first / previous frame (stabilization) --
the same estimate, differing only here. A `SeedPolicy` answers "what initial transform
does the estimator start from" and never decides how the transform is computed.

Series are indexed lazily (`series[t]` before `np.asarray`) so dask- or zarr-backed
inputs only materialize the frame requested.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform


@runtime_checkable
class ReferencePolicy(Protocol):
    """Returns the reference array for timepoint `t`, given the full moving series."""

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike: ...


class CrossChannel:
    """Registration: a fixed external reference series (a different channel)."""

    def __init__(self, ref: ArrayLike):
        self.ref = ref

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike:
        return np.asarray(self.ref[t])


class FixedFrame:
    """Stabilization `t_reference: "first"`: every t compares against one fixed frame."""

    def __init__(self, t_ref: int = 0):
        self.t_ref = t_ref

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike:
        return np.asarray(mov[self.t_ref])


def is_empty(volume: ArrayLike) -> bool:
    """Return whether a frame has no data (all zeros or all NaN)."""
    return not np.any(np.nan_to_num(np.asarray(volume)))


class PreviousFrame:
    """Stabilization `t_reference: "previous"`: t compares against the previous frame with data.

    Usually t-1. Empty frames (no data) are skipped, so after a gap the first frame with
    data is compared against the last frame before the gap and the accumulated chain
    keeps the movement that happened during it. A frame with no earlier frame with data
    (t=0, or after leading empty frames) is compared against itself: it starts the chain.
    """

    def reference_index(self, mov: ArrayLike, t: int) -> int | None:
        for earlier in range(t - 1, -1, -1):
            if not is_empty(mov[earlier]):
                return earlier
        return None

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike:
        for earlier in range(t - 1, -1, -1):
            frame = np.asarray(mov[earlier])  # read once: checked, then returned
            if not is_empty(frame):
                return frame
        return np.asarray(mov[t])


@runtime_checkable
class SeedPolicy(Protocol):
    """Returns the initial transform guess for timepoint `t`."""

    def seed_for(self, t: int) -> Transform: ...


class FixedSeed:
    """A fixed seed for every t -- today's default.

    Also what `optimize-registration`'s "refine an existing transform" case is: pass the
    existing transform as this seed instead of a fresh default. No separate "optimize"
    concept is needed once seed policies exist.
    """

    def __init__(self, transform: Transform):
        self.transform = transform

    def seed_for(self, t: int) -> Transform:
        return self.transform


class PreviousSeed:
    """Propagation: seed t from the last accepted transform.

    Falls back to another policy for t=0 or whenever no previous result exists yet.
    Reads from an explicit, caller-owned history mapping rather than mutating any
    settings object in place, so a "config seed" elsewhere keeps meaning the config's
    seed. The caller writes `history[t] = accepted_transform` after each accepted
    estimate; this policy never mutates it.
    """

    def __init__(self, history: dict[int, Transform], fallback: SeedPolicy):
        self.history = history
        self.fallback = fallback

    def seed_for(self, t: int) -> Transform:
        if t - 1 in self.history:
            return self.history[t - 1]
        return self.fallback.seed_for(t)


class ConsensusSeed:
    """Repair seed from the run's own well-scoring timepoints.

    Element-wise median transform over every timepoint (other than `t` itself) whose
    score is >= `score_threshold`. Recomputed fresh on every call from a caller-owned
    history/scores mapping rather than memoized, so a repair pass sees results it has
    already accepted earlier in the same run. Built only from good timepoints:
    including the ones being repaired would contaminate the very reference used to
    detect and fix them.

    Raises `ValueError` when fewer than `min_good` timepoints qualify, so a caller
    trying this as one of several candidates (e.g. `registration.engine.repair`,
    whose per-candidate try/except already skips a candidate that raises) treats "no
    consensus yet" the same as any other failed candidate rather than crashing.
    """

    def __init__(
        self,
        history: dict[int, Transform],
        scores: dict[int, float],
        score_threshold: float = 0.75,
        min_good: int = 5,
    ):
        self.history = history
        self.scores = scores
        self.score_threshold = score_threshold
        self.min_good = min_good

    def seed_for(self, t: int) -> Transform:
        good = [
            t2
            for t2 in self.history
            if t2 != t
            and np.isfinite(self.scores.get(t2, np.nan))
            and self.scores[t2] >= self.score_threshold
        ]
        if len(good) < self.min_good:
            raise ValueError(
                f"Consensus seed: only {len(good)} timepoints score >= "
                f"{self.score_threshold}; need at least {self.min_good}."
            )
        stack = np.stack([self.history[t2].matrix for t2 in good])
        median = np.median(stack, axis=0)
        ndim = median.shape[0] - 1
        median[-1] = [0.0] * ndim + [1.0]
        return Transform(median, transform_type=self.history[good[0]].transform_type)
