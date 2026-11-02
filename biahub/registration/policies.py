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
    """Stabilization `t_reference: "first"`: every t compares against one fixed frame.

    The frame is `t_ref`, or the first frame with data after it when `t_ref` is empty (an
    empty frame cannot be a reference: every timepoint would read as empty).
    """

    def __init__(self, t_ref: int = 0):
        self.t_ref = t_ref
        self._index: int | None = None
        self._frame: np.ndarray | None = None

    def _resolve(self, mov: ArrayLike) -> None:
        """Find the reference frame once, reading each frame at most once."""
        t = self.t_ref
        while True:
            try:
                frame = np.asarray(mov[t])
            except IndexError:  # past the last frame: every frame is empty
                self._index, self._frame = self.t_ref, None
                return
            if not is_empty(frame):
                self._index, self._frame = t, frame
                return
            t += 1

    def reference_index(self, mov: ArrayLike) -> int:
        """Return the frame every t compares against: `t_ref`, else the first with data."""
        if self._index is None:
            self._resolve(mov)
        return self._index

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike:
        index = self.reference_index(mov)
        return self._frame if self._frame is not None else np.asarray(mov[index])


def is_empty(volume: ArrayLike) -> bool:
    """Return whether a frame has no data (all zeros or all NaN)."""
    return not np.any(np.nan_to_num(np.asarray(volume)))


class PreviousFrame:
    """Stabilization `t_reference: "previous"`: t compares against the previous frame with data.

    Usually t-1. Empty frames (no data) are skipped, so after a gap the first frame with
    data is compared against the last frame before the gap and the accumulated chain
    keeps the movement that happened during it. A frame with no earlier frame with data
    (the run's `first` timepoint, or after leading empty frames) is compared against itself:
    it starts the chain. Frames before `first` are outside the run and never compared.
    """

    def __init__(self, first: int = 0):
        self.first = first

    def reference_index(self, mov: ArrayLike, t: int) -> int | None:
        for earlier in range(t - 1, self.first - 1, -1):
            if not is_empty(mov[earlier]):
                return earlier
        return None

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike:
        for earlier in range(t - 1, self.first - 1, -1):
            frame = np.asarray(mov[earlier])  # read once: checked, then returned
            if not is_empty(frame):
                return frame
        return np.asarray(mov[t])


class NeighbourMedian:
    """Stabilization `reference.frame: neighbours`: t against the median of the frames around it.

    The reference is the voxel-wise median of the frames with data in t - `window` .. t +
    `window`, t itself excluded, kept inside the run (`first` .. `last`). A frame that
    jumps for one timepoint and snaps back does not move the median, so the jump is
    measured against where the sample should be; the transforms are absolute (no chain,
    nothing accumulates), but slow drift moves the median with it. A timepoint with no
    neighbour with data compares against itself.
    """

    def __init__(self, window: int, first: int = 0, last: int | None = None):
        self.window = window
        self.first = first
        self.last = last
        self._t: int | None = None  # the last timepoint gathered, its neighbours, median
        self._indices: list[int] = []
        self._median: np.ndarray | None = None

    def _gather(self, mov: ArrayLike, t: int) -> None:
        """Read t's neighbours once and keep their median (both calls for t reuse it).

        The frames go straight into one preallocated array and the median is taken in
        place: the peak is 2 x window frames, the one being read and the median.
        """
        if self._t == t:
            return
        last = t + self.window if self.last is None else min(t + self.window, self.last)
        stack, indices = None, []
        for k in range(max(self.first, t - self.window), last + 1):
            if k == t:
                continue
            try:
                frame = np.asarray(mov[k])
            except IndexError:  # past the series' last frame
                break
            if is_empty(frame):
                continue
            if stack is None:
                stack = np.empty((2 * self.window, *frame.shape), dtype=np.float32)
            stack[len(indices)] = frame
            indices.append(k)
        self._t, self._indices, self._median = t, indices, None
        if indices:
            self._median = np.median(stack[: len(indices)], axis=0, overwrite_input=True)
            self._median = self._median.astype(np.float32, copy=False)

    def neighbour_indices(self, mov: ArrayLike, t: int) -> list[int]:
        """Return the frames t's reference is the median of (empty: t is its own)."""
        self._gather(mov, t)
        return list(self._indices)

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike:
        self._gather(mov, t)
        return self._median if self._median is not None else np.asarray(mov[t])


def intensity_centroid(
    volume: ArrayLike, *, bg_percentile: float, downsample: int, blur_sigma: float
) -> np.ndarray:
    """Return the ZYX centroid of `volume`'s signal, in full-resolution voxels.

    The anchor's measure of where the sample is (dynatrack's intensity centre of mass):
    the volume binned by `downsample` (an axis shorter than it, e.g. a projection's z,
    is not binned), blurred by `blur_sigma` binned voxels, and weighted by its intensity
    above the `bg_percentile` percentile. NaN when nothing is above it.
    """
    from scipy import ndimage

    v = np.asarray(volume, dtype=np.float32)
    factors = [downsample if n >= downsample else 1 for n in v.shape]
    v = v[tuple(slice(0, n // f * f) for n, f in zip(v.shape, factors, strict=True))]
    shape = [x for n, f in zip(v.shape, factors, strict=True) for x in (n // f, f)]
    v = v.reshape(shape).mean(axis=(1, 3, 5))
    v = ndimage.gaussian_filter(v, blur_sigma)
    weights = np.clip(v - np.percentile(v, bg_percentile), 0, None)
    if not weights.sum() > 0:
        return np.full(3, np.nan)
    factors = np.asarray(factors, dtype=float)
    # binned index i covers full-resolution voxels i*f .. i*f + f - 1
    return np.asarray(ndimage.center_of_mass(weights)) * factors + (factors - 1) / 2


def smooth_track(track: ArrayLike, *, median: int, sigma: float) -> np.ndarray:
    """Return a (T, 3) centroid track smoothed over time, the anchor's slow component.

    NaN rows (frames with no centroid) are interpolated from their neighbours; a median
    over `median` timepoints drops single-frame jumps, then a Gaussian of `sigma`
    timepoints smooths what is left. Rows are taken in order, one per timepoint.
    """
    from scipy import ndimage

    track = np.array(track, dtype=float)
    rows = np.arange(len(track))
    for axis in range(track.shape[1]):
        known = np.isfinite(track[:, axis])
        if not known.any():
            raise ValueError("no timepoint has a centroid to anchor on")
        track[:, axis] = np.interp(rows, rows[known], track[known, axis])
        track[:, axis] = ndimage.median_filter(track[:, axis], median, mode="nearest")
        track[:, axis] = ndimage.gaussian_filter1d(track[:, axis], sigma, mode="nearest")
    return track


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
