"""Policies: what a moving frame is compared against, and where the optimizer starts.

A `ReferencePolicy` answers "what does `mov` at time `t` register against": another
channel (registration) or the channel's own first / previous frame (stabilization) --
the same estimate, differing only here. An `InitialTransformPolicy` answers "what
initial transform does the estimator start from" and never decides how the transform is
computed.

Series are indexed lazily (`series[t]` before `np.asarray`) so dask- or zarr-backed
inputs only materialize the frame requested.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.estimators import CandidateUnavailableError


@runtime_checkable
class ReferencePolicy(Protocol):
    """Returns the reference array for timepoint `t`, given the full moving series.

    `compares_with_itself`: t is its own reference, so it starts the stabilization chain
    (identity, not estimated, when propagating). `spans_gap`: t's reference reaches back
    over empty frames (where focus finding holds yx).
    """

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike: ...

    def compares_with_itself(self, mov: ArrayLike, t: int) -> bool: ...

    def spans_gap(self, mov: ArrayLike, t: int) -> bool: ...


class CrossChannel:
    """Registration: a fixed external reference series (a different channel)."""

    def __init__(self, ref: ArrayLike):
        self.ref = ref

    def reference_for(self, mov: ArrayLike, t: int) -> ArrayLike:
        return np.asarray(self.ref[t])

    def compares_with_itself(self, mov: ArrayLike, t: int) -> bool:
        return False

    def spans_gap(self, mov: ArrayLike, t: int) -> bool:
        return False


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

    def compares_with_itself(self, mov: ArrayLike, t: int) -> bool:
        return t == self.reference_index(mov)

    def spans_gap(self, mov: ArrayLike, t: int) -> bool:
        return False


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

    def compares_with_itself(self, mov: ArrayLike, t: int) -> bool:
        return self.reference_index(mov, t) is None

    def spans_gap(self, mov: ArrayLike, t: int) -> bool:
        """Whether t's previous frame with data is before t-1 (t-1 has no data)."""
        index = self.reference_index(mov, t)
        return index is not None and index != t - 1


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

    def compares_with_itself(self, mov: ArrayLike, t: int) -> bool:
        return not self.neighbour_indices(mov, t)

    def spans_gap(self, mov: ArrayLike, t: int) -> bool:
        return False


@runtime_checkable
class InitialTransformPolicy(Protocol):
    """Returns the initial transform guess for timepoint `t`.

    `keep_unless_beaten`: the initial transform is an earlier result, so the engine keeps
    it unless the estimate from it scores strictly better.
    """

    keep_unless_beaten: bool

    def initial_for(self, t: int) -> Transform: ...


class FixedInitialTransform:
    """A fixed initial transform for every t -- today's default.

    Also what `optimize-registration`'s "refine an existing transform" case is: pass the
    existing transform as this initial transform instead of a fresh default. No separate
    "optimize" concept is needed once initial transform policies exist.
    """

    keep_unless_beaten = False

    def __init__(self, transform: Transform):
        self.transform = transform

    def initial_for(self, t: int) -> Transform:
        return self.transform


class ShapesInitialTransform:
    """`transform.initial: {from: shapes}`: the initial transform from the two grids.

    The moving grid, flipped left-right (`fliplr`) and turned by `rotate90` quarter
    turns in yx, then rescaled by the voxel-size ratio (z, and yx from the last axis)
    onto the reference grid. The voxel sizes are the stores' own (their OME-NGFF scale;
    only the last three axes are read): there is no default instrument. The same
    transform for every t, computed once. ZYX order.
    """

    keep_unless_beaten = False

    def __init__(
        self,
        mov_shape: tuple,
        ref_shape: tuple,
        mov_voxel_size: tuple,
        ref_voxel_size: tuple,
        rotate90: int,
        fliplr: bool,
        transform_type: str = "affine",
    ):
        mov_shape, ref_shape = tuple(mov_shape[-3:]), tuple(ref_shape[-3:])
        scale_z = mov_voxel_size[-3] / ref_voxel_size[-3]
        scale_yx = mov_voxel_size[-1] / ref_voxel_size[-1]
        forward = (
            Transform.from_scaling((scale_z, scale_yx, scale_yx), ref_shape, ref_shape).matrix
            @ Transform.from_rotation_yx(90 * rotate90, mov_shape, ref_shape).matrix
            @ (Transform.from_flip_x(mov_shape, ref_shape).matrix if fliplr else np.eye(4))
        )
        # reference -> moving, as on disk (init_run shows it)
        self.inverse_matrix = np.linalg.inv(forward)
        self.transform = Transform.from_inverse(self.inverse_matrix, transform_type)

    def initial_for(self, t: int) -> Transform:
        return self.transform


class FileInitialTransform:
    """`transform.initial: {from: file}`: each timepoint's transform from an earlier result.

    `by_t` maps each timepoint to its forward transform (e.g. a pooled plate transform,
    the same at every t). It is an earlier result, so `keep_unless_beaten`: the engine,
    which scores, keeps it unless the estimate from it scores strictly better.
    """

    keep_unless_beaten = True

    def __init__(self, by_t: dict[int, Transform]):
        self.by_t = by_t

    def initial_for(self, t: int) -> Transform:
        if t not in self.by_t:
            raise KeyError(
                f"no initial transform for t={t} in the file (it has {sorted(self.by_t)})"
            )
        return self.by_t[t]


class ConsensusInitialTransform:
    """Repair initial transform from the run's own well-scoring timepoints.

    Element-wise median transform over every timepoint (other than `t` itself) whose
    score is >= `score_threshold`. Recomputed fresh on every call from a caller-owned
    history/scores mapping rather than memoized, so a repair pass sees results it has
    already accepted earlier in the same run. Built only from good timepoints:
    including the ones being repaired would contaminate the very reference used to
    detect and fix them.

    Raises `CandidateUnavailableError` when fewer than `min_good` timepoints qualify, so
    a caller trying this as one of several candidates (e.g.
    `registration.engine.repair`, whose per-candidate try/except already skips a
    candidate that raises) treats "no consensus yet" the same as any other failed
    candidate rather than crashing.
    """

    keep_unless_beaten = False

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

    def initial_for(self, t: int) -> Transform:
        good = [
            t2
            for t2 in self.history
            if t2 != t
            and np.isfinite(self.scores.get(t2, np.nan))
            and self.scores[t2] >= self.score_threshold
        ]
        if len(good) < self.min_good:
            raise CandidateUnavailableError(
                f"Consensus initial transform: only {len(good)} timepoints score >= "
                f"{self.score_threshold}; need at least {self.min_good}."
            )
        stack = np.stack([self.history[t2].matrix for t2 in good])
        median = np.median(stack, axis=0)
        ndim = median.shape[0] - 1
        median[-1] = [0.0] * ndim + [1.0]
        return Transform(median, transform_type=self.history[good[0]].transform_type)
