"""Transform estimators: the protocol, its errors, and the compositions.

A `TransformEstimator` only promises `estimate(mov, ref, seed=None) -> Transform` in the
forward (moving -> reference) direction. Each method's estimator lives with the method in
`biahub.registration.methods`; this module holds what they share -- the protocols, the
failure type, and the two compositions (`ChainedEstimator`: acquire then refine;
`CompetingEstimator`: several arms, keep the best score).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform

ScoreFn = Callable[[Transform, np.ndarray, np.ndarray], float]


NodeMatcher = Callable[
    [np.ndarray, np.ndarray], np.ndarray
]  # (mov_nodes, ref_nodes) -> (N, 2)


@runtime_checkable
class TransformEstimator(Protocol):
    """Computes the Transform that maps `mov` onto `ref`.

    `seed`, when given, is an initial guess in this same contract's direction (true
    forward, moving -> reference) -- e.g. a previous timepoint's accepted result via
    `SeedPolicy`. Estimators that don't use a seed (correlation-based methods: PCC,
    stackreg) ignore it.
    """

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform: ...


@runtime_checkable
class NodeDetector(Protocol):
    """Extracts point coordinates (nodes) from an array."""

    def detect(self, array: ArrayLike) -> ArrayLike: ...


class EstimationError(RuntimeError):
    """`estimate()` could not produce a transform (too few nodes or matches, degenerate fit)."""


class ChainedEstimator:
    """Run estimators in sequence, each seeded by the previous result.

    Acquire-then-refine: e.g. a spectral matcher that can find the correspondence from a
    poor seed, followed by the Hungarian matcher that is more precise once close. With a
    `score_fn`, the best-scoring stage output is returned (a refinement that makes things
    worse is dropped); without one, the last stage's output.
    """

    def __init__(self, stages: list[TransformEstimator], score_fn: ScoreFn | None = None):
        if not stages:
            raise ValueError("ChainedEstimator needs at least one stage")
        self.stages = list(stages)
        self.score_fn = score_fn

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        current = seed
        best: Transform | None = None
        best_score = -np.inf
        for stage in self.stages:
            try:
                current = stage.estimate(mov, ref, seed=current)
            except EstimationError:
                if best is not None:
                    break
                raise
            if self.score_fn is None:
                best = current
                continue
            score = self.score_fn(current, mov, ref)
            if np.isfinite(score) and score > best_score:
                best, best_score = current, score
        if best is None:
            raise EstimationError("no stage of the chain produced a finite score")
        return best


class CompetingEstimator:
    """Run several estimators on the same input and keep the best-scoring result.

    An arm that raises `EstimationError` is skipped; only when every arm fails does the
    whole estimate fail, with each arm's reason. With `escalate_below`, arms after the
    first run only while the best score so far is below it. `last_winner` names the arm
    whose result was returned.
    """

    def __init__(
        self,
        arms: dict[str, TransformEstimator],
        score_fn: ScoreFn,
        escalate_below: float | None = None,
    ):
        if not arms:
            raise ValueError("CompetingEstimator needs at least one arm")
        self.arms = dict(arms)
        self.score_fn = score_fn
        self.escalate_below = escalate_below
        self.last_winner: str | None = None
        self.last_scores: dict[str, float] = {}

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        best: Transform | None = None
        best_score = -np.inf
        self.last_winner = None
        self.last_scores = {}
        failures: dict[str, str] = {}
        for index, (name, arm) in enumerate(self.arms.items()):
            if (
                index > 0
                and self.escalate_below is not None
                and best is not None
                and best_score >= self.escalate_below
            ):
                break
            try:
                transform = arm.estimate(mov, ref, seed=seed)
            except EstimationError as e:
                failures[name] = str(e)
                continue
            score = self.score_fn(transform, mov, ref)
            self.last_scores[name] = float(score)
            if np.isfinite(score) and score > best_score:
                best, best_score, self.last_winner = transform, score, name
        if best is None:
            raise EstimationError(
                "every arm failed: " + "; ".join(f"{k}: {v}" for k, v in failures.items())
            )
        return best
