"""Transform estimators: the protocol, its errors, and the compositions.

A `TransformEstimator` only promises `estimate(mov, ref, seed=None) -> Transform` in the
forward (moving -> reference) direction. Vocabulary, fixed across the package: at the
array level `mov` / `ref`; at the channel and CLI level `source` / `target`. `source` is
always the moving side and `target` always the reference -- the two pairs are the same
thing at two levels, never mixed within one signature. Each method's estimator lives with the method in
`biahub.registration.methods`; this module holds what they share -- the protocols, the
failure types, the one "keep the best" rule (`is_higher`, `select_best`) and the two
compositions (`ChainedEstimator`: acquire then refine; `CompetingEstimator`: several arms,
keep the best score).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
    `InitialTransformPolicy`. Estimators that don't use a seed (correlation-based methods: PCC,
    stackreg) ignore it and set `uses_seed = False`, so the engine skips re-seeding
    passes (repair, polish) for them; an estimator without the attribute uses its seed.
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


class CandidateUnavailableError(RuntimeError):
    """A candidate has nothing to offer yet (e.g. a consensus before enough good timepoints)."""


# The only exceptions that mean "this candidate failed"; anything else is a bug and propagates.
CANDIDATE_FAILURES = (EstimationError, CandidateUnavailableError)


def is_higher(candidate: float | None, current: float | None) -> bool:
    """Whether `candidate` scores higher than `current`: a finite score, strictly greater.

    A tie keeps the incumbent; NaN (or None) never wins; a NaN or missing incumbent is
    beaten by any finite score. The one rule every "keep the best" in the engine uses.
    """
    if candidate is None or not np.isfinite(candidate):
        return False
    return current is None or not np.isfinite(current) or candidate > current


@dataclass
class Selection:
    """What `select_best` kept, and what every candidate did.

    `name` is the winner's (the incumbent's, or None if nothing was usable), with its
    transform and score; `scores` every candidate's score, `failures` why any failed.
    """

    name: str | None
    transform: Transform | None
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


def select_best(
    attempts: dict[str, Callable[[], tuple[Transform, float]]],
    current: tuple[str, Transform, float] | None = None,
    stop_at: float | None = None,
) -> Selection:
    """Run `attempts` in order and keep the best-scoring transform, by `is_higher`.

    `current` is the incumbent (name, transform, score), kept unless a candidate beats
    it; ties go to the incumbent, then to the earlier candidate. A candidate fails on
    `CANDIDATE_FAILURES` or a non-finite score (kept in `failures`, never the winner);
    any other exception propagates. With `stop_at`, later candidates are not run once
    the best score reaches it. What to do when nothing is usable is the caller's.
    """
    name, transform, score = current if current is not None else (None, None, float("nan"))
    scores: dict[str, float] = {}
    failures: dict[str, str] = {}
    for candidate, attempt in attempts.items():
        if stop_at is not None and transform is not None and np.isfinite(score):
            if score >= stop_at:
                break
        try:
            candidate_transform, candidate_score = attempt()
        except CANDIDATE_FAILURES as error:
            failures[candidate] = f"{type(error).__name__}: {error}"
            continue
        scores[candidate] = float(candidate_score)
        if not np.isfinite(candidate_score):
            failures[candidate] = f"its transform scored {candidate_score}"
            continue
        if is_higher(candidate_score, score):
            name, transform, score = candidate, candidate_transform, float(candidate_score)
    return Selection(name, transform, score, scores, failures)


def fit_pairs(
    src, dst, transform_type: str = "affine", solver: str = "least_squares"
) -> np.ndarray:
    """Return `Transform.fit`'s matrix; a fit that fails is a failed estimate."""
    try:
        return Transform.fit(src, dst, transform_type, solver).matrix
    except ValueError as error:
        raise EstimationError(str(error)) from error


class ChainedEstimator:
    """Run estimators in sequence, each seeded by the previous result.

    Acquire-then-refine: e.g. a spectral matcher that can find the correspondence from a
    poor seed, followed by the Hungarian matcher that is more precise once close. With a
    `score_fn`, the best-scoring stage output is returned (a refinement that makes things
    worse is dropped); without one, the last stage's output.

    With `names` (the blocks of a stack), each estimate leaves what every stage did in
    `last_stages`: its transform, score, error and whether it won.
    """

    def __init__(
        self,
        stages: list[TransformEstimator],
        score_fn: ScoreFn | None = None,
        names: list[str] | None = None,
    ):
        if not stages:
            raise ValueError("ChainedEstimator needs at least one stage")
        if names is not None and len(names) != len(stages):
            raise ValueError(f"{len(names)} names for {len(stages)} stages")
        self.stages = list(stages)
        self.score_fn = score_fn
        self.names = names
        self.last_stages: list[dict] | None = None

    @property
    def uses_seed(self) -> bool:
        """Only the first stage starts from the seed; the others from the stage before."""
        return bool(getattr(self.stages[0], "uses_seed", True))

    @property
    def holds_xy_across_gaps(self) -> bool:
        """Focus-finding's across-gap rule (z measured, yx held), when every stage holds it.

        In a mixed stack, holding yx would drop what the other stages estimated.
        """
        return all(getattr(stage, "holds_xy_across_gaps", False) for stage in self.stages)

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        if self.names is not None:
            self.last_stages = None  # a failed estimate leaves no earlier one's stages
        current = seed
        best: Transform | None = None
        best_score = -np.inf
        stages = [
            {"block": name, "matrix": None, "score": None, "error": None, "won": False}
            for name in (self.names or [""] * len(self.stages))
        ]
        winner = None
        for k, stage in enumerate(self.stages):
            try:
                current = stage.estimate(mov, ref, seed=current)
            except EstimationError as error:
                stages[k]["error"] = str(error)
                if best is not None:
                    for later in stages[k + 1 :]:
                        later["error"] = "not run: an earlier block failed"
                    break
                self._keep(stages, winner)
                raise
            stages[k]["matrix"] = current.to_list()
            if self.score_fn is None:
                best, winner = current, k
                continue
            score = self.score_fn(current, mov, ref)
            if np.isfinite(score):
                stages[k]["score"] = float(score)
            if is_higher(score, best_score):
                best, best_score, winner = current, score, k
        self._keep(stages, winner)
        if best is None:
            raise EstimationError("no stage of the chain produced a finite score")
        return best

    def _keep(self, stages: list[dict], winner: int | None) -> None:
        if self.names is None:
            return
        if winner is not None:
            stages[winner]["won"] = True
        self.last_stages = stages


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

        def run(arm):
            def attempt():
                transform = arm.estimate(mov, ref, seed=seed)
                return transform, self.score_fn(transform, mov, ref)

            return attempt

        chosen = select_best(
            {name: run(arm) for name, arm in self.arms.items()}, stop_at=self.escalate_below
        )
        self.last_winner, self.last_scores = chosen.name, chosen.scores
        if chosen.name is None:
            raise EstimationError(
                "every arm failed: "
                + "; ".join(f"{k}: {v}" for k, v in chosen.failures.items())
            )
        return chosen.transform
