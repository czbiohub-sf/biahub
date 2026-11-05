"""Transform estimators: the protocol, its errors, and the compositions.

A `TransformEstimator` promises `estimate(mov, ref, seed=None) -> Transform` in the
forward (moving -> reference) direction, plus the few members the engine reads
(`EstimatorBase` gives their defaults). Vocabulary, fixed across the package: at the
array level `mov` / `ref`; at the channel and CLI level `source` / `target`. `source` is
always the moving side and `target` always the reference -- the two pairs are the same
thing at two levels, never mixed within one signature. Each method's estimator lives with the method in
`biahub.registration.methods`; this module holds what they share -- the protocols, the
failure types, the one "keep the best" rule (`is_higher`, `select_best`) and the two
compositions (`ChainedEstimator`: acquire then refine; `CompetingEstimator`: several
candidates, keep the best score).
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
    `InitialTransformPolicy`. The other members are what the engine reads; see
    `EstimatorBase`, which every estimator subclasses for their defaults.
    """

    uses_seed: bool
    holds_xy_across_gaps: bool
    accepts_competitor: bool

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform: ...

    def trace(self) -> dict: ...


class EstimatorBase:
    """The members the engine reads from an estimator, with the defaults most keep.

    - `uses_seed`: the estimate starts from `seed`. Correlation methods (PCC, stackreg)
      ignore it and set False, so the engine skips re-seeding passes (repair, polish).
    - `holds_xy_across_gaps`: across empty frames, z is measured and yx held (focus
      finding's legacy rule).
    - `accepts_competitor`: `estimate` also takes `competitor`, a second seed it refines
      alongside `seed` (beads). For the others the engine runs both starts and keeps the
      better score.
    - `trace()`: what the last `estimate` did, read right after it (failed or not) for
      the timepoint's record: `best_candidate`, the candidate a competition kept, and
      `blocks`, what each block of a stack did. None when it does not apply.
    """

    uses_seed: bool = True
    holds_xy_across_gaps: bool = False
    accepts_competitor: bool = False

    def trace(self) -> dict:
        return {"best_candidate": None, "blocks": None}


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


class ChainedEstimator(EstimatorBase):
    """Run estimators in sequence, each seeded by the previous result.

    Acquire-then-refine: e.g. a spectral matcher that can find the correspondence from a
    poor seed, followed by the Hungarian matcher that is more precise once close. With a
    `score_fn`, the best-scoring stage output is returned (a refinement that makes things
    worse is dropped); without one, the last stage's output.

    With `names` (the blocks of a stack), `trace()["blocks"]` holds what every stage of
    the last estimate did: its transform, score, error and whether it won.
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
        self._blocks: list[dict] | None = None

    @property
    def uses_seed(self) -> bool:
        """Only the first stage starts from the seed; the others from the stage before."""
        return self.stages[0].uses_seed

    @property
    def holds_xy_across_gaps(self) -> bool:
        """Focus-finding's across-gap rule (z measured, yx held), when every stage holds it.

        In a mixed stack, holding yx would drop what the other stages estimated.
        """
        return all(stage.holds_xy_across_gaps for stage in self.stages)

    def trace(self) -> dict:
        return {"best_candidate": None, "blocks": self._blocks}

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        self._blocks = None  # a failed estimate leaves no earlier one's blocks
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
        self._blocks = stages


class CompetingEstimator(EstimatorBase):
    """Run several estimators on the same input and keep the best-scoring result.

    A candidate that raises `EstimationError` is skipped; only when every candidate fails
    does the whole estimate fail, with each one's reason. With `escalate_below`,
    candidates after the first run only while the best score so far is below it.
    `trace()["best_candidate"]` names the one whose result the last estimate returned.
    """

    def __init__(
        self,
        candidates: dict[str, TransformEstimator],
        score_fn: ScoreFn,
        escalate_below: float | None = None,
    ):
        if not candidates:
            raise ValueError("CompetingEstimator needs at least one candidate")
        self.candidates = dict(candidates)
        self.score_fn = score_fn
        self.escalate_below = escalate_below
        self._best_candidate: str | None = None

    def trace(self) -> dict:
        return {"best_candidate": self._best_candidate, "blocks": None}

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, seed: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov)
        ref = np.asarray(ref)

        def run(candidate):
            def attempt():
                transform = candidate.estimate(mov, ref, seed=seed)
                return transform, self.score_fn(transform, mov, ref)

            return attempt

        chosen = select_best(
            {name: run(c) for name, c in self.candidates.items()},
            stop_at=self.escalate_below,
        )
        self._best_candidate = chosen.name
        if chosen.name is None:
            raise EstimationError(
                "every candidate failed: "
                + "; ".join(f"{k}: {v}" for k, v in chosen.failures.items())
            )
        return chosen.transform
