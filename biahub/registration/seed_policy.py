"""Seed policies: what initial transform guess to start the optimizer from.

Orthogonal to `ReferencePolicy` (what to compare against) and `TransformEstimator` (how
the transform is computed).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from biahub.core.transform import Transform


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
    settings object in place. `beads.py`'s `estimate_with_propagation` currently mutates
    `affine_transform_settings.approx_transform` directly, which the repair/sweep
    fallback passes also read, expecting the ORIGINAL config value -- so the
    "config_seed" reseed candidate ends up being whatever timepoint propagation last
    visited, not the config's actual seed (PR #339 review finding 4). The caller is
    responsible for writing `history[t] = accepted_transform` after each accepted
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
    trying this as one of several candidates (e.g. `registration.fallback.repair`,
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
