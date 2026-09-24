"""Seed policies: what initial transform guess to start the optimizer from.

Orthogonal to `ReferencePolicy` (what to compare against) and `TransformEstimator` (how
the transform is computed).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
