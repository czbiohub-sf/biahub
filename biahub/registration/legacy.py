"""The one place a legacy-direction matrix becomes an engine Transform, and back.

The legacy pipeline stores matrices in the pull direction (reference -> moving, ready to
hand straight to a resampler): `AffineTransformSettings.approx_transform`,
`RegistrationSettings.affine_transform_zyx`, `StabilizationSettings.affine_transform_zyx_list`,
and the return values of `beads.estimate_tzyx` / `ants.estimate` / `optimize_transform`.
The engine's contract (`TransformEstimator`, `SeedPolicy`, `fallback.repair`,
`Transform.apply`) is forward (moving -> reference). Variable names on the legacy side
("fwd_transform") do not reliably indicate direction; only this boundary does.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform, TransformType


def forward_from_legacy_pull(
    matrix: ArrayLike, transform_type: TransformType = "affine"
) -> Transform:
    """Engine (forward) Transform from a legacy pull-direction matrix."""
    return Transform(np.asarray(matrix, dtype=float), transform_type=transform_type).invert()


def legacy_pull_from_forward(transform: Transform) -> list[list[float]]:
    """Legacy pull-direction matrix (as a YAML-ready nested list) from an engine Transform."""
    return transform.invert().to_list()
