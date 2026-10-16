"""Registration evaluation: how well a transform registers, in µm, independent of the score.

Layer 1 (here): geometry -- the plausibility of a transform, and statistics of matched
residual displacements. Distances are in µm from the stores' voxel sizes (z, y, x); voxel
values are reported alongside where useful.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform

# Fewer matched points than this do not constrain an affine residual (12 parameters).
MIN_POINTS_FOR_AFFINE = 6


def _polar(linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a linear map into rotation @ symmetric stretch (polar decomposition)."""
    u, sigma, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:  # keep a proper rotation; the reflection goes in stretch
        u[:, -1] *= -1
        rotation = u @ vt
    stretch = rotation.T @ linear
    return rotation, stretch


def _rotation_angles(rotation: np.ndarray) -> tuple[float, float]:
    """Return (total rotation angle, in-plane yx angle), in degrees, of a ZYX rotation."""
    total = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
    yx = np.degrees(np.arctan2(rotation[2, 1], rotation[1, 1]))
    return float(total), float(yx)


def decompose(transform: Transform, mov_um: ArrayLike, ref_um: ArrayLike) -> dict:
    """Return a forward transform's geometry in physical units, to judge its plausibility.

    The voxel matrix is first expressed in µm (`diag(ref_um) @ L @ diag(mov_um)^-1`), so a
    scale of 1 means the same physical size; then split into rotation, per-axis scale and
    shear. Translation is the reference-side offset in µm.
    """
    matrix = np.asarray(transform.matrix, dtype=float)
    mov_um, ref_um = np.asarray(mov_um, float), np.asarray(ref_um, float)
    linear = np.diag(ref_um) @ matrix[:3, :3] @ np.diag(1.0 / mov_um)
    rotation, stretch = _polar(linear)
    total, yx = _rotation_angles(rotation)
    scale = np.diag(stretch).copy()
    shear = stretch - np.diag(scale)
    return {
        "scale_zyx": scale.tolist(),
        "rotation_deg": total,
        "rotation_yx_deg": yx,
        "shear": float(np.abs(shear).max()),
        "translation_um": (matrix[:3, 3] * ref_um).tolist(),
    }


def displacement_stats(diff_vox: ArrayLike, voxel_um: ArrayLike) -> dict:
    """Return statistics of matched residual displacements (N x 3, voxels, ZYX) in µm."""
    diff = np.asarray(diff_vox, dtype=float).reshape(-1, 3) * np.asarray(voxel_um, float)
    if len(diff) == 0:
        return {"n": 0}
    dist = np.linalg.norm(diff, axis=1)
    median_zyx = np.median(diff, axis=0)
    return {
        "n": int(len(diff)),
        "median_um": float(np.median(dist)),
        "p90_um": float(np.percentile(dist, 90)),
        "median_zyx_um": median_zyx.tolist(),
        "mad_zyx_um": np.median(np.abs(diff - median_zyx), axis=0).tolist(),
        "rms_zyx_um": np.sqrt((diff**2).mean(axis=0)).tolist(),
    }


def residual_affine(points_vox: ArrayLike, diff_vox: ArrayLike, voxel_um: ArrayLike) -> dict:
    """Fit residual displacement = a + B @ position over matched points, in µm.

    A residual that grows across the field is a scale or rotation error of the transform,
    which median distances alone hide; a constant residual is a shift. Returns the residual
    map's per-axis scale and rotation (1 and 0 for no such error), its shift in µm, and the
    displacement it accumulates across the extent of the points (`trend_um`, per axis).
    """
    pts = np.asarray(points_vox, dtype=float).reshape(-1, 3) * np.asarray(voxel_um, float)
    diff = np.asarray(diff_vox, dtype=float).reshape(-1, 3) * np.asarray(voxel_um, float)
    n = int(len(pts))
    if n < MIN_POINTS_FOR_AFFINE:
        return {"n": n, "scale_zyx": None, "rotation_yx_deg": None}
    design = np.c_[np.ones(n), pts]
    coef, *_ = np.linalg.lstsq(design, diff, rcond=None)
    shift, gradient = coef[0], coef[1:].T  # diff ~ shift + gradient @ pos
    rotation, stretch = _polar(np.eye(3) + gradient)
    _total, yx = _rotation_angles(rotation)
    extent = pts.max(axis=0) - pts.min(axis=0)
    return {
        "n": n,
        "scale_zyx": np.diag(stretch).tolist(),
        "rotation_yx_deg": yx,
        "shift_um": shift.tolist(),
        "trend_um": (gradient @ extent).tolist(),
    }
