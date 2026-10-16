"""Registration evaluation: how well a transform registers, in µm, independent of the score.

Layer 1 (here): geometry -- the plausibility of a transform, and statistics of matched
residual displacements. Distances are in µm from the stores' voxel sizes (z, y, x); voxel
values are reported alongside where useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from biahub.characterize_psf import detect_peaks
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


# --- layer 2: point-based residuals -----------------------------------------------------
# Landmarks seen in both channels (beads, nuclei, or points a person clicked), mapped through
# the transform without warping any image, and compared where they pair up.


@dataclass
class Landmarks:
    """Points in one volume: N x 3 voxel coordinates (z, y, x), with that volume's voxel size.

    `pair_ids` marks known correspondences (points a person clicked in both channels): the
    point with id i in the moving set is the one with id i in the reference set. Detected
    landmarks have none and are matched by distance.
    """

    points_vox: np.ndarray
    voxel_um: tuple
    source: str  # "beads", "nuclei" or "manual"
    pair_ids: list | None = None

    def __post_init__(self):
        self.points_vox = np.asarray(self.points_vox, dtype=float).reshape(-1, 3)


# Matching radius (µm) and the axes it is measured on, per landmark source. Nuclei are long
# in z, so they pair up in yx and their z is compared afterwards.
MATCHING = {"beads": (1.0, "zyx"), "nuclei": (3.0, "yx")}


def _mutual_nearest(
    a_um: np.ndarray, b_um: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices (i, j) where a[i] and b[j] are each other's nearest, within `radius`."""
    if len(a_um) == 0 or len(b_um) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d_ab, j = cKDTree(b_um).query(a_um)
    _d_ba, i_back = cKDTree(a_um).query(b_um)
    keep = (d_ab <= radius) & (i_back[j] == np.arange(len(a_um)))
    return np.flatnonzero(keep), j[keep]


def point_residuals(
    transform: Transform,
    mov: Landmarks,
    ref: Landmarks,
    max_distance_um: float | None = None,
) -> dict:
    """Return how far `transform` leaves each landmark from its partner, in µm.

    The moving points are mapped onto the reference (no warping). Clicked points pair by
    `pair_ids`; detected ones by mutual nearest neighbour within `max_distance_um` (default
    per source: `MATCHING`). Displacements are reference minus mapped moving, in reference
    voxels, summarised by `displacement_stats` and `residual_affine`.
    """
    mapped = transform.apply_points(mov.points_vox) if len(mov.points_vox) else mov.points_vox
    if mov.pair_ids is not None and ref.pair_ids is not None:
        ref_index = {pid: k for k, pid in enumerate(ref.pair_ids)}
        pairs = [(k, ref_index[pid]) for k, pid in enumerate(mov.pair_ids) if pid in ref_index]
        i = np.array([p[0] for p in pairs], int)
        j = np.array([p[1] for p in pairs], int)
        matched_by = "pair_ids"
    else:
        radius, axes = MATCHING.get(ref.source, (1.0, "zyx"))
        radius = radius if max_distance_um is None else max_distance_um
        cols = slice(1, 3) if axes == "yx" else slice(0, 3)
        um = np.asarray(ref.voxel_um, float)
        i, j = _mutual_nearest(
            mapped[:, cols] * um[cols], ref.points_vox[:, cols] * um[cols], radius
        )
        matched_by = f"mutual nearest within {radius} um ({axes})"
    diff = ref.points_vox[j] - mapped[i]
    return {
        "n_mov": int(len(mov.points_vox)),
        "n_ref": int(len(ref.points_vox)),
        "n_matched": int(len(i)),
        "matched_by": matched_by,
        "stats": displacement_stats(diff, ref.voxel_um),
        "affine": residual_affine(ref.points_vox[j], diff, ref.voxel_um),
    }


def detect_beads(
    volume: ArrayLike,
    voxel_um: tuple,
    threshold_abs: float,
    block_size: tuple = (8, 8, 8),
    nms_distance: int = 16,
    refine_halfwidth: tuple = (2, 4, 4),
) -> Landmarks:
    """Return bead centres to a fraction of a voxel: biahub's peaks, refined by centroid.

    Peaks sit on whole voxels; each is refined to the intensity centroid of a small box
    around it (background-subtracted), which a residual of a voxel or less needs.
    """
    vol = np.asarray(volume, dtype=np.float32)
    peaks = np.asarray(
        detect_peaks(
            vol,
            block_size=block_size,
            threshold_abs=threshold_abs,
            nms_distance=nms_distance,
            min_distance=0,
        ),
        dtype=int,
    ).reshape(-1, 3)
    background = float(np.median(vol))
    refined = []
    for z, y, x in peaks:
        box = tuple(
            slice(max(c - h, 0), min(c + h + 1, n))
            for c, h, n in zip((z, y, x), refine_halfwidth, vol.shape, strict=True)
        )
        weights = np.clip(vol[box] - background, 0, None)
        if weights.sum() <= 0:
            refined.append([z, y, x])
            continue
        grids = np.meshgrid(*[np.arange(s.start, s.stop) for s in box], indexing="ij")
        refined.append([(g * weights).sum() / weights.sum() for g in grids])
    return Landmarks(np.array(refined, dtype=float).reshape(-1, 3), tuple(voxel_um), "beads")


def detect_nuclei(
    volume: ArrayLike,
    voxel_um: tuple,
    min_area_um2: float = 8.0,
    max_area_um2: float = 320.0,
    min_solidity: float = 0.9,
    smooth_um: float = 0.2,
) -> Landmarks:
    """Return nucleus centres: yx from the max projection's objects, z from each profile.

    Objects are kept when compact (solidity), of nuclear size (area in µm², so the same
    settings hold across instruments) and whole (not cut by the frame). z is the centroid of
    the nucleus's mean intensity profile above half its maximum.
    """
    from scipy.ndimage import gaussian_filter
    from skimage import filters, measure

    vol = np.asarray(volume, dtype=np.float32)
    _vz, vy, vx = voxel_um
    mip = gaussian_filter(vol.max(axis=0), smooth_um / vy)
    if not np.isfinite(mip).all() or mip.std() == 0:
        return Landmarks(np.zeros((0, 3)), tuple(voxel_um), "nuclei")
    labels = measure.label(mip > filters.threshold_otsu(mip))
    background = np.median(vol[:, labels == 0], axis=1) if (labels == 0).any() else 0.0
    points = []
    for region in measure.regionprops(labels):
        area_um2 = region.area * vy * vx
        if not (min_area_um2 <= area_um2 <= max_area_um2) or region.solidity < min_solidity:
            continue
        y0, x0, y1, x1 = region.bbox
        if y0 == 0 or x0 == 0 or y1 == mip.shape[0] or x1 == mip.shape[1]:
            continue  # cut by the frame: its centre is not its centre
        profile = vol[:, y0:y1, x0:x1][:, region.image].mean(axis=1) - background
        profile = profile - profile.min()
        top = profile >= 0.5 * profile.max()
        z = float((np.arange(len(profile))[top] * profile[top]).sum() / profile[top].sum())
        points.append((z, *region.centroid))
    return Landmarks(np.array(points, dtype=float).reshape(-1, 3), tuple(voxel_um), "nuclei")
