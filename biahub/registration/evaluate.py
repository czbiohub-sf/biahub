"""Registration evaluation: how well a transform registers, in µm, independent of the score.

Layer 1: geometry -- the plausibility of a transform, and statistics of matched residual
displacements. Layer 2: residuals of landmarks (beads, nuclei, clicked points). Layer 3:
image-based residuals (tiles and z crops) for data without landmarks. Layer 4: `evaluate`,
every layer for one transform next to the unregistered pair. Distances are in µm from the
stores' voxel sizes (z, y, x); voxel values are reported alongside where useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from numpy.typing import ArrayLike
from scipy.ndimage import shift as ndi_shift
from scipy.spatial import cKDTree
from skimage import filters
from skimage.registration import phase_cross_correlation

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


def _residuals(points_vox: ArrayLike, diff_vox: ArrayLike, voxel_um: ArrayLike) -> dict:
    """Return each residual with where it was measured, in µm (for the overlays)."""
    um = np.asarray(voxel_um, float)
    return {
        "position_um": (np.asarray(points_vox, float).reshape(-1, 3) * um).tolist(),
        "diff_um": (np.asarray(diff_vox, float).reshape(-1, 3) * um).tolist(),
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
        "residuals": _residuals(ref.points_vox[j], diff, ref.voxel_um),
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


# --- layer 3: image-based residuals -----------------------------------------------------
# For data without landmarks: local shifts between the warped moving volume and the
# reference, measured by phase correlation and summarised like matched points.


def coverage(transform: Transform, mov_shape: tuple, ref_shape: tuple) -> np.ndarray:
    """Return where `transform.apply` has moving data on the reference grid (boolean).

    The same as warping a volume of ones, without the warp: each reference voxel maps back
    into the moving volume, and along a row (the last axis) an affine map is a line, so
    the covered part of each row is one interval. The edge is the resampler's: ANTs (3D)
    samples up to half a voxel beyond the outer centres, the upper edge excluded; scipy
    (2D) up to the outer centres.
    """
    inverse = np.linalg.inv(transform.matrix)
    ndim = len(ref_shape)
    starts = np.indices(ref_shape[:-1], dtype=float).reshape(ndim - 1, -1)  # each row's
    first = np.full(starts.shape[1], -np.inf)
    last = np.full(starts.shape[1], np.inf)
    for axis in range(ndim):
        at_zero = inverse[axis, : ndim - 1] @ starts + inverse[axis, ndim]  # moving coord
        step = inverse[axis, ndim - 1]  # its change per voxel along the row
        if ndim == 3:
            lower, upper = -0.5, mov_shape[axis] - 0.5 - 1e-9
        else:
            lower, upper = 0.0, mov_shape[axis] - 1 + 1e-9
        if step == 0:
            outside = (at_zero < lower) | (at_zero > upper)
            first[outside], last[outside] = np.inf, -np.inf
            continue
        a, b = (lower - at_zero) / step, (upper - at_zero) / step
        first = np.maximum(first, np.minimum(a, b))
        last = np.minimum(last, np.maximum(a, b))
    x = np.arange(ref_shape[-1])
    return ((x >= first[:, None]) & (x <= last[:, None])).reshape(ref_shape)


@dataclass
class ComparablePair:
    """The warped moving volume and the reference, prepared once for layer 3's measures."""

    warped: np.ndarray
    ref: np.ndarray
    has_data: np.ndarray  # where the warped moving volume has data
    compare: str


def prepare_pair(transform: Transform, mov: ArrayLike, ref: ArrayLike, compare: str = "edges"):
    """Return the pair `tile_residuals` and `z_residuals` compare, to share between them.

    `edges` takes the Sobel magnitude in each volume's own grid before warping, so the
    empty padding carries no edges and cross-modal pairs compare structure; `intensity`
    compares the volumes as they are.
    """
    if compare not in ("edges", "intensity"):
        raise ValueError(f"compare must be 'edges' or 'intensity', got {compare!r}")
    mov = np.nan_to_num(np.asarray(mov, dtype=np.float32))
    ref = np.nan_to_num(np.asarray(ref, dtype=np.float32))
    if compare == "edges":
        mov, ref = filters.sobel(mov).astype(np.float32), filters.sobel(ref).astype(np.float32)
    if transform.is_identity and mov.shape == ref.shape:
        # the unregistered pair: an identity warp returns the volume unchanged
        return ComparablePair(mov, ref, np.ones(ref.shape, dtype=bool), compare)
    warped = transform.apply(mov, reference=ref)
    return ComparablePair(warped, ref, coverage(transform, mov.shape, ref.shape), compare)


def _pair(transform, mov, ref, compare: str, pair: ComparablePair | None) -> ComparablePair:
    if pair is None:
        return prepare_pair(transform, mov, ref, compare)
    if pair.compare != compare:
        raise ValueError(f"the pair was prepared for {pair.compare!r}, not {compare!r}")
    return pair


def _zscore(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


def tile_residuals(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    voxel_um: tuple,
    compare: str = "edges",
    tile_px: int = 256,
    grid: int = 7,
    min_ncc: float = 0.5,
    max_shift_px: float | None = None,
    min_coverage: float = 0.95,
    pair: ComparablePair | None = None,
) -> dict:
    """Return local yx residuals over a grid of tiles of the max projections, in µm.

    Each tile of the warped moving projection is phase-correlated with the reference's; a
    tile counts when it is mostly real data, its shift is plausible (under `max_shift_px`,
    default a quarter tile) and its shifted copy correlates above `min_ncc`. Displacements
    follow `point_residuals`' convention (reference minus mapped moving) and are summarised
    by `displacement_stats` and `residual_affine` (a radial pattern = a scale error).
    `pair` (from `prepare_pair`) skips preparing the volumes again.
    """
    pair = _pair(transform, mov, ref, compare, pair)
    warped, ref_p, has_data = pair.warped, pair.ref, pair.has_data
    w_mip, r_mip = warped.max(axis=0), ref_p.max(axis=0)
    covered = has_data.mean(axis=0) >= 0.5
    max_shift = tile_px / 4 if max_shift_px is None else max_shift_px
    h = tile_px // 2
    _, ny, nx = ref_p.shape
    window = np.outer(np.hanning(tile_px), np.hanning(tile_px))
    points, diffs = [], []
    for cy in np.linspace(h, ny - h, grid).astype(int):
        for cx in np.linspace(h, nx - h, grid).astype(int):
            tile = (slice(cy - h, cy + h), slice(cx - h, cx + h))
            if covered[tile].mean() < min_coverage:
                continue
            r, w = r_mip[tile], w_mip[tile]
            if r.std() == 0 or w.std() == 0:
                continue
            r, w = _zscore(r), _zscore(w)
            shift, _error, _phase = phase_cross_correlation(
                r * window, w * window, upsample_factor=10, normalization=None
            )
            if np.abs(shift).max() > max_shift:
                continue
            moved = ndi_shift(w, shift, order=1, mode="nearest")
            if np.corrcoef(r.ravel(), moved.ravel())[0, 1] < min_ncc:
                continue
            points.append((ref_p.shape[0] / 2, cy, cx))
            diffs.append((0.0, *shift))  # moving shifted by `shift` matches: ref - mapped
    points, diffs = np.array(points).reshape(-1, 3), np.array(diffs).reshape(-1, 3)
    return {
        "compare": compare,
        "n_tiles": int(len(points)),
        "stats": displacement_stats(diffs, voxel_um),
        "affine": residual_affine(points, diffs, voxel_um),
        "residuals": _residuals(points, diffs, voxel_um),
    }


def z_residuals(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    voxel_um: tuple,
    compare: str = "edges",
    crop_px: int = 768,
    min_coverage: float = 0.5,
    pair: ComparablePair | None = None,
) -> dict:
    """Return the residual shift in the centre and 4 corner crops (3D phase correlation), µm.

    Different corners show a tilt (`tilt_dz_um`: the spread of their z residuals). A crop
    that is mostly padding is skipped; in intensity mode the padding is filled with the
    crop's median so its edge does not dominate the correlation. `pair` (from
    `prepare_pair`) skips preparing the volumes again.
    """
    pair = _pair(transform, mov, ref, compare, pair)
    warped, ref_p, has_data = pair.warped, pair.ref, pair.has_data
    _, ny, nx = ref_p.shape
    cy, cx = min(crop_px, ny), min(crop_px, nx)
    starts = {
        "centre": ((ny - cy) // 2, (nx - cx) // 2),
        "top_left": (0, 0),
        "top_right": (0, nx - cx),
        "bottom_left": (ny - cy, 0),
        "bottom_right": (ny - cy, nx - cx),
    }
    um = np.asarray(voxel_um, float)
    crops = {}
    for name, (y0, x0) in starts.items():
        box = (slice(None), slice(y0, y0 + cy), slice(x0, x0 + cx))
        w, r, cov = warped[box], ref_p[box], has_data[box]
        if cov.mean() < min_coverage or r.std() == 0 or w[cov].std() == 0:
            crops[name] = None
            continue
        if compare == "intensity":
            w = np.where(cov, w, np.median(w[cov]))
        shift, _error, _phase = phase_cross_correlation(
            _zscore(r), _zscore(w), upsample_factor=10, normalization=None
        )
        crops[name] = (np.asarray(shift) * um).tolist()
    corner_dz = [v[0] for k, v in crops.items() if k != "centre" and v is not None]
    return {
        "compare": compare,
        "crops_um": crops,
        "tilt_dz_um": float(max(corner_dz) - min(corner_dz)) if len(corner_dz) > 1 else None,
    }


# --- layer 4: one record per position and timepoint -------------------------------------
# Every layer for one transform, next to the same measures for doing nothing (identity).


@dataclass
class Evaluation:
    """How well one transform registers one position and timepoint, in µm.

    `geometry` is layer 1 (`decompose`); `points` layer 2 (`point_residuals`, None without
    landmarks); `tiles` and `z_crops` layer 3 (None when images are skipped). `baseline`
    holds the same measures for the unregistered pair (identity), so each number reads
    against doing nothing. `z_um` is the residual z shift (reference minus mapped moving)
    and `z_from` the measure it came from: points when there are any, since the image
    crops can be biased (cross-modal pairs), else the median of the z crops. Tiles see yx
    only (max projections); a z error shows in points and z crops alone.

    The baseline's points can match few: a large scale or shift error moves landmarks
    beyond the matching radius (DragonFly unregistered: 5-7 of ~18 nuclei), so read the
    unregistered pair from its tiles too.
    """

    geometry: dict
    points: dict | None
    tiles: dict | None
    z_crops: dict | None
    z_um: float | None
    z_from: str | None
    baseline: dict
    # Max projections for the views (`display_projections`); kept in memory, not stored.
    projections: dict | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "projections"}


# The views' projections are at most this many pixels on a side.
DISPLAY_PX = 512


def _shrink_yx(volume: np.ndarray, factor: int) -> np.ndarray:
    """Return `volume` averaged over factor x factor blocks in yx (edges cropped)."""
    z, y, x = volume.shape
    y, x = y - y % factor, x - x % factor
    blocks = volume[:, :y, :x].reshape(z, y // factor, factor, x // factor, factor)
    return blocks.mean(axis=(2, 4), dtype=np.float32)


def display_volumes(transform: Transform, mov: ArrayLike, ref: ArrayLike) -> tuple:
    """Return (warped moving, reference, downsample) averaged down in yx for display.

    Both volumes shrink to at most `DISPLAY_PX` a side and the moving one is warped there
    by the transform expressed on the small grid, so a tilt or z shift shows as it would at
    full size, for a fraction of the warp's cost. Full-size pixel = downsample * small
    pixel + (downsample - 1) / 2.
    """
    mov = np.nan_to_num(np.asarray(mov, dtype=np.float32))
    ref = np.nan_to_num(np.asarray(ref, dtype=np.float32))
    factor = max(1, int(np.ceil(max(ref.shape[1:]) / DISPLAY_PX)))
    small_mov, small_ref = _shrink_yx(mov, factor), _shrink_yx(ref, factor)
    # small index d sits at full index factor * d + (factor - 1) / 2 in y and x
    to_full = np.diag([1.0, factor, factor, 1.0])
    to_full[1:3, 3] = (factor - 1) / 2
    small = Transform(np.linalg.inv(to_full) @ np.asarray(transform.matrix) @ to_full)
    return prepare_pair(small, small_mov, small_ref, "intensity").warped, small_ref, factor


def display_projections(transform: Transform, mov: ArrayLike, ref: ArrayLike) -> dict:
    """Return max projections of the reference and the moving volume, registered and not.

    For the views only, from `display_volumes` (`downsample` converts back to full size).
    """
    registered, small_ref, factor = display_volumes(transform, mov, ref)
    unregistered, _ref, _factor = display_volumes(Transform.identity(3), mov, ref)
    return {
        "reference": small_ref.max(axis=0),
        "registered": registered.max(axis=0),
        "unregistered": unregistered.max(axis=0),
        "downsample": factor,
    }


LANDMARK_DETECTORS = {"nuclei": detect_nuclei, "beads": detect_beads}


def _landmarks(landmarks, mov, ref, mov_um, ref_um, detect_kwargs) -> tuple | None:
    if landmarks is None or isinstance(landmarks, tuple):
        return landmarks
    if landmarks not in LANDMARK_DETECTORS:
        raise ValueError(
            f"landmarks must be one of {sorted(LANDMARK_DETECTORS)}, a (moving, reference) "
            f"pair of Landmarks or None, got {landmarks!r}"
        )
    detect = LANDMARK_DETECTORS[landmarks]
    kwargs = detect_kwargs or {}
    return detect(mov, mov_um, **kwargs), detect(ref, ref_um, **kwargs)


def _z(points: dict | None, z_crops: dict | None, points_source: str | None) -> tuple:
    """Return (z residual in µm, the measure it came from)."""
    if points is not None and points["stats"]["n"] > 0:
        return points["stats"]["median_zyx_um"][0], f"points ({points_source})"
    if z_crops is not None:
        dz = [v[0] for v in z_crops["crops_um"].values() if v is not None]
        if dz:
            return float(np.median(dz)), f"z crops ({z_crops['compare']})"
    return None, None


def _measures(transform, mov, ref, ref_um, marks, images, compare, image_kwargs) -> dict:
    """Return layers 2-3 and the z residual for one transform."""
    points = point_residuals(transform, *marks) if marks is not None else None
    tiles = z_crops = None
    if images:
        tile_kwargs = {k: v for k, v in image_kwargs.items() if k in ("tile_px", "grid")}
        crop_kwargs = {k: v for k, v in image_kwargs.items() if k in ("crop_px",)}
        pair = prepare_pair(transform, mov, ref, compare)
        tiles = tile_residuals(transform, mov, ref, ref_um, compare, pair=pair, **tile_kwargs)
        z_crops = z_residuals(transform, mov, ref, ref_um, compare, pair=pair, **crop_kwargs)
    z_um, z_from = _z(points, z_crops, marks[1].source if marks is not None else None)
    return {
        "points": points,
        "tiles": tiles,
        "z_crops": z_crops,
        "z_um": z_um,
        "z_from": z_from,
    }


def evaluate(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    mov_um: tuple,
    ref_um: tuple,
    landmarks: str | tuple | None = None,
    images: bool = True,
    compare: str = "edges",
    detect_kwargs: dict | None = None,
    **image_kwargs,
) -> Evaluation:
    """Evaluate a forward transform on one moving / reference pair (ZYX volumes).

    `landmarks`: "nuclei" or "beads" (detected in both volumes; `detect_kwargs` go to the
    detector, e.g. beads' `threshold_abs`), a (moving, reference) pair of `Landmarks`
    (detected elsewhere or clicked), or None. `images` adds layer 3 (`compare`: "edges" or
    "intensity"; `tile_px`, `grid`, `crop_px` pass through), the slow part: one warp of
    the moving volume per transform. Thresholds for a verdict are left to the caller.
    """
    marks = _landmarks(landmarks, mov, ref, mov_um, ref_um, detect_kwargs)
    measures = _measures(transform, mov, ref, ref_um, marks, images, compare, image_kwargs)
    baseline = _measures(
        Transform.identity(3), mov, ref, ref_um, marks, images, compare, image_kwargs
    )
    return Evaluation(
        geometry=decompose(transform, mov_um, ref_um),
        baseline=baseline,
        projections={**display_projections(transform, mov, ref), "ref_um": tuple(ref_um)},
        **measures,
    )
