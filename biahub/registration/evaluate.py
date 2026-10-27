"""Registration evaluation: how well a transform registers, in µm, independent of the score.

Layer 1: geometry -- the plausibility of a transform, and statistics of matched residual
displacements. Layer 2: residuals of landmarks (beads, nuclei, clicked points). Layer 3:
image-based residuals (tiles and z crops) for data without landmarks. Layer 4: `evaluate`,
every layer for one transform next to the unregistered pair. Distances are in µm from the
stores' voxel sizes (z, y, x); voxel values are reported alongside where useful.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from numpy.typing import ArrayLike
from scipy.ndimage import shift as ndi_shift
from skimage import filters

from biahub.cli.parsing import position_key
from biahub.core.nodes.solvers import mutual_nearest
from biahub.core.transform import Transform, polar, rotation_angles
from biahub.registration.methods.intensity.phase_cross_corr import phase_cross_corr
from biahub.registration.methods.landmark.bead_matching import detect_beads
from biahub.registration.methods.landmark.nuclei_matching import detect_nuclei

# Fewer matched points than this do not constrain an affine residual (12 parameters,
# fitted to noisy centres: DragonFly nuclei gave a scale of 0.86 from 8 points).
MIN_POINTS_FOR_AFFINE = 10

# Fewer matched points than this give no z residual (one nucleus read -1.98 µm).
MIN_POINTS_FOR_Z = 5


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
    """Fit residual displacement = a + B @ mapped position over matched points, in µm.

    `points_vox` are the reference positions and `diff_vox` reference minus mapped moving,
    so what is left is a map acting on the mapped positions (`points - diff`): fitted
    there, a moving set at 0.8 x the reference reads a scale of 1.25.

    A residual that grows across the field is a scale or rotation error of the transform,
    which median distances alone hide; a constant residual is a shift. Returns the residual
    map's per-axis scale and rotation (1 and 0 for no such error), its shift in µm, and the
    displacement it accumulates across the extent of the points (`trend_um`, per axis).
    """
    diff = np.asarray(diff_vox, dtype=float).reshape(-1, 3) * np.asarray(voxel_um, float)
    ref = np.asarray(points_vox, dtype=float).reshape(-1, 3) * np.asarray(voxel_um, float)
    pts = ref - diff  # the mapped positions, where the residual map acts
    n = int(len(pts))
    if n < MIN_POINTS_FOR_AFFINE:
        return {"n": n, "scale_zyx": None, "rotation_yx_deg": None}
    # positions about their centre: points at one depth (tiles) make a constant column,
    # which would otherwise share the shift with a false z gradient
    centre = pts.mean(axis=0)
    design = np.c_[np.ones(n), pts - centre]
    coef, *_ = np.linalg.lstsq(design, diff, rcond=None)
    gradient = coef[1:].T  # diff ~ shift + gradient @ pos
    shift = coef[0] - gradient @ centre
    rotation, stretch = polar(np.eye(3) + gradient)
    _total, yx = rotation_angles(rotation)
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
        i, j = mutual_nearest(
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


# --- layer 3: image-based residuals -----------------------------------------------------
# For data without landmarks: local shifts between the warped moving volume and the
# reference, measured by phase correlation and summarised like matched points.


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
    return ComparablePair(warped, ref, transform.coverage(mov.shape, ref.shape), compare)


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
    _, ny, nx = ref_p.shape
    size = min(tile_px, ny, nx)  # a tile fits the image, and is exactly the window's size
    max_shift = size / 4 if max_shift_px is None else max_shift_px
    h = size // 2
    window = np.outer(np.hanning(size), np.hanning(size))
    points, diffs = [], []
    # distinct centres only: a tile that fits the image once is one measurement
    for cy in np.unique(np.linspace(h, ny - (size - h), grid).astype(int)):
        for cx in np.unique(np.linspace(h, nx - (size - h), grid).astype(int)):
            tile = (slice(cy - h, cy - h + size), slice(cx - h, cx - h + size))
            if covered[tile].mean() < min_coverage:
                continue
            r, w = r_mip[tile], w_mip[tile]
            if r.std() == 0 or w.std() == 0:
                continue
            r, w = _zscore(r), _zscore(w)
            shift, _ = phase_cross_corr(r * window, w * window, upsample_factor=10)
            if np.abs(shift).max() > max_shift:
                continue
            # the shared window pulls the peak toward zero shift (~5% short on nuclei):
            # measure the remainder once more, where that pull is a share of almost nothing
            moved = ndi_shift(w, shift, order=1, mode="nearest")
            remainder, _ = phase_cross_corr(r * window, moved * window, upsample_factor=10)
            shift = shift + remainder
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
        shift, _ = phase_cross_corr(_zscore(r), _zscore(w), upsample_factor=10)
        crops[name] = (np.asarray(shift) * um).tolist()
    # one value per distinct place: corners of a crop as big as the image are one place
    corner_dz = list(
        {starts[k]: v[0] for k, v in crops.items() if k != "centre" and v is not None}.values()
    )
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

    `geometry` is layer 1 (`Transform.decompose`); `points` layer 2 (`point_residuals`, None without
    landmarks); `tiles` and `z_crops` layer 3 (None when images are skipped). `baseline`
    holds the same measures for the unregistered pair (identity), so each number reads
    against doing nothing. `z_um` is the residual z shift (reference minus mapped moving)
    and `z_from` the measure it came from: points when there are any, since the image
    crops can be biased (cross-modal pairs), else the median of the z crops; `z_from`
    says when there were too few points (`MIN_POINTS_FOR_Z`). Tiles see yx only (max
    projections); a z error shows in points and z crops alone. Point distances are 3D and
    carry each landmark's centre error, so they read higher than tiles for the same
    registration.

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


def _detect(source: str, volume, voxel_um, **kwargs) -> Landmarks:
    """Landmarks of `source` ("beads" or "nuclei") found in `volume` by its method's detector."""
    points = LANDMARK_DETECTORS[source](volume, voxel_um, **kwargs)
    return Landmarks(points, tuple(voxel_um), source)


def _landmarks(landmarks, mov, ref, mov_um, ref_um, detect_kwargs) -> tuple | None:
    if landmarks is None or isinstance(landmarks, tuple):
        return landmarks
    if landmarks not in LANDMARK_DETECTORS:
        raise ValueError(
            f"landmarks must be one of {sorted(LANDMARK_DETECTORS)}, a (moving, reference) "
            f"pair of Landmarks or None, got {landmarks!r}"
        )
    kwargs = detect_kwargs or {}
    # one set for both, or {"moving": {...}, "reference": {...}} (e.g. bead thresholds)
    sides = kwargs if set(kwargs) <= {"moving", "reference"} and kwargs else None
    mov_kwargs = sides.get("moving", {}) if sides else kwargs
    ref_kwargs = sides.get("reference", {}) if sides else kwargs
    return (
        _detect(landmarks, mov, mov_um, **mov_kwargs),
        _detect(landmarks, ref, ref_um, **ref_kwargs),
    )


def _z(points: dict | None, z_crops: dict | None, points_source: str | None) -> tuple:
    """Return (z residual in µm, the measure it came from)."""
    n = 0 if points is None else points["stats"]["n"]
    if n >= MIN_POINTS_FOR_Z:
        return points["stats"]["median_zyx_um"][0], f"points ({points_source})"
    too_few = f"; {n} matched points are too few" if n else ""
    if z_crops is not None:
        dz = [v[0] for v in z_crops["crops_um"].values() if v is not None]
        if dz:
            return float(np.median(dz)), f"z crops ({z_crops['compare']}){too_few}"
    return None, (f"no z measure{too_few}" if too_few else None)


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
    baseline: bool = True,
    **image_kwargs,
) -> Evaluation:
    """Evaluate a forward transform on one moving / reference pair (ZYX volumes).

    `landmarks`: "nuclei" or "beads" (detected in both volumes; `detect_kwargs` go to the
    detector, e.g. beads' `threshold_abs`), a (moving, reference) pair of `Landmarks`
    (detected elsewhere or clicked), or None. `images` adds layer 3 (`compare`: "edges" or
    "intensity"; `tile_px`, `grid`, `crop_px` pass through), the slow part: one warp of
    the moving volume per transform. Thresholds for a verdict are left to the caller.
    `baseline=False` skips the unregistered pair (one warp less).
    """
    marks = _landmarks(landmarks, mov, ref, mov_um, ref_um, detect_kwargs)
    measures = _measures(transform, mov, ref, ref_um, marks, images, compare, image_kwargs)
    # the unregistered pair; `baseline=False` leaves it out (e.g. measured once, apart)
    baseline = (
        _measures(
            Transform.identity(3), mov, ref, ref_um, marks, images, compare, image_kwargs
        )
        if baseline
        else {}
    )
    return Evaluation(
        geometry=transform.decompose(mov_um, ref_um),
        baseline=baseline,
        projections={**display_projections(transform, mov, ref), "ref_um": tuple(ref_um)},
        **measures,
    )


# --- many candidates on many positions (the benchmark's core) ------------------------------


def _row(record: Evaluation, candidate: str, position: str, t: int) -> dict:
    """Flatten one evaluation into a table row."""
    points, tiles = record.points or {}, record.tiles or {}
    affine = points.get("affine") or {}
    scale = affine.get("scale_zyx")
    return {
        "candidate": candidate,
        "position": position,
        "t": t,
        "tiles_n": tiles.get("n_tiles"),
        "tiles_median_um": (tiles.get("stats") or {}).get("median_um"),
        "points_n": points.get("n_matched"),
        "points_median_um": (points.get("stats") or {}).get("median_um"),
        "scale_yx": None if scale is None else list(scale[1:]),
        "rotation_yx_deg": affine.get("rotation_yx_deg"),
        "z_um": record.z_um,
        "z_from": record.z_from,
        "transform_scale_zyx": record.geometry["scale_zyx"],
        "transform_rotation_yx_deg": record.geometry["rotation_yx_deg"],
        "empty": None,
    }


def _blank(volume: np.ndarray) -> bool:
    """Whether a volume holds no signal: never written, all NaN, or one constant value."""
    finite = volume[np.isfinite(volume)]
    return finite.size == 0 or finite.min() == finite.max()


_MEASURED = (
    "tiles_n",
    "tiles_median_um",
    "points_n",
    "points_median_um",
    "scale_yx",
    "rotation_yx_deg",
    "z_um",
    "z_from",
    "transform_scale_zyx",
    "transform_rotation_yx_deg",
)


def _empty_row(position: str, t: int, empty: str) -> dict:
    """Return the one row of a position with nothing to compare: `_row`'s columns, empty."""
    row = {"candidate": "unregistered", "position": position, "t": t}
    return {**row, **dict.fromkeys(_MEASURED), "empty": empty}


def evaluate_candidates(
    candidates: dict,
    position_pairs: list,
    moving_channel: str,
    reference_channel: str,
    t: int = 0,
    landmarks: str | None = None,
    images: bool = True,
    compare: str = "edges",
    detect_kwargs: dict | None = None,
    figures_dir=None,
    reference_t: int | None = None,
    **image_kwargs,
) -> list[dict]:
    """Evaluate every candidate on every position at `t`: one row each (`_row`).

    `candidates` maps a name to a forward `Transform` for every position, or to a dict
    keyed by `t` or by (position, t) (e.g. `block_candidates`); a candidate with no
    transform for a position is left out there. `position_pairs` are (moving, reference)
    position paths. Each position also gets an "unregistered" row (identity), measured
    once; landmarks are detected once per position. A position whose moving or reference
    image is blank (no signal) gets one row instead, its `empty` saying which, and a warning.
    With `figures_dir`, each candidate's
    figure (`plot_evaluation`, against that unregistered pair) is saved there.
    `reference_t` reads the reference at another timepoint than `t` (a stabilization's
    reference frame); by default the same.
    """
    from iohub import open_ome_zarr

    rows = []
    for mov_path, ref_path in position_pairs:
        key = position_key(mov_path)
        with open_ome_zarr(mov_path, mode="r") as m, open_ome_zarr(ref_path, mode="r") as r:
            mov = np.asarray(m.data[t, m.channel_names.index(moving_channel)], np.float32)
            t_ref = t if reference_t is None else reference_t
            ref = np.asarray(
                r.data[t_ref, r.channel_names.index(reference_channel)], np.float32
            )
            mov_um, ref_um = tuple(m.scale[-3:]), tuple(r.scale[-3:])
        sides = [side for side, v in (("moving", mov), ("reference", ref)) if _blank(v)]
        if sides:  # nothing to compare: one row says so, instead of a row per candidate
            empty = "both" if len(sides) == 2 else sides[0]
            warnings.warn(
                f"{key} t={t}: the {empty} image is blank; not evaluated", stacklevel=2
            )
            rows.append(_empty_row(key, t, empty))
            continue
        marks = _landmarks(landmarks, mov, ref, mov_um, ref_um, detect_kwargs)
        named = {"unregistered": Transform.identity(3), **candidates}
        for name, candidate in named.items():
            transform = (
                candidate
                if isinstance(candidate, Transform)
                else candidate.get((key, t), candidate.get(t))
            )
            if transform is None:
                continue
            record = evaluate(
                transform,
                mov,
                ref,
                mov_um,
                ref_um,
                landmarks=marks,
                images=images,
                compare=compare,
                baseline=False,
                **image_kwargs,
            )
            rows.append(_row(record, name, key, t))
            if name == "unregistered":
                unregistered = {
                    k: getattr(record, k)
                    for k in ("points", "tiles", "z_crops", "z_um", "z_from")
                }
            elif figures_dir is not None:
                from biahub.registration.evaluate_view import plot_evaluation

                record.baseline = unregistered  # measured once for the position
                Path(figures_dir).mkdir(parents=True, exist_ok=True)
                name_safe = f"{name}_{key}_t{t}".replace("/", "_").replace(":", "-")
                plot_evaluation(
                    record,
                    path=Path(figures_dir) / f"{name_safe}.png",
                    title=f"{name} {key} t={t}",
                )
    return rows


def block_candidates(run_dir) -> dict:
    """Return each block of a stack run as a candidate: {block: {t or (position, t): T}}.

    From the per-timepoint records the run wrote (`blocks`: every block's transform),
    under `<run>/timepoints/` or, one list per position, `<run>/positions/<r/c/f>/...` --
    only the layout of the positions in the run's `run.json`, not an earlier run's.
    """
    import json

    run_dir = Path(run_dir)
    run_file = run_dir / "run.json"  # the run's positions: only its own layout is read
    positions = json.loads(run_file.read_text())["positions"] if run_file.exists() else None
    if positions is None:
        record_dirs = [(None, run_dir / "timepoints")] + [
            ("/".join(d.parent.relative_to(run_dir / "positions").parts), d)
            for d in sorted((run_dir / "positions").glob("*/*/*/timepoints"))
        ]
    elif len(positions) == 1:
        record_dirs = [(None, run_dir / "timepoints")]
    else:
        record_dirs = [(key, run_dir / "positions" / key / "timepoints") for key in positions]
    candidates: dict = {}
    for position, directory in record_dirs:
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            record = json.loads(path.read_text())
            for block in record.get("blocks") or []:
                if block["matrix"] is None:
                    continue
                key = record["t"] if position is None else (position, record["t"])
                candidates.setdefault(block["block"], {})[key] = Transform(
                    np.asarray(block["matrix"], dtype=float)
                )
    return candidates
