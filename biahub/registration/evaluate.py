"""Registration evaluation: how well a transform registers, in µm, independent of the score.

`evaluate` runs the metrics (`biahub.registration.metrics`) for one transform next to the
unregistered pair: its geometry (`Transform.decompose`), residuals of landmarks (beads,
nuclei, clicked points) and image-based residuals (tiles and z crops) for data without
landmarks, and keeps them in one `Evaluation`. Distances are in µm from the stores' voxel
sizes (z, y, x); voxel values are reported alongside where useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.methods.landmark.bead_matching import detect_beads
from biahub.registration.methods.landmark.nuclei_matching import detect_nuclei
from biahub.registration.metrics import (
    Landmarks,
    point_residuals,
    prepare_pair,
    tile_residuals,
    z_residuals,
)

# Fewer matched points than this give no z residual (one nucleus read -1.98 µm).
MIN_POINTS_FOR_Z = 5


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


def detect_landmarks(landmarks, mov, ref, mov_um, ref_um, detect_kwargs) -> tuple | None:
    """Return (moving, reference) `Landmarks`: detected for "beads" / "nuclei", or as given.

    `detect_kwargs` hold one set of detector settings for both, or {"moving": {...},
    "reference": {...}}. None (no landmarks) and a ready pair pass through.
    """
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
    marks = detect_landmarks(landmarks, mov, ref, mov_um, ref_um, detect_kwargs)
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
