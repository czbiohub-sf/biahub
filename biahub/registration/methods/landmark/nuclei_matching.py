"""The nuclei method: landmark-based, with nuclei for landmarks.

The same parts as beads (`NodeGraphEstimator`): a detector per channel -- here nuclei
(`detect_nuclei`, below) -- the beads' graph matchers, and a fit. Nuclei seen in both
channels (DAPI or H2B onto virtual-stained nuclei, ...) are detected on the moving volume
warped by the current transform and on the reference, matched, and fitted. A cell layer
is thin, so its nuclei constrain yx and a z offset but no z scale or tilt: the fit is the
transform type in yx plus the median z shift.

The seed must bring the nuclei within the matcher's reach (a few µm); put a coarse block
before this one when it may be further off.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.estimators import ScoreFn, fit_pairs
from biahub.registration.methods.landmark.bead_matching import NodeGraphEstimator
from biahub.settings import AffineTransformSettings, BeadsMatchSettings, NucleiSettings


def detect_nuclei(
    volume: ArrayLike,
    voxel_um: tuple,
    min_area_um2: float = 8.0,
    max_area_um2: float = 320.0,
    min_solidity: float = 0.9,
    smooth_um: float = 0.2,
) -> np.ndarray:
    """Return nucleus centres (N x 3 voxels, z y x): yx from the projection, z from profiles.

    Objects are kept when compact (solidity), of nuclear size (area in µm², so the same
    settings hold across instruments) and whole (not cut by the frame or by missing data).
    z is the centroid of the nucleus's mean intensity profile above half its maximum. NaN
    marks voxels with no data (e.g. a warp's padding): they are left out, never read as
    dark, so padding cannot pull the threshold below the background.
    """
    import warnings

    from scipy.ndimage import binary_dilation, gaussian_filter
    from skimage import filters, measure

    vol = np.asarray(volume, dtype=np.float32)
    _vz, vy, vx = voxel_um
    empty = np.zeros((0, 3))
    with warnings.catch_warnings():  # columns with no data at all: NaN, left out below
        warnings.simplefilter("ignore", RuntimeWarning)
        mip = np.nanmax(vol, axis=0)
    has_data = np.isfinite(mip)
    if not has_data.any():
        return empty
    mip = gaussian_filter(np.where(has_data, mip, np.median(mip[has_data])), smooth_um / vy)
    if mip[has_data].std() == 0:
        return empty
    labels = measure.label((mip > filters.threshold_otsu(mip[has_data])) & has_data)
    near_missing = binary_dilation(~has_data) if not has_data.all() else None
    outside = (labels == 0) & has_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        background = np.nanmedian(vol[:, outside], axis=1) if outside.any() else 0.0
    points = []
    for region in measure.regionprops(labels):
        area_um2 = region.area * vy * vx
        if not (min_area_um2 <= area_um2 <= max_area_um2) or region.solidity < min_solidity:
            continue
        y0, x0, y1, x1 = region.bbox
        if y0 == 0 or x0 == 0 or y1 == mip.shape[0] or x1 == mip.shape[1]:
            continue  # cut by the frame: its centre is not its centre
        if near_missing is not None and near_missing[y0:y1, x0:x1][region.image].any():
            continue  # cut by missing data, likewise
        if vol.shape[0] == 1:  # a single slice: no depth to measure, z is that slice
            points.append((0.0, *region.centroid))
            continue
        with warnings.catch_warnings():  # depths with no data: NaN, left out
            warnings.simplefilter("ignore", RuntimeWarning)
            profile = np.nanmean(vol[:, y0:y1, x0:x1][:, region.image], axis=1) - background
        depth = np.isfinite(profile)
        if not depth.any():
            continue
        profile = np.where(depth, profile - profile[depth].min(), 0.0)
        if profile.max() <= 0:
            continue  # the same at every depth: no z to measure
        top = depth & (profile >= 0.5 * profile.max())
        z = float((np.arange(len(profile))[top] * profile[top]).sum() / profile[top].sum())
        points.append((z, *region.centroid))
    return np.array(points, dtype=float).reshape(-1, 3)


class NucleiNodeDetector:
    """Detects nucleus centres (voxels, z y x) as nodes for graph matching."""

    # `detect_nuclei` leaves NaN out, so the estimator marks a warp's padding as NaN
    missing_as_nan = True

    def __init__(self, settings: NucleiSettings, voxel_um: tuple[float, float, float]):
        self.settings = settings
        self.voxel_um = tuple(voxel_um)

    def detect(self, array: ArrayLike) -> np.ndarray:
        s = self.settings
        return detect_nuclei(
            array,
            self.voxel_um,
            min_area_um2=s.min_area_um2,
            max_area_um2=s.max_area_um2,
            min_solidity=s.min_solidity,
            smooth_um=s.smooth_um,
        )


def yx_and_z_shift_fit(transform_type: str, voxel_um: tuple[float, float, float]):
    """Return a fit for matched nodes on one grid: the type in yx (in µm), the z shift."""
    um = np.asarray(voxel_um, float)

    def fit(matches, mov_nodes, ref_nodes) -> Transform:
        src = np.asarray(mov_nodes)[matches[:, 0]] * um
        dst = np.asarray(ref_nodes)[matches[:, 1]] * um
        yx = fit_pairs(src[:, 1:], dst[:, 1:], transform_type)
        in_um = np.eye(4)
        in_um[1:3, 1:3] = yx[:2, :2]
        in_um[1:3, 3] = yx[:2, 2]
        in_um[0, 3] = float(np.median(dst[:, 0] - src[:, 0]))
        to_vox, from_vox = np.diag([*(1.0 / um), 1.0]), np.diag([*um, 1.0])
        return Transform(to_vox @ in_um @ from_vox, transform_type=transform_type)

    return fit


def nuclei_estimator(
    settings: NucleiSettings,
    transform_type: str,
    ref_um: tuple[float, float, float],
    score_fn: ScoreFn | None = None,
) -> NodeGraphEstimator:
    """Build the nuclei method: nucleus detectors, the configured matcher, a yx + z fit.

    Both channels are detected on the reference grid (the moving volume is warped onto
    it first), so `ref_um` is the voxel size for both.
    """
    matching = BeadsMatchSettings(
        algorithm=settings.algorithm,
        source_peaks_settings=None,
        target_peaks_settings=None,
        hungarian_match_settings=settings.hungarian_match_settings,
        spectral_match_settings=settings.spectral_match_settings,
        match_descriptor_settings=settings.match_descriptor_settings,
        filter_matches_settings=settings.filter_matches_settings,
    )
    return NodeGraphEstimator(
        mov_detector=NucleiNodeDetector(settings, ref_um),
        ref_detector=NucleiNodeDetector(settings, ref_um),
        beads_match_settings=matching,
        affine_transform_settings=AffineTransformSettings(transform_type=transform_type),
        iterations=settings.iterations,
        score_fn=score_fn,
        fit=yx_and_z_shift_fit(transform_type, ref_um),
    )
