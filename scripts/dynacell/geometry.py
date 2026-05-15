"""Geometry and overlap utilities for dynacell preprocessing."""

import numpy as np
from tqdm import tqdm
from waveorder.focus import focus_from_transverse_band


NA_DET = 1.35
LAMBDA_ILL = 0.500


def make_circular_mask(Y: int, X: int, radius_fraction: float = 0.95) -> np.ndarray:
    """Create a circular mask for the well border in LF data.

    Parameters
    ----------
    Y, X : int
        Dimensions of the mask.
    radius_fraction : float
        Fraction of the half-extent to use as radius.

    Returns
    -------
    np.ndarray of bool, shape (Y, X).
    """
    y, x = np.ogrid[:Y, :X]
    center_y, center_x = Y // 2, X // 2
    radius = int(radius_fraction * min(center_y, center_x))
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2


def find_overlap_mask(arrays: list[np.ndarray]) -> np.ndarray:
    """Find the overlap (non-zero, non-NaN) mask across 2D arrays.

    Arrays can have different Y,X dimensions. The overlap is computed within
    the common region min(Y), min(X), assuming top-left alignment.

    Parameters
    ----------
    arrays : list of np.ndarray
        List of 2D arrays (Y, X). Pixels that are zero or NaN in any
        array are marked as invalid.

    Returns
    -------
    np.ndarray of bool, shape (Y_common, X_common).
    """
    Y_common = min(arr.shape[0] for arr in arrays)
    X_common = min(arr.shape[1] for arr in arrays)
    overlap_mask = np.ones((Y_common, X_common), dtype=bool)
    for arr in arrays:
        cropped = arr[:Y_common, :X_common]
        overlap_mask &= (cropped != 0) & ~np.isnan(cropped)
    return overlap_mask


def find_inscribed_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the tightest axis-aligned rectangle whose borders are fully valid.

    Starts from the bounding box of the mask and iteratively shrinks each
    edge inward until the entire top/bottom row and left/right column lie
    inside the valid region. This handles sheared overlaps and circular
    masks without the area-optimization quirks of the maximal-rectangle
    algorithm.

    Parameters
    ----------
    mask : np.ndarray of bool, shape (Y, X).

    Returns
    -------
    (y_min, y_max, x_min, x_max) or None if no True pixels exist.
    """
    if not np.any(mask):
        return None

    ys, xs = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    # Shrink one pixel per side per iteration so all four sides
    # converge together (avoids collapsing one axis before the other).
    changed = True
    while changed:
        changed = False
        if y_min <= y_max and not mask[y_min, x_min : x_max + 1].all():
            y_min += 1
            changed = True
        if y_max >= y_min and not mask[y_max, x_min : x_max + 1].all():
            y_max -= 1
            changed = True
        if x_min <= x_max and not mask[y_min : y_max + 1, x_min].all():
            x_min += 1
            changed = True
        if x_max >= x_min and not mask[y_min : y_max + 1, x_max].all():
            x_max -= 1
            changed = True

    if y_max < y_min or x_max < x_min:
        return None

    return (y_min, y_max, x_min, x_max)


def find_overlap_bbox_across_time(
    arrays: list[np.ndarray],
    lf_mask_radius: float | None = None,
    pixel_size: float | None = None,
    skip_frames: list[int] | None = None,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """Find the intersection of overlap regions across time points.

    Returns the combined bounding box, the intersection mask, and per-timepoint
    bboxes as a (T, 4) array with columns [y_min, y_max, x_min, x_max].

    Parameters
    ----------
    arrays : list of np.ndarray
        List of arrays with shape (T, C, Z, Y, X).
        All arrays must share the same T.
    lf_mask_radius : float or None
        If provided, apply a circular mask to the first array (LF)
        with this radius fraction (e.g. 0.95) to exclude well borders.
        Pixels outside the circle are zeroed before the overlap check.
    pixel_size : float or None
        If provided, use focus_from_transverse_band on the LF phase channel
        (channel 0 of arrays[0]) to find the in-focus Z index per timepoint.
        If None, fall back to mid-Z.
    skip_frames : list of int or None
        Timepoint indices to skip (e.g. blank frames). Skipped frames are
        excluded from the combined overlap mask and get [0, 0, 0, 0] bboxes.

    Returns
    -------
    ((y_min, y_max, x_min, x_max), overlap_mask, per_t_bboxes) or None.
    per_t_bboxes: np.ndarray of shape (T, 4) with [y_min, y_max, x_min, x_max] per t.
    """
    T = arrays[0].shape[0]
    Y_common = min(arr.shape[-2] for arr in arrays)
    X_common = min(arr.shape[-1] for arr in arrays)

    # Circular mask for the LF well border
    circ_mask = None
    if lf_mask_radius is not None:
        Y_lf, X_lf = arrays[0].shape[-2], arrays[0].shape[-1]
        circ_mask = make_circular_mask(Y_lf, X_lf, lf_mask_radius)

    # Intersection mask across all time points
    combined_mask = np.ones((Y_common, X_common), dtype=bool)
    per_t_bboxes = np.zeros((T, 4), dtype=int)  # y_min, y_max, x_min, x_max
    skip_set = set(skip_frames) if skip_frames else set()

    for t in tqdm(range(T), desc="Finding overlap bboxes"):
        if t in skip_set:
            per_t_bboxes[t] = [0, 0, 0, 0]
            continue

        # Find in-focus Z for LF phase channel
        if pixel_size is not None:
            lf_phase_zyx = np.asarray(arrays[0][t, 0, :, :, :])
            z_focus_lf = focus_from_transverse_band(
                lf_phase_zyx,
                NA_det=NA_DET,
                lambda_ill=LAMBDA_ILL,
                pixel_size=pixel_size,
            )
        else:
            z_focus_lf = arrays[0].shape[2] // 2

        # Use z_focus for ALL arrays so incomplete-Z frames
        # (beginning/end of FOV) are handled correctly.
        slices = []
        for i, arr in enumerate(arrays):
            for c in range(arr.shape[1]):
                slc = np.asarray(arr[t, c, z_focus_lf, :, :]).copy()
                if i == 0 and circ_mask is not None:
                    slc[~circ_mask] = 0
                slices.append(slc)
        t_mask = find_overlap_mask(slices)
        combined_mask &= t_mask

        t_bbox = find_inscribed_bbox(t_mask)
        if t_bbox is not None:
            per_t_bboxes[t] = list(t_bbox)
        else:
            per_t_bboxes[t] = [0, 0, 0, 0]

    bbox = find_inscribed_bbox(combined_mask)
    if bbox is None:
        return None

    return bbox, combined_mask, per_t_bboxes
