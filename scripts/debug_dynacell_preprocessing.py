from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import submitit
from glob import glob
from pathlib import Path
from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from tqdm import tqdm
from waveorder.focus import focus_from_transverse_band
from dynacell_qc_report import generate_dataset_report


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


def plot_bbox_over_time(
    per_t_bboxes: np.ndarray,
    bbox: tuple[int, int, int, int],
    save_path: str | None = None,
):
    """Plot per-timepoint bbox coordinates and their statistics.

    Parameters
    ----------
    per_t_bboxes : np.ndarray, shape (T, 4)
        Columns: y_min, y_max, x_min, x_max.
    bbox : tuple
        The combined (intersection) bbox.
    save_path : str or None
        If provided, save the figure to this path.
    """
    T = per_t_bboxes.shape[0]
    t_axis = np.arange(T)

    y_min_t = per_t_bboxes[:, 0]
    y_max_t = per_t_bboxes[:, 1]
    x_min_t = per_t_bboxes[:, 2]
    x_max_t = per_t_bboxes[:, 3]
    height_t = y_max_t - y_min_t + 1
    width_t = x_max_t - x_min_t + 1

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    def _plot_panel(ax, t_axis, values, combined_val, color, ylabel):
        mu = np.mean(values)
        sigma = np.std(values)
        ax.plot(t_axis, values, color=color, alpha=0.7, linewidth=1.0, label="per timepoint")
        ax.axhline(combined_val, color="r", linestyle="--", linewidth=1.5, label=f"combined = {combined_val}")
        ax.axhline(mu, color="orange", linestyle=":", linewidth=1.5, label=f"mean = {mu:.1f}")
        ax.fill_between(t_axis, mu - sigma, mu + sigma, color="orange", alpha=0.15, label=f"std = {sigma:.1f}")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")

    _plot_panel(axes[0, 0], t_axis, y_min_t, bbox[0], "tab:blue", "y_min")
    _plot_panel(axes[0, 1], t_axis, y_max_t, bbox[1], "tab:blue", "y_max")
    _plot_panel(axes[1, 0], t_axis, x_min_t, bbox[2], "tab:green", "x_min")
    _plot_panel(axes[1, 1], t_axis, x_max_t, bbox[3], "tab:green", "x_max")
    _plot_panel(axes[2, 0], t_axis, height_t, bbox[1] - bbox[0] + 1, "tab:purple", "Height (Y)")
    _plot_panel(axes[2, 1], t_axis, width_t, bbox[3] - bbox[2] + 1, "tab:purple", "Width (X)")
    axes[2, 0].set_xlabel("Time point")
    axes[2, 1].set_xlabel("Time point")

    # --- Statistics text box ---
    stats_lines = [
        "Bbox statistics over time:",
        f"  y_min: mean={np.mean(y_min_t):.1f}, std={np.std(y_min_t):.1f}, "
        f"range=[{y_min_t.min()}, {y_min_t.max()}]",
        f"  y_max: mean={np.mean(y_max_t):.1f}, std={np.std(y_max_t):.1f}, "
        f"range=[{y_max_t.min()}, {y_max_t.max()}]",
        f"  x_min: mean={np.mean(x_min_t):.1f}, std={np.std(x_min_t):.1f}, "
        f"range=[{x_min_t.min()}, {x_min_t.max()}]",
        f"  x_max: mean={np.mean(x_max_t):.1f}, std={np.std(x_max_t):.1f}, "
        f"range=[{x_max_t.min()}, {x_max_t.max()}]",
        f"  height: mean={np.mean(height_t):.1f}, std={np.std(height_t):.1f}, "
        f"range=[{height_t.min()}, {height_t.max()}]",
        f"  width:  mean={np.mean(width_t):.1f}, std={np.std(width_t):.1f}, "
        f"range=[{width_t.min()}, {width_t.max()}]",
        f"  Combined bbox: Y=[{bbox[0]}:{bbox[1]+1}], X=[{bbox[2]}:{bbox[3]+1}]",
    ]
    stats_text = "\n".join(stats_lines)
    print(stats_text)

    fig.suptitle("Overlap bbox over time", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved bbox-over-time plot to {save_path}")
    else:
        plt.show()
    plt.close()


def test_overlap_bbox():
    """Test with synthetic numpy arrays."""
    T, C_lf, C_ls, Z, Y, X = 3, 2, 1, 5, 100, 120
    rng = np.random.default_rng(42)

    # Label-free: zero padding on the left, increasing per t
    lf = rng.random((T, C_lf, Z, Y, X), dtype=np.float32) + 0.1
    for t in range(T):
        lf[t, :, :, :, : 10 + t] = 0

    # Light-sheet: zero padding on the right, increasing per t
    ls = rng.random((T, C_ls, Z, Y, X), dtype=np.float32) + 0.1
    for t in range(T):
        ls[t, :, :, :, -(15 + t) :] = 0

    # Run
    result = find_overlap_bbox_across_time([lf, ls])
    assert result is not None, "Expected overlap, got None"
    (y_min, y_max, x_min, x_max), mask, per_t = result

    # Expected per-t:
    #   t=0: lf zeros cols 0-9,  ls zeros cols 105-119 -> X overlap [10, 104]
    #   t=1: lf zeros cols 0-10, ls zeros cols 104-119 -> X overlap [11, 103]
    #   t=2: lf zeros cols 0-11, ls zeros cols 103-119 -> X overlap [12, 102]
    # Intersection: x_min=12, x_max=102, y_min=0, y_max=99
    assert y_min == 0, f"y_min={y_min} != 0"
    assert y_max == 99, f"y_max={y_max} != 99"
    assert x_min == 12, f"x_min={x_min} != 12"
    assert x_max == 102, f"x_max={x_max} != 102"
    assert mask.shape == (100, 120), f"mask shape={mask.shape}"
    assert mask[50, 50] is np.True_, "center should be valid"
    assert mask[50, 0] is np.False_, "left edge should be invalid"

    print(f"Overlap bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
    print(f"Crop size: Y={y_max - y_min + 1}, X={x_max - x_min + 1}")
    print("TEST PASSED!")


def test_overlap_bbox_no_overlap():
    """Test with arrays that have no overlapping non-zero region."""
    T, C, Z, Y, X = 2, 1, 3, 50, 50
    rng = np.random.default_rng(0)

    arr1 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr1[:, :, :, :, 25:] = 0  # only left half

    arr2 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr2[:, :, :, :, :25] = 0  # only right half

    result = find_overlap_bbox_across_time([arr1, arr2])
    assert result is None, f"Expected None, got {result}"
    print("NO-OVERLAP TEST PASSED!")


def test_overlap_bbox_with_y_padding():
    """Test with zero padding in both Y and X."""
    T, C, Z, Y, X = 2, 1, 5, 80, 100
    rng = np.random.default_rng(7)

    arr1 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr1[:, :, :, :5, :] = 0   # top 5 rows zero
    arr1[:, :, :, :, :8] = 0   # left 8 cols zero

    arr2 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr2[:, :, :, 70:, :] = 0  # bottom 10 rows zero
    arr2[:, :, :, :, 90:] = 0  # right 10 cols zero

    result = find_overlap_bbox_across_time([arr1, arr2])
    assert result is not None
    (y_min, y_max, x_min, x_max), mask, per_t = result
    assert y_min == 5, f"y_min={y_min} != 5"
    assert y_max == 69, f"y_max={y_max} != 69"
    assert x_min == 8, f"x_min={x_min} != 8"
    assert x_max == 89, f"x_max={x_max} != 89"
    print(f"Y+X padding bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
    print("Y+X PADDING TEST PASSED!")


def test_overlap_bbox_different_spatial_dims():
    """Test with arrays that have different Y,X dimensions (like real data)."""
    T, Z = 2, 5
    rng = np.random.default_rng(99)

    # arr1: (T, 1, Z, 100, 80) - zeros on left 5 cols
    arr1 = rng.random((T, 1, Z, 100, 80), dtype=np.float32) + 0.1
    arr1[:, :, :, :, :5] = 0

    # arr2: (T, 2, Z, 60, 120) - zeros on bottom 10 rows
    arr2 = rng.random((T, 2, Z, 60, 120), dtype=np.float32) + 0.1
    arr2[:, :, :, 50:, :] = 0

    # Common region: Y=min(100,60)=60, X=min(80,120)=80
    # arr1 in common: zeros cols 0-4, valid Y=[0:60], X=[5:80]
    # arr2 in common: zeros rows 50-59, valid Y=[0:50], X=[0:80]
    # Overlap: Y=[0:50], X=[5:80]
    result = find_overlap_bbox_across_time([arr1, arr2])
    assert result is not None, f"Expected overlap, got None"
    (y_min, y_max, x_min, x_max), mask, per_t = result
    assert y_min == 0, f"y_min={y_min} != 0"
    assert y_max == 49, f"y_max={y_max} != 49"
    assert x_min == 5, f"x_min={x_min} != 5"
    assert x_max == 79, f"x_max={x_max} != 79"
    # mask shape should be min(Y), min(X) = 60, 80
    assert mask.shape == (60, 80), f"mask shape={mask.shape}"
    print(f"Different dims bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
    print("DIFFERENT SPATIAL DIMS TEST PASSED!")


def test_overlap_bbox_with_nans():
    """Test that NaN regions are treated as invalid (like zeros)."""
    T, C, Z, Y, X = 2, 1, 3, 50, 60
    rng = np.random.default_rng(13)

    arr1 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr1[:, :, :, :, :10] = np.nan  # NaN on left

    arr2 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr2[:, :, :, :, 50:] = 0  # zeros on right

    result = find_overlap_bbox_across_time([arr1, arr2])
    assert result is not None
    (y_min, y_max, x_min, x_max), mask, per_t = result
    assert x_min == 10, f"x_min={x_min} != 10"
    assert x_max == 49, f"x_max={x_max} != 49"
    print(f"NaN bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
    print("NAN TEST PASSED!")


def test_blank_first_frame():
    """Test that a blank first timepoint is properly skipped."""
    T, C, Z, Y, X = 5, 1, 5, 50, 60
    rng = np.random.default_rng(77)

    arr1 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr1[0, :, :, :, :] = 0  # t=0 completely blank
    arr1[:, :, :, :, :5] = 0  # left 5 cols always zero

    arr2 = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1
    arr2[0, :, :, :, :] = 0  # t=0 completely blank
    arr2[:, :, :, :, 50:] = 0  # right 10 cols always zero

    # Without skip_frames: blank t=0 poisons combined_mask -> None
    result_no_skip = find_overlap_bbox_across_time([arr1, arr2])
    assert result_no_skip is None, "Without skip_frames, blank t=0 should cause no overlap"

    # With skip_frames: blank t=0 is excluded
    result = find_overlap_bbox_across_time([arr1, arr2], skip_frames=[0])
    assert result is not None, "With skip_frames=[0], should find valid overlap"
    (y_min, y_max, x_min, x_max), mask, per_t = result
    assert x_min == 5, f"x_min={x_min} != 5"
    assert x_max == 49, f"x_max={x_max} != 49"
    assert per_t[0].tolist() == [0, 0, 0, 0], "Blank frame should have zero bbox"
    assert per_t[1, 2] > 0, "Non-blank frame should have valid x_min"

    print(f"Blank t=0 bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
    print("BLANK FIRST FRAME TEST PASSED!")


def test_inscribed_bbox():
    """Test find_inscribed_bbox on a mildly sheared parallelogram (realistic LS data)."""
    # 100x120 mask with mild shear: ~0.1 px shift per row (10px total over 100 rows)
    # This matches real deskewed LS data where shear is small relative to width.
    mask = np.zeros((100, 120), dtype=bool)
    width = 80
    for y in range(100):
        x_start = y // 10  # shifts right by 1 every 10 rows
        mask[y, x_start : x_start + width] = True

    # Bounding box would be (0, 99, 0, 89) — includes invalid sheared corners.
    # Border-shrink should find a rectangle fully inside the parallelogram.
    result = find_inscribed_bbox(mask)
    assert result is not None
    y_min, y_max, x_min, x_max = result
    # Verify all pixels in the bbox are True
    assert mask[y_min : y_max + 1, x_min : x_max + 1].all(), "Bbox has False pixels!"
    # Should keep most of the area (mild shear loses only ~10 cols)
    area = (y_max - y_min + 1) * (x_max - x_min + 1)
    assert area >= 80 * 70, f"Area {area} too small for mild shear"
    print(f"Inscribed bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}], area={area}")
    print("INSCRIBED BBOX TEST PASSED!")


def test_overlap_bbox_sheared():
    """Test that sheared LS overlap is handled via inscribed bbox."""
    T, C, Z, Y, X = 2, 1, 5, 100, 120
    rng = np.random.default_rng(55)

    # LF: valid everywhere
    lf = rng.random((T, C, Z, Y, X), dtype=np.float32) + 0.1

    # LS: mildly sheared parallelogram (realistic: ~0.1 px shift per row)
    ls = np.zeros((T, C, Z, Y, X), dtype=np.float32)
    for y in range(Y):
        x_start = y // 10  # 10px total shift over 100 rows
        x_end = x_start + 80
        if x_end <= X:
            ls[:, :, :, y, x_start:x_end] = (
                rng.random((T, C, Z, x_end - x_start), dtype=np.float32) + 0.1
            )

    result = find_overlap_bbox_across_time([lf, ls])
    assert result is not None
    (y_min, y_max, x_min, x_max), mask, per_t = result

    # The bbox must be fully inside the valid LS region
    for y in range(y_min, y_max + 1):
        ls_start = y // 10
        assert x_min >= ls_start, (
            f"Row {y}: x_min={x_min} < ls_start={ls_start}"
        )
        assert x_max < ls_start + 80, (
            f"Row {y}: x_max={x_max} >= ls_end={ls_start + 80}"
        )

    area = (y_max - y_min + 1) * (x_max - x_min + 1)
    print(
        f"Sheared bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}], area={area}"
    )
    print("SHEARED OVERLAP TEST PASSED!")


def plot_overlap(
    arrays: list[np.ndarray],
    bbox,
    overlap_mask: np.ndarray | None = None,
    t: int = 0,
    save_path: str | None = None,
    lf_mask_radius: float | None = None,
):
    """Plot mid-Z slices on a common canvas (napari-style, origin-aligned).

    All channels are placed on the same max(Y) x max(X) canvas so they
    share the same coordinate system, like napari with no translation.
    """
    Y_max = max(arr.shape[-2] for arr in arrays)
    X_max = max(arr.shape[-1] for arr in arrays)

    # Circular mask for LF well border
    lf_circ_mask = None
    if lf_mask_radius is not None:
        Y_lf, X_lf = arrays[0].shape[-2], arrays[0].shape[-1]
        lf_circ_mask = make_circular_mask(Y_lf, X_lf, lf_mask_radius)

    # Collect all channel slices padded to common canvas
    channel_slices = []
    channel_labels = []
    for i, arr in enumerate(arrays):
        z_mid = arr.shape[2] // 2
        for c in range(arr.shape[1]):
            raw = np.asarray(arr[t, c, z_mid, :, :])
            canvas = np.zeros((Y_max, X_max), dtype=raw.dtype)
            canvas[: raw.shape[0], : raw.shape[1]] = raw
            channel_slices.append(canvas)
            channel_labels.append(f"arr{i}_ch{c}")

    n = len(channel_slices)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

    y_min, y_max_val, x_min, x_max_val = bbox
    rect_h = y_max_val - y_min + 1
    rect_w = x_max_val - x_min + 1

    for i, (slc, label) in enumerate(zip(channel_slices, channel_labels)):
        ax = axes[i]
        ax.imshow(slc, cmap="gray", origin="upper")
        rect = patches.Rectangle(
            (x_min, y_min), rect_w, rect_h,
            linewidth=2, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
        # Draw circular mask on LF panels
        if lf_circ_mask is not None and label.startswith("arr0"):
            Y_lf, X_lf = arrays[0].shape[-2], arrays[0].shape[-1]
            center_x, center_y = X_lf // 2, Y_lf // 2
            radius = int(lf_mask_radius * min(center_y, center_x))
            circle = patches.Circle(
                (center_x, center_y), radius,
                linewidth=2, edgecolor="cyan", facecolor="none", linestyle="--",
            )
            ax.add_patch(circle)
        ax.set_title(label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    # Last panel: intersection mask on common canvas
    mask_canvas = np.zeros((Y_max, X_max), dtype=bool)
    if overlap_mask is not None:
        mask_canvas[: overlap_mask.shape[0], : overlap_mask.shape[1]] = overlap_mask

    ax = axes[-1]
    ax.imshow(mask_canvas, cmap="gray", origin="upper")
    rect = patches.Rectangle(
        (x_min, y_min), rect_w, rect_h,
        linewidth=2, edgecolor="red", facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title("Intersection mask + bbox")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Overlay panel: composite RGB with all channels
    def _normalize(img):
        vmin, vmax = np.nanpercentile(img[img != 0], [1, 99]) if np.any(img != 0) else (0, 1)
        if vmax == vmin:
            return np.zeros_like(img, dtype=np.float32)
        return np.clip((img - vmin) / (vmax - vmin), 0, 1).astype(np.float32)

    # Assign colors: arr0 channels -> gray, arr1 channels -> green, red, cyan...
    colors = []
    for i, arr in enumerate(arrays):
        for c in range(arr.shape[1]):
            if i == 0:
                colors.append(np.array([0.7, 0.7, 0.7]))  # gray for LF
            elif c == 0:
                colors.append(np.array([0.0, 1.0, 0.0]))  # green
            elif c == 1:
                colors.append(np.array([1.0, 0.0, 0.0]))  # red
            else:
                colors.append(np.array([0.0, 1.0, 1.0]))  # cyan

    rgb = np.zeros((Y_max, X_max, 3), dtype=np.float32)
    for slc, color in zip(channel_slices, colors):
        normed = _normalize(slc)
        rgb += normed[:, :, None] * color[None, None, :]
    rgb = np.clip(rgb, 0, 1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    ax2.imshow(rgb, origin="upper")
    rect = patches.Rectangle(
        (x_min, y_min), rect_w, rect_h,
        linewidth=2, edgecolor="yellow", facecolor="none", linestyle="--",
    )
    ax2.add_patch(rect)
    ax2.set_title(f"t={t} overlay | bbox Y=[{y_min}:{y_max_val+1}], X=[{x_min}:{x_max_val+1}]")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    fig2.tight_layout()
    if save_path:
        overlay_path = save_path.replace(".png", "_overlay.png")
        fig2.savefig(overlay_path, dpi=150, bbox_inches="tight")
        print(f"Saved overlay to {overlay_path}")
    else:
        plt.show()
    plt.close(fig2)

    fig.suptitle(f"t={t} | bbox Y=[{y_min}:{y_max_val+1}], X=[{x_min}:{x_max_val+1}]")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_z_focus(
    z_focus: list[int],
    save_path: str | None = None,
    n_std: float = 2.5,
) -> dict:
    """Plot z focus index over time with statistics and outlier detection.

    Returns a dict with statistics and outlier timepoints.
    """
    z_arr = np.array(z_focus, dtype=float)
    t_axis = np.arange(len(z_arr))
    mu = np.mean(z_arr)
    sigma = np.std(z_arr)
    upper = mu + n_std * sigma
    lower = mu - n_std * sigma

    # Outlier timepoints
    t_above = np.where(z_arr > upper)[0]
    t_below = np.where(z_arr < lower)[0]
    t_outliers = np.sort(np.concatenate([t_above, t_below]))

    # Print statistics
    print(f"Z focus statistics:")
    print(f"  mean = {mu:.1f}, std = {sigma:.1f}, median = {np.median(z_arr):.0f}")
    print(f"  range = [{z_arr.min():.0f}, {z_arr.max():.0f}]")
    print(f"  2*std band = [{lower:.1f}, {upper:.1f}]")
    print(f"  Timepoints above mean+2*std (>{upper:.1f}): {t_above.tolist()}")
    print(f"  Timepoints below mean-2*std (<{lower:.1f}): {t_below.tolist()}")
    print(f"  Total outliers: {len(t_outliers)} / {len(z_arr)}")

    # Save outlier info
    if save_path:
        outlier_path = save_path.replace(".png", "_outliers.csv")
        pd.DataFrame({"t": t_outliers, "z_focus": z_arr[t_outliers]}).to_csv(
            outlier_path, index=False
        )
        print(f"Saved outlier timepoints to {outlier_path}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t_axis, z_arr, "tab:blue", alpha=0.7, linewidth=1.0, label="per timepoint")
    ax.axhline(mu, color="orange", linestyle=":", linewidth=1.5, label=f"mean = {mu:.1f}")
    ax.fill_between(t_axis, mu - sigma, mu + sigma, color="orange", alpha=0.15, label=f"1 std = {sigma:.1f}")
    ax.fill_between(t_axis, lower, upper, color="red", alpha=0.07, label=f"2 std = [{lower:.1f}, {upper:.1f}]")
    ax.axhline(np.median(z_arr), color="green", linestyle="--", linewidth=1.2, label=f"median = {np.median(z_arr):.0f}")
    # Mark outliers
    if len(t_above) > 0:
        ax.scatter(t_above, z_arr[t_above], color="red", s=30, zorder=5, label=f"above 2std (n={len(t_above)})")
    if len(t_below) > 0:
        ax.scatter(t_below, z_arr[t_below], color="purple", s=30, zorder=5, label=f"below 2std (n={len(t_below)})")
    ax.set_xlabel("Time point")
    ax.set_ylabel("Z focus index")
    ax.set_title(
        f"Z focus over time | mean={mu:.1f}, std={sigma:.1f}, "
        f"range=[{z_arr.min():.0f}, {z_arr.max():.0f}]"
    )
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved z focus plot to {save_path}")
    else:
        plt.show()
    plt.close()

    return {
        "mean": mu, "std": sigma, "median": np.median(z_arr),
        "upper_2std": upper, "lower_2std": lower,
        "t_above": t_above, "t_below": t_below, "t_outliers": t_outliers,
    }


def find_blank_frames(
    arrays: list[np.ndarray],
    threshold: float = 1e-6,
) -> list[int]:
    """Find timepoints where any array has an all-zero (blank/black) mid-Z slice.

    Parameters
    ----------
    arrays : list of np.ndarray
        Arrays with shape (T, C, Z, Y, X).
    threshold : float
        A frame is blank if its absolute max value is below this threshold.

    Returns
    -------
    List of blank timepoint indices.
    """
    T = arrays[0].shape[0]
    blank_frames = []
    # Track per-timepoint max intensity for statistics
    max_intensities = np.zeros(T, dtype=np.float64)
    for t in tqdm(range(T), desc="Finding blank frames"):
        t_max = 0.0
        is_blank = False
        for i, arr in enumerate(arrays):
            z_mid = arr.shape[2] // 2
            for c in range(arr.shape[1]):
                slc = np.asarray(arr[t, c, z_mid, :, :])
                slc_max = float(np.nanmax(np.abs(slc)))
                t_max = max(t_max, slc_max)
                if slc_max < threshold and not is_blank:
                    blank_frames.append(t)
                    is_blank = True
        max_intensities[t] = t_max

    blank_frames = sorted(set(blank_frames))
    # Print statistics
    print(f"\nBlank frame statistics:")
    print(f"  Total frames: {T}")
    print(f"  Blank frames: {len(blank_frames)} ({100*len(blank_frames)/T:.1f}%)")
    print(f"  Max intensity per frame: mean={np.mean(max_intensities):.4f}, "
          f"std={np.std(max_intensities):.4f}, "
          f"range=[{max_intensities.min():.4f}, {max_intensities.max():.4f}]")
    if len(blank_frames) > 0:
        print(f"  Blank timepoints: {blank_frames}")
        # Check for consecutive runs
        runs = []
        start = blank_frames[0]
        for i in range(1, len(blank_frames)):
            if blank_frames[i] != blank_frames[i-1] + 1:
                runs.append((start, blank_frames[i-1]))
                start = blank_frames[i]
        runs.append((start, blank_frames[-1]))
        print(f"  Consecutive blank runs: {['t={}-{}'.format(s, e) if s != e else 't={}'.format(s) for s, e in runs]}")

    return blank_frames


def build_drop_list(
    blank_frames: list[int],
    z_focus_outliers: np.ndarray,
    T: int,
    save_path: str | None = None,
) -> dict:
    """Combine blank frames and z_focus outliers into a drop list with reasons.

    Returns
    -------
    dict with keys:
        'drop_indices': sorted array of timepoints to drop
        'keep_indices': sorted array of timepoints to keep
        'reasons': dict mapping timepoint -> list of reason strings
    """
    reasons = {}
    for t in blank_frames:
        reasons.setdefault(t, []).append("blank_frame")
    for t in z_focus_outliers:
        reasons.setdefault(int(t), []).append("z_focus_outlier")

    drop_indices = np.array(sorted(reasons.keys()), dtype=int)
    keep_indices = np.array(sorted(set(range(T)) - set(drop_indices)), dtype=int)

    print(f"\n=== Drop list ===")
    print(f"Total timepoints: {T}")
    print(f"Blank frames ({len(blank_frames)}): {blank_frames}")
    print(f"Z focus outliers ({len(z_focus_outliers)}): {z_focus_outliers.tolist()}")
    print(f"Total to drop: {len(drop_indices)}")
    print(f"Remaining after drop: {len(keep_indices)}")
    for t in drop_indices:
        print(f"  t={t}: {', '.join(reasons[t])}")

    if save_path:
        save_path = save_path.replace(".npy", ".csv")
        rows = []
        for t in drop_indices:
            rows.append({"t": t, "reason": ", ".join(reasons[t])})
        pd.DataFrame(rows, columns=["t", "reason"]).to_csv(save_path, index=False)
        print(f"Saved drop list to {save_path}")

    return {'drop_indices': drop_indices, 'keep_indices': keep_indices, 'reasons': reasons}


## ===== Single-pass metadata computation =====

def compute_per_timepoint_metadata(
    arrays: list[np.ndarray],
    pixel_size: float,
    lf_mask_radius: float | None = None,
    blank_threshold: float = 1e-6,
) -> dict | None:
    """Single-pass computation of blank frames, overlap bbox, and z_focus.

    Iterates each timepoint once, loading mid-Z slices for blank detection.
    For non-blank frames, computes z_focus from the LF phase Z-stack, then
    loads z_focus slices for overlap computation (handles incomplete Z
    volumes at beginning/end of FOV). The bbox is the largest inscribed
    axis-aligned rectangle in the overlap mask (handles sheared LS data).

    Parameters
    ----------
    arrays : list of np.ndarray
        Arrays with shape (T, C, Z, Y, X). The first array is assumed
        to be label-free (LF) data.
    pixel_size : float
        Physical pixel size for z_focus computation.
    lf_mask_radius : float or None
        Circular mask radius for the LF well border.
    blank_threshold : float
        A frame is blank if any channel in any array has
        nanmax(abs(slice)) below this threshold at mid-Z.

    Returns
    -------
    dict or None
        None if no valid overlap is found. Otherwise dict with keys:
        bbox, overlap_mask, per_t_bboxes, blank_frames, z_focus,
        max_intensities.
    """
    T = arrays[0].shape[0]
    Y_common = min(arr.shape[-2] for arr in arrays)
    X_common = min(arr.shape[-1] for arr in arrays)

    circ_mask = None
    if lf_mask_radius is not None:
        circ_mask = make_circular_mask(
            arrays[0].shape[-2], arrays[0].shape[-1], lf_mask_radius
        )

    combined_mask = np.ones((Y_common, X_common), dtype=bool)
    per_t_bboxes = np.zeros((T, 4), dtype=int)
    blank_frames = []
    z_focus = []
    max_intensities = np.zeros(T, dtype=np.float64)
    mid_z_placeholder = arrays[0].shape[2] // 2

    for t in tqdm(range(T), desc="Processing timepoints"):
        # Load mid-Z slices once for all arrays/channels
        mid_z_slices = []  # (array_index, 2D slice)
        t_max = 0.0
        is_blank = False

        for i, arr in enumerate(arrays):
            z_mid = arr.shape[2] // 2
            for c in range(arr.shape[1]):
                slc = np.asarray(arr[t, c, z_mid, :, :])
                slc_max = float(np.nanmax(np.abs(slc)))
                t_max = max(t_max, slc_max)
                if slc_max < blank_threshold:
                    is_blank = True
                mid_z_slices.append((i, slc))

        max_intensities[t] = t_max

        if is_blank:
            blank_frames.append(t)
            per_t_bboxes[t] = [0, 0, 0, 0]
            z_focus.append(mid_z_placeholder)
            continue

        # Z focus from LF phase Z-stack
        lf_phase_zyx = np.asarray(arrays[0][t, 0, :, :, :])
        z_f = focus_from_transverse_band(
            lf_phase_zyx,
            NA_det=NA_DET,
            lambda_ill=LAMBDA_ILL,
            pixel_size=pixel_size,
        )
        z_focus.append(z_f)

        # Overlap from z_focus slices (not mid-Z) so that frames with
        # incomplete Z volumes at the beginning/end are handled correctly.
        overlap_slices = []
        for i, arr in enumerate(arrays):
            for c in range(arr.shape[1]):
                slc = np.asarray(arr[t, c, z_f, :, :]).copy()
                if i == 0 and circ_mask is not None:
                    slc[~circ_mask] = 0
                overlap_slices.append(slc)

        t_mask = find_overlap_mask(overlap_slices)
        combined_mask &= t_mask

        t_bbox = find_inscribed_bbox(t_mask)
        if t_bbox is not None:
            per_t_bboxes[t] = list(t_bbox)
        else:
            per_t_bboxes[t] = [0, 0, 0, 0]

    # Print blank statistics
    print(f"\nBlank frame statistics:")
    print(f"  Total frames: {T}")
    print(f"  Blank frames: {len(blank_frames)} ({100 * len(blank_frames) / T:.1f}%)")
    print(
        f"  Max intensity per frame: mean={np.mean(max_intensities):.4f}, "
        f"std={np.std(max_intensities):.4f}, "
        f"range=[{max_intensities.min():.4f}, {max_intensities.max():.4f}]"
    )
    if blank_frames:
        print(f"  Blank timepoints: {blank_frames}")
        runs = []
        start = blank_frames[0]
        for i in range(1, len(blank_frames)):
            if blank_frames[i] != blank_frames[i - 1] + 1:
                runs.append((start, blank_frames[i - 1]))
                start = blank_frames[i]
        runs.append((start, blank_frames[-1]))
        print(
            f"  Consecutive blank runs: "
            f"{['t={}-{}'.format(s, e) if s != e else 't={}'.format(s) for s, e in runs]}"
        )

    # Combined bbox (bounding box of valid region — a few zero pixels
    # Border-shrink: tightest rectangle with fully valid borders.
    bbox = find_inscribed_bbox(combined_mask)
    if bbox is None:
        return None

    return {
        "bbox": bbox,
        "overlap_mask": combined_mask,
        "per_t_bboxes": per_t_bboxes,
        "blank_frames": blank_frames,
        "z_focus": z_focus,
        "max_intensities": max_intensities,
    }


## ===== STAGE 1: Compute metadata per FOV =====

def compute_fov_metadata(
    im_lf_path: Path,
    im_ls_path: Path,
    output_plots_dir: Path,
    fov: str,
    lf_mask_radius: float = 0.75,
    n_std: float = 2.5,
    DEBUG: bool = True,
) -> dict:
    """Stage 1: Compute bbox, z_focus, blank frames, and drop list for one FOV.

    Uses a single pass over all timepoints to detect blank frames,
    compute the overlap bounding box, and find z_focus simultaneously.

    Saves plots/CSVs to output_plots_dir. Returns a summary dict with
    crop dimensions and keep_indices so stage 2 can determine min T/Y/X.
    """
    with open_ome_zarr(im_lf_path) as im_lf_ds, open_ome_zarr(im_ls_path) as im_ls_ds:
        im_lf_arr = im_lf_ds.data.dask_array()
        im_ls_arr = im_ls_ds.data.dask_array()
        T = im_lf_arr.shape[0]
        print(f"LF shape: {im_lf_arr.shape}")
        print(f"LS shape: {im_ls_arr.shape}")

        pixel_size = im_lf_ds.scale[-1]
        print(f"Pixel size: {pixel_size} um")

        # --- Single pass: blank detection + overlap bbox + z_focus ---
        metadata = compute_per_timepoint_metadata(
            [im_lf_arr, im_ls_arr],
            pixel_size=pixel_size,
            lf_mask_radius=lf_mask_radius,
        )
        if metadata is None:
            print("ERROR: No valid overlap found")
            return {"status": "no_overlap"}

        bbox = metadata["bbox"]
        y_min, y_max, x_min, x_max = bbox
        overlap_mask = metadata["overlap_mask"]
        per_t_bboxes = metadata["per_t_bboxes"]
        blank_frames = metadata["blank_frames"]
        z_focus = metadata["z_focus"]

        print(f"Blank frames: {blank_frames}")
        print(f"Overlap bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
        print(f"Crop size: Y={y_max - y_min + 1}, X={x_max - x_min + 1}")

        if DEBUG:
            print(f"Overlap mask shape: {overlap_mask.shape}")
            print(f"Valid pixels in mask: {overlap_mask.sum()} / {overlap_mask.size}")
            blank_set = set(blank_frames)
            t_debug = next((t for t in range(T) if t not in blank_set), 0)
            plot_overlap(
                [im_lf_arr, im_ls_arr],
                bbox,
                overlap_mask=overlap_mask,
                t=t_debug,
                save_path=str(output_plots_dir / f"overlap_t{t_debug}.png"),
                lf_mask_radius=lf_mask_radius,
            )
            bbox_csv = output_plots_dir / "per_t_bboxes.csv"
            pd.DataFrame(
                per_t_bboxes, columns=["y_min", "y_max", "x_min", "x_max"]
            ).to_csv(bbox_csv, index_label="t")
            print(f"Saved per-timepoint bboxes to {bbox_csv}")
            plot_bbox_over_time(
                per_t_bboxes, bbox,
                save_path=str(output_plots_dir / "bbox_over_time.png"),
            )

        # --- Z focus stats (on non-blank frames only) ---
        print(f"Z focus: {z_focus}")
        z_focus_csv = output_plots_dir / "z_focus.csv"
        pd.DataFrame({"z_focus": z_focus}).to_csv(z_focus_csv, index_label="t")
        print(f"Saved z focus to {z_focus_csv}")

        blank_set = set(blank_frames)
        z_focus_valid = [z_focus[t] for t in range(T) if t not in blank_set]
        z_focus_stats = plot_z_focus(
            z_focus_valid,
            save_path=str(output_plots_dir / "z_focus.png"),
            n_std=n_std,
        )

        # Remap outlier indices back to original timepoint indices
        valid_t_indices = [t for t in range(T) if t not in blank_set]
        z_focus_outliers_original = np.array(
            [valid_t_indices[i] for i in z_focus_stats["t_outliers"]], dtype=int
        )

        # --- Drop list ---
        drop_info = build_drop_list(
            blank_frames=blank_frames,
            z_focus_outliers=z_focus_outliers_original,
            T=T,
            save_path=str(output_plots_dir / "drop_list.csv"),
        )

        # --- Recompute bbox using only kept frames ---
        # Dropped frames should not constrain the overlap bbox.
        # Quick second pass: load z_focus slices for kept frames only,
        # accumulate combined mask, then compute inscribed bbox.
        keep_indices = drop_info["keep_indices"]
        arrays = [im_lf_arr, im_ls_arr]
        Y_common = min(arr.shape[-2] for arr in arrays)
        X_common = min(arr.shape[-1] for arr in arrays)
        circ_mask = None
        if lf_mask_radius is not None:
            circ_mask = make_circular_mask(
                arrays[0].shape[-2], arrays[0].shape[-1], lf_mask_radius
            )

        kept_mask = np.ones((Y_common, X_common), dtype=bool)
        for t in tqdm(keep_indices, desc="Recomputing bbox from kept frames"):
            z_f = int(z_focus[t])
            slices = []
            for i, arr in enumerate(arrays):
                for c in range(arr.shape[1]):
                    slc = np.asarray(arr[t, c, z_f, :, :]).copy()
                    if i == 0 and circ_mask is not None:
                        slc[~circ_mask] = 0
                    slices.append(slc)
            kept_mask &= find_overlap_mask(slices)

        # Use bounding box (not inscribed) — a few zero pixels at circle
        kept_bbox = find_inscribed_bbox(kept_mask)
        if kept_bbox is not None:
            print(f"Recomputed bbox from {len(keep_indices)} kept frames "
                  f"(was Y=[{bbox[0]}:{bbox[1]+1}], X=[{bbox[2]}:{bbox[3]+1}])")
            y_min, y_max, x_min, x_max = kept_bbox
        else:
            print("WARNING: kept-frame mask has no valid overlap, using all-frame bbox")

        print(f"Final bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
        print(f"Final crop size: Y={y_max - y_min + 1}, X={x_max - x_min + 1}")

    # Save FOV summary metadata for stage 2
    summary = {
        "status": "ok",
        "fov": fov,
        "bbox": [y_min, y_max, x_min, x_max],
        "Y_crop": y_max - y_min + 1,
        "X_crop": x_max - x_min + 1,
        "T_out": len(keep_indices),
        "T_total": T,
    }
    summary_csv = output_plots_dir / "fov_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(f"Saved FOV summary to {summary_csv}")

    return summary


## ===== STAGE 2: Crop and save into plate =====

def crop_fov(
    input_zarr_paths: list[Path],
    output_zarr: Path,
    output_plots_dir: Path,
    fov: str,
    global_drop_csv: Path,
    z_final: int = 64,
    T_out: int | None = None,
    Y_out: int | None = None,
    X_out: int | None = None,
):
    """Stage 2: Crop one FOV and write into the pre-created plate.

    Reads bbox, z_focus from stage 1 CSVs and keep_indices from the
    global drop list.
    Crops to uniform (T_out, C, z_final, Y_out, X_out) with padding.

    Parameters
    ----------
    input_zarr_paths : list of Path
        Paths to input zarr FOV positions (e.g. [lf_zarr/fov, ls_zarr/fov]).
    global_drop_csv : Path
        Path to drop_list_all_fovs.csv (fov, t, reason). Unique t values are dropped.
    """
    # Read stage 1 metadata
    z_focus_df = pd.read_csv(output_plots_dir / "z_focus.csv")
    z_focus = z_focus_df["z_focus"].tolist()

    # Read global drop list — unique timepoints across all FOVs
    drop_set = set()
    if global_drop_csv.exists() and global_drop_csv.stat().st_size > 0:
        drop_df = pd.read_csv(global_drop_csv)
        if len(drop_df) > 0:
            drop_set = set(drop_df["t"].unique().tolist())
    T_total = len(z_focus)
    keep_indices = np.array(sorted(set(range(T_total)) - drop_set), dtype=int)

    summary = pd.read_csv(output_plots_dir / "fov_summary.csv")
    bbox_str = summary["bbox"].iloc[0]
    bbox = [int(x.strip()) for x in bbox_str.strip("[]").split(",")]
    y_min, y_max, x_min, x_max = bbox

    Y_crop = y_max - y_min + 1
    X_crop = x_max - x_min + 1
    z_below = z_final // 3
    z_above = z_final - z_below - 1
    Z_out = z_final

    # Center-crop / center-pad offsets when per-FOV crop differs from plate dims
    y_src_off = max(0, (Y_crop - Y_out) // 2)
    x_src_off = max(0, (X_crop - X_out) // 2)
    y_dst_off = max(0, (Y_out - Y_crop) // 2)
    x_dst_off = max(0, (X_out - X_crop) // 2)
    y_size = min(Y_crop, Y_out)
    x_size = min(X_crop, X_out)

    # Limit to T_out timepoints (min across all FOVs after dropping)
    if T_out is not None and len(keep_indices) > T_out:
        keep_indices = keep_indices[:T_out]

    fov_str = fov

    print(f"\nCropping FOV {fov_str} to {output_zarr}")
    print(f"  FOV crop: ({len(keep_indices)}, ?, {Z_out}, {Y_crop}, {X_crop})")
    print(f"  Plate dims: T={T_out}, Y={Y_out}, X={X_out}")
    if Y_crop != Y_out or X_crop != X_out:
        print(f"  Center-crop src[{y_src_off}:{y_src_off+y_size}, {x_src_off}:{x_src_off+x_size}]"
              f" -> dst[{y_dst_off}:{y_dst_off+y_size}, {x_dst_off}:{x_dst_off+x_size}]")
    print(f"  z_final: {z_final} (1/3 below={z_below}, 2/3 above={z_above})")
    print(f"  Input zarrs: {len(input_zarr_paths)}")

    # Open all input zarrs
    from contextlib import ExitStack
    with ExitStack() as stack:
        src_arrays = []
        for zarr_path in input_zarr_paths:
            ds = stack.enter_context(open_ome_zarr(zarr_path))
            src_arrays.append(ds.data.dask_array())

        pos_path = output_zarr / fov_str
        with open_ome_zarr(pos_path, mode="r+") as out_ds:
            out_img = out_ds["0"]

            for t_out, t_in in enumerate(tqdm(keep_indices, desc=f"Cropping {fov_str}")):
                z_center = int(z_focus[t_in])

                c_out = 0
                for src_arr in src_arrays:
                    Z_total = src_arr.shape[2]
                    z_start = max(0, z_center - z_below)
                    z_end = min(Z_total, z_center + z_above + 1)
                    pad_top = max(0, z_below - z_center)

                    for c in range(src_arr.shape[1]):
                        slc = np.asarray(
                            src_arr[t_in, c, z_start:z_end, y_min:y_max+1, x_min:x_max+1]
                        )
                        out_slice = np.zeros((Z_out, Y_out, X_out), dtype=np.float32)
                        z_actual = slc.shape[0]
                        out_slice[
                            pad_top:pad_top + z_actual,
                            y_dst_off:y_dst_off + y_size,
                            x_dst_off:x_dst_off + x_size,
                        ] = slc[:, y_src_off:y_src_off + y_size, x_src_off:x_src_off + x_size]
                        out_img[t_out, c_out] = out_slice
                        c_out += 1

    print(f"Saved cropped FOV {fov_str}")


## ===== Beads registration QC =====

def compute_beads_registration_qc(
    im_lf_path: Path,
    im_ls_path: Path,
    output_plots_dir: Path,
    n_std: float = 2.5,
) -> dict:
    """Compute per-timepoint registration QC using phase cross-correlation on beads.

    For each timepoint, computes PCC between mid-Z slices of the LF and LS
    beads channels. Reports the Pearson correlation and residual shift.
    Timepoints with high shift or low correlation are flagged as outliers.

    Returns a dict with 'drop_indices' (timepoints to drop due to bad registration).
    """
    from skimage.registration import phase_cross_correlation as pcc_skimage

    with open_ome_zarr(im_lf_path) as im_lf_ds, open_ome_zarr(im_ls_path) as im_ls_ds:
        im_lf_arr = im_lf_ds.data.dask_array()
        im_ls_arr = im_ls_ds.data.dask_array()
        T = im_lf_arr.shape[0]
        print(f"Beads QC: LF shape={im_lf_arr.shape}, LS shape={im_ls_arr.shape}")

        pearson_corrs = np.zeros(T, dtype=np.float64)
        pcc_shifts_z = np.zeros(T, dtype=np.float64)
        pcc_shifts_y = np.zeros(T, dtype=np.float64)
        pcc_shifts_x = np.zeros(T, dtype=np.float64)
        pcc_errors = np.zeros(T, dtype=np.float64)

        z_mid_lf = im_lf_arr.shape[2] // 2
        z_mid_ls = im_ls_arr.shape[2] // 2

        for t in tqdm(range(T), desc="Beads registration QC"):
            # Use channel 0 (phase) for LF and channel 0 for LS
            lf_slice = np.asarray(im_lf_arr[t, 0, z_mid_lf, :, :]).astype(np.float64)
            ls_slice = np.asarray(im_ls_arr[t, 0, z_mid_ls, :, :]).astype(np.float64)

            # Crop to common region
            Y_common = min(lf_slice.shape[0], ls_slice.shape[0])
            X_common = min(lf_slice.shape[1], ls_slice.shape[1])
            lf_crop = lf_slice[:Y_common, :X_common]
            ls_crop = ls_slice[:Y_common, :X_common]

            # Pearson correlation
            lf_flat = lf_crop.ravel()
            ls_flat = ls_crop.ravel()
            lf_centered = lf_flat - lf_flat.mean()
            ls_centered = ls_flat - ls_flat.mean()
            denom = np.sqrt(np.sum(lf_centered**2) * np.sum(ls_centered**2))
            if denom > 0:
                pearson_corrs[t] = np.sum(lf_centered * ls_centered) / denom
            else:
                pearson_corrs[t] = np.nan

            # Phase cross-correlation (residual shift after registration)
            shift, error, _phasediff = pcc_skimage(
                lf_crop, ls_crop, upsample_factor=10
            )
            pcc_shifts_y[t] = shift[0]
            pcc_shifts_x[t] = shift[1]
            pcc_errors[t] = error

    # Shift magnitude
    shift_mag = np.sqrt(pcc_shifts_y**2 + pcc_shifts_x**2)

    # Outlier detection on shift magnitude
    mu_shift = np.nanmean(shift_mag)
    sigma_shift = np.nanstd(shift_mag)
    upper_shift = mu_shift + n_std * sigma_shift
    shift_outliers = np.where(shift_mag > upper_shift)[0]

    # Outlier detection on Pearson correlation (low is bad)
    mu_corr = np.nanmean(pearson_corrs)
    sigma_corr = np.nanstd(pearson_corrs)
    lower_corr = mu_corr - n_std * sigma_corr
    corr_outliers = np.where(pearson_corrs < lower_corr)[0]

    all_outliers = np.array(sorted(set(shift_outliers) | set(corr_outliers)), dtype=int)

    print(f"\nBeads registration QC results:")
    print(f"  Pearson corr: mean={mu_corr:.4f}, std={sigma_corr:.4f}")
    print(f"  Shift magnitude: mean={mu_shift:.2f}, std={sigma_shift:.2f}")
    print(f"  Shift outliers (>{upper_shift:.2f} px): {len(shift_outliers)}")
    print(f"  Correlation outliers (<{lower_corr:.4f}): {len(corr_outliers)}")
    print(f"  Total bad timepoints: {len(all_outliers)}")
    if len(all_outliers) > 0:
        print(f"  Bad timepoints: {all_outliers.tolist()}")

    # Save CSV
    qc_df = pd.DataFrame({
        "t": np.arange(T),
        "pearson_corr": pearson_corrs,
        "pcc_shift_y": pcc_shifts_y,
        "pcc_shift_x": pcc_shifts_x,
        "shift_magnitude": shift_mag,
        "pcc_error": pcc_errors,
    })
    qc_csv = output_plots_dir / "registration_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved QC CSV to {qc_csv}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax = axes[0]
    ax.plot(pearson_corrs, "tab:blue", alpha=0.7, linewidth=0.8)
    ax.axhline(mu_corr, color="orange", linestyle=":", linewidth=1)
    ax.fill_between(range(T), mu_corr - sigma_corr, mu_corr + sigma_corr,
                    color="orange", alpha=0.12)
    ax.axhline(lower_corr, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    if len(corr_outliers) > 0:
        ax.scatter(corr_outliers, pearson_corrs[corr_outliers],
                   color="red", s=20, zorder=5)
    ax.set_ylabel("Pearson correlation")
    ax.set_title(f"Beads registration QC | mean r={mu_corr:.4f}")

    ax = axes[1]
    ax.plot(pcc_shifts_y, "tab:blue", alpha=0.7, linewidth=0.8, label="shift Y")
    ax.plot(pcc_shifts_x, "tab:green", alpha=0.7, linewidth=0.8, label="shift X")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("PCC shift (px)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(shift_mag, "tab:purple", alpha=0.7, linewidth=0.8)
    ax.axhline(mu_shift, color="orange", linestyle=":", linewidth=1)
    ax.axhline(upper_shift, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    if len(shift_outliers) > 0:
        ax.scatter(shift_outliers, shift_mag[shift_outliers],
                   color="red", s=20, zorder=5)
    ax.set_ylabel("Shift magnitude (px)")
    ax.set_xlabel("Time point")

    fig.suptitle("Beads registration QC", fontsize=13)
    plt.tight_layout()
    plot_path = output_plots_dir / "registration_qc.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved QC plot to {plot_path}")

    return {
        "drop_indices": all_outliers,
        "pearson_corrs": pearson_corrs,
        "shift_magnitude": shift_mag,
    }


## ===== Two-stage orchestrator =====

def run_all_fovs(
    root_path: Path,
    dataset: str,
    lf_mask_radius: float = 0.75,
    z_final: int = 64,
    n_std: float = 2.5,
    local: bool = False,
    stage1_run_dir: Path | None = None,
    beads_fov: str | None = None,
    max_drops: int = 5,
    overlay_channels: list[str] | None = None,
    exclude_fovs: list[str] | None = None,
):
    """Two-stage pipeline:
    Stage 1: Compute bbox, z_focus, drop list per FOV (parallel submitit jobs).
    Stage 2: After stage 1 completes, gather min T/Y/X, create plate, crop all FOVs.

    Parameters
    ----------
    stage1_run_dir : Path or None
        If provided, skip stage 1 and read its results from this existing run
        directory (e.g. run_20260312_112144). Stage 2 output goes into a new run dir.
    beads_fov : str or None
        FOV key for the beads position (e.g. "C/1/000000"). Used for
        registration QC: per-timepoint phase cross-correlation between
        LF and LS beads channels is computed and timepoints with poor
        registration are added to the global drop list.
        This FOV is excluded from the regular processing pipeline.
    """
    lf_zarr = root_path / dataset / "1-preprocess" / "label-free" / "0-reconstruct" / f"{dataset}.zarr"
    ls_zarr = root_path / dataset / "1-preprocess" / "light-sheet" / "raw" / "1-register" / f"{dataset}.zarr"
    ls_deconvolved_zarr = root_path / dataset / "1-preprocess" / "light-sheet" / "deconvolved" / "2-register" / f"{dataset}.zarr"
    bf_zarr = root_path / dataset / "0-convert" / f"{dataset}_symlink" / f"{dataset}_labelfree_1.zarr"
   # vs_zarr = root_path / dataset / "1-preprocess" / "label-free" / "1-virtual-stain" / f"{dataset}.zarr"

    # If reusing stage 1, read from existing run; create new dir for stage 2 output
    if stage1_run_dir is not None:
        stage1_dir = root_path / dataset / "dynacell" / stage1_run_dir
        plots_dir = stage1_dir / "plots"
        print(f"Reusing stage 1 results from: {stage1_dir}")
    else:
        stage1_dir = None

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root_path / dataset / "dynacell" / f"run_{run_id}"
    output_zarr = output_dir / f"{dataset}.zarr"
    if stage1_dir is None:
        plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Output dir: {output_dir}")

    # Discover FOVs
    position_dirpaths = sorted([Path(p) for p in glob(str(lf_zarr / "*" / "*" / "*"))])
    position_keys = [p.parts[-3:] for p in position_dirpaths]
    print(f"Found {len(position_keys)} FOVs")

    # Exclude user-specified FOVs
    if exclude_fovs:
        exclude_set = {tuple(f.split("/")) for f in exclude_fovs}
        excluded = [k for k in position_keys if k in exclude_set]
        position_keys = [k for k in position_keys if k not in exclude_set]
        if excluded:
            print(f"  Excluded by user: {len(excluded)} FOVs: {['/'.join(k) for k in excluded]}")

    # Validate beads FOV
    if beads_fov is not None:
        beads_key = tuple(beads_fov.split("/"))
        if beads_key not in position_keys:
            print(f"WARNING: beads_fov '{beads_fov}' not found in dataset, skipping beads QC")
            beads_fov = None
        else:
            print(f"  Beads FOV: {beads_fov} (will run registration QC in addition to standard processing)")

    # Get metadata from first FOV for resource estimation
    # All zarrs that will be cropped in stage 2
    all_zarrs = [lf_zarr, ls_zarr, ls_deconvolved_zarr, bf_zarr]  # vs_zarr commented out
    first_fov = "/".join(position_keys[0])

    all_channel_names = []
    total_elements_per_t = 0
    bytes_per_element = np.dtype(np.float32).itemsize
    scale = None

    for zarr_path in all_zarrs:
        fov_path = zarr_path / first_fov
        with open_ome_zarr(fov_path) as ds:
            _, C_i, Z_i, Y_i, X_i = ds.data.shape
            all_channel_names.extend(list(ds.channel_names))
            total_elements_per_t += C_i * Z_i * Y_i * X_i
            if scale is None:
                scale = list(ds.scale)

    # Resource estimation
    gb_per_timepoint = total_elements_per_t * bytes_per_element / 1e9
    gb_ram_per_cpu = max(4, int(np.ceil(gb_per_timepoint * 4)))
    num_cpus = 4

    cluster = "local" if local else "slurm"
    slurm_out_path = output_dir / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    if stage1_dir is None:
        # ========================
        # STAGE 1: Compute metadata
        # ========================
        print(f"\n=== STAGE 1: Computing metadata for {len(position_keys)} FOVs ===")

        slurm_args_s1 = {
            "slurm_job_name": "dynacell_s1_metadata",
            "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
            "slurm_cpus_per_task": num_cpus,
            "slurm_array_parallelism": 100,
            "slurm_time": 120,
            "slurm_partition": "cpu",
        }

        executor_s1 = submitit.AutoExecutor(folder=slurm_out_path / "stage1", cluster=cluster)
        executor_s1.update_parameters(**slurm_args_s1)

        print(f"  Resources: {num_cpus} CPUs, {gb_ram_per_cpu}G RAM/CPU")
        jobs_s1 = []

        with submitit.helpers.clean_env(), executor_s1.batch():
            for position_key in position_keys:
                fov = "/".join(position_key)
                fov_name = "_".join(position_key)
                im_lf_path = lf_zarr / fov
                im_ls_path = ls_zarr / fov
                output_plots_dir = plots_dir / fov_name
                output_plots_dir.mkdir(parents=True, exist_ok=True)

                job = executor_s1.submit(
                    compute_fov_metadata,
                    im_lf_path=im_lf_path,
                    im_ls_path=im_ls_path,
                    output_plots_dir=output_plots_dir,
                    fov=fov,
                    lf_mask_radius=lf_mask_radius,
                    n_std=n_std,
                    DEBUG=True,
                )
                jobs_s1.append(job)

        job_ids_s1 = [job.job_id for job in jobs_s1]
        log_path_s1 = slurm_out_path / "stage1_job_ids.log"
        with log_path_s1.open("w") as f:
            f.write("\n".join(job_ids_s1))
        print(f"Stage 1: Submitted {len(jobs_s1)} jobs. IDs: {log_path_s1}")

        # Submit beads registration QC job (runs in parallel with stage 1)
        beads_qc_job = None
        if beads_fov is not None:
            beads_fov_name = "_".join(beads_fov.split("/"))
            beads_plots_dir = plots_dir / beads_fov_name
            beads_plots_dir.mkdir(parents=True, exist_ok=True)

            executor_beads = submitit.AutoExecutor(
                folder=slurm_out_path / "beads_qc", cluster=cluster
            )
            executor_beads.update_parameters(**slurm_args_s1)
            beads_qc_job = executor_beads.submit(
                compute_beads_registration_qc,
                im_lf_path=lf_zarr / beads_fov,
                im_ls_path=ls_zarr / beads_fov,
                output_plots_dir=beads_plots_dir,
                n_std=n_std,
            )
            print(f"Beads QC: Submitted job {beads_qc_job.job_id}")

        # ========================
        # Wait for stage 1 to complete
        # ========================
        print("\nWaiting for stage 1 jobs to complete...")
        results_s1 = [job.result() for job in jobs_s1]

        # Check for failures
        failed = [r for r in results_s1 if r.get("status") != "ok"]
        if failed:
            print(f"WARNING: {len(failed)} FOVs failed in stage 1:")
            for r in failed:
                print(f"  {r}")

        ok_results = [r for r in results_s1 if r.get("status") == "ok"]
        if not ok_results:
            print("ERROR: No FOVs succeeded in stage 1")
            return
    else:
        # ========================
        # Read stage 1 results from existing run
        # ========================
        print(f"\n=== Skipping stage 1, reading results from {stage1_dir} ===")
        global_summary_path = stage1_dir / "global_summary.csv"
        if global_summary_path.exists():
            ok_results = pd.read_csv(global_summary_path).to_dict("records")
        else:
            # Reconstruct from per-FOV summaries
            ok_results = []
            for position_key in position_keys:
                fov_name = "_".join(position_key)
                summary_path = plots_dir / fov_name / "fov_summary.csv"
                if summary_path.exists():
                    summary = pd.read_csv(summary_path).to_dict("records")[0]
                    if summary.get("status") == "ok":
                        ok_results.append(summary)
        if not ok_results:
            print("ERROR: No valid stage 1 results found")
            return
        print(f"  Loaded {len(ok_results)} FOV results from stage 1")
        beads_qc_job = None  # no beads job when reusing stage 1

    # ========================
    # Collect beads registration QC results
    # ========================
    beads_drop_indices = np.array([], dtype=int)
    if beads_qc_job is not None:
        print("\nWaiting for beads QC job to complete...")
        beads_qc_result = beads_qc_job.result()
        beads_drop_indices = beads_qc_result["drop_indices"]
        print(f"Beads QC: {len(beads_drop_indices)} bad registration timepoints")
    elif beads_fov is not None:
        # Reusing stage 1: read beads QC from existing CSV
        beads_fov_name = "_".join(beads_fov.split("/"))
        beads_qc_csv = plots_dir / beads_fov_name / "registration_qc.csv"
        if beads_qc_csv.exists():
            beads_qc_df = pd.read_csv(beads_qc_csv)
            shift_mag = beads_qc_df["shift_magnitude"].values
            pearson = beads_qc_df["pearson_corr"].values
            mu_s, sigma_s = np.nanmean(shift_mag), np.nanstd(shift_mag)
            mu_c, sigma_c = np.nanmean(pearson), np.nanstd(pearson)
            shift_bad = np.where(shift_mag > mu_s + n_std * sigma_s)[0]
            corr_bad = np.where(pearson < mu_c - n_std * sigma_c)[0]
            beads_drop_indices = np.array(
                sorted(set(shift_bad) | set(corr_bad)), dtype=int
            )
            print(f"Beads QC (from CSV): {len(beads_drop_indices)} bad registration timepoints")
        else:
            print(f"WARNING: No beads QC CSV found at {beads_qc_csv}")

    # ========================
    # Disqualify FOVs with too many drops
    # ========================
    fov_drop_counts = {}
    ok_fov_set = {"_".join(r["fov"].split("/")) for r in ok_results}

    for fov_result in ok_results:
        fov = fov_result["fov"]
        fov_name = "_".join(fov.split("/"))
        drop_path = plots_dir / fov_name / "drop_list.csv"
        n_drops = 0
        if drop_path.exists() and drop_path.stat().st_size > 0:
            drop_df = pd.read_csv(drop_path)
            n_drops = len(drop_df)
        fov_drop_counts[fov_name] = n_drops

    disqualified_fovs = {
        fov_name for fov_name, n in fov_drop_counts.items() if n > max_drops
    }

    # If the beads FOV is disqualified, registration is too poor to continue
    beads_fov_name = "_".join(beads_fov.split("/")) if beads_fov else None
    if beads_fov_name is not None and beads_fov_name in disqualified_fovs:
        print(f"\n{'='*60}")
        print(f"ERROR: Beads FOV '{beads_fov}' is disqualified "
              f"({fov_drop_counts[beads_fov_name]} dropped frames > {max_drops}).")
        print(f"Registration quality is insufficient for this dataset.")
        print(f"Please improve the registration before running dynacell preprocessing.")
        print(f"{'='*60}")
        return

    # Generate annotations.csv (status: 0=not checked, 1=visual checked, -1=unfit)
    annotation_rows = []
    for position_key in position_keys:
        fov_name = "_".join(position_key)
        if fov_name == beads_fov_name:
            status = 1
            well_map = "Beads"
            comments = f"beads: {len(beads_drop_indices)} bad registration timepoints"
        elif fov_name not in ok_fov_set:
            status = -1
            well_map = ""
            comments = "auto: failed stage 1"
        elif fov_name in disqualified_fovs:
            status = -1
            well_map = ""
            comments = (
                f"auto: {fov_drop_counts[fov_name]} dropped frames (>{max_drops})"
            )
        else:
            status = 0
            well_map = ""
            comments = ""
        annotation_rows.append({
            "fov": fov_name,
            "status": status,
            "Well-map": well_map,
            "comments": comments,
        })
    # Add user-excluded FOVs
    if exclude_fovs:
        for fov in exclude_fovs:
            annotation_rows.append({
                "fov": "_".join(fov.split("/")),
                "status": -1,
                "Well-map": "",
                "comments": "user: excluded",
            })
    annotations_df = pd.DataFrame(
        annotation_rows, columns=["fov", "status", "Well-map", "comments"]
    )
    annotations_df = annotations_df.sort_values("fov").reset_index(drop=True)
    annotations_path = output_dir / "annotations.csv"
    annotations_df.to_csv(annotations_path, index=False)
    print(f"Saved annotations to {annotations_path}")

    # Filter to only qualified FOVs for stage 2
    qualified_results = [
        r for r in ok_results
        if "_".join(r["fov"].split("/")) not in disqualified_fovs
    ]

    print(f"\n=== FOV qualification (max {max_drops} drops) ===")
    print(f"  Total FOVs: {len(position_keys)}")
    print(f"  Failed stage 1: {len(position_keys) - len(ok_results)}")
    print(f"  Disqualified (>{max_drops} drops): {len(disqualified_fovs)}")
    for fov_name in sorted(disqualified_fovs):
        print(f"    {fov_name}: {fov_drop_counts[fov_name]} drops")
    print(f"  Qualified for stage 2: {len(qualified_results)}")

    if not qualified_results:
        print("ERROR: No qualified FOVs remaining after disqualification")
        return

    ok_results = qualified_results

    # ========================
    # Unify drop list from QUALIFIED FOVs only + beads QC
    # (disqualified FOVs are already excluded from ok_results above)
    # ========================
    global_drop_set = set()
    T_total = ok_results[0]["T_total"]
    for fov_result in ok_results:
        fov = fov_result["fov"]
        fov_name = "_".join(fov.split("/"))
        drop_path = plots_dir / fov_name / "drop_list.csv"
        if drop_path.exists() and drop_path.stat().st_size > 0:
            drop_df = pd.read_csv(drop_path)
            if len(drop_df) > 0:
                global_drop_set.update(drop_df["t"].tolist())

    # Merge beads registration QC bad timepoints
    beads_drop_set = set(int(t) for t in beads_drop_indices)
    n_beads_new = len(beads_drop_set - global_drop_set)
    global_drop_set.update(beads_drop_set)

    global_keep_indices = np.array(
        sorted(set(range(T_total)) - global_drop_set), dtype=int
    )
    print(f"\n=== Unified drop list (qualified FOVs + beads QC) ===")
    print(f"  Drops from qualified FOVs: {len(global_drop_set) - len(beads_drop_set & global_drop_set)}")
    if len(beads_drop_indices) > 0:
        print(f"  Beads QC bad timepoints: {len(beads_drop_indices)} "
              f"({n_beads_new} new)")
    print(f"  Total global drops: {len(global_drop_set)}")
    print(f"  Remaining after unified drop: {len(global_keep_indices)} / {T_total}")

    global_drop_csv = output_dir / "drop_list_all_fovs.csv"

    # ========================
    # Gather min T, Y, X across FOVs
    # ========================
    T_min = len(global_keep_indices)
    Y_min = min(r["Y_crop"] for r in ok_results)
    X_min = min(r["X_crop"] for r in ok_results)

    C_out = len(all_channel_names)
    Z_out = z_final

    print(f"\n=== Stage 1 summary ===")
    print(f"  FOVs OK: {len(ok_results)}")
    print(f"  T_min (after drops): {T_min}")
    print(f"  Y_min: {Y_min}, X_min: {X_min}")
    print(f"  Output plate shape: ({T_min}, {C_out}, {Z_out}, {Y_min}, {X_min})")

    # Save global summary — use actual cropped values, not per-FOV stage 1 values
    global_summary = pd.DataFrame(ok_results)
    global_summary["T_out"] = T_min
    global_summary["Y_crop"] = Y_min
    global_summary["X_crop"] = X_min
    global_summary.to_csv(output_dir / "global_summary.csv", index=False)
    print(f"Saved global summary to {output_dir / 'global_summary.csv'}")

    # ========================
    # Combined z_focus plot across all FOVs
    # ========================
    ok_fovs = [r["fov"] for r in ok_results]
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    all_z_focus_data = {}
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        z_df = pd.read_csv(plots_dir / fov_name / "z_focus.csv")
        z_vals = z_df["z_focus"].values
        all_z_focus_data[fov_name] = z_vals
        ax.plot(z_vals, alpha=0.5, linewidth=0.8, label=fov_name)
    ax.set_xlabel("Time point")
    ax.set_ylabel("Z focus index")
    ax.set_title(f"Z focus across all FOVs ({len(ok_fovs)} FOVs)")
    ax.legend(fontsize=6, loc="best", ncol=2)
    plt.tight_layout()
    zfocus_all_path = output_dir / "z_focus_all_fovs.png"
    plt.savefig(zfocus_all_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined z_focus plot to {zfocus_all_path}")

    # Save combined z_focus CSV (fov x timepoint)
    z_focus_all_df = pd.DataFrame(all_z_focus_data)
    z_focus_all_df.index.name = "t"
    z_focus_all_df.to_csv(output_dir / "z_focus_all_fovs.csv")
    print(f"Saved combined z_focus CSV to {output_dir / 'z_focus_all_fovs.csv'}")

    # ========================
    # Combined blank frames summary across all FOVs
    # ========================
    drop_rows = []
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        drop_path = plots_dir / fov_name / "drop_list.csv"
        if drop_path.stat().st_size > 0:
            drop_df = pd.read_csv(drop_path)
            for _, row in drop_df.iterrows():
                drop_rows.append({
                    "fov": fov_name,
                    "t": int(row["t"]),
                    "reason": row["reason"],
                })
    # Add beads registration QC drops
    for t in sorted(beads_drop_set):
        drop_rows.append({
            "fov": "beads_qc",
            "t": int(t),
            "reason": "bad_registration",
        })
    drop_all_df = pd.DataFrame(drop_rows, columns=["fov", "t", "reason"])
    drop_all_path = output_dir / "drop_list_all_fovs.csv"
    drop_all_df.to_csv(drop_all_path, index=False)
    print(f"Saved combined drop list to {drop_all_path}")
    print(f"  Total entries: {len(drop_all_df)}, unique timepoints dropped: {drop_all_df['t'].nunique()}")

    # Per-FOV drop count summary
    if len(drop_all_df) > 0:
        drop_counts = drop_all_df.groupby("fov").size().reset_index(name="n_dropped")
        print("  Drops per FOV:")
        for _, row in drop_counts.iterrows():
            print(f"    {row['fov']}: {row['n_dropped']}")

    # ========================
    # STAGE 2: Create plate and crop
    # ========================
    print(f"\n=== STAGE 2: Cropping {len(ok_results)} FOVs ===")

    ok_position_keys = [tuple(fov.split("/")) for fov in ok_fovs]

    create_empty_plate(
        store_path=output_zarr,
        position_keys=ok_position_keys,
        shape=(T_min, C_out, Z_out, Y_min, X_min),
        chunks=(1, 1, Z_out, Y_min, X_min),
        scale=scale,
        channel_names=all_channel_names,
        dtype=np.float32,
        version = '0.5',
    )
    print(f"Created output plate at {output_zarr}")

    slurm_args_s2 = {
        "slurm_job_name": "dynacell_s2_crop",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 120,
        "slurm_partition": "cpu",
    }

    executor_s2 = submitit.AutoExecutor(folder=slurm_out_path / "stage2", cluster=cluster)
    executor_s2.update_parameters(**slurm_args_s2)

    jobs_s2 = []
    with submitit.helpers.clean_env(), executor_s2.batch():
        for fov in ok_fovs:
            fov_name = "_".join(fov.split("/"))
            input_zarr_paths = [lf_zarr / fov, ls_zarr / fov, ls_deconvolved_zarr / fov, bf_zarr / fov] #, vs_zarr / fov]
            fov_plots_dir = plots_dir / fov_name

            job = executor_s2.submit(
                crop_fov,
                input_zarr_paths=input_zarr_paths,
                output_zarr=output_zarr,
                output_plots_dir=fov_plots_dir,
                fov=fov,
                global_drop_csv=global_drop_csv,
                z_final=z_final,
                T_out=T_min,
                Y_out=Y_min,
                X_out=X_min,
            )
            jobs_s2.append(job)

    job_ids_s2 = [job.job_id for job in jobs_s2]
    log_path_s2 = slurm_out_path / "stage2_job_ids.log"
    with log_path_s2.open("w") as f:
        f.write("\n".join(job_ids_s2))
    print(f"Stage 2: Submitted {len(jobs_s2)} jobs. IDs: {log_path_s2}")

    # Wait for stage 2
    print("\nWaiting for stage 2 jobs to complete...")
    for job in jobs_s2:
        job.result()
    print(f"\nDone! Output plate at {output_zarr}")

    # ========================
    # Generate QC report
    # ========================
    print(f"\n=== Generating QC report ===")
    generate_dataset_report(output_dir, overlay_channels=overlay_channels)


if __name__ == "__main__":
    test_inscribed_bbox()
    test_overlap_bbox()
    test_overlap_bbox_no_overlap()
    test_overlap_bbox_with_y_padding()
    test_overlap_bbox_different_spatial_dims()
    test_overlap_bbox_with_nans()
    test_blank_first_frame()
    test_overlap_bbox_sheared()
    print("\n=== SUBMITTING ALL FOVs ===")
    root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
    dataset = "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
    run_all_fovs(
        root_path=root_path,
        dataset=dataset,
        lf_mask_radius=0.98,
        local=False,
        stage1_run_dir=None,
        beads_fov="A/3/000001",
        overlay_channels=["Phase3D", "raw GFP EX488 EM525-45"],
        exclude_fovs=["A/3/000000", "A/3/001000", "A/3/001001"],
        z_final = 44,
        
    )
