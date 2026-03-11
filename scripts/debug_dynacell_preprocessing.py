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

        slices = []
        for i, arr in enumerate(arrays):
            if i == 0:
                z_idx = z_focus_lf
            else:
                z_idx = arr.shape[2] // 2
            for c in range(arr.shape[1]):
                slc = np.asarray(arr[t, c, z_idx, :, :]).copy()
                # Mask the LF phase with the circular mask
                if i == 0 and circ_mask is not None:
                    slc[~circ_mask] = 0
                slices.append(slc)
        t_mask = find_overlap_mask(slices)
        combined_mask &= t_mask

        ys_t, xs_t = np.where(t_mask)
        if len(ys_t) > 0:
            per_t_bboxes[t] = [ys_t.min(), ys_t.max(), xs_t.min(), xs_t.max()]
        else:
            per_t_bboxes[t] = [0, 0, 0, 0]

    ys, xs = np.where(combined_mask)
    if len(ys) == 0:
        return None

    bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
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
    plt.tight_layout()
    if save_path:
        overlay_path = save_path.replace(".png", "_overlay.png")
        plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
        print(f"Saved overlay to {overlay_path}")
    else:
        plt.show()
    plt.close()

    fig.suptitle(f"t={t} | bbox Y=[{y_min}:{y_max_val+1}], X=[{x_min}:{x_max_val+1}]")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


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

        # --- Blank frames (must run first) ---
        blank_frames = find_blank_frames([im_lf_arr, im_ls_arr])
        print(f"Blank frames: {blank_frames}")

        # --- Overlap bbox (skip blank frames) ---
        result = find_overlap_bbox_across_time(
            [im_lf_arr, im_ls_arr],
            lf_mask_radius=lf_mask_radius,
            pixel_size=None,
            skip_frames=blank_frames,
        )
        if result is None:
            print("ERROR: No valid overlap found")
            return {"status": "no_overlap"}

        (y_min, y_max, x_min, x_max), overlap_mask, per_t_bboxes = result
        print(f"Overlap bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
        print(f"Crop size: Y={y_max - y_min + 1}, X={x_max - x_min + 1}")

        if DEBUG:
            print(f"Overlap mask shape: {overlap_mask.shape}")
            print(f"Valid pixels in mask: {overlap_mask.sum()} / {overlap_mask.size}")
            # Use first non-blank timepoint for the debug plot
            blank_set = set(blank_frames)
            t_debug = next((t for t in range(T) if t not in blank_set), 0)
            plot_overlap(
                [im_lf_arr, im_ls_arr],
                (y_min, y_max, x_min, x_max),
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
                per_t_bboxes,
                (y_min, y_max, x_min, x_max),
                save_path=str(output_plots_dir / "bbox_over_time.png"),
            )

        # --- Z focus (skip blank frames, use mid-Z as placeholder) ---
        blank_set = set(blank_frames)
        z_focus = []
        for t in tqdm(range(T), desc="Finding z focus"):
            if t in blank_set:
                # Placeholder for blank frames; will be dropped later
                z_focus.append(im_lf_arr.shape[2] // 2)
                continue
            lf_phase_zyx = np.asarray(im_lf_arr[t, 0, :, :, :])
            z_focus_lf = focus_from_transverse_band(
                lf_phase_zyx,
                NA_det=NA_DET,
                lambda_ill=LAMBDA_ILL,
                pixel_size=pixel_size,
            )
            z_focus.append(z_focus_lf)
        print(f"Z focus: {z_focus}")
        z_focus_csv = output_plots_dir / "z_focus.csv"
        pd.DataFrame({"z_focus": z_focus}).to_csv(z_focus_csv, index_label="t")
        print(f"Saved z focus to {z_focus_csv}")

        # Compute z_focus stats only on non-blank frames
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

    # Save FOV summary metadata for stage 2
    keep_indices = drop_info["keep_indices"]
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
    im_lf_path: Path,
    im_ls_path: Path,
    output_zarr: Path,
    output_plots_dir: Path,
    fov: str,
    z_final: int = 64,
    T_out: int | None = None,
    Y_out: int | None = None,
    X_out: int | None = None,
):
    """Stage 2: Crop one FOV and write into the pre-created plate.

    Reads bbox, z_focus, and keep_indices from stage 1 CSVs.
    Crops to uniform (T_out, C, z_final, Y_out, X_out) with padding.
    """
    # Read stage 1 metadata
    z_focus_df = pd.read_csv(output_plots_dir / "z_focus.csv")
    z_focus = z_focus_df["z_focus"].tolist()

    drop_path = output_plots_dir / "drop_list.csv"
    if drop_path.stat().st_size > 0:
        drop_df = pd.read_csv(drop_path)
        drop_set = set(drop_df["t"].tolist()) if len(drop_df) > 0 else set()
    else:
        drop_set = set()
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

    # Limit to T_out timepoints (min across all FOVs after dropping)
    if T_out is not None and len(keep_indices) > T_out:
        keep_indices = keep_indices[:T_out]

    position_key = tuple(fov.split("/"))
    fov_str = fov

    print(f"\nCropping FOV {fov_str} to {output_zarr}")
    print(f"  Crop: ({len(keep_indices)}, ?, {Z_out}, {Y_crop}, {X_crop})")
    print(f"  Plate dims: T={T_out}, Y={Y_out}, X={X_out}")
    print(f"  z_final: {z_final} (1/3 below={z_below}, 2/3 above={z_above})")

    with open_ome_zarr(im_lf_path) as im_lf_ds, open_ome_zarr(im_ls_path) as im_ls_ds:
        im_lf_arr = im_lf_ds.data.dask_array()
        im_ls_arr = im_ls_ds.data.dask_array()
        src_arrays = [im_lf_arr, im_ls_arr]

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
                        out_slice[pad_top:pad_top + z_actual, :Y_crop, :X_crop] = slc
                        out_img[t_out, c_out] = out_slice
                        c_out += 1

    print(f"Saved cropped FOV {fov_str}")


## ===== Two-stage orchestrator =====

def run_all_fovs(
    root_path: Path,
    dataset: str,
    lf_mask_radius: float = 0.75,
    z_final: int = 64,
    n_std: float = 2.5,
    local: bool = False,
):
    """Two-stage pipeline:
    Stage 1: Compute bbox, z_focus, drop list per FOV (parallel submitit jobs).
    Stage 2: After stage 1 completes, gather min T/Y/X, create plate, crop all FOVs.
    """
    lf_zarr = root_path / dataset / "1-preprocess" / "label-free" / "0-reconstruct" / f"{dataset}.zarr"
    ls_zarr = root_path / dataset / "1-preprocess" / "light-sheet" / "raw" / "1-register" / f"{dataset}.zarr"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root_path / dataset / "dynacell" / f"run_{run_id}"
    output_zarr = output_dir / f"{dataset}.zarr"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Output dir: {output_dir}")

    # Discover FOVs
    position_dirpaths = sorted([Path(p) for p in glob(str(lf_zarr / "*" / "*" / "*"))])
    position_keys = [p.parts[-3:] for p in position_dirpaths]
    print(f"Found {len(position_keys)} FOVs")

    # Get metadata from first FOV for resource estimation
    with open_ome_zarr(position_dirpaths[0]) as lf_ds:
        _, C_lf, Z_lf, Y_lf, X_lf = lf_ds.data.shape
        scale = list(lf_ds.scale)
        lf_channel_names = list(lf_ds.channel_names)

    ls_first = ls_zarr / "/".join(position_keys[0])
    with open_ome_zarr(ls_first) as ls_ds:
        _, C_ls, Z_ls, Y_ls, X_ls = ls_ds.data.shape
        ls_channel_names = list(ls_ds.channel_names)

    # Resource estimation
    bytes_per_element = np.dtype(np.float32).itemsize
    gb_per_timepoint = (
        (C_lf * Z_lf * Y_lf * X_lf + C_ls * Z_ls * Y_ls * X_ls)
        * bytes_per_element / 1e9
    )
    gb_ram_per_cpu = max(4, int(np.ceil(gb_per_timepoint * 4)))
    num_cpus = 4

    cluster = "local" if local else "slurm"
    slurm_out_path = output_dir / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)

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

    # ========================
    # Gather min T, Y, X across FOVs
    # ========================
    T_min = min(r["T_out"] for r in ok_results)
    Y_min = min(r["Y_crop"] for r in ok_results)
    X_min = min(r["X_crop"] for r in ok_results)

    all_channel_names = lf_channel_names + ls_channel_names
    C_out = len(all_channel_names)
    Z_out = z_final

    print(f"\n=== Stage 1 summary ===")
    print(f"  FOVs OK: {len(ok_results)} / {len(results_s1)}")
    print(f"  T_min (after drops): {T_min}")
    print(f"  Y_min: {Y_min}, X_min: {X_min}")
    print(f"  Output plate shape: ({T_min}, {C_out}, {Z_out}, {Y_min}, {X_min})")

    # Save global summary
    global_summary = pd.DataFrame(ok_results)
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
    blank_rows = []
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        drop_path = plots_dir / fov_name / "drop_list.csv"
        if drop_path.stat().st_size > 0:
            drop_df = pd.read_csv(drop_path)
            for _, row in drop_df.iterrows():
                blank_rows.append({
                    "fov": fov_name,
                    "t": int(row["t"]),
                    "reason": row["reason"],
                })
    blank_summary_df = pd.DataFrame(blank_rows, columns=["fov", "t", "reason"])
    blank_summary_path = output_dir / "drop_list_all_fovs.csv"
    blank_summary_df.to_csv(blank_summary_path, index=False)
    print(f"Saved combined drop list to {blank_summary_path}")
    print(f"  Total dropped frames across FOVs: {len(blank_summary_df)}")

    # Per-FOV drop count summary
    if len(blank_summary_df) > 0:
        drop_counts = blank_summary_df.groupby("fov").size().reset_index(name="n_dropped")
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
            im_lf_path = lf_zarr / fov
            im_ls_path = ls_zarr / fov
            fov_plots_dir = plots_dir / fov_name

            job = executor_s2.submit(
                crop_fov,
                im_lf_path=im_lf_path,
                im_ls_path=im_ls_path,
                output_zarr=output_zarr,
                output_plots_dir=fov_plots_dir,
                fov=fov,
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


if __name__ == "__main__":
    test_overlap_bbox()
    test_overlap_bbox_no_overlap()
    test_overlap_bbox_with_y_padding()
    test_overlap_bbox_different_spatial_dims()
    test_overlap_bbox_with_nans()
    print("\n=== SUBMITTING ALL FOVs ===")
    root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
    dataset = "2024_11_07_A549_SEC61_DENV"
    run_all_fovs(
        root_path=root_path,
        dataset=dataset,
        lf_mask_radius=0.75,
        local=False,
    )
