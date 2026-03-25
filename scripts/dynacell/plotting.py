"""Visualization utilities for dynacell preprocessing.

Also contains replot helpers to regenerate all QC plots from existing CSVs
(formerly in dynacell_replot.py).

Usage (replot mode):
    python dynacell_plotting.py <run_dir>
"""

import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from pathlib import Path

from dynacell.geometry import make_circular_mask

# ---------------------------------------------------------------------------
# Unified plot style
# ---------------------------------------------------------------------------

STYLE = {
    # Font sizes
    "fs_suptitle": 12,
    "fs_title": 11,
    "fs_label": 10,
    "fs_tick": 8,
    "fs_legend": 7,
    "fs_legend_multi": 6,   # all-FOV legends with many entries
    "fs_annotation": 8,

    # Colors
    "c_primary": "#1f77b4",    # blue
    "c_secondary": "#2ca02c",  # green
    "c_tertiary": "#9467bd",   # purple
    "c_mean": "#ff7f0e",       # orange
    "c_median": "#2ca02c",     # green
    "c_threshold": "#d62728",  # red
    "c_outlier": "#d62728",    # red
    "c_fill": "#ff7f0e",       # orange (for std bands)
    "c_grid": "#cccccc",

    "c_zscore": "#9467bd",         # purple — dedicated for local z-score panels

    # Semantic QC colors
    "c_qc": {
        "blank_frame": "#888888",
        "z_focus_outlier": "#ff7f0e",
        "hf_blur": "#d62728",
        "entropy": "#9467bd",
        "frc": "#1f77b4",
        "fov_reg": "#2ca02c",
        "laplacian": "#8c564b",
        "max_intensity": "#e377c2",
        "bleach": "#bcbd22",
    },

    # Lines
    "lw_data": 0.8,
    "lw_ref": 1.2,
    "lw_threshold": 1.0,
    "marker_size": 3,
    "scatter_size": 20,
    "alpha_data": 0.7,
    "alpha_fill": 0.12,
    "alpha_multi": 0.5,       # all-FOV overlay traces

    # Figure
    "dpi": 150,
    "fig_single": (12, 4),    # single-panel per-FOV
    "fig_double": (12, 7),    # two-panel per-FOV
    "fig_triple": (12, 10),   # three-panel per-FOV
    "fig_wide": (14, 5),      # wide single-panel
    "fig_all_single": (12, 5),  # all-FOV single panel
    "fig_all_double": (12, 8),  # all-FOV two panels
    "fig_all_triple": (12, 10), # all-FOV three panels

    # Y-axis limits per metric.
    # None means data-adaptive (auto-scale with margin).
    # Tuple means fixed (ymin, ymax) — only for semantically meaningful ranges.
    "ylim": {
        "z_focus": None,
        "laplacian": None,
        "entropy": None,
        "entropy_local_z": None,         # data-adaptive z-score
        "hf_ratio": None,
        "hf_local_z": None,             # data-adaptive z-score
        "frc": None,
        "frc_local_z": None,            # data-adaptive z-score
        "max_intensity": None,
        "max_intensity_local_z": None,   # data-adaptive z-score
        "fov_reg_pearson": None,
        "fov_reg_local_z": None,         # data-adaptive z-score
        "laplacian_local_z": None,       # data-adaptive z-score
        "bleach_norm": (0, 1.15),        # normalised to 1
        "bleach_local_z": None,          # data-adaptive z-score
        "shift_magnitude": None,
        "pearson_beads": None,
    },
    "ylim_margin": 0.25,  # fractional padding for data-adaptive limits
}

# Ordered color cycle for multi-FOV plots
COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def _set_ylim(ax, key, data=None):
    """Set y-axis limits from STYLE['ylim'] if fixed, else let matplotlib auto-scale.

    When ylim is None, matplotlib auto-scales to fit all plotted elements
    (data, axhlines, fill_between, scatter, etc.) with a small margin.
    """
    lim = STYLE["ylim"].get(key)
    if lim is not None:
        ax.set_ylim(lim)
    else:
        # Let matplotlib auto-scale to include all plotted elements,
        # then add a small margin so points at edges aren't clipped.
        ax.autoscale_view()
        lo, hi = ax.get_ylim()
        span = hi - lo
        if span > 0:
            margin = STYLE.get("ylim_margin", 0.05)
            ax.set_ylim(lo - margin * span, hi + margin * span)


def _apply_style(fig, axes=None):
    """Apply uniform tick sizes and grid to figure axes."""
    if axes is None:
        axes = fig.get_axes()
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.tick_params(labelsize=STYLE["fs_tick"])
        ax.grid(True, alpha=0.15, color=STYLE["c_grid"])
        ax.set_xlim(left=0)
    fig.tight_layout()


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

    # Skip blank frames (bbox = [0, 0, 0, 0])
    valid_mask = ~((per_t_bboxes == 0).all(axis=1))
    valid_t = np.where(valid_mask)[0]

    y_min_t = per_t_bboxes[valid_t, 0]
    y_max_t = per_t_bboxes[valid_t, 1]
    x_min_t = per_t_bboxes[valid_t, 2]
    x_max_t = per_t_bboxes[valid_t, 3]
    height_t = y_max_t - y_min_t + 1
    width_t = x_max_t - x_min_t + 1

    n_blank = T - len(valid_t)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    S = STYLE

    def _plot_panel(ax, t_axis, values, combined_val, color, ylabel):
        mu = np.mean(values)
        sigma = np.std(values)
        ax.plot(t_axis, values, color=color, alpha=S["alpha_data"], linewidth=S["lw_data"], label="per timepoint")
        ax.axhline(combined_val, color=S["c_threshold"], linestyle="--", linewidth=S["lw_ref"], label=f"combined = {combined_val}")
        ax.axhline(mu, color=S["c_mean"], linestyle=":", linewidth=S["lw_ref"], label=f"mean = {mu:.1f}")
        ax.fill_between(t_axis, mu - sigma, mu + sigma, color=S["c_fill"], alpha=S["alpha_fill"], label=f"std = {sigma:.1f}")
        ax.set_ylabel(ylabel, fontsize=S["fs_label"])
        ax.legend(fontsize=S["fs_legend"], loc="best")

    _plot_panel(axes[0, 0], valid_t, y_min_t, bbox[0], S["c_primary"], "y_min")
    _plot_panel(axes[0, 1], valid_t, y_max_t, bbox[1], S["c_primary"], "y_max")
    _plot_panel(axes[1, 0], valid_t, x_min_t, bbox[2], S["c_secondary"], "x_min")
    _plot_panel(axes[1, 1], valid_t, x_max_t, bbox[3], S["c_secondary"], "x_max")
    _plot_panel(axes[2, 0], valid_t, height_t, bbox[1] - bbox[0] + 1, S["c_tertiary"], "Height (Y)")
    _plot_panel(axes[2, 1], valid_t, width_t, bbox[3] - bbox[2] + 1, S["c_tertiary"], "Width (X)")
    axes[2, 0].set_xlabel("Timepoint", fontsize=S["fs_label"])
    axes[2, 1].set_xlabel("Timepoint", fontsize=S["fs_label"])

    # --- Statistics text box ---
    stats_lines = [
        f"Bbox statistics over time (excl. {n_blank} blank):",
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

    title = "Overlap bbox over time"
    if n_blank > 0:
        title += f" (excl. {n_blank} blank)"
    fig.suptitle(title, fontsize=S["fs_suptitle"])
    _apply_style(fig)
    if save_path:
        plt.savefig(save_path, dpi=S["dpi"], bbox_inches="tight")
        print(f"Saved bbox-over-time plot to {save_path}")
    else:
        plt.show()
    plt.close()


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

    S = STYLE
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
        ax.set_title(label, fontsize=S["fs_title"])
        ax.set_xlabel("X", fontsize=S["fs_label"])
        ax.set_ylabel("Y", fontsize=S["fs_label"])
        ax.tick_params(labelsize=S["fs_tick"])

    # Last panel: intersection mask on common canvas
    mask_canvas = np.zeros((Y_max, X_max), dtype=bool)
    if overlap_mask is not None:
        mask_canvas[: overlap_mask.shape[0], : overlap_mask.shape[1]] = overlap_mask

    ax = axes[-1]
    ax.imshow(mask_canvas, cmap="gray", origin="upper")
    rect = patches.Rectangle(
        (x_min, y_min), rect_w, rect_h,
        linewidth=2, edgecolor=S["c_threshold"], facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title("Intersection mask + bbox", fontsize=S["fs_title"])
    ax.set_xlabel("X", fontsize=S["fs_label"])
    ax.set_ylabel("Y", fontsize=S["fs_label"])
    ax.tick_params(labelsize=S["fs_tick"])

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
    ax2.set_title(f"t={t} overlay | bbox Y=[{y_min}:{y_max_val+1}], X=[{x_min}:{x_max_val+1}]",
                  fontsize=S["fs_title"])
    ax2.set_xlabel("X", fontsize=S["fs_label"])
    ax2.set_ylabel("Y", fontsize=S["fs_label"])
    ax2.tick_params(labelsize=S["fs_tick"])
    fig2.tight_layout()
    if save_path:
        overlay_path = save_path.replace(".png", "_overlay.png")
        fig2.savefig(overlay_path, dpi=S["dpi"], bbox_inches="tight")
        print(f"Saved overlay to {overlay_path}")
    else:
        plt.show()
    plt.close(fig2)

    fig.suptitle(f"t={t} | bbox Y=[{y_min}:{y_max_val+1}], X=[{x_min}:{x_max_val+1}]",
                 fontsize=S["fs_suptitle"])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=S["dpi"], bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_z_focus(
    z_focus: list[int],
    save_path: str | None = None,
    n_std: float = 2.5,
    z_window: int | None = None,
) -> dict:
    """Plot z focus index over time with statistics and outlier detection.

    Returns a dict with statistics and outlier timepoints.
    """
    z_arr = np.array(z_focus, dtype=float)
    t_axis = np.arange(len(z_arr))
    mu = np.mean(z_arr)
    sigma = np.std(z_arr)
    if z_window is not None:
        upper = mu + z_window
        lower = mu - z_window
    else:
        upper = mu + n_std * sigma
        lower = mu - n_std * sigma

    print(f"Z window: {z_window}")
    print(f"Upper: {upper}, Lower: {lower}")
    print(f"N std: {n_std}")

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
    S = STYLE
    fig, ax = plt.subplots(1, 1, figsize=S["fig_single"])
    ax.plot(t_axis, z_arr, ".-", color=S["c_primary"], ms=S["marker_size"], alpha=S["alpha_data"], linewidth=S["lw_data"], label="per timepoint")
    ax.axhline(mu, color=S["c_mean"], linestyle=":", linewidth=S["lw_ref"], label=f"mean = {mu:.1f}")
    ax.fill_between(t_axis, mu - sigma, mu + sigma, color=S["c_fill"], alpha=S["alpha_fill"], label=f"1 std = {sigma:.1f}")
    if z_window is not None:
        band_label = f"z_window={z_window} [{lower:.1f}, {upper:.1f}]"
        outlier_label_above = f"above mean+{z_window} (n={len(t_above)})"
        outlier_label_below = f"below mean-{z_window} (n={len(t_below)})"
    else:
        band_label = f"{n_std} std = [{lower:.1f}, {upper:.1f}]"
        outlier_label_above = f"above {n_std}std (n={len(t_above)})"
        outlier_label_below = f"below {n_std}std (n={len(t_below)})"
    ax.fill_between(t_axis, lower, upper, color=S["c_threshold"], alpha=0.07, label=band_label)
    ax.axhline(np.median(z_arr), color=S["c_median"], linestyle="--", linewidth=S["lw_ref"], label=f"median = {np.median(z_arr):.0f}")
    # Mark outliers
    if len(t_above) > 0:
        ax.scatter(t_above, z_arr[t_above], color=S["c_outlier"], s=S["scatter_size"], zorder=5, label=outlier_label_above)
    if len(t_below) > 0:
        ax.scatter(t_below, z_arr[t_below], color=S["c_tertiary"], s=S["scatter_size"], zorder=5, label=outlier_label_below)
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.set_ylabel("Z focus index", fontsize=S["fs_label"])
    ax.set_title(
        f"Z focus over time | mean={mu:.1f}, std={sigma:.1f}, "
        f"range=[{z_arr.min():.0f}, {z_arr.max():.0f}]",
        fontsize=S["fs_title"],
    )
    ax.legend(fontsize=S["fs_legend"], loc="best")
    _set_ylim(ax, "z_focus", data=z_arr)
    _apply_style(fig)
    if save_path:
        plt.savefig(save_path, dpi=S["dpi"], bbox_inches="tight")
        print(f"Saved z focus plot to {save_path}")
    else:
        plt.show()
    plt.close()

    return {
        "mean": mu, "std": sigma, "median": np.median(z_arr),
        "upper_2std": upper, "lower_2std": lower,
        "t_above": t_above, "t_below": t_below, "t_outliers": t_outliers,
    }


# ---------------------------------------------------------------------------
# QC plot functions — called from dynacell_qc.py compute_* functions
# Each returns a matplotlib Figure (caller is responsible for saving/closing).
# ---------------------------------------------------------------------------

def plot_registration_qc(
    pearson_corrs: np.ndarray,
    pcc_shifts_y: np.ndarray,
    pcc_shifts_x: np.ndarray,
    shift_mag: np.ndarray,
    mu_corr: float,
    sigma_corr: float,
    lower_corr: float,
    mu_shift: float,
    upper_shift: float,
    corr_outliers: np.ndarray,
    shift_outliers: np.ndarray,
) -> plt.Figure:
    """Plot beads registration QC: Pearson correlation, PCC shifts, shift magnitude."""
    S = STYLE
    T = len(pearson_corrs)
    fig, axes = plt.subplots(3, 1, figsize=S["fig_triple"], sharex=True)

    ax = axes[0]
    ax.plot(pearson_corrs, ".-", color=S["c_primary"], ms=S["marker_size"], alpha=S["alpha_data"], linewidth=S["lw_data"])
    ax.axhline(mu_corr, color=S["c_mean"], linestyle=":", linewidth=S["lw_ref"])
    ax.fill_between(range(T), mu_corr - sigma_corr, mu_corr + sigma_corr,
                    color=S["c_fill"], alpha=S["alpha_fill"])
    ax.axhline(lower_corr, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"], alpha=0.5)
    if len(corr_outliers) > 0:
        ax.scatter(corr_outliers, pearson_corrs[corr_outliers],
                   color=S["c_outlier"], s=S["scatter_size"], zorder=5)
    ax.set_ylabel("Pearson correlation", fontsize=S["fs_label"])
    ax.set_title(f"Beads registration QC | mean r={mu_corr:.4f}", fontsize=S["fs_title"])
    _set_ylim(ax, "pearson_beads", data=pearson_corrs)

    ax = axes[1]
    ax.plot(pcc_shifts_y, ".-", color=S["c_primary"], ms=S["marker_size"], alpha=S["alpha_data"], linewidth=S["lw_data"], label="shift Y")
    ax.plot(pcc_shifts_x, ".-", color=S["c_secondary"], ms=S["marker_size"], alpha=S["alpha_data"], linewidth=S["lw_data"], label="shift X")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("PCC shift (px)", fontsize=S["fs_label"])
    ax.legend(fontsize=S["fs_legend"])

    ax = axes[2]
    ax.plot(shift_mag, ".-", color=S["c_tertiary"], ms=S["marker_size"], alpha=S["alpha_data"], linewidth=S["lw_data"])
    ax.axhline(mu_shift, color=S["c_mean"], linestyle=":", linewidth=S["lw_ref"])
    ax.axhline(upper_shift, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"], alpha=0.5)
    if len(shift_outliers) > 0:
        ax.scatter(shift_outliers, shift_mag[shift_outliers],
                   color=S["c_outlier"], s=S["scatter_size"], zorder=5)
    ax.set_ylabel("Shift magnitude (px)", fontsize=S["fs_label"])
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    _set_ylim(ax, "shift_magnitude", data=shift_mag)

    fig.suptitle("Beads registration QC", fontsize=S["fs_suptitle"])
    _apply_style(fig)
    return fig


def plot_laplacian_qc(
    lap_vars: np.ndarray,
    mu: float,
    sigma: float,
    lower: float,
    n_std: float,
    outliers: np.ndarray,
    channel_name: str,
    local_z: np.ndarray | None = None,
) -> plt.Figure:
    """Plot Laplacian variance and local z-score per timepoint."""
    S = STYLE
    valid_idx = np.where(~np.isnan(lap_vars))[0]
    n_blank = len(lap_vars) - len(valid_idx)
    has_lz = local_z is not None

    n_panels = 2 if has_lz else 1
    figsize = S["fig_double"] if has_lz else S["fig_single"]
    if n_panels == 1:
        fig, ax_single = plt.subplots(1, 1, figsize=figsize)
        axes = [ax_single]
    else:
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)

    T = len(lap_vars)

    # Panel 1: Laplacian variance
    ax = axes[0]
    ax.plot(valid_idx, lap_vars[valid_idx], ".-", ms=S["marker_size"], alpha=S["alpha_data"], linewidth=S["lw_data"])
    ax.axhline(mu, color=S["c_mean"], linestyle=":", linewidth=S["lw_ref"], label=f"mean={mu:.1f}")
    ax.fill_between(valid_idx, mu - sigma, mu + sigma, color=S["c_fill"], alpha=S["alpha_fill"])
    ax.axhline(lower, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"], alpha=0.5,
               label=f"mean-{n_std}*std={lower:.1f}")
    if len(outliers) > 0:
        ax.scatter(outliers, lap_vars[outliers], color=S["c_outlier"], s=S["scatter_size"], zorder=5,
                   label=f"outliers ({len(outliers)})")
    ax.set_ylabel("Laplacian variance (max-Z proj)", fontsize=S["fs_label"])
    title = f"Laplacian blur QC (max-Z proj) | {channel_name}"
    if n_blank > 0:
        title += f" (excl. {n_blank} blank)"
    ax.set_title(title, fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"], loc="best")
    _set_ylim(ax, "laplacian", data=lap_vars[valid_idx])

    # Panel 2: Local z-score
    if has_lz:
        ax = axes[1]
        plot_lz = local_z[valid_idx]
        ax.plot(valid_idx, plot_lz, ".-", ms=S["marker_size"], alpha=S["alpha_data"], color=S["c_zscore"],
                label="Local z-score")
        ax.axhline(-n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"], alpha=0.6,
                   label=f"threshold (-{n_std})")
        if len(outliers) > 0:
            ax.scatter(outliers, local_z[outliers], color=S["c_outlier"], s=S["scatter_size"], zorder=5)
        ax.set_ylabel("Local z-score", fontsize=S["fs_label"])
        ax.legend(fontsize=S["fs_legend"])
        _set_ylim(ax, "laplacian_local_z", data=plot_lz)

    axes[-1].set_xlabel("Timepoint", fontsize=S["fs_label"])
    axes[-1].set_xlim(0, T - 1)

    _apply_style(fig)
    return fig


def plot_entropy_qc(
    entropies: np.ndarray,
    local_z: np.ndarray,
    blank_mask: np.ndarray,
    outliers: np.ndarray,
    med: float,
    n_std: float,
    channel_name: str,
    global_lower: float | None = None,
    global_upper: float | None = None,
) -> plt.Figure:
    """Plot Shannon entropy and local z-score per timepoint (blank frames excluded)."""
    S = STYLE
    valid_idx = np.where(~blank_mask)[0]
    plot_ent = entropies[valid_idx]
    plot_local_z = local_z[valid_idx]

    T = len(entropies)
    fig, axes = plt.subplots(2, 1, figsize=S["fig_double"], sharex=True)

    ax = axes[0]
    ax.plot(valid_idx, plot_ent, ".-", ms=S["marker_size"], alpha=S["alpha_data"])
    ax.axhline(med, color=S["c_median"], ls="--", lw=S["lw_ref"], label=f"median={med:.4f}")
    if global_lower is not None:
        ax.axhline(global_lower, color=S["c_mean"], ls=":", lw=S["lw_ref"],
                    label=f"IQR lower={global_lower:.4f}")
    if global_upper is not None:
        ax.axhline(global_upper, color=S["c_mean"], ls=":", lw=S["lw_ref"],
                    label=f"IQR upper={global_upper:.4f}")
        ax.fill_between(
            [0, T - 1], global_lower, global_upper,
            color=S["c_median"], alpha=0.05, zorder=0,
        )
    if len(outliers) > 0:
        ax.scatter(outliers, entropies[outliers], color=S["c_outlier"],
                   s=S["scatter_size"], zorder=5, label=f"outliers ({len(outliers)})")
    ax.set_ylabel("Shannon entropy", fontsize=S["fs_label"])
    ax.set_title(f"Entropy QC | {channel_name} (global+local threshold)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"])
    _set_ylim(ax, "entropy", data=plot_ent)

    ax = axes[1]
    ax.plot(valid_idx, plot_local_z, ".-", ms=S["marker_size"], alpha=S["alpha_data"], color=S["c_zscore"],
            label="Local z-score")
    ax.axhline(n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"],
               label=f"threshold (+{n_std})")
    ax.axhline(-n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"],
               label=f"threshold (-{n_std})")
    if len(outliers) > 0:
        ax.scatter(outliers, local_z[outliers], color=S["c_outlier"],
                   s=S["scatter_size"], zorder=5)
    ax.set_ylabel("Local z-score", fontsize=S["fs_label"])
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.legend(fontsize=S["fs_legend"])
    ax.set_xlim(0, T - 1)
    _set_ylim(ax, "entropy_local_z", data=plot_local_z)

    _apply_style(fig)
    return fig


def plot_hf_ratio_qc(
    hf_ratios: np.ndarray,
    hf_local_z: np.ndarray,
    blank_mask: np.ndarray,
    outliers: np.ndarray,
    med: float,
    n_std: float,
    channel_name: str,
) -> plt.Figure:
    """Plot HF energy ratio and local z-score (2-panel)."""
    S = STYLE
    valid_idx = np.where(~blank_mask)[0]
    plot_hf = hf_ratios[valid_idx]
    plot_hf_lz = hf_local_z[valid_idx]

    T = len(hf_ratios)
    fig, axes = plt.subplots(2, 1, figsize=S["fig_double"], sharex=True)

    # Panel 1: HF ratio (max-Z projection)
    ax = axes[0]
    ax.plot(valid_idx, plot_hf, ".-", ms=S["marker_size"], alpha=S["alpha_data"])
    ax.axhline(med, color=S["c_median"], ls="--", lw=S["lw_ref"], label=f"median={med:.6f}")
    if len(outliers) > 0:
        ax.scatter(outliers, hf_ratios[outliers], color=S["c_outlier"],
                   s=S["scatter_size"], zorder=5, label=f"outliers ({len(outliers)})")
    ax.set_ylabel("HF energy ratio", fontsize=S["fs_label"])
    ax.set_title(f"HF ratio QC (max-Z proj) | {channel_name}", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"])
    _set_ylim(ax, "hf_ratio", data=plot_hf)

    # Panel 2: HF local z-score
    ax = axes[1]
    ax.plot(valid_idx, plot_hf_lz, ".-", ms=S["marker_size"], alpha=S["alpha_data"], color=S["c_zscore"],
            label="Local z-score")
    ax.axhline(-n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"], alpha=0.6,
               label=f"threshold (-{n_std})")
    if len(outliers) > 0:
        ax.scatter(outliers, hf_local_z[outliers], color=S["c_outlier"],
                   s=S["scatter_size"], zorder=5)
    ax.set_ylabel("Local z-score", fontsize=S["fs_label"])
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.legend(fontsize=S["fs_legend"])
    ax.set_xlim(0, T - 1)
    _set_ylim(ax, "hf_local_z", data=plot_hf_lz)

    _apply_style(fig)
    return fig


def plot_frc_qc(
    frc_values: np.ndarray,
    local_z: np.ndarray,
    blank_mask: np.ndarray,
    med: float,
    channel_name: str,
    mean_frc_curve: np.ndarray | None = None,
    n_std: float = 2.5,
) -> plt.Figure:
    """Plot FRC mean correlation, local z-score, and mean FRC curve."""
    S = STYLE
    valid_idx = np.where(~blank_mask)[0]
    plot_frc = frc_values[valid_idx]
    plot_lz = local_z[valid_idx]
    outlier_idx = valid_idx[plot_lz < -n_std] if len(plot_lz) > 0 else np.array([])

    T = len(frc_values)
    n_panels = 3 if mean_frc_curve is not None else 2
    figsize = S["fig_triple"] if n_panels == 3 else S["fig_double"]
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=(n_panels == 2))

    # Panel 1: FRC mean correlation over time
    ax = axes[0]
    ax.plot(valid_idx, plot_frc, ".-", ms=S["marker_size"], alpha=S["alpha_data"])
    ax.axhline(med, color=S["c_median"], ls="--", lw=S["lw_ref"], label=f"median={med:.4f}")
    if len(outlier_idx) > 0:
        ax.scatter(outlier_idx, frc_values[outlier_idx], color=S["c_outlier"],
                   s=S["scatter_size"], zorder=5, label=f"outliers ({len(outlier_idx)})")
    ax.set_ylabel("FRC mean correlation", fontsize=S["fs_label"])
    ax.set_title(f"FRC blur QC (max-Z proj) | {channel_name}", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"])
    _set_ylim(ax, "frc", data=plot_frc)

    # Panel 2: Local z-score
    ax = axes[1]
    ax.plot(valid_idx, plot_lz, ".-", ms=S["marker_size"], alpha=S["alpha_data"],
            color=S["c_zscore"], label="Local z-score")
    ax.axhline(-n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"],
               label=f"threshold (-{n_std})")
    if len(outlier_idx) > 0:
        ax.scatter(outlier_idx, local_z[outlier_idx], color=S["c_outlier"],
                   s=S["scatter_size"], zorder=5)
    ax.set_ylabel("Local z-score", fontsize=S["fs_label"])
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.legend(fontsize=S["fs_legend"])
    ax.set_xlim(0, T - 1)
    _set_ylim(ax, "frc_local_z", data=plot_lz)

    # Panel 3: Mean FRC correlation curve (correlation vs frequency ring)
    if mean_frc_curve is not None:
        ax = axes[2]
        freq = np.linspace(0, 1, len(mean_frc_curve))
        ax.plot(freq, mean_frc_curve, color=S["c_primary"], lw=S["lw_ref"],
                label="mean FRC curve")
        ax.axhline(1 / 7, color=S["c_threshold"], ls="--", lw=S["lw_threshold"],
                    alpha=0.7, label="1/7 threshold")
        ax.set_xlabel("Spatial frequency (Nyquist fraction)", fontsize=S["fs_label"])
        ax.set_ylabel("FRC correlation", fontsize=S["fs_label"])
        ax.set_title("Mean FRC curve (averaged over timepoints)", fontsize=S["fs_title"])
        ax.legend(fontsize=S["fs_legend"])
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)

    _apply_style(fig)
    return fig


def plot_max_intensity_qc(
    max_vals: np.ndarray,
    blank_mask: np.ndarray,
    med: float,
    channel_name: str,
    local_z: np.ndarray | None = None,
    n_std: float = 2.5,
) -> plt.Figure:
    """Plot max intensity and local z-score over time."""
    S = STYLE
    valid_idx = np.where(~blank_mask)[0]
    plot_vals = max_vals[valid_idx]
    has_lz = local_z is not None and not np.all(np.isnan(local_z))
    T = len(max_vals)

    if has_lz:
        fig, axes = plt.subplots(2, 1, figsize=S["fig_double"], sharex=True)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(1, 1, figsize=S["fig_single"])

    ax.plot(valid_idx, plot_vals, ".-", ms=S["marker_size"], alpha=S["alpha_data"])
    ax.axhline(med, color=S["c_median"], ls="--", lw=S["lw_ref"], label=f"median={med:.2f}")
    ax.set_ylabel("Max intensity (full Z)", fontsize=S["fs_label"])
    ax.set_title(f"Max intensity QC | {channel_name}", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"])
    ax.set_xlim(0, T - 1)
    _set_ylim(ax, "max_intensity", data=plot_vals)

    if has_lz:
        ax = axes[1]
        plot_lz = local_z[valid_idx]
        ax.plot(valid_idx, plot_lz, ".-", ms=S["marker_size"], alpha=S["alpha_data"], color=S["c_zscore"],
                label="Local z-score")
        ax.axhline(n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"], alpha=0.6,
                   label=f"threshold (+{n_std})")
        ax.axhline(-n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"], alpha=0.6,
                   label=f"threshold (-{n_std})")
        outlier_mask = np.abs(local_z) > n_std
        outlier_valid = outlier_mask[valid_idx]
        if outlier_valid.any():
            ax.scatter(valid_idx[outlier_valid], plot_lz[outlier_valid],
                       color=S["c_outlier"], s=S["scatter_size"], zorder=5)
        ax.set_ylabel("Local z-score", fontsize=S["fs_label"])
        ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
        ax.legend(fontsize=S["fs_legend"])
        ax.set_xlim(0, T - 1)
        _set_ylim(ax, "max_intensity_local_z", data=plot_lz)
    else:
        ax.set_xlabel("Timepoint", fontsize=S["fs_label"])

    _apply_style(fig)
    return fig


def plot_fov_registration_qc(
    pearson_corrs: np.ndarray,
    mu: float,
    n_outliers: int,
    outlier_idx: np.ndarray,
    blank_count: int = 0,
    local_z: np.ndarray | None = None,
    n_std: float = 2.5,
) -> plt.Figure:
    """Plot per-FOV Pearson correlation and local z-score."""
    S = STYLE
    valid_mask = ~np.isnan(pearson_corrs)
    valid_idx = np.where(valid_mask)[0]
    T = len(pearson_corrs)
    has_lz = local_z is not None and not np.all(np.isnan(local_z))

    if has_lz:
        fig, axes = plt.subplots(2, 1, figsize=S["fig_double"], sharex=True)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=S["fig_single"])

    ax.plot(valid_idx, pearson_corrs[valid_idx], ".-",
            markersize=S["marker_size"], linewidth=S["lw_data"])
    title = "Per-timepoint LF–LS registration (Phase ch 0)"
    if blank_count > 0:
        title += f" (excl. {blank_count} blank)"
    ax.set_title(title, fontsize=S["fs_title"])
    ax.axhline(mu, color=S["c_mean"], linestyle="--", linewidth=S["lw_ref"],
               label=f"mean={mu:.4f}")
    if n_outliers > 0:
        ax.scatter(outlier_idx, pearson_corrs[outlier_idx], color=S["c_outlier"],
                   s=S["scatter_size"], zorder=5, label=f"outliers ({n_outliers})")
    ax.set_ylabel("Pearson correlation (LF vs LS)", fontsize=S["fs_label"])
    ax.legend(fontsize=S["fs_legend"])
    ax.set_xlim(0, T - 1)
    _set_ylim(ax, "fov_reg_pearson", data=pearson_corrs[valid_idx])

    if has_lz:
        ax = axes[1]
        plot_lz = local_z[valid_idx]
        ax.plot(valid_idx, plot_lz, ".-", ms=S["marker_size"], alpha=S["alpha_data"],
                color=S["c_zscore"], label="Local z-score")
        ax.axhline(-n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"], alpha=0.6,
                   label=f"threshold (-{n_std})")
        if n_outliers > 0:
            ax.scatter(outlier_idx, local_z[outlier_idx], color=S["c_outlier"],
                       s=S["scatter_size"], zorder=5)
        ax.set_ylabel("Local z-score", fontsize=S["fs_label"])
        ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
        ax.legend(fontsize=S["fs_legend"])
        ax.set_xlim(0, T - 1)
        _set_ylim(ax, "fov_reg_local_z", data=plot_lz)
    else:
        ax.set_xlabel("Timepoint", fontsize=S["fs_label"])

    _apply_style(fig)
    return fig


def plot_tilt_qc(
    tilt_ranges: np.ndarray,
    tilt_slopes: np.ndarray,
    blank_mask: np.ndarray,
    med: float,
    local_z: np.ndarray | None = None,
    n_std: float = 2.5,
    example_grid: np.ndarray | None = None,
    grid_size: int = 3,
) -> plt.Figure:
    """Plot tilt range over time, local z-score, and example grid heatmap."""
    S = STYLE
    valid_idx = np.where(~blank_mask)[0]
    has_lz = local_z is not None and not np.all(np.isnan(local_z))
    T = len(tilt_ranges)

    n_rows = 2 if has_lz else 1
    has_grid = example_grid is not None and not np.all(np.isnan(example_grid))
    if has_grid:
        fig = plt.figure(figsize=(S["fig_double"][0] + 3, S["fig_double"][1]))
        gs = fig.add_gridspec(n_rows, 2, width_ratios=[3, 1], hspace=0.3, wspace=0.3)
        ax_range = fig.add_subplot(gs[0, 0])
        ax_grid = fig.add_subplot(gs[0, 1])
        if has_lz:
            ax_lz = fig.add_subplot(gs[1, 0], sharex=ax_range)
    else:
        if has_lz:
            fig, axes = plt.subplots(2, 1, figsize=S["fig_double"], sharex=True)
            ax_range = axes[0]
            ax_lz = axes[1]
        else:
            fig, ax_range = plt.subplots(1, 1, figsize=S["fig_single"])

    # Tilt range trace
    ax_range.plot(valid_idx, tilt_ranges[valid_idx], ".-",
                  ms=S["marker_size"], alpha=S["alpha_data"])
    ax_range.axhline(med, color=S["c_median"], ls="--", lw=S["lw_ref"],
                     label=f"median={med:.2f}")
    ax_range.set_ylabel("Tilt range (z-slices)", fontsize=S["fs_label"])
    ax_range.set_title("Tilt QC | sub-region z-focus range", fontsize=S["fs_title"])
    ax_range.legend(fontsize=S["fs_legend"])
    ax_range.set_xlim(0, T - 1)

    # Example grid heatmap
    if has_grid:
        im = ax_grid.imshow(example_grid, cmap="RdYlBu_r", aspect="equal",
                            interpolation="nearest")
        ax_grid.set_title(f"Z-focus grid (t=0)", fontsize=S["fs_title"])
        ax_grid.set_xlabel("X region", fontsize=S["fs_label"])
        ax_grid.set_ylabel("Y region", fontsize=S["fs_label"])
        for iy in range(grid_size):
            for ix in range(grid_size):
                val = example_grid[iy, ix]
                if not np.isnan(val):
                    ax_grid.text(ix, iy, f"{val:.0f}", ha="center", va="center",
                                fontsize=8, color="black")
        fig.colorbar(im, ax=ax_grid, label="z-focus", shrink=0.8)

    # Local z-score
    if has_lz:
        plot_lz = local_z[valid_idx]
        ax_lz.plot(valid_idx, plot_lz, ".-", ms=S["marker_size"],
                   alpha=S["alpha_data"], color=S["c_zscore"], label="Local z-score")
        ax_lz.axhline(n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"],
                      alpha=0.6, label=f"threshold (+{n_std})")
        outlier_mask = local_z > n_std
        outlier_valid = outlier_mask[valid_idx]
        if outlier_valid.any():
            ax_lz.scatter(valid_idx[outlier_valid], plot_lz[outlier_valid],
                         color=S["c_outlier"], s=S["scatter_size"], zorder=5)
        ax_lz.set_ylabel("Local z-score", fontsize=S["fs_label"])
        ax_lz.set_xlabel("Timepoint", fontsize=S["fs_label"])
        ax_lz.legend(fontsize=S["fs_legend"])
        ax_lz.set_xlim(0, T - 1)
    else:
        ax_range.set_xlabel("Timepoint", fontsize=S["fs_label"])

    _apply_style(fig)
    return fig


def plot_bleach_fov_qc(
    normalized: np.ndarray,
    fov_name: str,
    local_z: np.ndarray | None = None,
    n_std: float = 3.0,
) -> plt.Figure:
    """Plot per-FOV bleach curve and local z-score (residual from trend)."""
    S = STYLE
    valid_idx = np.where(~np.isnan(normalized))[0]
    pct_rem = float(normalized[valid_idx[-1]] * 100) if len(valid_idx) > 0 else 0
    has_lz = local_z is not None and not np.all(np.isnan(local_z))

    if has_lz:
        fig, axes = plt.subplots(2, 1, figsize=S["fig_double"], sharex=True)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=S["fig_single"])

    ax.plot(valid_idx, normalized[valid_idx], ".-",
            markersize=S["marker_size"], linewidth=S["lw_data"])
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_ylabel("Normalized intensity", fontsize=S["fs_label"])
    ax.set_title(f"Bleach QC | {fov_name} | {pct_rem:.1f}% remaining", fontsize=S["fs_title"])
    _set_ylim(ax, "bleach_norm")

    if has_lz:
        ax = axes[1]
        plot_lz = local_z[valid_idx]
        ax.plot(valid_idx, plot_lz, ".-", ms=S["marker_size"], alpha=S["alpha_data"],
                color=S["c_zscore"], label="Local z-score (residual)")
        ax.axhline(n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"], alpha=0.6,
                   label=f"threshold (\u00b1{n_std})")
        ax.axhline(-n_std, color=S["c_threshold"], ls="--", lw=S["lw_threshold"], alpha=0.6)
        # Mark outliers
        outlier_mask = np.abs(local_z) > n_std
        outlier_valid = outlier_mask[valid_idx]
        if outlier_valid.any():
            ax.scatter(valid_idx[outlier_valid], plot_lz[outlier_valid],
                       color=S["c_outlier"], s=S["scatter_size"], zorder=5)
        ax.set_ylabel("Local z-score", fontsize=S["fs_label"])
        ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
        ax.legend(fontsize=S["fs_legend"])
        _set_ylim(ax, "bleach_local_z", data=plot_lz)
    else:
        ax.set_xlabel("Timepoint", fontsize=S["fs_label"])

    _apply_style(fig)
    return fig


def plot_dust_qc(
    dust_fractions: np.ndarray,
    worst_z: int,
    worst_median: np.ndarray,
    dust_mask_worst: np.ndarray | None,
    circ_mask: np.ndarray,
    n_fovs: int,
    n_sample_t: int,
    lf_mask_radius: float,
    total_spots_worst: int,
    max_dust_fraction: float,
) -> plt.Figure:
    """Plot dust QC: fraction per Z, worst-Z median image, dust mask overlay."""
    S = STYLE
    Y, X = worst_median.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: dust fraction vs Z
    ax = axes[0]
    ax.plot(dust_fractions, S["c_primary"], alpha=S["alpha_data"], linewidth=S["lw_data"])
    ax.axhline(dust_fractions.mean(), color=S["c_mean"], linestyle=":", linewidth=S["lw_ref"],
               label=f"mean={dust_fractions.mean():.4f}")
    ax.axvline(worst_z, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"],
               label=f"worst z={worst_z}")
    ax.set_xlabel("Z plane", fontsize=S["fs_label"])
    ax.set_ylabel("Dust fraction", fontsize=S["fs_label"])
    ax.set_title("Dust fraction per Z plane", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"])

    # Panel 2: worst-Z median image (within mask)
    ax = axes[1]
    display_img = worst_median.copy()
    display_img[~circ_mask] = np.nan
    vmin, vmax = np.nanpercentile(display_img[circ_mask], [1, 99])
    ax.imshow(display_img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(f"Median Phase (z={worst_z})", fontsize=S["fs_title"])

    # Panel 3: dust mask at worst Z
    ax = axes[2]
    norm_img = (worst_median - vmin) / (vmax - vmin + 1e-8)
    norm_img = np.clip(norm_img, 0, 1)
    overlay = np.zeros((Y, X, 3), dtype=np.float32)
    overlay[:, :, 0] = norm_img
    overlay[:, :, 1] = norm_img
    overlay[:, :, 2] = norm_img
    if dust_mask_worst is not None:
        overlay[dust_mask_worst, 0] = 1.0
        overlay[dust_mask_worst, 1] = 0.0
        overlay[dust_mask_worst, 2] = 0.0
    overlay[~circ_mask] = 0.0
    ax.imshow(overlay, origin="upper")
    ax.set_title(f"Dust spots (z={worst_z}, {total_spots_worst} spots, "
                 f"{max_dust_fraction*100:.2f}%)", fontsize=S["fs_title"])

    fig.suptitle(f"Dust QC | {n_fovs} FOVs x {n_sample_t} t | mask_radius={lf_mask_radius}",
                 fontsize=S["fs_suptitle"])
    _apply_style(fig)
    return fig


def plot_bleach_qc(
    norm_curves: dict,
    common_t: np.ndarray,
    mean_curve: np.ndarray,
    std_curve: np.ndarray,
    valid_fit: np.ndarray,
    fit_params: dict | None,
    half_life: float,
    summary_rows: list[dict],
    channel_name: str,
) -> plt.Figure:
    """Plot bleach QC: per-FOV normalized curves + mean fit, and per-FOV % remaining."""
    from scipy.optimize import curve_fit  # noqa: F401 — for _exp_decay
    S = STYLE

    def _exp_decay(t, a, tau, c):
        return a * np.exp(-t / tau) + c

    fig, axes = plt.subplots(1, 2, figsize=S["fig_wide"])

    # Panel 1: per-FOV normalized curves + mean
    ax = axes[0]
    for fov_key, (t_arr, norm_vals) in norm_curves.items():
        ax.plot(t_arr, norm_vals, alpha=0.3, linewidth=0.6, color=S["c_primary"])
    ax.plot(common_t[valid_fit], mean_curve[valid_fit], "k-", linewidth=1.5, label="mean")
    ax.fill_between(
        common_t[valid_fit],
        (mean_curve - std_curve)[valid_fit],
        (mean_curve + std_curve)[valid_fit],
        color="gray", alpha=0.2, label="std",
    )
    if fit_params is not None:
        t_fit = common_t[valid_fit]
        t_smooth = np.linspace(t_fit[0], t_fit[-1], 200)
        ax.plot(t_smooth, _exp_decay(t_smooth, **fit_params), color=S["c_threshold"],
                linestyle="--", linewidth=S["lw_ref"], label=f"fit: t1/2={half_life:.1f}")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.set_ylabel("Normalized intensity", fontsize=S["fs_label"])
    ax.set_title(f"GFP bleaching ({len(norm_curves)} FOVs)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"])
    _set_ylim(ax, "bleach_norm")

    # Panel 2: per-FOV % remaining (bar chart)
    ax = axes[1]
    fov_names = [r["fov"] for r in summary_rows]
    pct_vals = [r["pct_remaining"] for r in summary_rows]
    colors = [S["c_secondary"] if p > 70 else S["c_mean"] if p > 40 else S["c_threshold"]
              for p in pct_vals]
    ax.barh(range(len(fov_names)), pct_vals, color=colors, alpha=S["alpha_data"])
    ax.set_yticks(range(len(fov_names)))
    ax.set_yticklabels(fov_names, fontsize=S["fs_legend_multi"])
    ax.set_xlabel("% signal remaining", fontsize=S["fs_label"])
    ax.set_title(f"Signal at last t | mean={np.mean(pct_vals):.1f}%", fontsize=S["fs_title"])
    ax.axvline(100, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlim(0, max(110, max(pct_vals) * 1.05))

    fig.suptitle(f"Bleach QC | {channel_name}", fontsize=S["fs_suptitle"])
    _apply_style(fig)
    return fig


# ---------------------------------------------------------------------------
# Combined all-FOV plots
# ---------------------------------------------------------------------------

def plot_z_focus_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, dict]:
    """Z focus across all FOVs. Returns (fig, {fov_name: z_values})."""
    S = STYLE
    fig, ax = plt.subplots(1, 1, figsize=S["fig_all_single"])
    all_data = {}
    for i, fov in enumerate(ok_fovs):
        fov_name = "_".join(fov.split("/"))
        z_df = pd.read_csv(plots_dir / fov_name / "z_focus.csv")
        z_vals = z_df["z_focus"].values
        all_data[fov_name] = z_vals
        ax.plot(z_vals, alpha=S["alpha_multi"], linewidth=S["lw_data"],
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=fov_name)
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.set_ylabel("Z focus index", fontsize=S["fs_label"])
    ax.set_title(f"Z focus across all FOVs ({len(ok_fovs)} FOVs)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _set_ylim(ax, "z_focus", data=np.concatenate(list(all_data.values())) if all_data else None)
    _apply_style(fig)
    return fig, all_data


def plot_laplacian_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, dict]:
    """Laplacian variance across all FOVs. Returns (fig, {fov_name: series})."""
    S = STYLE
    fig, ax = plt.subplots(1, 1, figsize=S["fig_all_single"])
    all_data = {}
    for i, fov in enumerate(ok_fovs):
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "laplacian_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        col = "lap_var" if "lap_var" in df.columns else "lap3d_var"
        all_data[fov_name] = df.set_index("t")[col]
        ax.plot(df["t"], df[col], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=fov_name)
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.set_ylabel("Laplacian variance", fontsize=S["fs_label"])
    ax.set_title(f"Laplacian QC across all FOVs ({len(all_data)} FOVs)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _set_ylim(ax, "laplacian", data=np.concatenate([s.values for s in all_data.values()]) if all_data else None)
    _apply_style(fig)
    return fig, all_data


def plot_hf_ratio_all_fovs(
    ok_fovs: list[str],
    plots_dir,
    n_std: float = 2.5,
) -> tuple[plt.Figure, dict]:
    """HF ratio + local z-score across all FOVs. Returns (fig, {fov_name: df})."""
    S = STYLE
    fig, axes = plt.subplots(2, 1, figsize=S["fig_all_double"], sharex=True)
    all_data = {}
    for i, fov in enumerate(ok_fovs):
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "hf_ratio_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        all_data[fov_name] = df.set_index("t")[["hf_ratio", "hf_local_z"]]
        c = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        axes[0].plot(df["t"], df["hf_ratio"], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                     color=c, label=fov_name)
        axes[1].plot(df["t"], df["hf_local_z"], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                     color=c, label=fov_name)
    axes[0].set_ylabel("HF energy ratio (max-Z proj)", fontsize=S["fs_label"])
    axes[0].set_title(f"HF ratio QC across all FOVs ({len(all_data)} FOVs)", fontsize=S["fs_title"])
    axes[1].set_xlabel("Timepoint", fontsize=S["fs_label"])
    axes[1].set_ylabel("HF local z-score", fontsize=S["fs_label"])
    axes[1].axhline(-n_std, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"],
                    alpha=0.7, label=f"threshold (n_std={n_std})")
    axes[1].axhline(n_std, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"], alpha=0.7)
    hf_vals = np.concatenate([d["hf_ratio"].values for d in all_data.values()]) if all_data else None
    hf_lz_vals = np.concatenate([d["hf_local_z"].values for d in all_data.values()]) if all_data else None
    _set_ylim(axes[0], "hf_ratio", data=hf_vals)
    _set_ylim(axes[1], "hf_local_z", data=hf_lz_vals)
    for a in axes:
        a.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _apply_style(fig)
    return fig, all_data


def plot_entropy_all_fovs(
    ok_fovs: list[str],
    plots_dir,
    n_std: float = 2.5,
) -> tuple[plt.Figure, dict]:
    """Entropy + local z-score across all FOVs. Returns (fig, {fov_name: df})."""
    S = STYLE
    fig, axes = plt.subplots(2, 1, figsize=S["fig_all_double"], sharex=True)
    all_data = {}
    for i, fov in enumerate(ok_fovs):
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "entropy_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        all_data[fov_name] = df.set_index("t")[["entropy", "local_z"]]
        c = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        axes[0].plot(df["t"], df["entropy"], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                     color=c, label=fov_name)
        axes[1].plot(df["t"], df["local_z"], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                     color=c, label=fov_name)
    axes[0].set_ylabel("Shannon entropy", fontsize=S["fs_label"])
    axes[0].set_title(f"Entropy QC across all FOVs ({len(all_data)} FOVs)", fontsize=S["fs_title"])
    axes[1].set_xlabel("Timepoint", fontsize=S["fs_label"])
    axes[1].set_ylabel("Entropy local z-score", fontsize=S["fs_label"])
    axes[1].axhline(n_std, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"],
                    alpha=0.7, label=f"threshold (n_std={n_std})")
    axes[1].axhline(-n_std, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"], alpha=0.7)
    ent_vals = np.concatenate([d["entropy"].values for d in all_data.values()]) if all_data else None
    ent_lz_vals = np.concatenate([d["local_z"].values for d in all_data.values()]) if all_data else None
    _set_ylim(axes[0], "entropy", data=ent_vals)
    _set_ylim(axes[1], "entropy_local_z", data=ent_lz_vals)
    for a in axes:
        a.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _apply_style(fig)
    return fig, all_data


def plot_blur_detection_all_fovs(
    ok_fovs: list[str],
    plots_dir,
    hf_n_std: float = 2.5,
    ent_n_std: float = 2.5,
) -> tuple[plt.Figure, list[dict]]:
    """Combined HF+entropy blur detection across all FOVs. Returns (fig, blur_summary)."""
    S = STYLE
    fig, axes = plt.subplots(3, 1, figsize=S["fig_all_triple"], sharex=True)
    blur_summary = []
    for i, fov in enumerate(ok_fovs):
        fov_name = "_".join(fov.split("/"))
        hf_csv = plots_dir / fov_name / "hf_ratio_qc.csv"
        if not hf_csv.exists():
            continue
        df = pd.read_csv(hf_csv)
        hf_lz = df["hf_local_z"].values if "hf_local_z" in df.columns else None
        ent_lz = df["ent_local_z"].values if "ent_local_z" in df.columns else None
        is_outlier = df["is_outlier"].values.astype(bool) if "is_outlier" in df.columns else None
        if hf_lz is None:
            continue
        c = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        axes[0].plot(df["t"], hf_lz, alpha=S["alpha_multi"], linewidth=S["lw_data"],
                     color=c, label=fov_name)
        if ent_lz is not None:
            axes[1].plot(df["t"], ent_lz, alpha=S["alpha_multi"], linewidth=S["lw_data"],
                         color=c, label=fov_name)
        if is_outlier is not None and is_outlier.any():
            outlier_t = df["t"].values[is_outlier]
            blur_summary.append({"fov": fov_name, "blur_frames": outlier_t.tolist(),
                                 "n_blur": int(is_outlier.sum())})
            axes[2].scatter(outlier_t, [fov_name] * len(outlier_t),
                            color=S["c_outlier"], s=S["scatter_size"], alpha=0.8)
    axes[0].set_ylabel("HF local z-score", fontsize=S["fs_label"])
    axes[0].set_title("Combined blur detection (HF+entropy) across all FOVs", fontsize=S["fs_title"])
    axes[0].axhline(-hf_n_std, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"],
                    alpha=0.7, label=f"HF thresh (n_std={hf_n_std})")
    axes[0].legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _set_ylim(axes[0], "hf_local_z")
    axes[1].set_ylabel("Entropy local z-score", fontsize=S["fs_label"])
    axes[1].axhline(ent_n_std, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"],
                    alpha=0.7, label=f"Ent thresh (n_std={ent_n_std})")
    axes[1].legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _set_ylim(axes[1], "entropy_local_z")
    axes[2].set_xlabel("Timepoint", fontsize=S["fs_label"])
    axes[2].set_ylabel("FOV", fontsize=S["fs_label"])
    n_blur_total = sum(b["n_blur"] for b in blur_summary)
    axes[2].set_title(f"Flagged blur frames: {n_blur_total} total across {len(blur_summary)} FOVs",
                      fontsize=S["fs_title"])
    _apply_style(fig)
    return fig, blur_summary


def plot_drop_correlation_all_fovs(
    ok_fovs: list[str],
    plots_dir,
    T_total: int,
) -> tuple[plt.Figure | None, pd.DataFrame | None, list[str]]:
    """Drop frame correlation across FOVs by reason.

    Returns (fig, drop_df, correlated_drop_lines). fig is None if no drops.
    """
    all_drop_rows = []
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        drop_csv = plots_dir / fov_name / "drop_list.csv"
        if not drop_csv.exists():
            continue
        df = pd.read_csv(drop_csv)
        for _, row in df.iterrows():
            for reason in str(row["reason"]).split(", "):
                all_drop_rows.append({
                    "fov": fov_name, "t": int(row["t"]), "reason": reason.strip()
                })
    if not all_drop_rows:
        return None, None, []

    S = STYLE
    drop_all_df = pd.DataFrame(all_drop_rows)
    reasons_unique = sorted(drop_all_df["reason"].unique())
    fov_names_sorted = sorted(drop_all_df["fov"].unique())
    n_reasons = len(reasons_unique)
    reason_colors = S["c_qc"]

    fig, axes = plt.subplots(n_reasons + 2, 1,
                             figsize=(14, 3 * (n_reasons + 2)),
                             gridspec_kw={"height_ratios": [2] * n_reasons + [1, 2]})

    # Per-reason scatter heatmaps
    for i, reason in enumerate(reasons_unique):
        ax = axes[i]
        subset = drop_all_df[drop_all_df["reason"] == reason]
        color = reason_colors.get(reason, S["c_primary"])
        for j, fn in enumerate(fov_names_sorted):
            fov_drops = subset[subset["fov"] == fn]["t"].values
            if len(fov_drops) > 0:
                ax.scatter(fov_drops, [j] * len(fov_drops),
                           color=color, s=15, alpha=0.8)
        ax.set_yticks(range(len(fov_names_sorted)))
        ax.set_yticklabels(fov_names_sorted, fontsize=S["fs_legend_multi"])
        ax.set_xlim(-0.5, T_total - 0.5)
        ax.set_title(f"{reason} ({len(subset)} drops across {subset['fov'].nunique()} FOVs)",
                     fontsize=S["fs_title"])

    # Stacked histogram
    ax = axes[n_reasons]
    t_range = np.arange(T_total)
    bottom = np.zeros(T_total)
    for reason in reasons_unique:
        subset = drop_all_df[drop_all_df["reason"] == reason]
        counts = np.zeros(T_total)
        for t in subset["t"].values:
            if 0 <= t < T_total:
                counts[t] += 1
        color = reason_colors.get(reason, S["c_primary"])
        ax.bar(t_range, counts, bottom=bottom, width=1.0, alpha=S["alpha_data"],
               color=color, label=reason)
        bottom += counts
    ax.set_ylabel("# FOVs dropped", fontsize=S["fs_label"])
    ax.set_title("Timepoints dropped across FOVs (stacked by reason)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"])
    ax.set_xlim(-0.5, T_total - 0.5)

    # Correlated drops (>=2 FOVs)
    ax = axes[n_reasons + 1]
    corr_text = []
    for reason in reasons_unique:
        subset = drop_all_df[drop_all_df["reason"] == reason]
        t_counts = subset.groupby("t")["fov"].nunique()
        shared = t_counts[t_counts >= 2].sort_values(ascending=False)
        if len(shared) > 0:
            color = reason_colors.get(reason, S["c_primary"])
            ax.bar(shared.index, shared.values, width=0.8, alpha=S["alpha_data"],
                   color=color, label=reason)
            for t, n in shared.items():
                fovs_at_t = subset[subset["t"] == t]["fov"].tolist()
                corr_text.append(f"  t={t} ({reason}): {n} FOVs — {', '.join(fovs_at_t)}")
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.set_ylabel("# FOVs", fontsize=S["fs_label"])
    ax.set_title("Correlated drops: timepoints dropped in >=2 FOVs", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"])
    ax.set_xlim(-0.5, T_total - 0.5)
    if not corr_text:
        ax.text(0.5, 0.5, "No correlated drops found", transform=ax.transAxes,
                ha="center", va="center", fontsize=S["fs_annotation"], color="gray")

    _apply_style(fig)
    return fig, drop_all_df, corr_text


def plot_outlier_correlation_all_fovs(
    ok_fovs: list[str],
    plots_dir,
    T_total: int,
) -> tuple[plt.Figure | None, pd.DataFrame | None, list[str]]:
    """QC outlier correlation across FOVs (reporting-only outliers).

    Reads per-FOV QC CSVs for entropy, HF, FRC, and registration outliers.
    For CSVs without is_outlier column, computes z < -2.5 on the fly.

    Returns (fig, outlier_df, correlated_lines). fig is None if no outliers.
    """
    qc_sources = {
        "entropy": ("entropy_qc.csv", "is_outlier", "entropy", None),
        "hf_blur": ("hf_ratio_qc.csv", "is_outlier", "hf_ratio", None),
        "frc": ("frc_qc.csv", "is_outlier", "frc_mean_corr", -2.5),
        "fov_reg": ("fov_registration_qc.csv", "is_outlier", "pearson_corr", -2.5),
    }

    all_rows = []
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        for reason, (csv_name, outlier_col, value_col, z_thresh) in qc_sources.items():
            csv_path = plots_dir / fov_name / csv_name
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if outlier_col in df.columns:
                outlier_ts = df.loc[df[outlier_col] == 1, "t"].values
            elif value_col in df.columns and z_thresh is not None:
                # Compute z-score on the fly for old CSVs
                vals = df[value_col].values
                valid = ~np.isnan(vals)
                if valid.sum() < 3:
                    continue
                mu = np.nanmean(vals)
                sigma = np.nanstd(vals)
                if sigma < 1e-12:
                    continue
                z_scores = np.full(len(vals), np.nan)
                z_scores[valid] = (vals[valid] - mu) / sigma
                outlier_ts = df.loc[z_scores < z_thresh, "t"].values
            else:
                continue
            for t in outlier_ts:
                all_rows.append({"fov": fov_name, "t": int(t), "reason": reason})

    if not all_rows:
        return None, None, []

    S = STYLE
    outlier_df = pd.DataFrame(all_rows)
    reasons_unique = sorted(outlier_df["reason"].unique())
    fov_names_sorted = sorted(outlier_df["fov"].unique())
    n_reasons = len(reasons_unique)
    reason_colors = S["c_qc"]

    fig, axes = plt.subplots(n_reasons + 2, 1,
                             figsize=(14, 3 * (n_reasons + 2)),
                             gridspec_kw={"height_ratios": [2] * n_reasons + [1, 2]})
    if n_reasons + 2 == 1:
        axes = [axes]

    # Per-reason scatter
    for i, reason in enumerate(reasons_unique):
        ax = axes[i]
        subset = outlier_df[outlier_df["reason"] == reason]
        color = reason_colors.get(reason, S["c_primary"])
        for j, fn in enumerate(fov_names_sorted):
            fov_outliers = subset[subset["fov"] == fn]["t"].values
            if len(fov_outliers) > 0:
                ax.scatter(fov_outliers, [j] * len(fov_outliers),
                           color=color, s=15, alpha=0.8)
        ax.set_yticks(range(len(fov_names_sorted)))
        ax.set_yticklabels(fov_names_sorted, fontsize=S["fs_legend_multi"])
        ax.set_xlim(-0.5, T_total - 0.5)
        ax.set_title(f"{reason} outliers ({len(subset)} across {subset['fov'].nunique()} FOVs)",
                     fontsize=S["fs_title"])

    # Stacked histogram
    ax = axes[n_reasons]
    t_range = np.arange(T_total)
    bottom = np.zeros(T_total)
    for reason in reasons_unique:
        subset = outlier_df[outlier_df["reason"] == reason]
        counts = np.zeros(T_total)
        for t in subset["t"].values:
            if 0 <= t < T_total:
                counts[t] += 1
        color = reason_colors.get(reason, S["c_primary"])
        ax.bar(t_range, counts, bottom=bottom, width=1.0, alpha=S["alpha_data"],
               color=color, label=reason)
        bottom += counts
    ax.set_ylabel("# FOVs flagged", fontsize=S["fs_label"])
    ax.set_title("Outlier timepoints across FOVs (stacked by QC metric)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"])
    ax.set_xlim(-0.5, T_total - 0.5)

    # Correlated outliers (>=2 FOVs)
    ax = axes[n_reasons + 1]
    corr_text = []
    for reason in reasons_unique:
        subset = outlier_df[outlier_df["reason"] == reason]
        t_counts = subset.groupby("t")["fov"].nunique()
        shared = t_counts[t_counts >= 2].sort_values(ascending=False)
        if len(shared) > 0:
            color = reason_colors.get(reason, S["c_primary"])
            ax.bar(shared.index, shared.values, width=0.8, alpha=S["alpha_data"],
                   color=color, label=reason)
            for t, n in shared.items():
                fovs_at_t = subset[subset["t"] == t]["fov"].tolist()
                corr_text.append(f"  t={t} ({reason}): {n} FOVs — {', '.join(fovs_at_t)}")
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.set_ylabel("# FOVs", fontsize=S["fs_label"])
    ax.set_title("Correlated outliers: timepoints flagged in >=2 FOVs (reporting only)",
                 fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"])
    ax.set_xlim(-0.5, T_total - 0.5)
    if not corr_text:
        ax.text(0.5, 0.5, "No correlated outliers found", transform=ax.transAxes,
                ha="center", va="center", fontsize=S["fs_annotation"], color="gray")

    _apply_style(fig)
    return fig, outlier_df, corr_text


# ---------------------------------------------------------------------------
# Outlier correlation analysis (3-view decomposition)
# ---------------------------------------------------------------------------

# All QC CSVs with is_outlier column
_OUTLIER_SOURCES = {
    "laplacian": "laplacian_qc.csv",
    "entropy": "entropy_qc.csv",
    "hf_blur": "hf_ratio_qc.csv",
    "frc": "frc_qc.csv",
    "max_intensity": "max_intensity_qc.csv",
    "fov_reg": "fov_registration_qc.csv",
    "bleach": "bleach_qc.csv",
}


def _gather_all_outliers(ok_fovs, plots_dir):
    """Read is_outlier from all per-FOV QC CSVs.

    Returns a DataFrame with columns: fov, t, metric (one row per outlier event).
    """
    rows = []
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        for metric, csv_name in _OUTLIER_SOURCES.items():
            csv_path = plots_dir / fov_name / csv_name
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if "is_outlier" not in df.columns:
                continue
            outlier_ts = df.loc[df["is_outlier"] == 1, "t"].values
            for t in outlier_ts:
                rows.append({"fov": fov_name, "t": int(t), "metric": metric})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["fov", "t", "metric"])


def plot_outlier_heatmap_fov_metric(
    ok_fovs: list[str],
    plots_dir,
    T_total: int,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Heatmap: FOVs (rows) x metrics (cols), color = fraction of timepoints flagged.

    Answers: 'Which FOVs are problematic, and which metrics flag them?'
    """
    S = STYLE
    outlier_df = _gather_all_outliers(ok_fovs, plots_dir)
    metrics = list(_OUTLIER_SOURCES.keys())
    fov_names = sorted(["_".join(f.split("/")) for f in ok_fovs])

    # Build matrix: fraction of timepoints flagged per (FOV, metric)
    matrix = np.zeros((len(fov_names), len(metrics)), dtype=np.float64)
    for i, fov in enumerate(fov_names):
        for j, metric in enumerate(metrics):
            sub = outlier_df[(outlier_df["fov"] == fov) & (outlier_df["metric"] == metric)]
            matrix[i, j] = len(sub) / T_total if T_total > 0 else 0

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.2), max(6, len(fov_names) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0,
                   vmax=max(0.15, np.max(matrix) * 1.1) if matrix.max() > 0 else 0.1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=S["fs_label"], rotation=45, ha="right")
    ax.set_yticks(range(len(fov_names)))
    ax.set_yticklabels(fov_names, fontsize=S["fs_legend_multi"])

    # Annotate cells with count
    for i in range(len(fov_names)):
        for j in range(len(metrics)):
            count = int(matrix[i, j] * T_total)
            if count > 0:
                text_color = "white" if matrix[i, j] > 0.08 else "black"
                ax.text(j, i, str(count), ha="center", va="center",
                        fontsize=S["fs_legend_multi"], color=text_color)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Fraction of timepoints flagged", fontsize=S["fs_label"])

    ax.set_title("Outlier rate per FOV and metric", fontsize=S["fs_suptitle"])
    fig.tight_layout()
    return fig, outlier_df


def plot_outlier_cooccurrence_matrix(
    ok_fovs: list[str],
    plots_dir,
    T_total: int,
) -> plt.Figure:
    """Symmetric heatmap: Jaccard similarity between each pair of metrics.

    For each pair (A, B), the Jaccard index is |A∩B| / |A∪B| where each
    element is a (fov, t) tuple flagged by that metric.

    Answers: 'When entropy flags a frame, does HF also flag it?'
    """
    S = STYLE
    outlier_df = _gather_all_outliers(ok_fovs, plots_dir)
    metrics = list(_OUTLIER_SOURCES.keys())
    n = len(metrics)

    # Build sets of (fov, t) tuples per metric
    sets = {}
    for metric in metrics:
        sub = outlier_df[outlier_df["metric"] == metric]
        sets[metric] = set(zip(sub["fov"], sub["t"]))

    # Jaccard matrix
    jaccard = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            a, b = sets[metrics[i]], sets[metrics[j]]
            union = len(a | b)
            if union > 0:
                jaccard[i, j] = len(a & b) / union
            elif i == j:
                jaccard[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
    im = ax.imshow(jaccard, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(metrics, fontsize=S["fs_label"], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(metrics, fontsize=S["fs_label"])

    # Annotate
    for i in range(n):
        for j in range(n):
            val = jaccard[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=S["fs_annotation"], color=text_color)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Jaccard similarity", fontsize=S["fs_label"])

    ax.set_title("Outlier co-occurrence between metrics (Jaccard index)", fontsize=S["fs_suptitle"])
    fig.tight_layout()
    return fig


# Mapping: metric name → (csv filename, local z-score column)
_METRIC_Z_COLUMNS = {
    "laplacian": ("laplacian_qc.csv", "local_z"),
    "entropy": ("entropy_qc.csv", "local_z"),
    "hf_blur": ("hf_ratio_qc.csv", "hf_local_z"),
    "frc": ("frc_qc.csv", "local_z"),
    "max_intensity": ("max_intensity_qc.csv", "local_z"),
    "fov_reg": ("fov_registration_qc.csv", "local_z"),
    "bleach": ("bleach_qc.csv", "local_z"),
}


def _gather_metric_zscores(ok_fovs, plots_dir, T_total):
    """Build a DataFrame of local z-scores: rows = (fov, t), columns = metrics."""
    from dynacell.plotting import _recompute_bleach_local_z, _arr_from_csv

    records = []
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        row_base = {}
        for metric, (csv_name, z_col) in _METRIC_Z_COLUMNS.items():
            csv_path = plots_dir / fov_name / csv_name
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if z_col not in df.columns:
                continue
            T = int(df["t"].max()) + 1
            z_arr = _arr_from_csv(df, z_col, T)
            # Recompute bleach z if all NaN
            if metric == "bleach" and np.all(np.isnan(z_arr)):
                norm_arr = _arr_from_csv(df, "normalized", T)
                z_arr = _recompute_bleach_local_z(norm_arr)
            for t in range(min(T, T_total)):
                key = (fov_name, t)
                if key not in row_base:
                    row_base[key] = {"fov": fov_name, "t": t}
                row_base[key][metric] = z_arr[t] if t < len(z_arr) else np.nan
        records.extend(row_base.values())
    return pd.DataFrame(records)


def plot_metric_correlation_matrix(
    ok_fovs: list[str],
    plots_dir,
    T_total: int,
) -> plt.Figure:
    """Spearman correlation matrix between continuous local z-scores of all metrics.

    Each observation is a (FOV, timepoint). NaN pairs are excluded per-pair.

    Answers: 'Do metrics move together — e.g., when laplacian z drops, does HF z also drop?'
    """
    from scipy.stats import spearmanr

    S = STYLE
    zscore_df = _gather_metric_zscores(ok_fovs, plots_dir, T_total)
    metrics = [m for m in _METRIC_Z_COLUMNS if m in zscore_df.columns]
    n = len(metrics)

    # Compute pairwise Spearman correlation
    corr = np.full((n, n), np.nan, dtype=np.float64)
    pvals = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            a = zscore_df[metrics[i]].values
            b = zscore_df[metrics[j]].values
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() > 10:
                r, p = spearmanr(a[valid], b[valid])
                corr[i, j] = r
                pvals[i, j] = p

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(metrics, fontsize=S["fs_label"], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(metrics, fontsize=S["fs_label"])

    # Annotate with rho and significance stars
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if np.isnan(val):
                continue
            stars = ""
            p = pvals[i, j]
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}{stars}", ha="center", va="center",
                    fontsize=S["fs_annotation"], color=text_color)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Spearman ρ", fontsize=S["fs_label"])

    ax.set_title("Metric correlation (local z-scores, all FOVs×timepoints)",
                 fontsize=S["fs_suptitle"])
    fig.tight_layout()
    return fig


def plot_outlier_temporal_density(
    ok_fovs: list[str],
    plots_dir,
    T_total: int,
) -> plt.Figure:
    """Heatmap strips: metrics (rows) x timepoints (cols), color = # FOVs flagged.

    Answers: 'Which timepoints are globally bad, and do multiple metrics agree?'
    """
    S = STYLE
    outlier_df = _gather_all_outliers(ok_fovs, plots_dir)
    metrics = list(_OUTLIER_SOURCES.keys())
    n_fovs = len(ok_fovs)

    # Build matrix: # FOVs flagged per (metric, timepoint)
    matrix = np.zeros((len(metrics), T_total), dtype=np.float64)
    for i, metric in enumerate(metrics):
        sub = outlier_df[outlier_df["metric"] == metric]
        for _, row in sub.iterrows():
            t = int(row["t"])
            if 0 <= t < T_total:
                matrix[i, t] += 1

    fig, ax = plt.subplots(figsize=(14, max(3, len(metrics) * 0.6 + 1)))
    vmax = max(3, matrix.max()) if matrix.max() > 0 else 1
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
                   interpolation="nearest")

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=S["fs_label"])
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("# FOVs flagged", fontsize=S["fs_label"])

    ax.set_title(
        f"Temporal outlier density ({n_fovs} FOVs, {int(matrix.sum())} total outlier events)",
        fontsize=S["fs_suptitle"],
    )
    fig.tight_layout()
    return fig


def plot_frc_all_fovs(
    ok_fovs: list[str],
    plots_dir,
    n_std: float = 2.5,
) -> tuple[plt.Figure, dict]:
    """FRC mean correlation + local z-score across all FOVs. Returns (fig, {fov_name: df})."""
    S = STYLE
    fig, axes = plt.subplots(2, 1, figsize=S["fig_all_double"], sharex=True)
    all_data = {}
    for i, fov in enumerate(ok_fovs):
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "frc_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        cols = ["frc_mean_corr"]
        if "local_z" in df.columns:
            cols.append("local_z")
        all_data[fov_name] = df.set_index("t")[cols]
        c = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        axes[0].plot(df["t"], df["frc_mean_corr"], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                     color=c, label=fov_name)
        if "local_z" in df.columns:
            axes[1].plot(df["t"], df["local_z"], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                         color=c, label=fov_name)
    axes[0].set_ylabel("FRC mean correlation", fontsize=S["fs_label"])
    axes[0].set_title(f"FRC QC across all FOVs ({len(all_data)} FOVs)", fontsize=S["fs_title"])
    axes[1].set_xlabel("Timepoint", fontsize=S["fs_label"])
    axes[1].set_ylabel("FRC local z-score", fontsize=S["fs_label"])
    axes[1].axhline(-n_std, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"],
                    alpha=0.7, label=f"threshold (n_std={n_std})")
    frc_vals = np.concatenate([d["frc_mean_corr"].values for d in all_data.values()]) if all_data else None
    frc_lz_vals = np.concatenate([d["local_z"].values for d in all_data.values() if "local_z" in d.columns]) if all_data else None
    _set_ylim(axes[0], "frc", data=frc_vals)
    _set_ylim(axes[1], "frc_local_z", data=frc_lz_vals)
    for a in axes:
        a.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _apply_style(fig)
    return fig, all_data


def plot_max_intensity_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, dict]:
    """Max intensity across all FOVs (reporting only). Returns (fig, {fov_name: series})."""
    S = STYLE
    fig, ax = plt.subplots(1, 1, figsize=S["fig_all_single"])
    all_data = {}
    for i, fov in enumerate(ok_fovs):
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "max_intensity_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        all_data[fov_name] = df.set_index("t")["max_intensity"]
        ax.plot(df["t"], df["max_intensity"], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=fov_name)
    ax.set_ylabel("Max intensity (full Z)", fontsize=S["fs_label"])
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.set_title(f"Max intensity QC across all FOVs ({len(all_data)} FOVs)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _apply_style(fig)
    return fig, all_data


def plot_registration_pcc_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, dict]:
    """Pearson correlation across all FOVs. Returns (fig, {fov_name: series})."""
    S = STYLE
    fig, ax = plt.subplots(1, 1, figsize=S["fig_all_single"])
    all_data = {}
    for i, fov in enumerate(ok_fovs):
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "fov_registration_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        all_data[fov_name] = df.set_index("t")["pearson_corr"]
        ax.plot(df["t"], df["pearson_corr"], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=fov_name)
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.set_ylabel("Pearson correlation (LF vs LS)", fontsize=S["fs_label"])
    ax.set_title(f"Registration QC across all FOVs ({len(all_data)} FOVs)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    reg_vals = np.concatenate([s.values for s in all_data.values()]) if all_data else None
    _set_ylim(ax, "fov_reg_pearson", data=reg_vals)
    _apply_style(fig)
    return fig, all_data


# ---------------------------------------------------------------------------
# Replot helpers — regenerate QC plots from existing CSVs
# ---------------------------------------------------------------------------

def _arr_from_csv(df, col, T):
    """Build a full-length array from a CSV with a 't' column."""
    arr = np.full(T, np.nan, dtype=np.float64)
    for _, row in df.iterrows():
        t = int(row["t"])
        if t < T:
            arr[t] = row[col]
    return arr


def _blank_mask_from_arr(arr):
    return np.isnan(arr)


def _recompute_bleach_local_z(normalized):
    """Recompute bleach local z-score from normalized intensity (for legacy CSVs)."""
    from scipy.ndimage import median_filter
    T = len(normalized)
    local_z = np.full(T, np.nan, dtype=np.float64)
    valid_mask = ~np.isnan(normalized)
    if valid_mask.sum() > 5:
        smoothed = np.full(T, np.nan)
        smoothed[valid_mask] = median_filter(normalized[valid_mask], size=5, mode="nearest")
        residuals = np.full(T, np.nan)
        residuals[valid_mask] = normalized[valid_mask] - smoothed[valid_mask]
        valid_res = residuals[valid_mask]
        mad = float(np.median(np.abs(valid_res - np.median(valid_res))))
        scale = 1.4826 * mad if mad > 1e-12 else float(np.std(valid_res))
        if scale > 1e-12:
            local_z[valid_mask] = (residuals[valid_mask] - np.median(valid_res)) / scale
    return local_z


def replot_fov(fov_dir, fov_name, n_std, z_window):
    """Regenerate all per-FOV plots from CSVs."""
    fov_dir = Path(fov_dir)
    count = 0

    # --- bbox over time ---
    bbox_csv = fov_dir / "per_t_bboxes.csv"
    summary_csv = fov_dir / "fov_summary.csv"
    if bbox_csv.exists() and summary_csv.exists():
        bbox_df = pd.read_csv(bbox_csv)
        per_t = bbox_df[["y_min", "y_max", "x_min", "x_max"]].values
        summary = pd.read_csv(summary_csv).iloc[0]
        bbox = ast.literal_eval(summary["bbox"])
        plot_bbox_over_time(per_t, tuple(bbox), save_path=str(fov_dir / "bbox_over_time.png"))
        count += 1

    # --- z_focus ---
    z_csv = fov_dir / "z_focus.csv"
    if z_csv.exists():
        z_df = pd.read_csv(z_csv)
        plot_z_focus(
            z_df["z_focus"].tolist(),
            save_path=str(fov_dir / "z_focus.png"),
            n_std=n_std,
            z_window=z_window,
        )
        count += 1

    # --- laplacian ---
    lap_csv = fov_dir / "laplacian_qc.csv"
    if lap_csv.exists():
        df = pd.read_csv(lap_csv)
        col = "lap_var" if "lap_var" in df.columns else "lap3d_var"
        T = int(df["t"].max()) + 1
        lap_vars = _arr_from_csv(df, col, T)
        valid = lap_vars[~np.isnan(lap_vars)]
        mu, sigma = float(np.mean(valid)), float(np.std(valid))
        lower = mu - n_std * sigma
        outlier_mask = df.get("is_outlier", pd.Series([0] * len(df))).values.astype(bool)
        outliers = df["t"].values[outlier_mask]
        local_z = _arr_from_csv(df, "local_z", T) if "local_z" in df.columns else None
        fig = plot_laplacian_qc(lap_vars, mu, sigma, lower, n_std, outliers,
                                "raw GFP EX488 EM525-45", local_z=local_z)
        fig.savefig(fov_dir / "laplacian_qc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    # --- entropy ---
    ent_csv = fov_dir / "entropy_qc.csv"
    if ent_csv.exists():
        df = pd.read_csv(ent_csv)
        T = int(df["t"].max()) + 1
        entropies = _arr_from_csv(df, "entropy", T)
        local_z = _arr_from_csv(df, "local_z", T)
        blank_mask = _blank_mask_from_arr(entropies)
        outlier_mask = df.get("is_outlier", pd.Series([0] * len(df))).values.astype(bool)
        outliers = df["t"].values[outlier_mask]
        med = float(np.nanmedian(entropies))
        global_lower = None
        global_upper = None
        if "global_outlier" in df.columns:
            valid_ent = entropies[~blank_mask]
            q1, q3 = np.percentile(valid_ent, [25, 75])
            iqr = q3 - q1
            global_lower = q1 - 1.5 * iqr
            global_upper = q3 + 1.5 * iqr
        fig = plot_entropy_qc(entropies, local_z, blank_mask, outliers, med, n_std,
                              "raw GFP EX488 EM525-45",
                              global_lower=global_lower, global_upper=global_upper)
        fig.savefig(fov_dir / "entropy_qc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    # --- hf_ratio ---
    hf_csv = fov_dir / "hf_ratio_qc.csv"
    if hf_csv.exists():
        df = pd.read_csv(hf_csv)
        T = int(df["t"].max()) + 1
        hf_ratios = _arr_from_csv(df, "hf_ratio", T)
        hf_local_z = _arr_from_csv(df, "hf_local_z", T)
        blank_mask = _blank_mask_from_arr(hf_ratios)
        outlier_mask = df.get("is_outlier", pd.Series([0] * len(df))).values.astype(bool)
        outliers = df["t"].values[outlier_mask]
        med = float(np.nanmedian(hf_ratios))
        fig = plot_hf_ratio_qc(hf_ratios, hf_local_z, blank_mask, outliers, med,
                                n_std, "raw GFP EX488 EM525-45")
        fig.savefig(fov_dir / "hf_ratio_qc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    # --- frc ---
    frc_csv = fov_dir / "frc_qc.csv"
    if frc_csv.exists():
        df = pd.read_csv(frc_csv)
        T = int(df["t"].max()) + 1
        frc_vals = _arr_from_csv(df, "frc_mean_corr", T)
        local_z = _arr_from_csv(df, "local_z", T) if "local_z" in df.columns else np.zeros(T)
        blank_mask = _blank_mask_from_arr(frc_vals)
        med = float(np.nanmedian(frc_vals))
        fig = plot_frc_qc(frc_vals, local_z, blank_mask, med,
                          "raw GFP EX488 EM525-45", mean_frc_curve=None, n_std=n_std)
        fig.savefig(fov_dir / "frc_qc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    # --- max_intensity ---
    mi_csv = fov_dir / "max_intensity_qc.csv"
    if mi_csv.exists():
        df = pd.read_csv(mi_csv)
        T = int(df["t"].max()) + 1
        max_vals = _arr_from_csv(df, "max_intensity", T)
        blank_mask = _blank_mask_from_arr(max_vals)
        med = float(np.nanmedian(max_vals))
        local_z = _arr_from_csv(df, "local_z", T) if "local_z" in df.columns else None
        fig = plot_max_intensity_qc(max_vals, blank_mask, med, "raw GFP EX488 EM525-45",
                                    local_z=local_z, n_std=n_std)
        fig.savefig(fov_dir / "max_intensity_qc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    # --- fov_registration ---
    reg_csv = fov_dir / "fov_registration_qc.csv"
    if reg_csv.exists():
        df = pd.read_csv(reg_csv)
        T = int(df["t"].max()) + 1
        pearson = _arr_from_csv(df, "pearson_corr", T)
        mu = float(np.nanmean(pearson))
        outlier_mask = df.get("is_outlier", pd.Series([0] * len(df))).values.astype(bool)
        outlier_idx = df["t"].values[outlier_mask]
        n_blank = T - int((~np.isnan(pearson)).sum())
        local_z = _arr_from_csv(df, "local_z", T) if "local_z" in df.columns else None
        fig = plot_fov_registration_qc(pearson, mu, len(outlier_idx), outlier_idx,
                                        blank_count=n_blank, local_z=local_z, n_std=n_std)
        fig.savefig(fov_dir / "fov_registration_qc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    # --- bleach ---
    bleach_csv = fov_dir / "bleach_qc.csv"
    if bleach_csv.exists():
        df = pd.read_csv(bleach_csv)
        T = int(df["t"].max()) + 1
        normalized = _arr_from_csv(df, "normalized", T)
        local_z = _arr_from_csv(df, "local_z", T) if "local_z" in df.columns else None
        # Recompute local_z if CSV has all-NaN values (legacy data)
        if local_z is None or np.all(np.isnan(local_z)):
            local_z = _recompute_bleach_local_z(normalized)
        fig = plot_bleach_fov_qc(normalized, fov_name, local_z=local_z, n_std=n_std)
        fig.savefig(fov_dir / "bleach_qc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    return count


def replot_all_fovs(run_dir):
    """Regenerate all per-FOV + all-FOV summary plots from CSVs."""
    run_dir = Path(run_dir)
    plots_dir = run_dir / "per_fov_analysis"

    # Read run parameters
    run_log_path = run_dir / "run_log.yaml"
    if run_log_path.exists():
        with open(run_log_path) as f:
            run_log = yaml.safe_load(f)
        params = run_log.get("parameters", {})
        n_std = params.get("n_std", 2.5)
        z_window = params.get("z_window", None)
    else:
        print("WARNING: run_log.yaml not found, using default parameters")
        n_std = 2.5
        z_window = None

    # Discover FOVs from global_summary or directory listing
    global_csv = run_dir / "global_summary.csv"
    if global_csv.exists():
        summary_df = pd.read_csv(global_csv)
        ok_fovs = summary_df["fov"].tolist()
    else:
        ok_fovs = []
        for d in sorted(plots_dir.iterdir()):
            if d.is_dir() and (d / "fov_summary.csv").exists():
                ok_fovs.append("/".join(d.name.split("_")))

    print(f"Run dir: {run_dir}")
    print(f"Parameters: n_std={n_std}, z_window={z_window}")
    print(f"Found {len(ok_fovs)} FOVs to replot")

    # --- Per-FOV plots ---
    total_plots = 0
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        fov_dir = plots_dir / fov_name
        if not fov_dir.exists():
            print(f"  Skipping {fov_name}: directory not found")
            continue
        n = replot_fov(fov_dir, fov_name, n_std, z_window)
        total_plots += n

    print(f"\nRegenerated {total_plots} per-FOV plots across {len(ok_fovs)} FOVs")

    # --- All-FOV summary plots ---
    print("\n=== Regenerating all-FOV summary plots ===")

    def _save(fig, name):
        fig.savefig(run_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {name}.png")

    fig, _ = plot_z_focus_all_fovs(ok_fovs, plots_dir)
    _save(fig, "z_focus_all_fovs")

    fig, _ = plot_laplacian_all_fovs(ok_fovs, plots_dir)
    _save(fig, "laplacian_all_fovs")

    fig, _ = plot_hf_ratio_all_fovs(ok_fovs, plots_dir)
    _save(fig, "hf_ratio_all_fovs")

    fig, _ = plot_entropy_all_fovs(ok_fovs, plots_dir)
    _save(fig, "entropy_all_fovs")

    fig, _ = plot_frc_all_fovs(ok_fovs, plots_dir)
    _save(fig, "frc_all_fovs")

    fig, _ = plot_max_intensity_all_fovs(ok_fovs, plots_dir)
    _save(fig, "max_intensity_all_fovs")

    # T_total from first FOV summary
    T_total = 66
    if global_csv.exists():
        T_total = int(summary_df.iloc[0].get("T_total", 66))

    drop_fig, drop_df, corr_text = plot_drop_correlation_all_fovs(ok_fovs, plots_dir, T_total)
    if drop_fig is not None:
        _save(drop_fig, "drop_correlation_all_fovs")

    fig, _ = plot_registration_pcc_all_fovs(ok_fovs, plots_dir)
    _save(fig, "registration_pcc_all_fovs")

    out_fig, out_df, out_text = plot_outlier_correlation_all_fovs(ok_fovs, plots_dir, T_total)
    if out_fig is not None:
        _save(out_fig, "outlier_correlation_all_fovs")

    # --- New 3-view outlier correlation analysis ---
    fig, _ = plot_outlier_heatmap_fov_metric(ok_fovs, plots_dir, T_total)
    _save(fig, "outlier_heatmap_fov_metric")

    fig = plot_outlier_cooccurrence_matrix(ok_fovs, plots_dir, T_total)
    _save(fig, "outlier_cooccurrence_matrix")

    fig = plot_outlier_temporal_density(ok_fovs, plots_dir, T_total)
    _save(fig, "outlier_temporal_density")

    fig = plot_metric_correlation_matrix(ok_fovs, plots_dir, T_total)
    _save(fig, "metric_correlation_matrix")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate dynacell QC plots from existing CSVs"
    )
    parser.add_argument(
        "run_dir", type=str,
        help="Path to the run directory (e.g. run_20260323_174801)",
    )
    args = parser.parse_args()
    replot_all_fovs(args.run_dir)
