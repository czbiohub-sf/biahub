"""Visualization utilities for dynacell preprocessing."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dynacell_geometry import make_circular_mask

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

    # Semantic QC colors
    "c_qc": {
        "blank_frame": "#888888",
        "z_focus_outlier": "#ff7f0e",
        "hf_blur": "#d62728",
        "entropy": "#9467bd",
        "frc": "#1f77b4",
        "fov_reg": "#2ca02c",
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

    # Y-axis limits per metric (based on prior dataset runs)
    # None means auto-scale; tuple means (ymin, ymax)
    "ylim": {
        "z_focus": (15, 65),
        "laplacian": (0, 650),
        "entropy": (0, 4.5),
        "entropy_local_z": (-5, 5),
        "hf_ratio": (0, 0.015),
        "hf_local_z": (-6, 6),
        "frc": (0.3, 0.8),             # room for outliers
        "fov_reg_pearson": (-0.02, 0.06),
        "fov_reg_local_z": (-5, 5),
        "bleach_norm": (0, 1.15),
        "shift_magnitude": None,        # auto
        "pearson_beads": None,           # auto
    },
}

# Ordered color cycle for multi-FOV plots
COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def _set_ylim(ax, key):
    """Set y-axis limits from STYLE['ylim'] if defined for the given key."""
    lim = STYLE["ylim"].get(key)
    if lim is not None:
        ax.set_ylim(lim)


def _apply_style(fig, axes=None):
    """Apply uniform tick sizes and grid to figure axes."""
    if axes is None:
        axes = fig.get_axes()
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.tick_params(labelsize=STYLE["fs_tick"])
        ax.grid(axis="x", alpha=0.15, color=STYLE["c_grid"])
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
    axes[2, 0].set_xlabel("Time point", fontsize=S["fs_label"])
    axes[2, 1].set_xlabel("Time point", fontsize=S["fs_label"])

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
    ax.plot(t_axis, z_arr, S["c_primary"], alpha=S["alpha_data"], linewidth=S["lw_data"], label="per timepoint")
    ax.axhline(mu, color=S["c_mean"], linestyle=":", linewidth=S["lw_ref"], label=f"mean = {mu:.1f}")
    ax.fill_between(t_axis, mu - sigma, mu + sigma, color=S["c_fill"], alpha=S["alpha_fill"], label=f"1 std = {sigma:.1f}")
    ax.fill_between(t_axis, lower, upper, color=S["c_threshold"], alpha=0.07, label=f"{n_std} std = [{lower:.1f}, {upper:.1f}]")
    ax.axhline(np.median(z_arr), color=S["c_median"], linestyle="--", linewidth=S["lw_ref"], label=f"median = {np.median(z_arr):.0f}")
    # Mark outliers
    if len(t_above) > 0:
        ax.scatter(t_above, z_arr[t_above], color=S["c_outlier"], s=S["scatter_size"], zorder=5, label=f"above {n_std}std (n={len(t_above)})")
    if len(t_below) > 0:
        ax.scatter(t_below, z_arr[t_below], color=S["c_tertiary"], s=S["scatter_size"], zorder=5, label=f"below {n_std}std (n={len(t_below)})")
    ax.set_xlabel("Time point", fontsize=S["fs_label"])
    ax.set_ylabel("Z focus index", fontsize=S["fs_label"])
    ax.set_title(
        f"Z focus over time | mean={mu:.1f}, std={sigma:.1f}, "
        f"range=[{z_arr.min():.0f}, {z_arr.max():.0f}]",
        fontsize=S["fs_title"],
    )
    ax.legend(fontsize=S["fs_legend"], loc="best")
    _set_ylim(ax, "z_focus")
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
    ax.plot(pearson_corrs, S["c_primary"], alpha=S["alpha_data"], linewidth=S["lw_data"])
    ax.axhline(mu_corr, color=S["c_mean"], linestyle=":", linewidth=S["lw_ref"])
    ax.fill_between(range(T), mu_corr - sigma_corr, mu_corr + sigma_corr,
                    color=S["c_fill"], alpha=S["alpha_fill"])
    ax.axhline(lower_corr, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"], alpha=0.5)
    if len(corr_outliers) > 0:
        ax.scatter(corr_outliers, pearson_corrs[corr_outliers],
                   color=S["c_outlier"], s=S["scatter_size"], zorder=5)
    ax.set_ylabel("Pearson correlation", fontsize=S["fs_label"])
    ax.set_title(f"Beads registration QC | mean r={mu_corr:.4f}", fontsize=S["fs_title"])
    _set_ylim(ax, "pearson_beads")

    ax = axes[1]
    ax.plot(pcc_shifts_y, S["c_primary"], alpha=S["alpha_data"], linewidth=S["lw_data"], label="shift Y")
    ax.plot(pcc_shifts_x, S["c_secondary"], alpha=S["alpha_data"], linewidth=S["lw_data"], label="shift X")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("PCC shift (px)", fontsize=S["fs_label"])
    ax.legend(fontsize=S["fs_legend"])

    ax = axes[2]
    ax.plot(shift_mag, S["c_tertiary"], alpha=S["alpha_data"], linewidth=S["lw_data"])
    ax.axhline(mu_shift, color=S["c_mean"], linestyle=":", linewidth=S["lw_ref"])
    ax.axhline(upper_shift, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"], alpha=0.5)
    if len(shift_outliers) > 0:
        ax.scatter(shift_outliers, shift_mag[shift_outliers],
                   color=S["c_outlier"], s=S["scatter_size"], zorder=5)
    ax.set_ylabel("Shift magnitude (px)", fontsize=S["fs_label"])
    ax.set_xlabel("Time point", fontsize=S["fs_label"])
    _set_ylim(ax, "shift_magnitude")

    fig.suptitle("Beads registration QC", fontsize=S["fs_suptitle"])
    _apply_style(fig)
    return fig


def plot_laplacian_qc(
    lap3d_vars: np.ndarray,
    max_ints: np.ndarray,
    mu: float,
    sigma: float,
    lower: float,
    n_std: float,
    outliers: np.ndarray,
    channel_name: str,
) -> plt.Figure:
    """Plot Laplacian variance (sharpness) and max intensity per timepoint (blank frames excluded)."""
    S = STYLE
    valid_idx = np.where(~np.isnan(lap3d_vars))[0]
    n_blank = len(lap3d_vars) - len(valid_idx)

    fig, axes = plt.subplots(2, 1, figsize=S["fig_double"], sharex=True)

    ax = axes[0]
    ax.plot(valid_idx, lap3d_vars[valid_idx], S["c_primary"], alpha=S["alpha_data"], linewidth=S["lw_data"])
    ax.axhline(mu, color=S["c_mean"], linestyle=":", linewidth=S["lw_ref"], label=f"mean={mu:.1f}")
    ax.fill_between(valid_idx, mu - sigma, mu + sigma, color=S["c_fill"], alpha=S["alpha_fill"])
    ax.axhline(lower, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"], alpha=0.5,
               label=f"mean-{n_std}*std={lower:.1f}")
    if len(outliers) > 0:
        ax.scatter(outliers, lap3d_vars[outliers], color=S["c_outlier"], s=S["scatter_size"], zorder=5,
                   label=f"outliers ({len(outliers)})")
    ax.set_ylabel("3D Laplacian variance", fontsize=S["fs_label"])
    title = f"Laplacian blur QC | {channel_name}"
    if n_blank > 0:
        title += f" (excl. {n_blank} blank)"
    ax.set_title(title, fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"], loc="best")
    _set_ylim(ax, "laplacian")

    ax = axes[1]
    ax.plot(valid_idx, max_ints[valid_idx], S["c_secondary"], alpha=S["alpha_data"], linewidth=S["lw_data"])
    if len(outliers) > 0:
        ax.scatter(outliers, max_ints[outliers], color=S["c_outlier"], s=S["scatter_size"], zorder=5)
    ax.set_ylabel("Max intensity (ROI)", fontsize=S["fs_label"])
    ax.set_xlabel("Time point", fontsize=S["fs_label"])

    _apply_style(fig)
    return fig


def plot_entropy_qc(
    entropies: np.ndarray,
    local_z: np.ndarray,
    blank_mask: np.ndarray,
    outliers: np.ndarray,
    med: float,
    local_z_threshold: float,
    channel_name: str,
    global_lower: float | None = None,
    global_upper: float | None = None,
) -> plt.Figure:
    """Plot Shannon entropy and local z-score per timepoint (blank frames excluded)."""
    S = STYLE
    valid_idx = np.where(~blank_mask)[0]
    plot_ent = entropies[valid_idx]
    plot_local_z = local_z[valid_idx]

    fig, axes = plt.subplots(2, 1, figsize=S["fig_double"], sharex=True)

    ax = axes[0]
    ax.plot(valid_idx, plot_ent, "o-", ms=S["marker_size"], alpha=S["alpha_data"])
    ax.axhline(med, color=S["c_median"], ls="--", lw=S["lw_ref"], label=f"median={med:.4f}")
    if global_lower is not None:
        ax.axhline(global_lower, color=S["c_mean"], ls=":", lw=S["lw_ref"],
                    label=f"IQR lower={global_lower:.4f}")
    if global_upper is not None:
        ax.axhline(global_upper, color=S["c_mean"], ls=":", lw=S["lw_ref"],
                    label=f"IQR upper={global_upper:.4f}")
        ax.fill_between(
            ax.get_xlim(), global_lower, global_upper,
            color=S["c_median"], alpha=0.05, zorder=0,
        )
    for idx in outliers:
        ax.axvline(idx, color=S["c_outlier"], alpha=0.4, lw=1.5)
    ax.set_ylabel("Shannon entropy", fontsize=S["fs_label"])
    ax.set_title(f"Entropy blur QC | {channel_name} (global+local threshold)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"])
    _set_ylim(ax, "entropy")

    ax = axes[1]
    ax.plot(valid_idx, plot_local_z, "o-", ms=S["marker_size"], alpha=S["alpha_data"], color=S["c_tertiary"],
            label="Local z-score")
    ax.axhline(local_z_threshold, color=S["c_threshold"], ls="--", lw=S["lw_threshold"],
               label=f"threshold=+{local_z_threshold}")
    ax.axhline(-local_z_threshold, color=S["c_threshold"], ls="--", lw=S["lw_threshold"],
               label=f"threshold=-{local_z_threshold}")
    for idx in outliers:
        ax.axvline(idx, color=S["c_outlier"], alpha=0.4, lw=1.5)
    ax.set_ylabel("Local z-score", fontsize=S["fs_label"])
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.legend(fontsize=S["fs_legend"])
    _set_ylim(ax, "entropy_local_z")

    _apply_style(fig)
    return fig


def plot_hf_ratio_qc(
    hf_ratios: np.ndarray,
    hf_local_z: np.ndarray,
    blank_mask: np.ndarray,
    outliers: np.ndarray,
    med: float,
    hf_z_threshold: float,
    ent_z_threshold: float,
    channel_name: str,
    z_range: int = 5,
    ent_local_z: np.ndarray | None = None,
) -> plt.Figure:
    """Plot HF energy ratio, local z-scores (HF + entropy), and combined decision."""
    S = STYLE
    valid_idx = np.where(~blank_mask)[0]
    plot_hf = hf_ratios[valid_idx]
    plot_hf_lz = hf_local_z[valid_idx]
    T = len(hf_ratios)
    t_ax = np.arange(T)
    has_entropy = ent_local_z is not None and len(ent_local_z) == T

    n_panels = 3 if has_entropy else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels), sharex=True)

    # Panel 1: HF ratio (multi-Z median)
    ax = axes[0]
    ax.plot(valid_idx, plot_hf, ".-", ms=S["marker_size"], alpha=S["alpha_data"])
    ax.axhline(med, color=S["c_median"], ls="--", lw=S["lw_ref"], label=f"median={med:.6f}")
    for idx in outliers:
        ax.axvline(idx, color=S["c_outlier"], alpha=0.4, lw=1.5)
    ax.set_ylabel("HF energy ratio", fontsize=S["fs_label"])
    ax.set_title(f"HF ratio blur QC (multi-Z ±{z_range}) | {channel_name}", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"])
    _set_ylim(ax, "hf_ratio")

    # Panel 2: HF local z-score + entropy local z-score
    ax = axes[1]
    ax.plot(valid_idx, plot_hf_lz, ".-", ms=S["marker_size"], alpha=S["alpha_data"], color=S["c_primary"],
            label="HF local z-score")
    ax.axhline(-hf_z_threshold, color=S["c_primary"], ls="--", lw=S["lw_threshold"], alpha=0.6,
               label=f"HF thresh (-{hf_z_threshold})")
    for idx in outliers:
        ax.axvline(idx, color=S["c_outlier"], alpha=0.4, lw=1.5)
    ax.set_ylabel("HF local z-score", fontsize=S["fs_label"], color=S["c_primary"])
    _set_ylim(ax, "hf_local_z")

    if has_entropy:
        ax2 = ax.twinx()
        plot_ent_lz = ent_local_z[valid_idx]
        ax2.plot(valid_idx, plot_ent_lz, ".-", ms=S["marker_size"], alpha=S["alpha_data"], color=S["c_mean"],
                 label="Entropy local z-score")
        ax2.axhline(+ent_z_threshold, color=S["c_mean"], ls="--", lw=S["lw_threshold"], alpha=0.6,
                     label=f"Ent thresh (+{ent_z_threshold})")
        ax2.set_ylabel("Entropy local z-score", fontsize=S["fs_label"], color=S["c_mean"])
        ax2.tick_params(labelsize=S["fs_tick"])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=S["fs_legend"], loc="lower left")
    else:
        ax.legend(fontsize=S["fs_legend"])

    if has_entropy:
        # Panel 3: Combined decision
        ax = axes[2]
        combined = np.zeros(T, dtype=float)
        combined[outliers] = 1.0
        ax.fill_between(t_ax, combined, alpha=0.3, color=S["c_outlier"], label="FLAGGED")
        for idx in outliers:
            ax.axvline(idx, color=S["c_outlier"], alpha=0.5)
        ax.set_ylabel("Flagged", fontsize=S["fs_label"])
        ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
        method_str = f"hf_lz<-{hf_z_threshold} AND ent_lz>+{ent_z_threshold}"
        ax.set_title(f"Combined blur: {method_str} | outliers: {outliers.tolist()}", fontsize=S["fs_title"])
        ax.legend(fontsize=S["fs_legend"])
    else:
        axes[-1].set_xlabel("Timepoint", fontsize=S["fs_label"])

    _apply_style(fig)
    return fig


def plot_frc_qc(
    frc_values: np.ndarray,
    blank_mask: np.ndarray,
    med: float,
    channel_name: str,
    z_range: int = 5,
) -> plt.Figure:
    """Plot FRC mean correlation per timepoint (reporting only)."""
    S = STYLE
    valid_idx = np.where(~blank_mask)[0]
    plot_frc = frc_values[valid_idx]

    fig, ax = plt.subplots(1, 1, figsize=S["fig_single"])

    ax.plot(valid_idx, plot_frc, ".-", ms=S["marker_size"], alpha=S["alpha_data"])
    ax.axhline(med, color=S["c_median"], ls="--", lw=S["lw_ref"], label=f"median={med:.4f}")
    ax.set_ylabel("FRC mean correlation", fontsize=S["fs_label"])
    ax.set_xlabel("Timepoint", fontsize=S["fs_label"])
    ax.set_title(f"FRC blur QC (multi-Z ±{z_range}) | {channel_name}", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend"])
    _set_ylim(ax, "frc")

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
    ax.set_xlabel("Time point", fontsize=S["fs_label"])
    ax.set_ylabel("Z focus index", fontsize=S["fs_label"])
    ax.set_title(f"Z focus across all FOVs ({len(ok_fovs)} FOVs)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _set_ylim(ax, "z_focus")
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
        all_data[fov_name] = df.set_index("t")["lap3d_var"]
        ax.plot(df["t"], df["lap3d_var"], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=fov_name)
    ax.set_xlabel("Time point", fontsize=S["fs_label"])
    ax.set_ylabel("Laplacian variance", fontsize=S["fs_label"])
    ax.set_title(f"Laplacian QC across all FOVs ({len(all_data)} FOVs)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _set_ylim(ax, "laplacian")
    _apply_style(fig)
    return fig, all_data


def plot_hf_ratio_all_fovs(
    ok_fovs: list[str],
    plots_dir,
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
    axes[0].set_ylabel("HF energy ratio (multi-Z)", fontsize=S["fs_label"])
    axes[0].set_title(f"HF ratio QC across all FOVs ({len(all_data)} FOVs)", fontsize=S["fs_title"])
    axes[1].set_xlabel("Time point", fontsize=S["fs_label"])
    axes[1].set_ylabel("HF local z-score", fontsize=S["fs_label"])
    axes[1].axhline(-3.0, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"],
                    alpha=0.7, label="threshold")
    axes[1].axhline(3.0, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"], alpha=0.7)
    _set_ylim(axes[0], "hf_ratio")
    _set_ylim(axes[1], "hf_local_z")
    for a in axes:
        a.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _apply_style(fig)
    return fig, all_data


def plot_entropy_all_fovs(
    ok_fovs: list[str],
    plots_dir,
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
    axes[1].set_xlabel("Time point", fontsize=S["fs_label"])
    axes[1].set_ylabel("Entropy local z-score", fontsize=S["fs_label"])
    axes[1].axhline(2.0, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"],
                    alpha=0.7, label="threshold")
    axes[1].axhline(-2.0, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"], alpha=0.7)
    _set_ylim(axes[0], "entropy")
    _set_ylim(axes[1], "entropy_local_z")
    for a in axes:
        a.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _apply_style(fig)
    return fig, all_data


def plot_blur_detection_all_fovs(
    ok_fovs: list[str],
    plots_dir,
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
    axes[0].axhline(-3.0, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"],
                    alpha=0.7, label="HF thresh")
    axes[0].legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _set_ylim(axes[0], "hf_local_z")
    axes[1].set_ylabel("Entropy local z-score", fontsize=S["fs_label"])
    axes[1].axhline(2.0, color=S["c_threshold"], linestyle="--", linewidth=S["lw_threshold"],
                    alpha=0.7, label="Ent thresh")
    axes[1].legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _set_ylim(axes[1], "entropy_local_z")
    axes[2].set_xlabel("Time point", fontsize=S["fs_label"])
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
    ax.set_xlabel("Time point", fontsize=S["fs_label"])
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
    ax.set_xlabel("Time point", fontsize=S["fs_label"])
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


def plot_frc_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, dict]:
    """FRC mean correlation across all FOVs (reporting only). Returns (fig, {fov_name: df})."""
    S = STYLE
    fig, ax = plt.subplots(1, 1, figsize=S["fig_all_single"])
    all_data = {}
    for i, fov in enumerate(ok_fovs):
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "frc_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        all_data[fov_name] = df.set_index("t")[["frc_mean_corr"]]
        ax.plot(df["t"], df["frc_mean_corr"], alpha=S["alpha_multi"], linewidth=S["lw_data"],
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=fov_name)
    ax.set_ylabel("FRC mean correlation", fontsize=S["fs_label"])
    ax.set_xlabel("Time point", fontsize=S["fs_label"])
    ax.set_title(f"FRC QC across all FOVs ({len(all_data)} FOVs)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _set_ylim(ax, "frc")
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
    ax.set_xlabel("Time point", fontsize=S["fs_label"])
    ax.set_ylabel("Pearson correlation (LF vs LS)", fontsize=S["fs_label"])
    ax.set_title(f"Registration QC across all FOVs ({len(all_data)} FOVs)", fontsize=S["fs_title"])
    ax.legend(fontsize=S["fs_legend_multi"], loc="best", ncol=2)
    _set_ylim(ax, "fov_reg_pearson")
    _apply_style(fig)
    return fig, all_data
