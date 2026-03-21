"""Visualization utilities for dynacell preprocessing."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dynacell_geometry import make_circular_mask


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

    def _plot_panel(ax, t_axis, values, combined_val, color, ylabel):
        mu = np.mean(values)
        sigma = np.std(values)
        ax.plot(t_axis, values, color=color, alpha=0.7, linewidth=1.0, label="per timepoint")
        ax.axhline(combined_val, color="r", linestyle="--", linewidth=1.5, label=f"combined = {combined_val}")
        ax.axhline(mu, color="orange", linestyle=":", linewidth=1.5, label=f"mean = {mu:.1f}")
        ax.fill_between(t_axis, mu - sigma, mu + sigma, color="orange", alpha=0.15, label=f"std = {sigma:.1f}")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")

    _plot_panel(axes[0, 0], valid_t, y_min_t, bbox[0], "tab:blue", "y_min")
    _plot_panel(axes[0, 1], valid_t, y_max_t, bbox[1], "tab:blue", "y_max")
    _plot_panel(axes[1, 0], valid_t, x_min_t, bbox[2], "tab:green", "x_min")
    _plot_panel(axes[1, 1], valid_t, x_max_t, bbox[3], "tab:green", "x_max")
    _plot_panel(axes[2, 0], valid_t, height_t, bbox[1] - bbox[0] + 1, "tab:purple", "Height (Y)")
    _plot_panel(axes[2, 1], valid_t, width_t, bbox[3] - bbox[2] + 1, "tab:purple", "Width (X)")
    axes[2, 0].set_xlabel("Time point")
    axes[2, 1].set_xlabel("Time point")

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
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t_axis, z_arr, "tab:blue", alpha=0.7, linewidth=1.0, label="per timepoint")
    ax.axhline(mu, color="orange", linestyle=":", linewidth=1.5, label=f"mean = {mu:.1f}")
    ax.fill_between(t_axis, mu - sigma, mu + sigma, color="orange", alpha=0.15, label=f"1 std = {sigma:.1f}")
    ax.fill_between(t_axis, lower, upper, color="red", alpha=0.07, label=f"{n_std} std = [{lower:.1f}, {upper:.1f}]")
    ax.axhline(np.median(z_arr), color="green", linestyle="--", linewidth=1.2, label=f"median = {np.median(z_arr):.0f}")
    # Mark outliers
    if len(t_above) > 0:
        ax.scatter(t_above, z_arr[t_above], color="red", s=30, zorder=5, label=f"above {n_std}std (n={len(t_above)})")
    if len(t_below) > 0:
        ax.scatter(t_below, z_arr[t_below], color="purple", s=30, zorder=5, label=f"below {n_std}std (n={len(t_below)})")
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
    T = len(pearson_corrs)
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
    fig.tight_layout()
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
    valid_idx = np.where(~np.isnan(lap3d_vars))[0]
    n_blank = len(lap3d_vars) - len(valid_idx)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax = axes[0]
    ax.plot(valid_idx, lap3d_vars[valid_idx], "tab:blue", alpha=0.7, linewidth=0.8)
    ax.axhline(mu, color="orange", linestyle=":", linewidth=1, label=f"mean={mu:.1f}")
    ax.fill_between(valid_idx, mu - sigma, mu + sigma, color="orange", alpha=0.12)
    ax.axhline(lower, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"mean-{n_std}*std={lower:.1f}")
    if len(outliers) > 0:
        ax.scatter(outliers, lap3d_vars[outliers], color="red", s=20, zorder=5,
                   label=f"outliers ({len(outliers)})")
    ax.set_ylabel("3D Laplacian variance")
    title = f"Laplacian blur QC | {channel_name}"
    if n_blank > 0:
        title += f" (excl. {n_blank} blank)"
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    ax.plot(valid_idx, max_ints[valid_idx], "tab:green", alpha=0.7, linewidth=0.8)
    if len(outliers) > 0:
        ax.scatter(outliers, max_ints[outliers], color="red", s=20, zorder=5)
    ax.set_ylabel("Max intensity (ROI)")
    ax.set_xlabel("Time point")

    fig.tight_layout()
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
    valid_idx = np.where(~blank_mask)[0]
    plot_ent = entropies[valid_idx]
    plot_local_z = local_z[valid_idx]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax = axes[0]
    ax.plot(valid_idx, plot_ent, "o-", ms=3, alpha=0.7)
    ax.axhline(med, color="green", ls="--", lw=1, label=f"median={med:.4f}")
    if global_lower is not None:
        ax.axhline(global_lower, color="orange", ls=":", lw=1,
                    label=f"IQR lower={global_lower:.4f}")
    if global_upper is not None:
        ax.axhline(global_upper, color="orange", ls=":", lw=1,
                    label=f"IQR upper={global_upper:.4f}")
        ax.fill_between(
            ax.get_xlim(), global_lower, global_upper,
            color="green", alpha=0.05, zorder=0,
        )
    for idx in outliers:
        ax.axvline(idx, color="red", alpha=0.4, lw=1.5)
    ax.set_ylabel("Shannon entropy")
    ax.set_title(f"Entropy blur QC | {channel_name} (global+local threshold)")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(valid_idx, plot_local_z, "o-", ms=3, alpha=0.7, color="tab:purple",
            label="Local z-score")
    ax.axhline(local_z_threshold, color="red", ls="--", lw=1,
               label=f"threshold=+{local_z_threshold}")
    ax.axhline(-local_z_threshold, color="red", ls="--", lw=1,
               label=f"threshold=-{local_z_threshold}")
    for idx in outliers:
        ax.axvline(idx, color="red", alpha=0.4, lw=1.5)
    ax.set_ylabel("Local z-score")
    ax.set_xlabel("Timepoint")
    ax.legend(fontsize=7)

    fig.tight_layout()
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
    valid_idx = np.where(~blank_mask)[0]
    plot_hf = hf_ratios[valid_idx]
    plot_hf_lz = hf_local_z[valid_idx]
    T = len(hf_ratios)
    t_ax = np.arange(T)
    has_entropy = ent_local_z is not None and len(ent_local_z) == T

    n_panels = 3 if has_entropy else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels), sharex=True)

    # Panel 1: HF ratio (multi-Z median)
    ax = axes[0]
    ax.plot(valid_idx, plot_hf, ".-", ms=3, alpha=0.7)
    ax.axhline(med, color="green", ls="--", lw=1, label=f"median={med:.6f}")
    for idx in outliers:
        ax.axvline(idx, color="red", alpha=0.4, lw=1.5)
    ax.set_ylabel("HF energy ratio")
    ax.set_title(f"HF ratio blur QC (multi-Z ±{z_range}) | {channel_name}")
    ax.legend(fontsize=7)

    # Panel 2: HF local z-score + entropy local z-score
    ax = axes[1]
    ax.plot(valid_idx, plot_hf_lz, ".-", ms=3, alpha=0.7, color="tab:blue",
            label="HF local z-score")
    ax.axhline(-hf_z_threshold, color="tab:blue", ls="--", lw=1, alpha=0.6,
               label=f"HF thresh (-{hf_z_threshold})")
    for idx in outliers:
        ax.axvline(idx, color="red", alpha=0.4, lw=1.5)
    ax.set_ylabel("HF local z-score", color="tab:blue")

    if has_entropy:
        ax2 = ax.twinx()
        plot_ent_lz = ent_local_z[valid_idx]
        ax2.plot(valid_idx, plot_ent_lz, ".-", ms=3, alpha=0.7, color="tab:orange",
                 label="Entropy local z-score")
        ax2.axhline(+ent_z_threshold, color="tab:orange", ls="--", lw=1, alpha=0.6,
                     label=f"Ent thresh (+{ent_z_threshold})")
        ax2.set_ylabel("Entropy local z-score", color="tab:orange")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower left")
    else:
        ax.legend(fontsize=7)

    if has_entropy:
        # Panel 3: Combined decision
        ax = axes[2]
        combined = np.zeros(T, dtype=float)
        combined[outliers] = 1.0
        ax.fill_between(t_ax, combined, alpha=0.3, color="red", label="FLAGGED")
        for idx in outliers:
            ax.axvline(idx, color="red", alpha=0.5)
        ax.set_ylabel("Flagged")
        ax.set_xlabel("Timepoint")
        method_str = f"hf_lz<-{hf_z_threshold} AND ent_lz>+{ent_z_threshold}"
        ax.set_title(f"Combined blur: {method_str} | outliers: {outliers.tolist()}")
        ax.legend(fontsize=7)
    else:
        axes[-1].set_xlabel("Timepoint")

    fig.tight_layout()
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
    Y, X = worst_median.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: dust fraction vs Z
    ax = axes[0]
    ax.plot(dust_fractions, "tab:blue", alpha=0.7, linewidth=1.0)
    ax.axhline(dust_fractions.mean(), color="orange", linestyle=":", linewidth=1,
               label=f"mean={dust_fractions.mean():.4f}")
    ax.axvline(worst_z, color="red", linestyle="--", linewidth=0.8,
               label=f"worst z={worst_z}")
    ax.set_xlabel("Z plane")
    ax.set_ylabel("Dust fraction")
    ax.set_title("Dust fraction per Z plane")
    ax.legend(fontsize=7)

    # Panel 2: worst-Z median image (within mask)
    ax = axes[1]
    display_img = worst_median.copy()
    display_img[~circ_mask] = np.nan
    vmin, vmax = np.nanpercentile(display_img[circ_mask], [1, 99])
    ax.imshow(display_img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(f"Median Phase (z={worst_z})")
    ax.tick_params(labelsize=7)

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
                 f"{max_dust_fraction*100:.2f}%)")
    ax.tick_params(labelsize=7)

    fig.suptitle(f"Dust QC | {n_fovs} FOVs x {n_sample_t} t | mask_radius={lf_mask_radius}",
                 fontsize=11)
    fig.tight_layout()
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

    def _exp_decay(t, a, tau, c):
        return a * np.exp(-t / tau) + c

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: per-FOV normalized curves + mean
    ax = axes[0]
    for fov_key, (t_arr, norm_vals) in norm_curves.items():
        ax.plot(t_arr, norm_vals, alpha=0.3, linewidth=0.6, color="tab:blue")
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
        ax.plot(t_smooth, _exp_decay(t_smooth, **fit_params), "r--", linewidth=1.2,
                label=f"fit: t1/2={half_life:.1f}")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(f"GFP bleaching ({len(norm_curves)} FOVs)")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)

    # Panel 2: per-FOV % remaining (bar chart)
    ax = axes[1]
    fov_names = [r["fov"] for r in summary_rows]
    pct_vals = [r["pct_remaining"] for r in summary_rows]
    colors = ["tab:green" if p > 70 else "tab:orange" if p > 40 else "tab:red"
              for p in pct_vals]
    ax.barh(range(len(fov_names)), pct_vals, color=colors, alpha=0.7)
    ax.set_yticks(range(len(fov_names)))
    ax.set_yticklabels(fov_names, fontsize=6)
    ax.set_xlabel("% signal remaining")
    ax.set_title(f"Signal at last t | mean={np.mean(pct_vals):.1f}%")
    ax.axvline(100, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlim(0, max(110, max(pct_vals) * 1.05))

    fig.suptitle(f"Bleach QC | {channel_name}", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined all-FOV plots
# ---------------------------------------------------------------------------

def plot_z_focus_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, dict]:
    """Z focus across all FOVs. Returns (fig, {fov_name: z_values})."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    all_data = {}
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        z_df = pd.read_csv(plots_dir / fov_name / "z_focus.csv")
        z_vals = z_df["z_focus"].values
        all_data[fov_name] = z_vals
        ax.plot(z_vals, alpha=0.5, linewidth=0.8, label=fov_name)
    ax.set_xlabel("Time point")
    ax.set_ylabel("Z focus index")
    ax.set_title(f"Z focus across all FOVs ({len(ok_fovs)} FOVs)")
    ax.legend(fontsize=6, loc="best", ncol=2)
    fig.tight_layout()
    return fig, all_data


def plot_laplacian_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, dict]:
    """Laplacian variance across all FOVs. Returns (fig, {fov_name: series})."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    all_data = {}
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "laplacian_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        all_data[fov_name] = df.set_index("t")["lap3d_var"]
        ax.plot(df["t"], df["lap3d_var"], alpha=0.5, linewidth=0.8, label=fov_name)
    ax.set_xlabel("Time point")
    ax.set_ylabel("Laplacian variance")
    ax.set_title(f"Laplacian QC across all FOVs ({len(all_data)} FOVs)")
    ax.legend(fontsize=6, loc="best", ncol=2)
    fig.tight_layout()
    return fig, all_data


def plot_hf_ratio_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, dict]:
    """HF ratio + local z-score across all FOVs. Returns (fig, {fov_name: df})."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    all_data = {}
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "hf_ratio_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        all_data[fov_name] = df.set_index("t")[["hf_ratio", "hf_local_z"]]
        axes[0].plot(df["t"], df["hf_ratio"], alpha=0.5, linewidth=0.8, label=fov_name)
        axes[1].plot(df["t"], df["hf_local_z"], alpha=0.5, linewidth=0.8, label=fov_name)
    axes[0].set_ylabel("HF energy ratio (multi-Z)")
    axes[0].set_title(f"HF ratio QC across all FOVs ({len(all_data)} FOVs)")
    axes[1].set_xlabel("Time point")
    axes[1].set_ylabel("HF local z-score")
    axes[1].axhline(-3.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="threshold")
    axes[1].axhline(3.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    for a in axes:
        a.legend(fontsize=6, loc="best", ncol=2)
    fig.tight_layout()
    return fig, all_data


def plot_entropy_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, dict]:
    """Entropy + local z-score across all FOVs. Returns (fig, {fov_name: df})."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    all_data = {}
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "entropy_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        all_data[fov_name] = df.set_index("t")[["entropy", "local_z"]]
        axes[0].plot(df["t"], df["entropy"], alpha=0.5, linewidth=0.8, label=fov_name)
        axes[1].plot(df["t"], df["local_z"], alpha=0.5, linewidth=0.8, label=fov_name)
    axes[0].set_ylabel("Shannon entropy")
    axes[0].set_title(f"Entropy QC across all FOVs ({len(all_data)} FOVs)")
    axes[1].set_xlabel("Time point")
    axes[1].set_ylabel("Entropy local z-score")
    axes[1].axhline(2.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="threshold")
    axes[1].axhline(-2.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    for a in axes:
        a.legend(fontsize=6, loc="best", ncol=2)
    fig.tight_layout()
    return fig, all_data


def plot_blur_detection_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, list[dict]]:
    """Combined HF+entropy blur detection across all FOVs. Returns (fig, blur_summary)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    blur_summary = []
    for fov in ok_fovs:
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
        axes[0].plot(df["t"], hf_lz, alpha=0.5, linewidth=0.8, label=fov_name)
        if ent_lz is not None:
            axes[1].plot(df["t"], ent_lz, alpha=0.5, linewidth=0.8, label=fov_name)
        if is_outlier is not None and is_outlier.any():
            outlier_t = df["t"].values[is_outlier]
            blur_summary.append({"fov": fov_name, "blur_frames": outlier_t.tolist(),
                                 "n_blur": int(is_outlier.sum())})
            axes[2].scatter(outlier_t, [fov_name] * len(outlier_t),
                            color="red", s=20, alpha=0.8)
    axes[0].set_ylabel("HF local z-score")
    axes[0].set_title("Combined blur detection (HF+entropy) across all FOVs")
    axes[0].axhline(-3.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="HF thresh")
    axes[0].legend(fontsize=6, loc="best", ncol=2)
    axes[1].set_ylabel("Entropy local z-score")
    axes[1].axhline(2.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Ent thresh")
    axes[1].legend(fontsize=6, loc="best", ncol=2)
    axes[2].set_xlabel("Time point")
    axes[2].set_ylabel("FOV")
    n_blur_total = sum(b["n_blur"] for b in blur_summary)
    axes[2].set_title(f"Flagged blur frames: {n_blur_total} total across {len(blur_summary)} FOVs")
    axes[2].tick_params(axis="y", labelsize=6)
    fig.tight_layout()
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

    drop_all_df = pd.DataFrame(all_drop_rows)
    reasons_unique = sorted(drop_all_df["reason"].unique())
    fov_names_sorted = sorted(drop_all_df["fov"].unique())
    n_reasons = len(reasons_unique)
    reason_colors = {"blank_frame": "gray", "z_focus_outlier": "orange", "hf_blur": "red"}

    fig, axes = plt.subplots(n_reasons + 2, 1,
                             figsize=(14, 3 * (n_reasons + 2)),
                             gridspec_kw={"height_ratios": [2] * n_reasons + [1, 2]})

    # Per-reason scatter heatmaps
    for i, reason in enumerate(reasons_unique):
        ax = axes[i]
        subset = drop_all_df[drop_all_df["reason"] == reason]
        color = reason_colors.get(reason, "blue")
        for j, fn in enumerate(fov_names_sorted):
            fov_drops = subset[subset["fov"] == fn]["t"].values
            if len(fov_drops) > 0:
                ax.scatter(fov_drops, [j] * len(fov_drops),
                           color=color, s=15, alpha=0.8)
        ax.set_yticks(range(len(fov_names_sorted)))
        ax.set_yticklabels(fov_names_sorted, fontsize=5)
        ax.set_xlim(-0.5, T_total - 0.5)
        ax.set_title(f"{reason} ({len(subset)} drops across {subset['fov'].nunique()} FOVs)",
                     fontsize=9)
        ax.grid(axis="x", alpha=0.2)

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
        color = reason_colors.get(reason, "blue")
        ax.bar(t_range, counts, bottom=bottom, width=1.0, alpha=0.7,
               color=color, label=reason)
        bottom += counts
    ax.set_ylabel("# FOVs dropped")
    ax.set_title("Timepoints dropped across FOVs (stacked by reason)")
    ax.legend(fontsize=6)
    ax.set_xlim(-0.5, T_total - 0.5)

    # Correlated drops (>=2 FOVs)
    ax = axes[n_reasons + 1]
    corr_text = []
    for reason in reasons_unique:
        subset = drop_all_df[drop_all_df["reason"] == reason]
        t_counts = subset.groupby("t")["fov"].nunique()
        shared = t_counts[t_counts >= 2].sort_values(ascending=False)
        if len(shared) > 0:
            color = reason_colors.get(reason, "blue")
            ax.bar(shared.index, shared.values, width=0.8, alpha=0.7,
                   color=color, label=reason)
            for t, n in shared.items():
                fovs_at_t = subset[subset["t"] == t]["fov"].tolist()
                corr_text.append(f"  t={t} ({reason}): {n} FOVs — {', '.join(fovs_at_t)}")
    ax.set_xlabel("Time point")
    ax.set_ylabel("# FOVs")
    ax.set_title("Correlated drops: timepoints dropped in >=2 FOVs")
    ax.legend(fontsize=6)
    ax.set_xlim(-0.5, T_total - 0.5)
    if not corr_text:
        ax.text(0.5, 0.5, "No correlated drops found", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")

    fig.tight_layout()
    return fig, drop_all_df, corr_text


def plot_registration_pcc_all_fovs(
    ok_fovs: list[str],
    plots_dir,
) -> tuple[plt.Figure, dict]:
    """Pearson correlation across all FOVs. Returns (fig, {fov_name: series})."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    all_data = {}
    for fov in ok_fovs:
        fov_name = "_".join(fov.split("/"))
        csv_path = plots_dir / fov_name / "fov_registration_qc.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        all_data[fov_name] = df.set_index("t")["pearson_corr"]
        ax.plot(df["t"], df["pearson_corr"], alpha=0.5, linewidth=0.8, label=fov_name)
    ax.set_xlabel("Time point")
    ax.set_ylabel("Pearson correlation (LF vs LS)")
    ax.set_title(f"Registration QC across all FOVs ({len(all_data)} FOVs)")
    ax.legend(fontsize=6, loc="best", ncol=2)
    fig.tight_layout()
    return fig, all_data
