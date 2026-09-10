"""QC functions for dynacell preprocessing (beads registration, Laplacian, entropy, dust)."""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from iohub import open_ome_zarr

from dynacell.geometry import make_circular_mask
from dynacell.plotting import (
    plot_registration_qc,
    plot_laplacian_qc,
    plot_entropy_qc,
    plot_dust_qc,
    plot_bleach_qc,
)


def compute_beads_registration_qc(
    im_lf_path: Path,
    im_ls_path: Path,
    output_plots_dir: Path,
    n_std: float = 2.5,
    blank_frames: list[int] | None = None,
) -> dict:
    """Compute per-timepoint registration QC using 3D phase cross-correlation on beads.

    For each timepoint, computes 3D PCC between the full LF and LS volumes
    (channel 0). Reports the 3D Pearson correlation and residual YX shift.
    Beads are distributed across Z, so 3D analysis captures them all.
    Blank frames are skipped (set to NaN) and excluded from outlier statistics.

    Returns a dict with 'drop_indices' (timepoints to drop due to bad registration).
    """
    from skimage.registration import phase_cross_correlation as pcc_skimage

    with open_ome_zarr(im_lf_path) as im_lf_ds, open_ome_zarr(im_ls_path) as im_ls_ds:
        im_lf_arr = im_lf_ds.data.dask_array()
        im_ls_arr = im_ls_ds.data.dask_array()
        T = im_lf_arr.shape[0]
        Z_lf, Z_ls = im_lf_arr.shape[2], im_ls_arr.shape[2]
        print(f"Beads QC (3D): LF shape={im_lf_arr.shape}, LS shape={im_ls_arr.shape}")

        # Detect blank frames if not provided
        if blank_frames is None:
            blank_frames = []
            z_mid_check = Z_lf // 2
            for t in range(T):
                slc = np.asarray(im_lf_arr[t, 0, z_mid_check, :, :])
                if float(np.nanmax(np.abs(slc))) < 1e-6:
                    blank_frames.append(t)
            if blank_frames:
                print(f"  Detected {len(blank_frames)} blank frames: {blank_frames}")
        blank_set = set(blank_frames)

        pearson_corrs = np.full(T, np.nan, dtype=np.float64)
        pcc_shifts_z = np.full(T, np.nan, dtype=np.float64)
        pcc_shifts_y = np.full(T, np.nan, dtype=np.float64)
        pcc_shifts_x = np.full(T, np.nan, dtype=np.float64)
        pcc_errors = np.full(T, np.nan, dtype=np.float64)

        # Common spatial dimensions
        Z_common = min(Z_lf, Z_ls)
        Y_common = min(im_lf_arr.shape[3], im_ls_arr.shape[3])
        X_common = min(im_lf_arr.shape[4], im_ls_arr.shape[4])

        for t in tqdm(range(T), desc="Beads registration QC (3D)"):
            if t in blank_set:
                continue

            # Load full 3D volume (channel 0) for both modalities
            lf_vol = np.asarray(
                im_lf_arr[t, 0, :Z_common, :Y_common, :X_common]
            ).astype(np.float64)
            ls_vol = np.asarray(
                im_ls_arr[t, 0, :Z_common, :Y_common, :X_common]
            ).astype(np.float64)

            # Handle NaN values
            nan_mask = np.isnan(lf_vol) | np.isnan(ls_vol)
            if nan_mask.all():
                continue
            lf_clean = np.where(nan_mask, 0.0, lf_vol)
            ls_clean = np.where(nan_mask, 0.0, ls_vol)

            # 3D Pearson correlation (on non-NaN voxels only)
            valid = ~nan_mask
            lf_valid = lf_vol[valid]
            ls_valid = ls_vol[valid]
            lf_centered = lf_valid - lf_valid.mean()
            ls_centered = ls_valid - ls_valid.mean()
            denom = np.sqrt(np.sum(lf_centered**2) * np.sum(ls_centered**2))
            if denom > 0:
                pearson_corrs[t] = np.sum(lf_centered * ls_centered) / denom

            # 3D phase cross-correlation (returns [z, y, x] shifts)
            shift, error, _phasediff = pcc_skimage(
                lf_clean, ls_clean, upsample_factor=10
            )
            pcc_shifts_z[t] = shift[0]
            pcc_shifts_y[t] = shift[1]
            pcc_shifts_x[t] = shift[2]
            pcc_errors[t] = error

    # YX shift magnitude (registration quality in lateral plane)
    shift_mag = np.sqrt(pcc_shifts_y**2 + pcc_shifts_x**2)
    valid_mask = ~np.isnan(shift_mag)

    # Outlier detection on shift magnitude (only on non-blank frames)
    mu_shift = np.nanmean(shift_mag)
    sigma_shift = np.nanstd(shift_mag)
    upper_shift = mu_shift + n_std * sigma_shift
    shift_outliers = np.where(valid_mask & (shift_mag > upper_shift))[0]

    # Outlier detection on Pearson correlation (low is bad, only on non-blank frames)
    mu_corr = np.nanmean(pearson_corrs)
    sigma_corr = np.nanstd(pearson_corrs)
    lower_corr = mu_corr - n_std * sigma_corr
    corr_outliers = np.where(valid_mask & (pearson_corrs < lower_corr))[0]

    all_outliers = np.array(sorted(set(shift_outliers) | set(corr_outliers)), dtype=int)
    is_outlier = np.zeros(T, dtype=int)
    is_outlier[all_outliers] = 1

    print(f"\nBeads registration QC results (3D):")
    print(f"  Blank frames excluded: {len(blank_set)}")
    print(f"  Pearson corr (3D): mean={mu_corr:.4f}, std={sigma_corr:.4f}")
    print(f"  YX shift magnitude: mean={mu_shift:.2f}, std={sigma_shift:.2f}")
    print(f"  Z shift: mean={np.nanmean(pcc_shifts_z):.2f}, std={np.nanstd(pcc_shifts_z):.2f}")
    print(f"  Shift outliers (>{upper_shift:.2f} px): {len(shift_outliers)}")
    print(f"  Correlation outliers (<{lower_corr:.4f}): {len(corr_outliers)}")
    print(f"  Total bad timepoints: {len(all_outliers)}")
    if len(all_outliers) > 0:
        print(f"  Bad timepoints: {all_outliers.tolist()}")

    # Save CSV
    qc_df = pd.DataFrame({
        "t": np.arange(T),
        "pearson_corr": pearson_corrs,
        "pcc_shift_z": pcc_shifts_z,
        "pcc_shift_y": pcc_shifts_y,
        "pcc_shift_x": pcc_shifts_x,
        "shift_magnitude": shift_mag,
        "pcc_error": pcc_errors,
        "is_outlier": is_outlier,
    })
    qc_csv = output_plots_dir / "registration_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved QC CSV to {qc_csv}")

    # Plot
    fig = plot_registration_qc(
        pearson_corrs=pearson_corrs,
        pcc_shifts_y=pcc_shifts_y,
        pcc_shifts_x=pcc_shifts_x,
        shift_mag=shift_mag,
        mu_corr=mu_corr, sigma_corr=sigma_corr, lower_corr=lower_corr,
        mu_shift=mu_shift, upper_shift=upper_shift,
        corr_outliers=corr_outliers, shift_outliers=shift_outliers,
    )
    plot_path = output_plots_dir / "registration_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved QC plot to {plot_path}")

    # Outlier agreement between shift magnitude and pearson correlation
    shift_set = set(shift_outliers.tolist())
    corr_set = set(corr_outliers.tolist())
    both = shift_set & corr_set
    shift_only = shift_set - corr_set
    corr_only = corr_set - shift_set
    print(f"\n  Outlier agreement:")
    print(f"    Shift only: {len(shift_only)} {sorted(shift_only)}")
    print(f"    Corr only:  {len(corr_only)} {sorted(corr_only)}")
    print(f"    Both:       {len(both)} {sorted(both)}")

    # Scatter plot: shift magnitude vs pearson correlation
    fig_scatter, ax = plt.subplots(figsize=(7, 5))
    # Classify each valid point
    valid = ~np.isnan(shift_mag) & ~np.isnan(pearson_corrs)
    colors = np.full(T, "tab:blue", dtype=object)
    for i in range(T):
        if not valid[i]:
            colors[i] = "lightgray"
        elif i in both:
            colors[i] = "red"
        elif i in shift_set:
            colors[i] = "orange"
        elif i in corr_set:
            colors[i] = "purple"

    for label, color, marker in [
        ("OK", "tab:blue", "o"),
        ("Shift outlier", "orange", "s"),
        ("Corr outlier", "purple", "D"),
        ("Both", "red", "X"),
        ("Blank", "lightgray", "."),
    ]:
        mask = colors == color
        if mask.any():
            ax.scatter(
                shift_mag[mask], pearson_corrs[mask],
                c=color, marker=marker, s=30, alpha=0.7, label=label,
            )

    ax.axvline(upper_shift, color="orange", ls="--", alpha=0.6, label=f"Shift threshold ({upper_shift:.1f})")
    ax.axhline(lower_corr, color="purple", ls="--", alpha=0.6, label=f"Corr threshold ({lower_corr:.4f})")
    ax.set_xlabel("Shift magnitude (px)")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Beads QC: outlier agreement")
    ax.legend(fontsize=7, loc="best")
    fig_scatter.tight_layout()
    scatter_path = output_plots_dir / "registration_qc_agreement.png"
    fig_scatter.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig_scatter)
    print(f"  Saved agreement plot to {scatter_path}")

    return {
        "drop_indices": all_outliers,
        "pearson_corrs": pearson_corrs,
        "shift_magnitude": shift_mag,
    }


def compute_fov_registration_qc(
    im_lf_arr,
    im_ls_arr,
    z_focus: list[int],
    output_plots_dir: Path,
    lf_mask_radius: float | None = None,
    blank_frames: list[int] | None = None,
    n_std: float = 2.5,
) -> dict:
    """Compute per-timepoint 3D Pearson correlation between LF and LS phase channels.

    This is a lightweight registration QC that runs on every FOV (not just beads).
    For each timepoint, correlates the full 3D LF and LS volumes (channel 0)
    within the circular mask applied per Z-plane. Results are saved to CSV and
    plot for reporting only (not used for the drop list). Blank frames are set to NaN.
    """
    T = im_lf_arr.shape[0]
    Z_common = min(im_lf_arr.shape[2], im_ls_arr.shape[2])
    pearson_corrs = np.full(T, np.nan, dtype=np.float64)
    blank_set = set(blank_frames) if blank_frames else set()

    Y_common = min(im_lf_arr.shape[-2], im_ls_arr.shape[-2])
    X_common = min(im_lf_arr.shape[-1], im_ls_arr.shape[-1])
    mask_2d = None
    if lf_mask_radius is not None:
        circ = make_circular_mask(im_lf_arr.shape[-2], im_lf_arr.shape[-1], lf_mask_radius)
        mask_2d = circ[:Y_common, :X_common]

    for t in tqdm(range(T), desc="Registration QC (3D Pearson)"):
        if t in blank_set:
            continue
        # Load full 3D volume (channel 0)
        lf_vol = np.asarray(im_lf_arr[t, 0, :Z_common, :Y_common, :X_common]).astype(np.float64)
        ls_vol = np.asarray(im_ls_arr[t, 0, :Z_common, :Y_common, :X_common]).astype(np.float64)

        if mask_2d is not None:
            # Apply 2D mask to each Z-plane
            mask_3d = np.broadcast_to(mask_2d[np.newaxis, :, :], lf_vol.shape)
            lf_flat = lf_vol[mask_3d].ravel()
            ls_flat = ls_vol[mask_3d].ravel()
        else:
            lf_flat = lf_vol.ravel()
            ls_flat = ls_vol.ravel()

        lf_centered = lf_flat - lf_flat.mean()
        ls_centered = ls_flat - ls_flat.mean()
        denom = np.sqrt(np.sum(lf_centered**2) * np.sum(ls_centered**2))
        if denom > 0:
            pearson_corrs[t] = np.sum(lf_centered * ls_centered) / denom

    # Outlier detection (reporting only — low PCC = bad registration)
    valid_mask = ~np.isnan(pearson_corrs)
    valid_vals = pearson_corrs[valid_mask]
    mu = float(np.nanmean(pearson_corrs))
    sigma = float(np.nanstd(pearson_corrs))
    local_z = np.full(T, np.nan, dtype=np.float64)
    is_outlier = np.zeros(T, dtype=int)
    if sigma > 1e-12:
        local_z[valid_mask] = (pearson_corrs[valid_mask] - mu) / sigma
        is_outlier[valid_mask] = (local_z[valid_mask] < -n_std).astype(int)

    n_outliers = int(is_outlier.sum())
    print(f"  FOV registration: mean={mu:.4f}, std={sigma:.4f}, n_std={n_std}, outliers={n_outliers} (reporting only)")

    # Save CSV
    qc_df = pd.DataFrame({
        "t": np.arange(T),
        "pearson_corr": pearson_corrs,
        "local_z": local_z,
        "is_outlier": is_outlier,
    })
    qc_csv = output_plots_dir / "fov_registration_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved FOV registration QC to {qc_csv}")

    # Plot
    from dynacell.plotting import plot_fov_registration_qc
    fig = plot_fov_registration_qc(
        pearson_corrs=pearson_corrs,
        mu=mu,
        n_outliers=n_outliers,
        outlier_idx=np.where(is_outlier)[0],
        blank_count=len(blank_set),
        local_z=local_z,
        n_std=n_std,
    )
    plot_path = output_plots_dir / "fov_registration_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"pearson_corrs": pearson_corrs, "outliers": np.where(is_outlier)[0]}


def compute_laplacian_qc(
    im_ls_path: Path,
    output_plots_dir: Path,
    channel_name: str = "raw GFP EX488 EM525-45",
    roi_half: int = 300,
    n_std: float = 1.0,
    blank_frames: list[int] | None = None,
) -> dict:
    """Compute per-timepoint 2D Laplacian variance on max-Z projection to detect blur.

    For each timepoint, max-projects the full Z stack, crops to an XY ROI,
    and computes the variance of the 2D Laplacian. Blurry frames have
    lower variance because high-frequency content is suppressed.
    Blank frames are set to NaN and excluded from statistics.
    """
    from scipy.ndimage import laplace

    blank_set = set(blank_frames) if blank_frames else set()

    with open_ome_zarr(im_ls_path) as ds:
        arr = ds.data.dask_array()
        T, C, Z, Y, X = arr.shape
        channels = list(ds.channel_names)

        if channel_name not in channels:
            print(f"WARNING: channel '{channel_name}' not found in {channels}")
            return {"lap_vars": np.full(T, np.nan), "outliers": np.array([], dtype=int)}

        c_idx = channels.index(channel_name)
        y_c, x_c = Y // 2, X // 2

        y_start = max(0, y_c - roi_half)
        y_end = min(Y, y_c + roi_half)
        x_start = max(0, x_c - roi_half)
        x_end = min(X, x_c + roi_half)

        print(f"Laplacian QC: channel='{channel_name}' (c={c_idx}), "
              f"max-Z projection, "
              f"Y=[{y_start}:{y_end}], X=[{x_start}:{x_end}]")

        lap_vars = np.full(T, np.nan, dtype=np.float64)
        max_ints = np.full(T, np.nan, dtype=np.float64)

        for t in tqdm(range(T), desc="Laplacian QC (max-Z proj)"):
            if t in blank_set:
                continue

            vol = np.asarray(
                arr[t, c_idx, :, y_start:y_end, x_start:x_end]
            ).astype(np.float64)
            mip = np.max(vol, axis=0)  # max Z projection → (Y, X)
            max_ints[t] = float(np.max(mip))
            if max_ints[t] < 1e-6:
                continue

            lap2d = laplace(mip)
            lap_vars[t] = float(np.var(lap2d))

    # Stats on non-blank frames only
    valid_mask = ~np.isnan(lap_vars)
    if valid_mask.sum() == 0:
        print("WARNING: All frames blank, no Laplacian QC computed")
        return {"lap_vars": lap_vars, "outliers": np.array([], dtype=int)}

    valid_vals = lap_vars[valid_mask]
    mu = float(np.mean(valid_vals))
    sigma = float(np.std(valid_vals))
    lower = mu - n_std * sigma

    # Local z-score
    local_z = np.full(T, np.nan, dtype=np.float64)
    if sigma > 1e-12:
        local_z[valid_mask] = (lap_vars[valid_mask] - mu) / sigma

    outliers = np.where((lap_vars < lower) & valid_mask)[0]
    is_outlier = np.zeros(T, dtype=int)
    is_outlier[outliers] = 1

    print(f"\nLaplacian QC results ({channel_name}):")
    print(f"  Laplacian variance (max-Z proj): mean={mu:.2f}, std={sigma:.2f}")
    print(f"  Threshold (mean - {n_std}*std): {lower:.2f}")
    print(f"  Outliers (blurry frames): {len(outliers)}")
    if len(outliers) > 0:
        for t in outliers:
            print(f"    t={t}: lap_var={lap_vars[t]:.2f}, z-score={local_z[t]:.2f}")

    # Save CSV
    qc_df = pd.DataFrame({
        "t": np.arange(T),
        "lap_var": lap_vars,
        "max_intensity": max_ints,
        "local_z": local_z,
        "is_outlier": is_outlier,
    })
    qc_csv = output_plots_dir / "laplacian_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved CSV to {qc_csv}")

    # Plot
    fig = plot_laplacian_qc(
        lap_vars=lap_vars,
        mu=mu, sigma=sigma, lower=lower, n_std=n_std,
        outliers=outliers, channel_name=channel_name,
        local_z=local_z,
    )
    plot_path = output_plots_dir / "laplacian_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    return {
        "lap_vars": lap_vars,
        "outliers": outliers,
        "mean": mu,
        "std": sigma,
    }


def compute_entropy_qc(
    im_ls_path: Path,
    output_plots_dir: Path,
    channel_name: str = "raw GFP EX488 EM525-45",
    n_bins: int = 256,
    local_window: int = 3,
    n_std: float = 2.5,
    iqr_factor: float = 1.5,
    blank_frames: list[int] | None = None,
) -> dict:
    """Measure per-timepoint Shannon entropy (reporting only, not used for dropping).

    Blank frames are set to NaN and excluded from statistics.
    """

    def _entropy(vol, n_bins):
        flat = vol.ravel()
        hist, _ = np.histogram(flat, bins=n_bins, density=True)
        hist = hist[hist > 0]
        return -float(np.sum(hist * np.log2(hist + 1e-12)))

    blank_set = set(blank_frames) if blank_frames else set()

    with open_ome_zarr(im_ls_path) as ds:
        arr = ds.data.dask_array()
        T, C, Z, Y, X = arr.shape
        channels = list(ds.channel_names)

        if channel_name not in channels:
            print(f"WARNING: channel '{channel_name}' not found in {channels}")
            return {
                "entropies": np.zeros(T),
                "outliers": np.array([], dtype=int),
                "stats": {},
            }

        c_idx = channels.index(channel_name)
        print(
            f"Entropy QC: channel='{channel_name}' (c={c_idx}), "
            f"max-Z-projection, Z={Z}, bins={n_bins}"
        )

        entropies = np.full(T, np.nan, dtype=np.float64)
        for t in tqdm(range(T), desc="Entropy QC (max-Z proj)"):
            if t in blank_set:
                continue
            vol = np.asarray(arr[t, c_idx]).astype(np.float32)
            mip = np.max(vol, axis=0)  # max Z projection → (Y, X)
            if np.ptp(mip) < 1.0:
                continue
            entropies[t] = _entropy(mip, n_bins)

    # --- Exclude blank frames ---
    blank = np.isnan(entropies)
    valid = entropies[~blank]
    if len(valid) < 3:
        print("WARNING: too few valid frames for entropy QC")
        return {
            "entropies": entropies,
            "outliers": np.array([], dtype=int),
            "stats": {},
        }

    # --- Global statistics (IQR fence) ---
    med = np.median(valid)
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    global_lower = q1 - iqr_factor * iqr
    global_upper = q3 + iqr_factor * iqr

    global_outlier = np.zeros(T, dtype=bool)
    for t in range(T):
        if not blank[t]:
            global_outlier[t] = entropies[t] < global_lower or entropies[t] > global_upper

    # --- Local z-score (+/-local_window neighbors) ---
    local_z = np.zeros(T)
    for t in range(T):
        if blank[t]:
            continue
        neighbors = [
            entropies[t + d]
            for d in range(-local_window, local_window + 1)
            if d != 0 and 0 <= t + d < T and not blank[t + d]
        ]
        if len(neighbors) >= 4:
            loc_mean = np.mean(neighbors)
            loc_std = np.std(neighbors)
            local_z[t] = (entropies[t] - loc_mean) / loc_std if loc_std > 0 else 0

    # --- Flag outliers: must satisfy BOTH global AND local criteria ---
    outlier_mask = global_outlier & (np.abs(local_z) > n_std) & ~blank
    outliers = np.where(outlier_mask)[0]

    stats = {
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "global_lower": global_lower,
        "global_upper": global_upper,
        "n_std": n_std,
        "iqr_factor": iqr_factor,
    }

    print(f"\nEntropy QC results ({channel_name}):")
    print(f"  Median={med:.4f}, Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
    print(f"  Global fence: [{global_lower:.4f}, {global_upper:.4f}]")
    print(f"  Global outliers (before local filter): {global_outlier.sum()}")
    print(f"  Local z-score threshold: {n_std}")
    print(f"  Combined outliers: {len(outliers)}")
    for t in outliers:
        print(
            f"    t={t}: entropy={entropies[t]:.4f}, "
            f"local_z={local_z[t]:+.2f}, "
            f"global={'above' if entropies[t] > global_upper else 'below'}"
        )

    # Save CSV
    qc_df = pd.DataFrame(
        {
            "t": np.arange(T),
            "entropy": entropies,
            "local_z": local_z,
            "global_outlier": global_outlier.astype(int),
            "is_outlier": outlier_mask.astype(int),
        }
    )
    qc_csv = output_plots_dir / "entropy_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved CSV to {qc_csv}")

    # Plot
    fig = plot_entropy_qc(
        entropies=entropies, local_z=local_z, blank_mask=blank,
        outliers=outliers, med=med, n_std=n_std,
        channel_name=channel_name,
        global_lower=global_lower, global_upper=global_upper,
    )
    plot_path = output_plots_dir / "entropy_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    return {"entropies": entropies, "outliers": outliers, "stats": stats}


def _hf_energy_ratio(img_2d: np.ndarray, cutoff_fraction: float = 0.1) -> float:
    """Ratio of high-frequency to total energy in the 2D power spectrum.

    Blur suppresses high frequencies → ratio drops.
    Bleaching is multiplicative → ratio unchanged.
    Illumination gradient is low-frequency → ratio unaffected.
    """
    fft2 = np.fft.fft2(img_2d.astype(np.float64))
    power = np.abs(np.fft.fftshift(fft2)) ** 2
    Y, X = power.shape
    cy, cx = Y // 2, X // 2
    yy, xx = np.ogrid[:Y, :X]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_freq = min(cy, cx)
    cutoff = cutoff_fraction * max_freq
    hf_energy = np.sum(power[dist > cutoff])
    total_energy = np.sum(power)
    return float(hf_energy / total_energy) if total_energy > 0 else 0.0


def compute_hf_ratio_qc(
    im_ls_path: Path,
    output_plots_dir: Path,
    channel_name: str = "raw GFP EX488 EM525-45",
    cutoff_fraction: float = 0.1,
    local_window: int = 3,
    n_std: float = 2.5,
    blank_frames: list[int] | None = None,
) -> dict:
    """Measure per-timepoint HF energy ratio as a blur/quality indicator.

    For each timepoint, computes the HF energy ratio on the max-Z
    projection of the full volume. Lower ratio = less high-frequency
    content = blurrier.

    Outlier detection uses local z-score of the HF ratio. Frames with
    local_z < -n_std are flagged as outliers.

    Parameters
    ----------
    im_ls_path : Path
        Path to the LS zarr FOV position.
    output_plots_dir : Path
        Directory for saving plots and CSVs.
    channel_name : str
        Channel to measure blur on.
    cutoff_fraction : float
        Fraction of max frequency for the high/low boundary (default 0.1).
    local_window : int
        Half-window for local z-score comparison.
    n_std : float
        HF ratio local z-score threshold. Frames with local_z < -threshold
        are flagged as outliers (default 2.5).
    blank_frames : list of int or None
        Timepoints to skip (blank frames).

    Returns
    -------
    dict with 'hf_ratios', 'outliers', 'stats'.
    """
    from dynacell.plotting import plot_hf_ratio_qc

    with open_ome_zarr(im_ls_path) as ds:
        arr = ds.data.dask_array()
        T, C, Z, Y, X = arr.shape
        channels = list(ds.channel_names)

        if channel_name not in channels:
            print(f"WARNING: channel '{channel_name}' not found in {channels}")
            return {
                "hf_ratios": np.zeros(T),
                "outliers": np.array([], dtype=int),
                "stats": {},
            }

        c_idx = channels.index(channel_name)
        print(
            f"HF ratio QC: channel='{channel_name}' (c={c_idx}), "
            f"cutoff={cutoff_fraction}, max-Z projection"
        )

        blank_set = set(blank_frames) if blank_frames else set()
        hf_ratios = np.full(T, np.nan, dtype=np.float64)
        for t in tqdm(range(T), desc="HF ratio QC (max-Z proj)"):
            if t in blank_set:
                continue
            vol = np.asarray(arr[t, c_idx]).astype(np.float64)
            mip = np.max(vol, axis=0)  # max Z projection → (Y, X)
            if np.ptp(mip) < 1e-6:
                continue
            hf_ratios[t] = _hf_energy_ratio(mip, cutoff_fraction)

    # --- Exclude blank frames ---
    blank = np.isnan(hf_ratios)
    valid = hf_ratios[~blank]
    if len(valid) < 3:
        print("WARNING: too few valid frames for HF ratio QC")
        return {
            "hf_ratios": hf_ratios,
            "outliers": np.array([], dtype=int),
            "stats": {},
        }

    med = float(np.median(valid))

    # --- HF local z-score ---
    hf_local_z = np.zeros(T)
    for t in range(T):
        if blank[t]:
            continue
        neighbors = [
            hf_ratios[t + d]
            for d in range(-local_window, local_window + 1)
            if d != 0 and 0 <= t + d < T and not blank[t + d]
        ]
        if len(neighbors) >= 4:
            loc_mean = np.mean(neighbors)
            loc_std = np.std(neighbors)
            hf_local_z[t] = (hf_ratios[t] - loc_mean) / loc_std if loc_std > 0 else 0

    # --- Outlier detection: HF local z-score only ---
    outlier_mask = (hf_local_z < -n_std) & ~blank
    outliers = np.where(outlier_mask)[0]

    stats = {
        "median": med,
        "n_std": n_std,
        "cutoff_fraction": cutoff_fraction,
    }

    print(f"\nHF ratio QC results ({channel_name}):")
    print(f"  Median={med:.6f}, max-Z projection")
    print(f"  Threshold: local_z < -{n_std}")
    print(f"  Outliers (low HF): {len(outliers)}")
    for t in outliers:
        print(f"    t={t}: hf_ratio={hf_ratios[t]:.6f}, "
              f"hf_lz={hf_local_z[t]:+.2f}")

    # Save CSV
    qc_df = pd.DataFrame({
        "t": np.arange(T),
        "hf_ratio": hf_ratios,
        "hf_local_z": hf_local_z,
        "is_outlier": outlier_mask.astype(int),
    })
    qc_csv = output_plots_dir / "hf_ratio_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved CSV to {qc_csv}")

    # Plot
    fig = plot_hf_ratio_qc(
        hf_ratios=hf_ratios, hf_local_z=hf_local_z, blank_mask=blank,
        outliers=outliers, med=med,
        n_std=n_std,
        channel_name=channel_name,
    )
    plot_path = output_plots_dir / "hf_ratio_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    return {"hf_ratios": hf_ratios, "outliers": outliers, "stats": stats}


def compute_bleach_fov(
    im_ls_path: Path,
    output_plots_dir: Path,
    bbox: tuple[int, int, int, int],
    channel_name: str = "raw GFP EX488 EM525-45",
    blank_frames: list[int] | None = None,
    n_std: float = 3.0,
) -> dict:
    """Measure per-timepoint mean GFP intensity within the crop bbox on max-Z projection.

    Saves bleach_qc.csv and bleach_qc.png per FOV. The dataset-level
    bleach summary reads these CSVs instead of re-opening the zarrs.
    """
    with open_ome_zarr(im_ls_path) as ds:
        arr = ds.data.dask_array()
        T, C, Z, Y, X = arr.shape
        channels = list(ds.channel_names)

        if channel_name not in channels:
            print(f"WARNING: channel '{channel_name}' not in {channels} for bleach QC")
            return {"mean_intensities": np.full(T, np.nan), "normalized": np.full(T, np.nan)}

        c_idx = channels.index(channel_name)
        y_min, y_max, x_min, x_max = bbox
        blank_set = set(blank_frames) if blank_frames else set()

        means = np.full(T, np.nan, dtype=np.float64)
        for t in range(T):
            if t in blank_set:
                continue
            vol = np.asarray(
                arr[t, c_idx, :, y_min:y_max + 1, x_min:x_max + 1]
            ).astype(np.float64)
            mip = np.max(vol, axis=0)  # max Z projection → (Y, X)
            means[t] = float(np.mean(mip))

    # Normalize to first valid timepoint
    first_valid = np.where(~np.isnan(means))[0]
    normalized = np.full(T, np.nan, dtype=np.float64)
    if len(first_valid) > 0 and means[first_valid[0]] > 0:
        normalized = means / means[first_valid[0]]

    # Outlier detection: flag frames with sudden jumps/drops in normalized intensity
    # Use rolling-median residuals — large deviations from local trend are outliers
    local_z = np.full(T, np.nan, dtype=np.float64)
    is_outlier = np.zeros(T, dtype=int)
    valid_mask = ~np.isnan(normalized)
    if valid_mask.sum() > 5:
        from scipy.ndimage import median_filter
        smoothed = np.full(T, np.nan)
        smoothed[valid_mask] = median_filter(normalized[valid_mask], size=5, mode="nearest")
        residuals = np.full(T, np.nan)
        residuals[valid_mask] = normalized[valid_mask] - smoothed[valid_mask]
        valid_res = residuals[valid_mask]
        mad = float(np.median(np.abs(valid_res - np.median(valid_res))))
        scale = 1.4826 * mad if mad > 1e-12 else float(np.std(valid_res))
        if scale > 1e-12:
            local_z[valid_mask] = (residuals[valid_mask] - np.median(valid_res)) / scale
            is_outlier[valid_mask] = (np.abs(local_z[valid_mask]) > n_std).astype(int)

    # Save CSV
    fov_df = pd.DataFrame({
        "t": np.arange(T),
        "mean_intensity": means,
        "normalized": normalized,
        "local_z": local_z,
        "is_outlier": is_outlier,
    })
    fov_df.to_csv(output_plots_dir / "bleach_qc.csv", index=False)

    # Per-FOV plot
    from dynacell.plotting import plot_bleach_fov_qc
    fov_name = output_plots_dir.name
    fig = plot_bleach_fov_qc(normalized=normalized, fov_name=fov_name, local_z=local_z, n_std=n_std)
    fig.savefig(output_plots_dir / "bleach_qc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    valid_idx = np.where(~np.isnan(normalized))[0]
    pct_rem = float(normalized[valid_idx[-1]] * 100) if len(valid_idx) > 0 else 0

    print(f"  Bleach QC: {pct_rem:.1f}% remaining at last t")
    return {"mean_intensities": means, "normalized": normalized}


def compute_max_intensity_qc(
    im_ls_path: Path,
    output_plots_dir: Path,
    channel_name: str = "raw GFP EX488 EM525-45",
    blank_frames: list[int] | None = None,
    n_std: float = 2.5,
) -> dict:
    """Track per-timepoint max intensity across the full Z stack.

    For each timepoint, computes the global max pixel value over all Z
    slices of the given channel. Useful for detecting intensity drift,
    sudden drops (e.g. lost focus), or anomalous bright frames.

    Saves max_intensity_qc.csv and max_intensity_qc.png per FOV.
    """
    from dynacell.plotting import plot_max_intensity_qc

    with open_ome_zarr(im_ls_path) as ds:
        arr = ds.data.dask_array()
        T, C, Z, Y, X = arr.shape
        channels = list(ds.channel_names)

        if channel_name not in channels:
            print(f"WARNING: channel '{channel_name}' not found in {channels}")
            return {"max_intensities": np.full(T, np.nan)}

        c_idx = channels.index(channel_name)
        print(f"Max intensity QC: channel='{channel_name}' (c={c_idx}), full Z={Z}")

        blank_set = set(blank_frames) if blank_frames else set()
        max_vals = np.full(T, np.nan, dtype=np.float64)
        for t in tqdm(range(T), desc="Max intensity QC"):
            if t in blank_set:
                continue
            vol = np.asarray(arr[t, c_idx]).astype(np.float64)
            max_vals[t] = float(np.max(vol))

    blank = np.isnan(max_vals)
    valid = max_vals[~blank]
    if len(valid) < 2:
        print("WARNING: too few valid frames for max intensity QC")
        return {"max_intensities": max_vals}

    med = float(np.median(valid))
    mu = float(np.mean(valid))
    sigma = float(np.std(valid))

    # Outlier detection: local z-score, flag if |z| > n_std
    local_z = np.full(T, np.nan, dtype=np.float64)
    is_outlier = np.zeros(T, dtype=int)
    if sigma > 1e-12:
        local_z[~blank] = (max_vals[~blank] - mu) / sigma
        is_outlier[~blank] = (np.abs(local_z[~blank]) > n_std).astype(int)

    n_outliers = int(is_outlier.sum())
    print(f"\nMax intensity QC results ({channel_name}):")
    print(f"  Median={med:.2f}, Min={valid.min():.2f}, Max={valid.max():.2f}")
    print(f"  Outliers (|z| > {n_std}): {n_outliers}")

    # Save CSV
    qc_df = pd.DataFrame({
        "t": np.arange(T),
        "max_intensity": max_vals,
        "local_z": local_z,
        "is_outlier": is_outlier,
    })
    qc_csv = output_plots_dir / "max_intensity_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved CSV to {qc_csv}")

    # Plot
    fig = plot_max_intensity_qc(
        max_vals=max_vals, blank_mask=blank,
        med=med, channel_name=channel_name,
        local_z=local_z, n_std=n_std,
    )
    plot_path = output_plots_dir / "max_intensity_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    return {"max_intensities": max_vals}


def _frc_mean_corr(img_2d: np.ndarray) -> tuple[float, np.ndarray | None]:
    """Compute mean of FRC correlation curve — higher = sharper.

    Returns (mean_corr, correlation_curve). The curve is None on failure.
    """
    try:
        from cubic.metrics.frc.frc import calculate_frc
        result = calculate_frc(img_2d.astype(np.float32))
        corr = np.array(result.correlation["correlation"])
        if corr.ndim == 0 or len(corr) == 0:
            return np.nan, None
        return float(np.nanmean(corr)), corr
    except Exception:
        return np.nan, None


def compute_frc_qc(
    im_ls_path: Path,
    output_plots_dir: Path,
    channel_name: str = "raw GFP EX488 EM525-45",
    blank_frames: list[int] | None = None,
    n_std: float = 2.5,
) -> dict:
    """Measure per-timepoint FRC mean correlation as a blur/quality indicator.

    For each timepoint, computes the FRC curve on the max-Z projection
    of the full volume. Lower FRC correlation indicates blur or loss of
    high-frequency content.

    Reporting only — not used for frame dropping or outlier detection.

    Parameters
    ----------
    im_ls_path : Path
        Path to the LS zarr FOV position.
    output_plots_dir : Path
        Directory for saving plots and CSVs.
    channel_name : str
        Channel to measure.
    blank_frames : list of int or None
        Timepoints to skip.

    Returns
    -------
    dict with 'frc_values' and 'stats'.
    """
    from dynacell.plotting import plot_frc_qc

    with open_ome_zarr(im_ls_path) as ds:
        arr = ds.data.dask_array()
        T, C, Z, Y, X = arr.shape
        channels = list(ds.channel_names)

        if channel_name not in channels:
            print(f"WARNING: channel '{channel_name}' not found in {channels}")
            return {
                "frc_values": np.zeros(T),
                "stats": {},
            }

        c_idx = channels.index(channel_name)
        print(
            f"FRC QC: channel='{channel_name}' (c={c_idx}), "
            f"max-Z projection"
        )

        blank_set = set(blank_frames) if blank_frames else set()
        frc_values = np.full(T, np.nan, dtype=np.float64)
        frc_curves = {}  # t -> correlation curve
        for t in tqdm(range(T), desc="FRC QC (max-Z proj)"):
            if t in blank_set:
                continue
            vol = np.asarray(arr[t, c_idx]).astype(np.float32)
            mip = np.max(vol, axis=0)  # max Z projection → (Y, X)
            if np.ptp(mip) < 1e-6:
                continue
            mean_corr, curve = _frc_mean_corr(mip)
            frc_values[t] = mean_corr
            if curve is not None:
                frc_curves[t] = curve

    # --- Exclude blank frames ---
    blank = np.isnan(frc_values)
    valid = frc_values[~blank]
    if len(valid) < 3:
        print("WARNING: too few valid frames for FRC QC")
        return {
            "frc_values": frc_values,
            "stats": {},
        }

    med = float(np.median(valid))
    mu = float(np.mean(valid))
    sigma = float(np.std(valid))

    # Outlier detection (reporting only — low FRC = blurry)
    local_z = np.full(T, np.nan, dtype=np.float64)
    is_outlier = np.zeros(T, dtype=int)
    if sigma > 1e-12:
        local_z[~blank] = (frc_values[~blank] - mu) / sigma
        is_outlier[~blank] = (local_z[~blank] < -n_std).astype(int)

    n_outliers = int(is_outlier.sum())

    stats = {
        "median": med,
        "mean": mu,
        "std": sigma,
        "n_outliers": n_outliers,
    }

    print(f"\nFRC QC results ({channel_name}):")
    print(f"  Median={med:.6f}, Mean={mu:.6f}, Std={sigma:.6f}, max-Z projection")
    print(f"  Outliers (z < -{n_std}, reporting only): {n_outliers}")

    # Save CSV
    qc_df = pd.DataFrame({
        "t": np.arange(T),
        "frc_mean_corr": frc_values,
        "local_z": local_z,
        "is_outlier": is_outlier,
    })
    qc_csv = output_plots_dir / "frc_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved CSV to {qc_csv}")

    # Compute mean FRC curve across valid timepoints
    mean_frc_curve = None
    if frc_curves:
        # Pad/truncate to common length
        max_len = max(len(c) for c in frc_curves.values())
        curve_stack = np.full((len(frc_curves), max_len), np.nan)
        for i, c in enumerate(frc_curves.values()):
            curve_stack[i, :len(c)] = c
        mean_frc_curve = np.nanmean(curve_stack, axis=0)

    # Plot
    fig = plot_frc_qc(
        frc_values=frc_values, local_z=local_z, blank_mask=blank, med=med,
        channel_name=channel_name, mean_frc_curve=mean_frc_curve, n_std=n_std,
    )
    plot_path = output_plots_dir / "frc_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    return {"frc_values": frc_values, "outliers": np.where(is_outlier)[0], "stats": stats}


def compute_dust_qc(
    lf_zarr: Path,
    fov_keys: list[str],
    output_dir: Path,
    lf_mask_radius: float = 0.75,
    n_sample_fovs: int = 6,
    blank_frames_per_fov: dict[str, set[int]] | None = None,
    sigma_bg: float = 30.0,
    n_std: float = 3.0,
) -> dict:
    """Estimate dust on the LF optical path from Phase channel.

    Reads 1 timepoint from each of n_sample_fovs FOVs (evenly spaced).
    The pixel-wise median across FOVs isolates the static dust pattern
    (sample signal washes out because different FOVs have different cells).
    A Gaussian blur estimates the smooth illumination background, and the
    residual (median - background) reveals dust spots.

    Parameters
    ----------
    lf_zarr : Path
        Path to the LF zarr store (plate-level, e.g. dataset.zarr).
    fov_keys : list of str
        FOV keys in "A/1/000001" format.
    output_dir : Path
        Directory for saving plots and CSVs.
    lf_mask_radius : float
        Radius fraction for the circular mask to exclude well border.
    n_sample_fovs : int
        Number of FOVs to sample (evenly spaced). 1 timepoint per FOV.
    blank_frames_per_fov : dict or None
        Mapping fov_key -> set of blank timepoint indices. These are skipped.
    sigma_bg : float
        Gaussian sigma for smooth background estimation.
    n_std : float
        Threshold: pixels with |residual| > n_std * sigma are flagged as dust.

    Returns
    -------
    dict with per-Z dust statistics and aggregate severity.
    """
    from scipy.ndimage import gaussian_filter, label

    if blank_frames_per_fov is None:
        blank_frames_per_fov = {}

    # Sample FOVs evenly
    n_fovs = min(n_sample_fovs, len(fov_keys))
    fov_indices = np.linspace(0, len(fov_keys) - 1, n_fovs, dtype=int)
    sampled_fovs = [fov_keys[i] for i in fov_indices]

    # Read shape from first FOV
    with open_ome_zarr(lf_zarr / sampled_fovs[0]) as ds:
        T_total, C, Z_total, Y, X = ds.data.shape
    print(f"Dust QC: LF shape=({T_total}, {C}, {Z_total}, {Y}, {X})")
    print(f"  Sampling 1 timepoint from {n_fovs} FOVs")

    circ_mask = make_circular_mask(Y, X, lf_mask_radius)
    n_masked_pixels = int(circ_mask.sum())

    # Read 1 non-blank timepoint per FOV (middle of valid range)
    per_z_frames = {z: [] for z in range(Z_total)}

    for fov_key in tqdm(sampled_fovs, desc="Dust QC: reading FOVs"):
        blank_set = blank_frames_per_fov.get(fov_key, set())
        valid_t = [t for t in range(T_total) if t not in blank_set]
        if len(valid_t) == 0:
            continue
        t_mid = valid_t[len(valid_t) // 2]

        with open_ome_zarr(lf_zarr / fov_key) as ds:
            arr = ds.data.dask_array()
            for z in range(Z_total):
                per_z_frames[z].append(
                    np.asarray(arr[t_mid, 0, z, :, :]).astype(np.float32)
                )

    # Per-Z: median across FOVs, subtract Gaussian background, detect dust
    z_stats = []
    dust_maps = {}

    for z in tqdm(range(Z_total), desc="Dust QC: analyzing per Z"):
        frames = per_z_frames[z]
        if len(frames) == 0:
            z_stats.append({"z": z, "dust_fraction": 0.0, "n_spots": 0,
                            "max_residual": 0.0, "mean_residual": 0.0})
            continue

        stack = np.stack(frames, axis=0)
        median_img = np.median(stack, axis=0)

        # Subtract smooth background (Gaussian is fast and separable)
        background = gaussian_filter(median_img, sigma=sigma_bg)
        residual = median_img - background

        # Apply circular mask
        residual[~circ_mask] = 0.0

        # Threshold
        valid_residuals = residual[circ_mask]
        sigma = float(np.std(valid_residuals))
        threshold = n_std * sigma

        dust_mask = (np.abs(residual) > threshold) & circ_mask
        dust_fraction = float(dust_mask.sum()) / n_masked_pixels

        # Connected components
        _, n_spots = label(dust_mask)

        max_res = float(np.max(np.abs(valid_residuals)))
        mean_res = float(np.mean(np.abs(valid_residuals)))

        z_stats.append({
            "z": z,
            "dust_fraction": dust_fraction,
            "n_spots": n_spots,
            "max_residual": max_res,
            "mean_residual": mean_res,
            "sigma": sigma,
            "threshold": threshold,
        })
        dust_maps[z] = dust_mask

    # Aggregate
    dust_fractions = np.array([s["dust_fraction"] for s in z_stats])
    worst_z = int(np.argmax(dust_fractions))
    max_dust_fraction = float(dust_fractions[worst_z])
    total_spots_worst = z_stats[worst_z]["n_spots"]

    # Dust score: 0 (no dust) to 1 (very dusty).
    # Based on mean dust fraction across Z. A sigmoid-like mapping so that:
    #   0% dust → 0.0, ~0.5% → ~0.25, ~1% → ~0.5, ~2% → ~0.75, ~5%+ → ~1.0
    mean_dust_fraction = float(dust_fractions.mean())
    # Saturating mapping: score = 1 - exp(-k * fraction), k chosen so 2% → 0.75
    k = -np.log(1.0 - 0.75) / 0.02  # k ≈ 69.3
    dust_score = float(1.0 - np.exp(-k * mean_dust_fraction))

    print(f"\nDust QC results:")
    print(f"  Worst Z plane: z={worst_z}")
    print(f"  Max dust fraction: {max_dust_fraction:.4f} ({max_dust_fraction*100:.2f}%)")
    print(f"  Spots at worst Z: {total_spots_worst}")
    print(f"  Mean dust fraction across Z: {mean_dust_fraction:.4f}")
    print(f"  Dust score: {dust_score:.3f} (0=clean, 1=very dusty)")

    # Save CSV
    stats_df = pd.DataFrame(z_stats)
    stats_csv = output_dir / "dust_qc.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"  Saved CSV to {stats_csv}")

    # Plot
    worst_median = np.median(np.stack(per_z_frames[worst_z], axis=0), axis=0)
    fig = plot_dust_qc(
        dust_fractions=dust_fractions,
        worst_z=worst_z,
        worst_median=worst_median,
        dust_mask_worst=dust_maps.get(worst_z),
        circ_mask=circ_mask,
        n_fovs=n_fovs, n_sample_t=1,
        lf_mask_radius=lf_mask_radius,
        total_spots_worst=total_spots_worst,
        max_dust_fraction=max_dust_fraction,
    )
    plot_path = output_dir / "dust_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    # Save dust mask as zarr (1, 1, Z, Y, X) — binary uint8
    dust_volume = np.zeros((Z_total, Y, X), dtype=np.uint8)
    for z, mask in dust_maps.items():
        dust_volume[z] = mask.astype(np.uint8)
    dust_zarr_path = output_dir / "dust_mask.zarr"
    from iohub.ngff.utils import create_empty_plate
    create_empty_plate(
        store_path=dust_zarr_path,
        position_keys=[("0", "0", "000000")],
        shape=(1, 1, Z_total, Y, X),
        chunks=(1, 1, Z_total, Y, X),
        scale=(1.0, 1.0, 1.0, 1.0, 1.0),
        channel_names=["dust_mask"],
        dtype=np.uint8,
        version='0.5',
    )
    with open_ome_zarr(dust_zarr_path / "0" / "0" / "000000", mode="r+") as ds:
        ds["0"][:] = dust_volume[np.newaxis, np.newaxis, ...]
    print(f"  Saved dust mask zarr to {dust_zarr_path} — shape (1, 1, {Z_total}, {Y}, {X})")

    return {
        "z_stats": z_stats,
        "worst_z": worst_z,
        "max_dust_fraction": max_dust_fraction,
        "dust_score": dust_score,
        "n_spots_worst": total_spots_worst,
    }


def compute_bleach_qc(
    fov_keys: list[str],
    plots_dir: Path,
    output_dir: Path,
    channel_name: str = "raw GFP EX488 EM525-45",
    **_ignored,
) -> dict:
    """Aggregate per-FOV bleach QC CSVs into a dataset-level summary.

    Reads plots/<fov>/bleach_qc.csv (produced by compute_bleach_fov in stage 1)
    and fits an exponential decay to the mean normalized curve.

    Parameters
    ----------
    fov_keys : list of str
        FOV keys in "A/1/000001" format.
    plots_dir : Path
        Stage 1 plots directory (contains per-FOV bleach_qc.csv).
    output_dir : Path
        Directory for saving dataset-level plots and CSVs.
    channel_name : str
        Channel name (for plot labels).

    Returns
    -------
    dict with bleaching statistics.
    """
    from scipy.optimize import curve_fit

    # Read per-FOV bleach CSVs
    fov_curves = {}  # fov_key -> (t_arr, means)
    for fov_key in fov_keys:
        fov_name = "_".join(fov_key.split("/"))
        bleach_csv = plots_dir / fov_name / "bleach_qc.csv"
        if not bleach_csv.exists():
            continue
        df = pd.read_csv(bleach_csv)
        valid = df.dropna(subset=["mean_intensity"])
        if len(valid) == 0:
            continue
        fov_curves[fov_key] = (valid["t"].values, valid["mean_intensity"].values)

    if not fov_curves:
        print("WARNING: No valid FOV curves for bleach QC")
        return {"half_life": np.nan, "pct_remaining": np.nan}

    # Normalize each FOV curve to its first timepoint
    norm_curves = {}
    for fov_key, (t_arr, means) in fov_curves.items():
        if means[0] > 0:
            norm_curves[fov_key] = (t_arr, means / means[0])

    # Compute mean normalized curve across FOVs on a common time grid
    all_t = set()
    for t_arr, _ in norm_curves.values():
        all_t.update(t_arr.tolist())
    common_t = np.array(sorted(all_t))

    # Interpolate each FOV to common grid and average
    interp_matrix = []
    for fov_key, (t_arr, norm_vals) in norm_curves.items():
        interp_vals = np.interp(common_t, t_arr, norm_vals, left=np.nan, right=np.nan)
        interp_matrix.append(interp_vals)
    interp_matrix = np.array(interp_matrix)  # (n_fovs, n_timepoints)
    mean_curve = np.nanmean(interp_matrix, axis=0)
    std_curve = np.nanstd(interp_matrix, axis=0)

    # Fit exponential decay: I(t) = a * exp(-t/tau) + c
    def _exp_decay(t, a, tau, c):
        return a * np.exp(-t / tau) + c

    valid_fit = ~np.isnan(mean_curve)
    t_fit = common_t[valid_fit]
    y_fit = mean_curve[valid_fit]

    half_life = np.nan
    pct_remaining = float(y_fit[-1] * 100) if len(y_fit) > 0 else np.nan
    fit_params = None

    if len(t_fit) > 3:
        try:
            p0 = [1.0, max(t_fit[-1], 1.0), 0.0]
            bounds = ([0, 1, -0.5], [2, t_fit[-1] * 100, 1.0])
            popt, _ = curve_fit(_exp_decay, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=5000)
            fit_params = {"a": popt[0], "tau": popt[1], "c": popt[2]}
            half_life = float(popt[1] * np.log(2))
        except (RuntimeError, ValueError) as e:
            print(f"  Exponential fit failed: {e}")

    print(f"\nBleach QC results ({channel_name}):")
    print(f"  FOVs analyzed: {len(norm_curves)}")
    print(f"  Signal remaining at last t: {pct_remaining:.1f}%")
    if not np.isnan(half_life):
        print(f"  Estimated half-life: {half_life:.1f} timepoints")
    if fit_params:
        print(f"  Fit: I(t) = {fit_params['a']:.3f} * exp(-t/{fit_params['tau']:.1f}) + {fit_params['c']:.3f}")

    # Save CSV: per-FOV raw and normalized curves
    rows = []
    for fov_key, (t_arr, means) in fov_curves.items():
        _, norm_vals = norm_curves.get(fov_key, (t_arr, means))
        for i, t in enumerate(t_arr):
            rows.append({
                "fov": "_".join(fov_key.split("/")),
                "t": int(t),
                "mean_intensity": means[i],
                "normalized": norm_vals[i],
            })
    bleach_df = pd.DataFrame(rows)
    bleach_csv = output_dir / "bleach_qc.csv"
    bleach_df.to_csv(bleach_csv, index=False)
    print(f"  Saved CSV to {bleach_csv}")

    # Save summary
    summary_rows = []
    for fov_key, (t_arr, means) in fov_curves.items():
        _, norm_vals = norm_curves.get(fov_key, (t_arr, means))
        summary_rows.append({
            "fov": "_".join(fov_key.split("/")),
            "t0_intensity": means[0],
            "t_last_intensity": means[-1],
            "pct_remaining": float(norm_vals[-1] * 100),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "bleach_qc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Plot
    fig = plot_bleach_qc(
        norm_curves=norm_curves,
        common_t=common_t, mean_curve=mean_curve, std_curve=std_curve,
        valid_fit=valid_fit, fit_params=fit_params, half_life=half_life,
        summary_rows=summary_rows, channel_name=channel_name,
    )
    plot_path = output_dir / "bleach_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    return {
        "half_life": half_life,
        "pct_remaining": pct_remaining,
        "fit_params": fit_params,
        "n_fovs": len(norm_curves),
    }


def compute_tilt_qc(
    im_lf_path: Path,
    output_plots_dir: Path,
    blank_frames: list[int] | None = None,
    max_offset: float = 5.0,
    grid_size: int = 3,
    **kwargs,
) -> dict:
    """Measure per-timepoint sample tilt via plane fit to variance grid at mid-Z.

    For each timepoint:
    1. Take the mid-Z slice (channel 0)
    2. Divide into grid_size × grid_size sub-regions
    3. Compute variance per sub-region
    4. Fit a plane: variance = a*x + b*y + c
    5. Report tilt_slope (magnitude of gradient) and tilt_range (max-min fitted values)

    Outlier detection: frames with tilt_slope > median + max_offset are flagged.

    Parameters
    ----------
    im_lf_path : Path
        Path to the label-free zarr FOV position.
    output_plots_dir : Path
        Directory to save CSV and plot.
    blank_frames : list of int or None
        Timepoints to skip (blank frames).
    max_offset : float
        Frames with tilt_slope > median + max_offset are flagged as outliers.
    grid_size : int
        Number of sub-regions per axis (default 3 → 3×3 = 9 sub-regions).
    """
    from dynacell.plotting import plot_tilt_qc

    with open_ome_zarr(im_lf_path) as ds:
        arr = ds.data.dask_array()
        T, C, Z, Y, X = arr.shape

    z_mid = Z // 2
    print(f"Tilt QC: grid={grid_size}x{grid_size}, T={T}, z_mid={z_mid}, YxX={Y}x{X}")

    blank_set = set(blank_frames) if blank_frames else set()

    # Sub-region boundaries
    y_edges = np.linspace(0, Y, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, X, grid_size + 1, dtype=int)

    tilt_slopes = np.full(T, np.nan, dtype=np.float64)
    tilt_ranges = np.full(T, np.nan, dtype=np.float64)
    grid_var_all = np.full((T, grid_size, grid_size), np.nan, dtype=np.float64)

    with open_ome_zarr(im_lf_path) as ds:
        arr = ds.data.dask_array()

        for t in tqdm(range(T), desc="Tilt QC"):
            if t in blank_set:
                continue

            slc = np.asarray(arr[t, 0, z_mid, :, :]).astype(np.float64)
            grid_var = np.full((grid_size, grid_size), np.nan, dtype=np.float64)

            for iy in range(grid_size):
                for ix in range(grid_size):
                    sub = slc[y_edges[iy]:y_edges[iy + 1], x_edges[ix]:x_edges[ix + 1]]
                    if sub.size < 32 * 32:
                        continue
                    if np.max(np.abs(sub)) < 1e-10:
                        continue
                    grid_var[iy, ix] = float(np.var(sub))

            grid_var_all[t] = grid_var
            valid = grid_var[~np.isnan(grid_var)]
            if len(valid) < 3:
                continue

            tilt_ranges[t] = float(np.max(valid) - np.min(valid))

            # Fit plane: variance = a*x + b*y + c
            rows_fit = []
            for iy in range(grid_size):
                for ix in range(grid_size):
                    if not np.isnan(grid_var[iy, ix]):
                        yc = (y_edges[iy] + y_edges[iy + 1]) / 2
                        xc = (x_edges[ix] + x_edges[ix + 1]) / 2
                        rows_fit.append([xc, yc, grid_var[iy, ix]])
            if len(rows_fit) >= 3:
                pts = np.array(rows_fit)
                A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
                coeffs, _, _, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
                slope_x, slope_y = coeffs[0], coeffs[1]
                tilt_slopes[t] = float(np.sqrt(slope_x**2 + slope_y**2))

    # Outlier detection on tilt_slope: median + max_offset
    valid_mask = ~np.isnan(tilt_slopes)
    valid_vals = tilt_slopes[valid_mask]
    if len(valid_vals) < 2:
        print("WARNING: too few valid frames for tilt QC")
        return {"tilt_slopes": tilt_slopes}

    med = float(np.median(valid_vals))
    threshold = med + max_offset

    is_outlier = np.zeros(T, dtype=int)
    is_outlier[valid_mask] = (tilt_slopes[valid_mask] > threshold).astype(int)

    n_outliers = int(is_outlier.sum())
    print(f"\nTilt QC results:")
    print(f"  Median tilt slope: {med:.6f}")
    print(f"  Threshold: median + {max_offset} = {threshold:.6f}")
    print(f"  Max tilt slope: {valid_vals.max():.6f}")
    print(f"  Outliers (>{threshold:.4f}): {n_outliers}")
    if n_outliers > 0:
        print(f"  Outlier timepoints: {np.where(is_outlier)[0].tolist()}")

    # Save CSV
    qc_df = pd.DataFrame({
        "t": np.arange(T),
        "tilt_slope": tilt_slopes,
        "tilt_range": tilt_ranges,
        "is_outlier": is_outlier,
    })
    qc_csv = output_plots_dir / "tilt_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved CSV to {qc_csv}")

    # Save example grid for diagnostic plot
    first_valid = np.where(valid_mask)[0]
    example_grid = grid_var_all[first_valid[0]] if len(first_valid) > 0 else None

    # Plot
    fig = plot_tilt_qc(
        tilt_ranges=tilt_slopes,
        tilt_slopes=tilt_slopes,
        blank_mask=~valid_mask,
        med=med,
        threshold=threshold,
        max_offset=max_offset,
        is_outlier=is_outlier,
        example_grid=example_grid,
        grid_size=grid_size,
    )
    plot_path = output_plots_dir / "tilt_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    return {"tilt_slopes": tilt_slopes}
