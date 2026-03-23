"""QC functions for dynacell preprocessing (beads registration, Laplacian, entropy, dust)."""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from iohub import open_ome_zarr

from dynacell_geometry import make_circular_mask
from dynacell_plotting import (
    STYLE as _STYLE,
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

            # Handle NaN values
            nan_mask = np.isnan(lf_crop) | np.isnan(ls_crop)
            if nan_mask.all():
                pearson_corrs[t] = np.nan
                pcc_shifts_y[t] = np.nan
                pcc_shifts_x[t] = np.nan
                pcc_errors[t] = np.nan
                continue
            lf_clean = np.where(nan_mask, 0.0, lf_crop)
            ls_clean = np.where(nan_mask, 0.0, ls_crop)

            # Pearson correlation (on non-NaN pixels only)
            valid = ~nan_mask
            lf_valid = lf_crop[valid]
            ls_valid = ls_crop[valid]
            lf_centered = lf_valid - lf_valid.mean()
            ls_centered = ls_valid - ls_valid.mean()
            denom = np.sqrt(np.sum(lf_centered**2) * np.sum(ls_centered**2))
            if denom > 0:
                pearson_corrs[t] = np.sum(lf_centered * ls_centered) / denom
            else:
                pearson_corrs[t] = np.nan

            # Phase cross-correlation (residual shift after registration)
            shift, error, _phasediff = pcc_skimage(
                lf_clean, ls_clean, upsample_factor=10
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
) -> dict:
    """Compute per-timepoint Pearson correlation between LF and LS phase channels.

    This is a lightweight registration QC that runs on every FOV (not just beads).
    For each timepoint, correlates LF channel 0 and LS channel 0 at z_focus
    within the circular mask. Results are saved to CSV and plot for reporting only
    (not used for the drop list). Blank frames are set to NaN.
    """
    T = im_lf_arr.shape[0]
    pearson_corrs = np.full(T, np.nan, dtype=np.float64)
    blank_set = set(blank_frames) if blank_frames else set()

    Y_common = min(im_lf_arr.shape[-2], im_ls_arr.shape[-2])
    X_common = min(im_lf_arr.shape[-1], im_ls_arr.shape[-1])
    mask = None
    if lf_mask_radius is not None:
        circ = make_circular_mask(im_lf_arr.shape[-2], im_lf_arr.shape[-1], lf_mask_radius)
        mask = circ[:Y_common, :X_common]

    for t in tqdm(range(T), desc="Registration QC (Pearson)"):
        if t in blank_set:
            continue
        z_f = int(z_focus[t])
        lf_slice = np.asarray(im_lf_arr[t, 0, z_f, :Y_common, :X_common]).astype(np.float64)
        ls_slice = np.asarray(im_ls_arr[t, 0, z_f, :Y_common, :X_common]).astype(np.float64)

        if mask is not None:
            lf_flat = lf_slice[mask].ravel()
            ls_flat = ls_slice[mask].ravel()
        else:
            lf_flat = lf_slice.ravel()
            ls_flat = ls_slice.ravel()

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
        is_outlier[valid_mask] = (local_z[valid_mask] < -2.5).astype(int)

    n_outliers = int(is_outlier.sum())
    print(f"  FOV registration: mean={mu:.4f}, std={sigma:.4f}, outliers={n_outliers} (reporting only)")

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

    # Plot (valid frames only, keep time index)
    valid_idx = np.where(valid_mask)[0]
    fig, ax = plt.subplots(figsize=_STYLE["fig_single"])
    ax.plot(valid_idx, pearson_corrs[valid_idx], ".-",
            markersize=_STYLE["marker_size"], linewidth=_STYLE["lw_data"])
    ax.set_xlabel("Timepoint", fontsize=_STYLE["fs_label"])
    ax.set_ylabel("Pearson correlation (LF vs LS)", fontsize=_STYLE["fs_label"])
    title = f"Per-timepoint LF–LS registration (Phase ch 0)"
    if blank_set:
        title += f" (excl. {len(blank_set)} blank)"
    ax.set_title(title, fontsize=_STYLE["fs_title"])
    ax.axhline(mu, color=_STYLE["c_mean"], linestyle="--", linewidth=_STYLE["lw_ref"],
               label=f"mean={mu:.4f}")
    if n_outliers > 0:
        outlier_idx = np.where(is_outlier)[0]
        ax.scatter(outlier_idx, pearson_corrs[outlier_idx], color=_STYLE["c_outlier"],
                   s=_STYLE["scatter_size"], zorder=5, label=f"outliers ({n_outliers})")
    ax.legend(fontsize=_STYLE["fs_legend"])
    ax.set_xlim(0, T - 1)
    fov_reg_lim = _STYLE["ylim"].get("fov_reg_pearson")
    if fov_reg_lim is not None:
        ax.set_ylim(fov_reg_lim)
    ax.tick_params(labelsize=_STYLE["fs_tick"])
    fig.tight_layout()
    plot_path = output_plots_dir / "fov_registration_qc.png"
    fig.savefig(plot_path, dpi=_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)

    return {"pearson_corrs": pearson_corrs, "outliers": np.where(is_outlier)[0]}


def compute_laplacian_qc(
    im_ls_path: Path,
    output_plots_dir: Path,
    z_focus: list[int],
    channel_name: str = "raw GFP EX488 EM525-45",
    z_final: int = 64,
    roi_half: int = 300,
    n_std: float = 2.0,
    blank_frames: list[int] | None = None,
) -> dict:
    """Compute per-timepoint 3D Laplacian variance to detect blurry frames.

    For each timepoint, uses the LF z_focus to extract a sub-volume
    (1/3 below, 2/3 above z_focus -- same asymmetric window as crop_fov)
    and computes the variance of the 3D Laplacian. Blurry frames have
    lower variance because high-frequency content is suppressed.
    Blank frames are set to NaN and excluded from statistics.
    """
    from scipy.ndimage import laplace

    z_below = z_final // 3
    z_above = z_final - z_below - 1
    blank_set = set(blank_frames) if blank_frames else set()

    with open_ome_zarr(im_ls_path) as ds:
        arr = ds.data.dask_array()
        T, C, Z, Y, X = arr.shape
        channels = list(ds.channel_names)

        if channel_name not in channels:
            print(f"WARNING: channel '{channel_name}' not found in {channels}")
            return {"lap3d_vars": np.full(T, np.nan), "outliers": np.array([], dtype=int)}

        c_idx = channels.index(channel_name)
        y_c, x_c = Y // 2, X // 2

        y_start = max(0, y_c - roi_half)
        y_end = min(Y, y_c + roi_half)
        x_start = max(0, x_c - roi_half)
        x_end = min(X, x_c + roi_half)

        print(f"Laplacian QC: channel='{channel_name}' (c={c_idx}), "
              f"z_final={z_final} (1/3 below={z_below}, 2/3 above={z_above}), "
              f"Y=[{y_start}:{y_end}], X=[{x_start}:{x_end}]")

        lap3d_vars = np.full(T, np.nan, dtype=np.float64)
        max_ints = np.full(T, np.nan, dtype=np.float64)

        for t in tqdm(range(T), desc="Laplacian QC"):
            if t in blank_set:
                continue
            z_f = int(z_focus[t])
            z_start_vol = max(0, z_f - z_below)
            z_end_vol = min(Z, z_f + z_above + 1)

            vol = np.asarray(
                arr[t, c_idx, z_start_vol:z_end_vol, y_start:y_end, x_start:x_end]
            ).astype(np.float64)
            max_ints[t] = float(np.max(vol))
            if max_ints[t] < 1e-6:
                continue

            lap3d = laplace(vol)
            lap3d_vars[t] = float(np.var(lap3d))

    # Stats on non-blank frames only
    valid_mask = ~np.isnan(lap3d_vars)
    if valid_mask.sum() == 0:
        print("WARNING: All frames blank, no Laplacian QC computed")
        return {"lap3d_vars": lap3d_vars, "outliers": np.array([], dtype=int)}

    valid_vals = lap3d_vars[valid_mask]
    mu = float(np.mean(valid_vals))
    sigma = float(np.std(valid_vals))
    lower = mu - n_std * sigma

    outliers = np.where((lap3d_vars < lower) & valid_mask)[0]

    print(f"\nLaplacian QC results ({channel_name}):")
    print(f"  3D Laplacian variance: mean={mu:.2f}, std={sigma:.2f}")
    print(f"  Threshold (mean - {n_std}*std): {lower:.2f}")
    print(f"  Outliers (blurry frames): {len(outliers)}")
    if len(outliers) > 0:
        for t in outliers:
            zscore = (lap3d_vars[t] - mu) / sigma if sigma > 0 else 0
            print(f"    t={t}: lap3d_var={lap3d_vars[t]:.2f}, "
                  f"z_focus={z_focus[t]}, z-score={zscore:.2f}")

    # Save CSV
    qc_df = pd.DataFrame({
        "t": np.arange(T),
        "lap3d_var": lap3d_vars,
        "max_intensity": max_ints,
    })
    qc_csv = output_plots_dir / "laplacian_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved CSV to {qc_csv}")

    # Plot
    fig = plot_laplacian_qc(
        lap3d_vars=lap3d_vars, max_ints=max_ints,
        mu=mu, sigma=sigma, lower=lower, n_std=n_std,
        outliers=outliers, channel_name=channel_name,
    )
    plot_path = output_plots_dir / "laplacian_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    return {
        "lap3d_vars": lap3d_vars,
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
    local_z_threshold: float = 2.5,
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
            f"full volume Z={Z}, bins={n_bins}"
        )

        entropies = np.full(T, np.nan, dtype=np.float64)
        for t in tqdm(range(T), desc="Entropy QC"):
            if t in blank_set:
                continue
            vol = np.asarray(arr[t, c_idx]).astype(np.float32)
            if np.ptp(vol) < 1.0:
                continue
            entropies[t] = _entropy(vol, n_bins)

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
    outlier_mask = global_outlier & (np.abs(local_z) > local_z_threshold) & ~blank
    outliers = np.where(outlier_mask)[0]

    stats = {
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "global_lower": global_lower,
        "global_upper": global_upper,
        "local_z_threshold": local_z_threshold,
        "iqr_factor": iqr_factor,
    }

    print(f"\nEntropy QC results ({channel_name}):")
    print(f"  Median={med:.4f}, Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
    print(f"  Global fence: [{global_lower:.4f}, {global_upper:.4f}]")
    print(f"  Global outliers (before local filter): {global_outlier.sum()}")
    print(f"  Local z-score threshold: {local_z_threshold}")
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
        outliers=outliers, med=med, local_z_threshold=local_z_threshold,
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
    z_focus: list[int],
    channel_name: str = "raw GFP EX488 EM525-45",
    cutoff_fraction: float = 0.1,
    z_range: int = 5,
    local_window: int = 3,
    hf_z_threshold: float = 3.0,
    ent_z_threshold: float = 2.0,
    blank_frames: list[int] | None = None,
) -> dict:
    """Detect blurry LS frames using combined HF ratio + entropy.

    For each timepoint, computes the median HF energy ratio across
    z_focus ± z_range Z planes (more robust than single Z — blur affects
    all Z planes while biological dynamics vary per Z).

    Outlier detection combines HF ratio with entropy: real optical blur
    causes HF ratio to DROP and entropy to RISE simultaneously. Biological
    dynamics (cell division, infection) affect HF ratio but not entropy.

    A frame is flagged as blurry only if BOTH:
      - HF local z-score < -hf_z_threshold  (HF ratio dropped)
      - Entropy local z-score > +ent_z_threshold  (entropy rose)

    Entropy data is read from entropy_qc.csv (must be computed first).

    Parameters
    ----------
    im_ls_path : Path
        Path to the LS zarr FOV position.
    output_plots_dir : Path
        Directory for saving plots and CSVs.
    z_focus : list of int
        Per-timepoint z_focus indices.
    channel_name : str
        Channel to measure blur on.
    cutoff_fraction : float
        Fraction of max frequency for the high/low boundary (default 0.1).
    z_range : int
        Number of Z planes above and below z_focus to include (default 5).
    local_window : int
        Half-window for local z-score comparison.
    hf_z_threshold : float
        HF ratio local z-score threshold. Frames with local_z < -threshold
        are candidates for blur (default 3.0).
    ent_z_threshold : float
        Entropy local z-score threshold. Candidates are confirmed as blurry
        only if entropy local_z > +threshold (default 2.0).
    blank_frames : list of int or None
        Timepoints to skip (blank frames).

    Returns
    -------
    dict with 'hf_ratios', 'outliers', 'stats'.
    """
    from dynacell_plotting import plot_hf_ratio_qc

    # --- Read entropy local z-scores (computed before HF ratio QC) ---
    ent_csv = output_plots_dir / "entropy_qc.csv"
    ent_local_z = None
    if ent_csv.exists():
        ent_df = pd.read_csv(ent_csv)
        ent_local_z = ent_df["local_z"].values.astype(float)
        print(f"HF ratio QC: loaded entropy local_z from {ent_csv}")
    else:
        print(f"WARNING: {ent_csv} not found — falling back to HF-only detection")

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
            f"cutoff={cutoff_fraction}, z_range=±{z_range}"
        )

        blank_set = set(blank_frames) if blank_frames else set()
        hf_ratios = np.full(T, np.nan, dtype=np.float64)
        for t in tqdm(range(T), desc="HF ratio QC (multi-Z)"):
            if t in blank_set:
                continue
            z_f = int(z_focus[t])
            z_lo = max(0, z_f - z_range)
            z_hi = min(Z, z_f + z_range + 1)
            hf_zs = []
            for z in range(z_lo, z_hi):
                img = np.asarray(arr[t, c_idx, z, :, :]).astype(np.float64)
                if np.ptp(img) < 1e-6:
                    continue
                hf_zs.append(_hf_energy_ratio(img, cutoff_fraction))
            if hf_zs:
                hf_ratios[t] = np.median(hf_zs)

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

    # --- Combined outlier detection ---
    hf_flag = (hf_local_z < -hf_z_threshold) & ~blank

    if ent_local_z is not None and len(ent_local_z) == T:
        # Combined rule: HF drops AND entropy rises → real blur
        ent_flag = ent_local_z > +ent_z_threshold
        outlier_mask = hf_flag & ent_flag
        used_combined = True
    else:
        # Fallback: HF-only (less reliable but better than nothing)
        outlier_mask = hf_flag
        used_combined = False

    outliers = np.where(outlier_mask)[0]

    stats = {
        "median": med,
        "hf_z_threshold": hf_z_threshold,
        "ent_z_threshold": ent_z_threshold,
        "cutoff_fraction": cutoff_fraction,
        "z_range": z_range,
        "used_combined": used_combined,
    }

    method = "combined HF+entropy" if used_combined else "HF-only"
    print(f"\nHF ratio QC results ({channel_name}, {method}):")
    print(f"  Median={med:.6f}, multi-Z ±{z_range}")
    print(f"  HF threshold: local_z < -{hf_z_threshold}")
    if used_combined:
        print(f"  Entropy threshold: local_z > +{ent_z_threshold}")
        hf_only_count = int(hf_flag.sum())
        print(f"  HF-only candidates: {hf_only_count}")
    print(f"  Outliers (blurry): {len(outliers)}")
    for t in outliers:
        ent_str = f", ent_lz={ent_local_z[t]:+.2f}" if ent_local_z is not None else ""
        print(f"    t={t}: hf_ratio={hf_ratios[t]:.6f}, "
              f"hf_lz={hf_local_z[t]:+.2f}{ent_str}")

    # Save CSV
    csv_data = {
        "t": np.arange(T),
        "hf_ratio": hf_ratios,
        "hf_local_z": hf_local_z,
        "is_outlier": outlier_mask.astype(int),
    }
    if ent_local_z is not None and len(ent_local_z) == T:
        csv_data["ent_local_z"] = ent_local_z
    qc_df = pd.DataFrame(csv_data)
    qc_csv = output_plots_dir / "hf_ratio_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"  Saved CSV to {qc_csv}")

    # Plot
    fig = plot_hf_ratio_qc(
        hf_ratios=hf_ratios, hf_local_z=hf_local_z, blank_mask=blank,
        outliers=outliers, med=med,
        hf_z_threshold=hf_z_threshold, ent_z_threshold=ent_z_threshold,
        channel_name=channel_name, z_range=z_range,
        ent_local_z=ent_local_z,
    )
    plot_path = output_plots_dir / "hf_ratio_qc.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")

    return {"hf_ratios": hf_ratios, "outliers": outliers, "stats": stats}


def compute_bleach_fov(
    im_ls_path: Path,
    output_plots_dir: Path,
    z_focus: list[int],
    bbox: tuple[int, int, int, int],
    channel_name: str = "raw GFP EX488 EM525-45",
    blank_frames: list[int] | None = None,
) -> dict:
    """Measure per-timepoint mean GFP intensity within the crop bbox at z_focus.

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
            z_f = min(int(z_focus[t]), Z - 1)
            slc = np.asarray(
                arr[t, c_idx, z_f, y_min:y_max + 1, x_min:x_max + 1]
            ).astype(np.float64)
            means[t] = float(np.mean(slc))

    # Normalize to first valid timepoint
    first_valid = np.where(~np.isnan(means))[0]
    normalized = np.full(T, np.nan, dtype=np.float64)
    if len(first_valid) > 0 and means[first_valid[0]] > 0:
        normalized = means / means[first_valid[0]]

    # Save CSV
    fov_df = pd.DataFrame({
        "t": np.arange(T),
        "mean_intensity": means,
        "normalized": normalized,
    })
    fov_df.to_csv(output_plots_dir / "bleach_qc.csv", index=False)

    # Per-FOV plot
    valid_idx = np.where(~np.isnan(normalized))[0]
    fig, ax = plt.subplots(figsize=_STYLE["fig_single"])
    ax.plot(valid_idx, normalized[valid_idx], ".-",
            markersize=_STYLE["marker_size"], linewidth=_STYLE["lw_data"])
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    pct_rem = float(normalized[valid_idx[-1]] * 100) if len(valid_idx) > 0 else 0
    ax.set_xlabel("Timepoint", fontsize=_STYLE["fs_label"])
    ax.set_ylabel("Normalized intensity", fontsize=_STYLE["fs_label"])
    fov_name = output_plots_dir.name
    ax.set_title(f"Bleach QC | {fov_name} | {pct_rem:.1f}% remaining", fontsize=_STYLE["fs_title"])
    bleach_lim = _STYLE["ylim"].get("bleach_norm")
    if bleach_lim is not None:
        ax.set_ylim(bleach_lim)
    else:
        ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=_STYLE["fs_tick"])
    fig.tight_layout()
    fig.savefig(output_plots_dir / "bleach_qc.png", dpi=_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)

    print(f"  Bleach QC: {pct_rem:.1f}% remaining at last t")
    return {"mean_intensities": means, "normalized": normalized}


def _frc_mean_corr(img_2d: np.ndarray) -> float:
    """Compute mean of FRC correlation curve — higher = sharper."""
    try:
        from cubic.metrics.frc.frc import calculate_frc
        result = calculate_frc(img_2d.astype(np.float32))
        corr = np.array(result.correlation["correlation"])
        if corr.ndim == 0 or len(corr) == 0:
            return np.nan
        return float(np.nanmean(corr))
    except Exception:
        return np.nan


def compute_frc_qc(
    im_ls_path: Path,
    output_plots_dir: Path,
    z_focus: list[int],
    channel_name: str = "raw GFP EX488 EM525-45",
    z_range: int = 5,
    blank_frames: list[int] | None = None,
) -> dict:
    """Measure per-timepoint FRC mean correlation as a blur/quality indicator.

    For each timepoint, computes the FRC curve on z_focus ± z_range Z planes
    and takes the median of the mean correlation values. Lower FRC correlation
    indicates blur or loss of high-frequency content.

    Reporting only — not used for frame dropping or outlier detection.

    Parameters
    ----------
    im_ls_path : Path
        Path to the LS zarr FOV position.
    output_plots_dir : Path
        Directory for saving plots and CSVs.
    z_focus : list of int
        Per-timepoint z_focus indices.
    channel_name : str
        Channel to measure.
    z_range : int
        Number of Z planes above and below z_focus to include.
    blank_frames : list of int or None
        Timepoints to skip.

    Returns
    -------
    dict with 'frc_values' and 'stats'.
    """
    from dynacell_plotting import plot_frc_qc

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
            f"z_range=±{z_range}"
        )

        blank_set = set(blank_frames) if blank_frames else set()
        frc_values = np.full(T, np.nan, dtype=np.float64)
        for t in tqdm(range(T), desc="FRC QC (multi-Z)"):
            if t in blank_set:
                continue
            z_f = int(z_focus[t])
            z_lo = max(0, z_f - z_range)
            z_hi = min(Z, z_f + z_range + 1)
            frc_zs = []
            for z in range(z_lo, z_hi):
                img = np.asarray(arr[t, c_idx, z, :, :]).astype(np.float32)
                if np.ptp(img) < 1e-6:
                    continue
                frc_zs.append(_frc_mean_corr(img))
            valid_frc = [v for v in frc_zs if not np.isnan(v)]
            if valid_frc:
                frc_values[t] = np.median(valid_frc)

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
        is_outlier[~blank] = (local_z[~blank] < -2.5).astype(int)

    n_outliers = int(is_outlier.sum())

    stats = {
        "median": med,
        "mean": mu,
        "std": sigma,
        "z_range": z_range,
        "n_outliers": n_outliers,
    }

    print(f"\nFRC QC results ({channel_name}):")
    print(f"  Median={med:.6f}, Mean={mu:.6f}, Std={sigma:.6f}, multi-Z ±{z_range}")
    print(f"  Outliers (z < -2.5, reporting only): {n_outliers}")

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

    # Plot
    fig = plot_frc_qc(
        frc_values=frc_values, blank_mask=blank, med=med,
        channel_name=channel_name, z_range=z_range,
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
