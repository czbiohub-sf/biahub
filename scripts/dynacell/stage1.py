"""Stage 1: Per-FOV metadata computation for dynacell preprocessing."""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from iohub import open_ome_zarr
from waveorder.focus import focus_from_transverse_band

from dynacell.geometry import (
    NA_DET,
    LAMBDA_ILL,
    make_circular_mask,
    find_overlap_mask,
    find_inscribed_bbox,
)
from dynacell.plotting import plot_overlap, plot_bbox_over_time, plot_z_focus
from dynacell.qc import (
    compute_laplacian_qc,
    compute_entropy_qc,
    compute_hf_ratio_qc,
    compute_bleach_fov,
    compute_max_intensity_qc,
    compute_frc_qc,
    compute_fov_registration_qc,
    compute_tilt_qc,
)


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
    hf_blur_outliers: np.ndarray | None = None,
    manual_drop_frames: list[int] | None = None,
    bbox_outlier_frames: list[int] | None = None,
) -> dict:
    """Combine blank frames, z_focus outliers, HF blur outliers, bbox outliers, and manual drops.

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
    if hf_blur_outliers is not None:
        for t in hf_blur_outliers:
            reasons.setdefault(int(t), []).append("hf_blur")
    if manual_drop_frames is not None:
        for t in manual_drop_frames:
            reasons.setdefault(int(t), []).append("manual")
    if bbox_outlier_frames is not None:
        for t in bbox_outlier_frames:
            reasons.setdefault(int(t), []).append("bbox_outlier")

    drop_indices = np.array(sorted(reasons.keys()), dtype=int)
    keep_indices = np.array(sorted(set(range(T)) - set(drop_indices)), dtype=int)

    print(f"\n=== Drop list ===")
    print(f"Total timepoints: {T}")
    print(f"Blank frames ({len(blank_frames)}): {blank_frames}")
    print(f"Z focus outliers ({len(z_focus_outliers)}): {z_focus_outliers.tolist()}")
    if hf_blur_outliers is not None and len(hf_blur_outliers) > 0:
        print(f"HF blur ({len(hf_blur_outliers)}): {hf_blur_outliers.tolist()}")
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
    z_index: int | str | None = None,
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
    z_index : int, "mid", or None
        If an int, use this fixed z index for all timepoints.

        If None (default), compute z_focus per timepoint via
        focus_from_transverse_band.

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

        # Z focus: use fixed index, mid-Z, or compute from LF phase Z-stack
        if z_index is not None:
            Z_min = min(arr.shape[2] for arr in arrays)
            if z_index == "mid":
                z_f = Z_min // 2
            else:
                z_f = int(np.clip(z_index, 0, Z_min - 1))
        else:
            lf_phase_zyx = np.asarray(arrays[0][t, 0, :, :, :])
            z_f = focus_from_transverse_band(
                lf_phase_zyx,
                NA_det=NA_DET,
                lambda_ill=LAMBDA_ILL,
                pixel_size=pixel_size,
            )
            # Clamp z_f to min Z across all arrays (LF and LS may differ in Z)
            Z_min = min(arr.shape[2] for arr in arrays)
            z_f = int(np.clip(z_f, 0, Z_min - 1))
        z_focus.append(z_f)

        # Overlap from mid-Z slices (XY overlap is z-independent;
        # using z_focus here would break when z_focus is an outlier).
        overlap_slices = []
        for i, arr in enumerate(arrays):
            z_mid = arr.shape[2] // 2
            for c in range(arr.shape[1]):
                slc = np.asarray(arr[t, c, z_mid, :, :]).copy()
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

    # Combined bbox (bounding box of valid region)
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
    z_window: int | None = None,
    z_final: int = 64,
    z_index: int | str | None = None,
    DEBUG: bool = True,
    qc_thresholds: dict | None = None,
) -> dict:
    """Stage 1: Compute bbox, z_focus, blank frames, and drop list for one FOV.

    Uses a single pass over all timepoints to detect blank frames,
    compute the overlap bounding box, and find z_focus simultaneously.

    Parameters
    ----------
    z_index : int, "mid", or None
        If an int, use this fixed z index instead of focus finding.
        If "mid", use the mid-Z slice (Z // 2).
        If None (default), auto-detect z_focus per timepoint.

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
            z_index=z_index,
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

        if z_index is not None:
            # Fixed z index: no outlier detection needed
            z_label = f"mid (Z//2={z_focus[0]})" if z_index == "mid" else str(z_index)
            print(f"Using fixed z_index={z_label}, skipping z_focus outlier detection")
            z_focus_outliers_original = np.array([], dtype=int)
        else:
            blank_set = set(blank_frames)
            z_focus_valid = [z_focus[t] for t in range(T) if t not in blank_set]
            z_focus_stats = plot_z_focus(
                z_focus_valid,
                save_path=str(output_plots_dir / "z_focus.png"),
                n_std=n_std,
                z_window=z_window,
            )

            # Remap outlier indices back to original timepoint indices
            valid_t_indices = [t for t in range(T) if t not in blank_set]
            z_focus_outliers_original = np.array(
                [valid_t_indices[i] for i in z_focus_stats["t_outliers"]], dtype=int
            )

        # --- Laplacian blur QC on LS GFP channel (plot only, not used for dropping) ---
        blank_set = set(blank_frames)

        lap_n_std = 1.0  # default
        if qc_thresholds and "laplacian" in qc_thresholds:
            lap_n_std = qc_thresholds["laplacian"].get("n_std", 1.0)
        compute_laplacian_qc(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            channel_name="raw GFP EX488 EM525-45",
            n_std=lap_n_std,
            blank_frames=blank_frames,
        )

        # --- Entropy QC on LS GFP channel (reporting only, not used for dropping) ---
        ent_n_std = 2.5
        if qc_thresholds and "entropy" in qc_thresholds:
            ent_n_std = qc_thresholds["entropy"].get("n_std", 2.5)
        compute_entropy_qc(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            channel_name="raw GFP EX488 EM525-45",
            blank_frames=blank_frames,
            n_std=ent_n_std,
        )

        # --- HF ratio blur QC on LS GFP channel (reporting only) ---
        hf_n_std = 2.5
        if qc_thresholds and "hf_ratio" in qc_thresholds:
            hf_n_std = qc_thresholds["hf_ratio"].get("n_std", 2.5)
        compute_hf_ratio_qc(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            channel_name="raw GFP EX488 EM525-45",
            blank_frames=blank_frames,
            n_std=hf_n_std,
        )

        # --- FRC blur QC on LS GFP channel (reporting only) ---
        frc_n_std = 2.5
        if qc_thresholds and "frc" in qc_thresholds:
            frc_n_std = qc_thresholds["frc"].get("n_std", 2.5)
        compute_frc_qc(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            channel_name="raw GFP EX488 EM525-45",
            blank_frames=blank_frames,
            n_std=frc_n_std,
        )

        # --- Bleach QC on LS GFP channel (per-FOV) ---
        bleach_n_std = 3.0
        if qc_thresholds and "bleach" in qc_thresholds:
            bleach_n_std = qc_thresholds["bleach"].get("n_std", 3.0)
        compute_bleach_fov(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            bbox=(y_min, y_max, x_min, x_max),
            channel_name="raw GFP EX488 EM525-45",
            blank_frames=blank_frames,
            n_std=bleach_n_std,
        )

        # --- Max intensity QC on LS GFP channel (reporting only) ---
        mi_n_std = 2.5
        if qc_thresholds and "max_intensity" in qc_thresholds:
            mi_n_std = qc_thresholds["max_intensity"].get("n_std", 2.5)
        compute_max_intensity_qc(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            channel_name="raw GFP EX488 EM525-45",
            blank_frames=blank_frames,
            n_std=mi_n_std,
        )

        # --- Per-FOV registration QC (Pearson LF vs LS, reporting only) ---
        reg_n_std = 2.5
        if qc_thresholds and "fov_registration" in qc_thresholds:
            reg_n_std = qc_thresholds["fov_registration"].get("n_std", 2.5)
        compute_fov_registration_qc(
            im_lf_arr=im_lf_arr,
            im_ls_arr=im_ls_arr,
            z_focus=z_focus,
            output_plots_dir=output_plots_dir,
            lf_mask_radius=lf_mask_radius,
            blank_frames=blank_frames,
            n_std=reg_n_std,
        )

        # --- Drop list (blank frames + z_focus outliers only) ---
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
            slices = []
            for i, arr in enumerate(arrays):
                z_mid = arr.shape[2] // 2
                for c in range(arr.shape[1]):
                    slc = np.asarray(arr[t, c, z_mid, :, :]).copy()
                    if i == 0 and circ_mask is not None:
                        slc[~circ_mask] = 0
                    slices.append(slc)
            kept_mask &= find_overlap_mask(slices)

        # Use bounding box (not inscribed) -- a few zero pixels at circle
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


## ===== Two-phase stage 1: core + per-metric QC =====


def compute_fov_core(
    im_lf_path: Path,
    im_ls_path: Path,
    output_plots_dir: Path,
    fov: str,
    lf_mask_radius: float = 0.75,
    n_std: float = 2.5,
    z_window: int | None = None,
    z_index: int | str | None = None,
    manual_drop_frames: list[int] | None = None,
    DEBUG: bool = True,
    qc_thresholds: dict | None = None,
) -> dict:
    """Phase 1: Compute bbox, z_focus, blank frames, and drop list for one FOV.

    This is the core metadata needed before QC metrics can run.
    Saves z_focus.csv, drop_list.csv, fov_summary.csv to output_plots_dir.

    Parameters
    ----------
    manual_drop_frames : list of int or None
        Timepoints to drop from pre-annotations (added with reason "manual").
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
            z_index=z_index,
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

        if z_index is not None:
            z_label = f"mid (Z//2={z_focus[0]})" if z_index == "mid" else str(z_index)
            print(f"Using fixed z_index={z_label}, skipping z_focus outlier detection")
            z_focus_outliers_original = np.array([], dtype=int)
        else:
            blank_set = set(blank_frames)
            z_focus_valid = [z_focus[t] for t in range(T) if t not in blank_set]
            z_focus_stats = plot_z_focus(
                z_focus_valid,
                save_path=str(output_plots_dir / "z_focus.png"),
                n_std=n_std,
                z_window=z_window,
            )
            valid_t_indices = [t for t in range(T) if t not in blank_set]
            z_focus_outliers_original = np.array(
                [valid_t_indices[i] for i in z_focus_stats["t_outliers"]], dtype=int
            )

        # --- Detect per-t bbox area outliers (partial acquisitions) ---
        bbox_areas = np.array([
            (b[1] - b[0] + 1) * (b[3] - b[2] + 1) if b[1] > 0 else 0
            for b in per_t_bboxes
        ], dtype=float)
        # Only consider non-blank, non-zero area frames
        blank_set = set(blank_frames)
        valid_area_mask = np.array([
            t not in blank_set and bbox_areas[t] > 0 for t in range(T)
        ])
        bbox_outlier_frames = []
        if valid_area_mask.sum() > 2:
            valid_areas = bbox_areas[valid_area_mask]
            median_area = np.median(valid_areas)
            # Flag frames with area < 25% of median (severe partial acquisition)
            area_threshold = 0.25 * median_area
            for t in range(T):
                if valid_area_mask[t] and bbox_areas[t] < area_threshold:
                    bbox_outlier_frames.append(t)
            if bbox_outlier_frames:
                print(f"\nBbox area outliers (area < 25% of median={median_area:.0f}):")
                for t in bbox_outlier_frames:
                    print(f"  t={t}: area={bbox_areas[t]:.0f}, "
                          f"bbox=[{per_t_bboxes[t][0]}:{per_t_bboxes[t][1]}, "
                          f"{per_t_bboxes[t][2]}:{per_t_bboxes[t][3]}]")

        # --- Drop list (blank frames + z_focus outliers + manual drops + bbox outliers) ---
        if manual_drop_frames:
            print(f"Manual drops from pre-annotations: {manual_drop_frames}")
        drop_info = build_drop_list(
            blank_frames=blank_frames,
            z_focus_outliers=z_focus_outliers_original,
            T=T,
            save_path=str(output_plots_dir / "drop_list.csv"),
            manual_drop_frames=manual_drop_frames,
            bbox_outlier_frames=bbox_outlier_frames if bbox_outlier_frames else None,
        )

        # --- Recompute bbox using only kept frames ---
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
            slices = []
            for i, arr in enumerate(arrays):
                z_mid = arr.shape[2] // 2
                for c in range(arr.shape[1]):
                    slc = np.asarray(arr[t, c, z_mid, :, :]).copy()
                    if i == 0 and circ_mask is not None:
                        slc[~circ_mask] = 0
                    slices.append(slc)
            kept_mask &= find_overlap_mask(slices)

        kept_bbox = find_inscribed_bbox(kept_mask)
        if kept_bbox is not None:
            print(f"Recomputed bbox from {len(keep_indices)} kept frames "
                  f"(was Y=[{bbox[0]}:{bbox[1]+1}], X=[{bbox[2]}:{bbox[3]+1}])")
            y_min, y_max, x_min, x_max = kept_bbox
        else:
            print("WARNING: kept-frame mask has no valid overlap, using all-frame bbox")

        print(f"Final bbox: Y=[{y_min}:{y_max+1}], X=[{x_min}:{x_max+1}]")
        print(f"Final crop size: Y={y_max - y_min + 1}, X={x_max - x_min + 1}")

    # Save FOV summary metadata
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


def _read_core_metadata(output_plots_dir: Path) -> dict:
    """Read core metadata files written by compute_fov_core.

    Returns dict with z_focus, blank_frames, bbox, or None if files missing.
    """
    z_focus_csv = output_plots_dir / "z_focus.csv"
    drop_csv = output_plots_dir / "drop_list.csv"
    summary_csv = output_plots_dir / "fov_summary.csv"

    if not z_focus_csv.exists() or not summary_csv.exists():
        return None

    z_df = pd.read_csv(z_focus_csv)
    z_focus = z_df["z_focus"].tolist()

    blank_frames = []
    if drop_csv.exists() and drop_csv.stat().st_size > 0:
        drop_df = pd.read_csv(drop_csv)
        blank_rows = drop_df[drop_df["reason"].str.contains("blank", case=False, na=False)]
        blank_frames = blank_rows["t"].tolist()

    summary = pd.read_csv(summary_csv).to_dict("records")[0]
    import ast
    bbox_raw = summary["bbox"]
    if isinstance(bbox_raw, str):
        bbox = tuple(ast.literal_eval(bbox_raw))
    else:
        bbox = (bbox_raw,)

    return {
        "z_focus": z_focus,
        "blank_frames": blank_frames,
        "bbox": bbox,
    }


# Mapping of QC metric names to their functions and required arguments
QC_METRICS = [
    "laplacian",
    "entropy",
    "hf_ratio",
    "frc",
    "bleach",
    "max_intensity",
    "fov_registration",
    "tilt",
]


def run_fov_qc(
    metric: str,
    im_lf_path: Path,
    im_ls_path: Path,
    output_plots_dir: Path,
    lf_mask_radius: float = 0.75,
    z_final: int = 64,
    channel_name: str = "raw GFP EX488 EM525-45",
    n_std: float = 2.5,
    qc_thresholds: dict | None = None,
) -> str:
    """Phase 2: Run a single QC metric for one FOV.

    Reads core metadata (z_focus, blank_frames, bbox) from output_plots_dir,
    then dispatches to the appropriate QC function.

    Parameters
    ----------
    metric : str
        One of: "laplacian", "entropy", "hf_ratio", "frc", "bleach", "max_intensity", "fov_registration"
    """
    # Per-metric threshold: use qc_thresholds[metric]["n_std"] if available, else fallback to n_std
    metric_n_std = n_std
    if qc_thresholds and metric in qc_thresholds:
        metric_n_std = qc_thresholds[metric].get("n_std", n_std)

    core = _read_core_metadata(output_plots_dir)
    if core is None:
        print(f"ERROR: Core metadata not found in {output_plots_dir}")
        return f"error: no core metadata for {metric}"

    z_focus = core["z_focus"]
    blank_frames = core["blank_frames"]
    bbox = core["bbox"]

    print(f"Running QC metric: {metric}")
    print(f"  z_focus: {len(z_focus)} timepoints, blank_frames: {blank_frames}")

    if metric == "laplacian":
        compute_laplacian_qc(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            channel_name=channel_name,
            n_std=metric_n_std,
            blank_frames=blank_frames,
        )
    elif metric == "entropy":
        compute_entropy_qc(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            channel_name=channel_name,
            blank_frames=blank_frames,
            n_std=metric_n_std,
        )
    elif metric == "hf_ratio":
        compute_hf_ratio_qc(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            channel_name=channel_name,
            blank_frames=blank_frames,
            n_std=metric_n_std,
        )
    elif metric == "frc":
        compute_frc_qc(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            channel_name=channel_name,
            blank_frames=blank_frames,
            n_std=metric_n_std,
        )
    elif metric == "bleach":
        compute_bleach_fov(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            bbox=bbox,
            channel_name=channel_name,
            blank_frames=blank_frames,
            n_std=metric_n_std,
        )
    elif metric == "max_intensity":
        compute_max_intensity_qc(
            im_ls_path=im_ls_path,
            output_plots_dir=output_plots_dir,
            channel_name=channel_name,
            blank_frames=blank_frames,
            n_std=metric_n_std,
        )
    elif metric == "fov_registration":
        with open_ome_zarr(im_lf_path) as im_lf_ds, open_ome_zarr(im_ls_path) as im_ls_ds:
            compute_fov_registration_qc(
                im_lf_arr=im_lf_ds.data.dask_array(),
                im_ls_arr=im_ls_ds.data.dask_array(),
                z_focus=z_focus,
                output_plots_dir=output_plots_dir,
                lf_mask_radius=lf_mask_radius,
                blank_frames=blank_frames,
                n_std=metric_n_std,
            )
    elif metric == "tilt":
        compute_tilt_qc(
            im_lf_path=im_lf_path,
            output_plots_dir=output_plots_dir,
            blank_frames=blank_frames,
            n_std=metric_n_std,
        )
    else:
        raise ValueError(f"Unknown QC metric: {metric}. Must be one of {QC_METRICS}")

    print(f"Done: {metric}")
    return f"ok: {metric}"
