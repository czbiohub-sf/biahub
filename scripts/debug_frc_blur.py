"""Test FRC 2D as blur detector on known blurry frames.

For each timepoint: compute FRC resolution at each Z, take median across Z.
Compare across timepoints to find outliers (blurry frames).
Known blurry: A/1/001001 t=44, A/1/000001 t=44.
"""

import numpy as np
import time
from pathlib import Path
from iohub.ngff import open_ome_zarr
from cubic.metrics.frc.frc import calculate_frc
import matplotlib.pyplot as plt

# --- Config ---
root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
dataset = "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
ls_zarr = (
    root_path / dataset / "1-preprocess" / "light-sheet" / "raw" / "1-register"
    / f"{dataset}.zarr"
)
channel_name = "raw GFP EX488 EM525-45"

# FOVs with known blur at t=44
test_fovs = ["A/1/001001", "A/1/000001"]

# Sample Z planes (not all 105 — too slow for a test)
z_sample = list(range(0, 105, 10))  # every 10th Z → 11 planes


def frc_resolution_safe(img_2d: np.ndarray) -> float:
    """Compute FRC resolution, return NaN if it fails."""
    try:
        result = calculate_frc(img_2d.astype(np.float32))
        res = result.resolution["resolution"]
        if res is None or np.isnan(res):
            # Fallback: use mean of FRC curve as quality proxy
            return np.nan
        return float(res)
    except Exception:
        return np.nan


def frc_mean_curve(img_2d: np.ndarray) -> float:
    """Compute mean of FRC correlation curve — higher = sharper."""
    try:
        result = calculate_frc(img_2d.astype(np.float32))
        corr = np.array(result.correlation)
        if corr.ndim == 0 or len(corr) == 0:
            return np.nan
        return float(np.nanmean(corr))
    except Exception:
        return np.nan


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    for fov_key in test_fovs:
        print(f"\n{'='*60}")
        print(f"FOV: {fov_key}")
        print(f"{'='*60}")

        with open_ome_zarr(ls_zarr / fov_key) as ds:
            arr = ds.data.dask_array()
            T, C, Z, Y, X = arr.shape
            channels = list(ds.channel_names)
            c_idx = channels.index(channel_name)
            print(f"Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")

            # Test on a few timepoints: some normal + the known blurry t=44
            test_t = [0, 10, 20, 30, 40, 44, 50, 60, T - 1]
            test_t = [t for t in test_t if t < T]

            results = {}
            for t in test_t:
                t0 = time.time()
                resolutions = []
                for z in z_sample:
                    if z >= Z:
                        continue
                    img = np.asarray(arr[t, c_idx, z, :, :]).astype(np.float32)
                    res = frc_resolution_safe(img)
                    resolutions.append(res)

                resolutions = np.array(resolutions)
                valid = resolutions[~np.isnan(resolutions)]
                median_res = float(np.nanmedian(resolutions)) if len(valid) > 0 else np.nan
                elapsed = time.time() - t0

                results[t] = {
                    "median_resolution": median_res,
                    "n_valid": len(valid),
                    "n_total": len(resolutions),
                    "time_s": elapsed,
                }
                print(f"  t={t:3d}  median_frc_res={median_res:8.3f}  "
                      f"valid={len(valid)}/{len(resolutions)}  {elapsed:.1f}s")

            # Find outliers
            all_med = np.array([results[t]["median_resolution"] for t in test_t])
            valid_med = all_med[~np.isnan(all_med)]
            if len(valid_med) > 3:
                mean_val = np.nanmean(valid_med)
                std_val = np.nanstd(valid_med)
                print(f"\n  Stats: mean={mean_val:.3f}, std={std_val:.3f}")
                for t, m in zip(test_t, all_med):
                    if not np.isnan(m) and std_val > 0:
                        z_score = (m - mean_val) / std_val
                        flag = " *** OUTLIER" if abs(z_score) > 2 else ""
                        print(f"  t={t:3d}  z_score={z_score:+.2f}{flag}")
