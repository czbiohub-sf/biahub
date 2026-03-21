"""Debug: test HF ratio with center crop and multiple Z planes.

Hypothesis: uneven LS illumination causes HF ratio variability between timepoints.
Fix 1: compute on center crop (avoid illumination gradient).
Fix 2: median across multiple Z planes (more robust than single z_focus).
Fix 3: both.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from iohub.ngff import open_ome_zarr
from tqdm import tqdm

root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
dataset = "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
ls_zarr = (
    root_path / dataset / "1-preprocess" / "light-sheet" / "raw" / "1-register"
    / f"{dataset}.zarr"
)
run_dir = root_path / dataset / "dynacell" / "run_20260320_150224"
channel_name = "raw GFP EX488 EM525-45"

test_fovs = {
    "A/1/000000": {"known_blur": [44], "note": "t=44 missed, t=58 false pos"},
    "B/1/000001": {"known_blur": [], "note": "t=50,56 false positives"},
    "A/1/001001": {"known_blur": [44], "note": "t=44 was caught in previous test"},
}


def _hf_energy_ratio(img_2d: np.ndarray, cutoff_fraction: float = 0.1) -> float:
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


def compute_local_z(series, local_window=3):
    blank = np.isnan(series)
    T = len(series)
    lz = np.zeros(T)
    for t in range(T):
        if blank[t]:
            continue
        nbrs = [
            series[t + d]
            for d in range(-local_window, local_window + 1)
            if d != 0 and 0 <= t + d < T and not blank[t + d]
        ]
        if len(nbrs) >= 4:
            m, s = np.mean(nbrs), np.std(nbrs)
            lz[t] = (series[t] - m) / s if s > 0 else 0
    return lz


if __name__ == "__main__":
    for fov_key, info in test_fovs.items():
        fov_name = "_".join(fov_key.split("/"))
        print(f"\n{'='*60}")
        print(f"FOV: {fov_key} — {info['note']}")

        # Read z_focus
        z_csv = run_dir / "plots" / fov_name / "z_focus.csv"
        z_focus = pd.read_csv(z_csv, index_col=0)["z_focus"].values.astype(int)

        # Read blank frames from drop list
        drop_csv = run_dir / "plots" / fov_name / "drop_list.csv"
        blank_set = set()
        if drop_csv.exists():
            drop_df = pd.read_csv(drop_csv)
            blank_t = drop_df[drop_df["reason"].str.contains("blank", case=False, na=False)]["t"]
            blank_set = set(int(t) for t in blank_t)

        with open_ome_zarr(ls_zarr / fov_key) as ds:
            arr = ds.data.dask_array()
            T, C, Z, Y, X = arr.shape
            c_idx = list(ds.channel_names).index(channel_name)
            print(f"Shape: T={T}, Z={Z}, Y={Y}, X={X}")

            # Define center crop (middle 50%)
            cy, cx = Y // 2, X // 2
            crop_h, crop_w = Y // 4, X // 4  # half of half = quarter on each side
            y0, y1 = cy - crop_h, cy + crop_h
            x0, x1 = cx - crop_w, cx + crop_w
            print(f"Center crop: Y=[{y0}:{y1}], X=[{x0}:{x1}] ({y1-y0}x{x1-x0})")

            # Methods to test:
            # 1. Current: single z_focus, full image
            # 2. Single z_focus, center crop
            # 3. Multi-Z (z_focus ± 5), full image, median
            # 4. Multi-Z, center crop, median
            z_range = 5

            hf_current = np.full(T, np.nan)
            hf_crop = np.full(T, np.nan)
            hf_multiz = np.full(T, np.nan)
            hf_multiz_crop = np.full(T, np.nan)

            for t in tqdm(range(T), desc="Computing HF ratio variants"):
                if t in blank_set:
                    continue
                zf = int(z_focus[t])

                # Method 1: current (single Z, full)
                img_full = np.asarray(arr[t, c_idx, zf, :, :]).astype(np.float64)
                if np.ptp(img_full) < 1e-6:
                    continue
                hf_current[t] = _hf_energy_ratio(img_full)

                # Method 2: single Z, center crop
                img_crop = img_full[y0:y1, x0:x1]
                hf_crop[t] = _hf_energy_ratio(img_crop)

                # Method 3 & 4: multi-Z
                z_lo = max(0, zf - z_range)
                z_hi = min(Z, zf + z_range + 1)
                hf_zs_full = []
                hf_zs_crop = []
                for z in range(z_lo, z_hi):
                    img_z = np.asarray(arr[t, c_idx, z, :, :]).astype(np.float64)
                    if np.ptp(img_z) < 1e-6:
                        continue
                    hf_zs_full.append(_hf_energy_ratio(img_z))
                    hf_zs_crop.append(_hf_energy_ratio(img_z[y0:y1, x0:x1]))
                if hf_zs_full:
                    hf_multiz[t] = np.median(hf_zs_full)
                if hf_zs_crop:
                    hf_multiz_crop[t] = np.median(hf_zs_crop)

        # Compute local z-scores for each method
        methods = {
            "current (1Z, full)": hf_current,
            "center crop (1Z)": hf_crop,
            f"multi-Z ±{z_range} (full)": hf_multiz,
            f"multi-Z ±{z_range} (crop)": hf_multiz_crop,
        }

        fig, axes = plt.subplots(len(methods), 2, figsize=(16, 4 * len(methods)), sharex=True)

        for i, (name, series) in enumerate(methods.items()):
            lz = compute_local_z(series)
            outliers = np.where((lz < -4.0) & ~np.isnan(series))[0]
            valid = ~np.isnan(series)

            axes[i, 0].plot(np.arange(T)[valid], series[valid], ".-", alpha=0.7, markersize=3)
            axes[i, 0].set_ylabel("HF ratio")
            axes[i, 0].set_title(f"{name}")

            axes[i, 1].plot(np.arange(T)[valid], lz[valid], ".-", alpha=0.7, color="purple",
                            markersize=3)
            axes[i, 1].axhline(-4.0, color="red", ls="--", alpha=0.7)
            for t in outliers:
                axes[i, 1].axvline(t, color="red", alpha=0.3)
            for t in info["known_blur"]:
                axes[i, 1].axvline(t, color="green", alpha=0.5, ls=":", lw=2)
                if not np.isnan(lz[t]):
                    axes[i, 1].annotate(f"t={t}\nlz={lz[t]:+.1f}",
                                        (t, lz[t]), fontsize=7, color="green")
            axes[i, 1].set_ylabel("Local z-score")
            axes[i, 1].set_title(f"Outliers: {outliers.tolist()}")

            # Print key values
            for t in info["known_blur"]:
                print(f"  {name:30s}  t={t} lz={lz[t]:+.2f}  "
                      f"hf={series[t]:.6f}" if not np.isnan(series[t]) else f"  {name:30s}  t={t} NaN")
            print(f"  {name:30s}  outliers={outliers.tolist()}")

        axes[-1, 0].set_xlabel("Timepoint")
        axes[-1, 1].set_xlabel("Timepoint")
        fig.suptitle(f"{fov_key} — HF ratio method comparison", fontsize=14)
        plt.tight_layout()
        out = Path(f"/hpc/mydata/taylla.theodoro/repo/biahub_dev/scripts/hf_methods_{fov_name}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")
