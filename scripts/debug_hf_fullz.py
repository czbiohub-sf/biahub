"""Test HF ratio with median across ALL Z planes (full stack) vs current single-Z."""

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
    "A/1/001001": {"known_blur": [44], "note": "t=44 caught"},
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

        # Read blank frames
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

            # Sample Z planes evenly across full stack
            z_all = np.linspace(0, Z - 1, 20, dtype=int)  # 20 evenly-spaced Z
            print(f"Sampling {len(z_all)} Z planes: {z_all.tolist()}")

            hf_1z = np.full(T, np.nan)
            hf_fullz = np.full(T, np.nan)

            for t in tqdm(range(T), desc="Computing HF ratio"):
                if t in blank_set:
                    continue
                zf = int(z_focus[t])

                # Single Z (current)
                img = np.asarray(arr[t, c_idx, zf, :, :]).astype(np.float64)
                if np.ptp(img) < 1e-6:
                    continue
                hf_1z[t] = _hf_energy_ratio(img)

                # Full Z stack (sampled)
                hf_zs = []
                for z in z_all:
                    img_z = np.asarray(arr[t, c_idx, z, :, :]).astype(np.float64)
                    if np.ptp(img_z) < 1e-6:
                        continue
                    hf_zs.append(_hf_energy_ratio(img_z))
                if hf_zs:
                    hf_fullz[t] = np.median(hf_zs)

        # Local z-scores
        lz_1z = compute_local_z(hf_1z)
        lz_fullz = compute_local_z(hf_fullz)

        out_1z = np.where((lz_1z < -4.0) & ~np.isnan(hf_1z))[0]
        out_fullz = np.where((lz_fullz < -4.0) & ~np.isnan(hf_fullz))[0]

        print(f"  Current (1Z):     outliers={out_1z.tolist()}")
        print(f"  Full-Z (20 samp): outliers={out_fullz.tolist()}")
        for t in info["known_blur"]:
            print(f"  Known blur t={t}: 1Z lz={lz_1z[t]:+.2f}, fullZ lz={lz_fullz[t]:+.2f}")

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
        valid_1z = ~np.isnan(hf_1z)
        valid_fz = ~np.isnan(hf_fullz)

        axes[0, 0].plot(np.arange(T)[valid_1z], hf_1z[valid_1z], ".-", alpha=0.7, ms=3)
        axes[0, 0].set_ylabel("HF ratio")
        axes[0, 0].set_title(f"Current (1Z at z_focus) — outliers: {out_1z.tolist()}")

        axes[1, 0].plot(np.arange(T)[valid_1z], lz_1z[valid_1z], ".-", alpha=0.7, color="purple", ms=3)
        axes[1, 0].axhline(-4.0, color="red", ls="--", alpha=0.7)
        for t in out_1z:
            axes[1, 0].axvline(t, color="red", alpha=0.3)
        for t in info["known_blur"]:
            axes[1, 0].axvline(t, color="green", alpha=0.5, ls=":", lw=2)
        axes[1, 0].set_ylabel("Local z-score")
        axes[1, 0].set_xlabel("Timepoint")

        axes[0, 1].plot(np.arange(T)[valid_fz], hf_fullz[valid_fz], ".-", alpha=0.7, ms=3, color="tab:orange")
        axes[0, 1].set_ylabel("HF ratio")
        axes[0, 1].set_title(f"Full-Z (median of 20 Z) — outliers: {out_fullz.tolist()}")

        axes[1, 1].plot(np.arange(T)[valid_fz], lz_fullz[valid_fz], ".-", alpha=0.7, color="purple", ms=3)
        axes[1, 1].axhline(-4.0, color="red", ls="--", alpha=0.7)
        for t in out_fullz:
            axes[1, 1].axvline(t, color="red", alpha=0.3)
        for t in info["known_blur"]:
            axes[1, 1].axvline(t, color="green", alpha=0.5, ls=":", lw=2)
        axes[1, 1].set_ylabel("Local z-score")
        axes[1, 1].set_xlabel("Timepoint")

        fig.suptitle(f"{fov_key} — 1Z vs full-Z HF ratio", fontsize=14)
        plt.tight_layout()
        out_path = Path(f"/hpc/mydata/taylla.theodoro/repo/biahub_dev/scripts/hf_fullz_{fov_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
