"""Test combined HF ratio (multi-Z ±5) + entropy for blur detection.

Rule: flag frame if HF local_z < -hf_thresh AND entropy local_z > +ent_thresh
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
    "A/1/000000": {"known_blur": [44], "false_pos": [58]},
    "B/1/000001": {"known_blur": [], "false_pos": [50, 56]},
    "A/1/001001": {"known_blur": [44], "false_pos": []},
}


def _hf_energy_ratio(img_2d, cutoff_fraction=0.1):
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
    # Threshold combos to test
    combos = [
        (3.0, 2.0),
        (3.0, 2.5),
        (3.5, 2.0),
        (3.5, 2.5),
        (4.0, 2.0),
    ]

    all_fov_results = {}

    for fov_key, info in test_fovs.items():
        fov_name = "_".join(fov_key.split("/"))
        print(f"\n{'='*60}")
        print(f"FOV: {fov_key}")

        # Read existing entropy data
        ent_df = pd.read_csv(run_dir / "plots" / fov_name / "entropy_qc.csv")
        entropy = ent_df["entropy"].values.astype(float)
        ent_lz = compute_local_z(entropy)

        # Read z_focus and blank frames
        z_csv = run_dir / "plots" / fov_name / "z_focus.csv"
        z_focus = pd.read_csv(z_csv, index_col=0)["z_focus"].values.astype(int)
        drop_csv = run_dir / "plots" / fov_name / "drop_list.csv"
        blank_set = set()
        if drop_csv.exists():
            drop_df = pd.read_csv(drop_csv)
            blank_t = drop_df[drop_df["reason"].str.contains("blank", case=False, na=False)]["t"]
            blank_set = set(int(t) for t in blank_t)

        # Compute multi-Z ±5 HF ratio
        z_range = 5
        with open_ome_zarr(ls_zarr / fov_key) as ds:
            arr = ds.data.dask_array()
            T, C, Z, Y, X = arr.shape
            c_idx = list(ds.channel_names).index(channel_name)

            hf_multiz = np.full(T, np.nan)
            for t in tqdm(range(T), desc=f"HF multi-Z {fov_name}"):
                if t in blank_set:
                    continue
                zf = int(z_focus[t])
                z_lo = max(0, zf - z_range)
                z_hi = min(Z, zf + z_range + 1)
                hf_zs = []
                for z in range(z_lo, z_hi):
                    img = np.asarray(arr[t, c_idx, z, :, :]).astype(np.float64)
                    if np.ptp(img) < 1e-6:
                        continue
                    hf_zs.append(_hf_energy_ratio(img))
                if hf_zs:
                    hf_multiz[t] = np.median(hf_zs)

        hf_lz = compute_local_z(hf_multiz)
        blank = np.isnan(hf_multiz)

        # Test all threshold combos
        print(f"\n  {'combo':>12s}  {'outliers':>30s}  catches_blur  no_fp")
        for hf_th, ent_th in combos:
            combined_mask = (hf_lz < -hf_th) & (ent_lz > +ent_th) & ~blank
            outliers = np.where(combined_mask)[0].tolist()

            catches = all(t in outliers for t in info["known_blur"])
            no_fp = not any(t in outliers for t in info["false_pos"])

            mark = ""
            if catches and no_fp:
                mark = " <-- PERFECT"
            elif catches:
                mark = " (catches blur)"
            elif no_fp:
                mark = " (no FP)"

            print(f"  hf<-{hf_th} ent>+{ent_th}  {str(outliers):>30s}  "
                  f"{'YES' if catches else 'no':>12s}  {'YES' if no_fp else 'no':>5s}{mark}")

        all_fov_results[fov_name] = {
            "hf_multiz": hf_multiz, "hf_lz": hf_lz,
            "entropy": entropy, "ent_lz": ent_lz,
            "blank": blank, "info": info,
        }

    # --- Plot best combo for all FOVs ---
    best_hf_th, best_ent_th = 3.0, 2.0  # we'll see from results

    for fov_name, data in all_fov_results.items():
        info = data["info"]
        hf_lz = data["hf_lz"]
        ent_lz = data["ent_lz"]
        hf_multiz = data["hf_multiz"]
        entropy = data["entropy"]
        blank = data["blank"]
        valid = ~blank
        T = len(hf_lz)
        t_ax = np.arange(T)

        combined = (hf_lz < -best_hf_th) & (ent_lz > +best_ent_th) & valid
        outliers = np.where(combined)[0]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # HF ratio
        axes[0].plot(t_ax[valid], data["hf_multiz"][valid], ".-", ms=3, alpha=0.7)
        axes[0].set_ylabel("HF ratio (multi-Z ±5)")
        axes[0].set_title(f"{fov_name} — Combined: hf<-{best_hf_th} AND ent>+{best_ent_th}")

        # HF local z + entropy local z
        axes[1].plot(t_ax[valid], hf_lz[valid], ".-", ms=3, alpha=0.7, label="HF local_z", color="tab:blue")
        axes[1].axhline(-best_hf_th, color="tab:blue", ls="--", alpha=0.5, label=f"HF thresh (-{best_hf_th})")
        ax2 = axes[1].twinx()
        ax2.plot(t_ax[valid], ent_lz[valid], ".-", ms=3, alpha=0.7, label="Entropy local_z", color="tab:orange")
        ax2.axhline(+best_ent_th, color="tab:orange", ls="--", alpha=0.5, label=f"Ent thresh (+{best_ent_th})")
        axes[1].set_ylabel("HF local z-score", color="tab:blue")
        ax2.set_ylabel("Entropy local z-score", color="tab:orange")
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower left")

        # Combined decision
        axes[2].fill_between(t_ax, combined.astype(float), alpha=0.3, color="red", label="FLAGGED")
        for t in outliers:
            axes[2].axvline(t, color="red", alpha=0.5)
        for t in info["known_blur"]:
            axes[2].axvline(t, color="green", ls=":", lw=2, alpha=0.8, label=f"known blur t={t}")
        for t in info["false_pos"]:
            axes[2].axvline(t, color="orange", ls=":", lw=2, alpha=0.8, label=f"was FP t={t}")
        axes[2].set_ylabel("Flagged")
        axes[2].set_xlabel("Timepoint")
        axes[2].set_title(f"Outliers: {outliers.tolist()}")
        axes[2].legend(fontsize=7)

        plt.tight_layout()
        out = Path(f"/hpc/mydata/taylla.theodoro/repo/biahub_dev/scripts/hf_ent_combined_{fov_name}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved: {out}")
