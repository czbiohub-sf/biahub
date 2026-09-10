"""Debug: test detrended HF ratio local z-score.

Problem: bleaching trend makes local z-score miss real blur (A_1_000000 t=44)
and flag false positives (B_1_000001 t=50,56).

Fix: detrend with rolling median before computing local z-score.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

run_dir = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/"
    "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/dynacell/run_20260320_150224"
)

test_fovs = {
    "A_1_000000": {"known_blur": [44], "false_pos": [58]},
    "B_1_000001": {"known_blur": [], "false_pos": [50, 56]},
}

LOCAL_WINDOW = 3
THRESHOLD = 4.0


def rolling_median(arr, window):
    """Rolling median ignoring NaNs, same length as input."""
    out = np.full_like(arr, np.nan)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        vals = arr[lo:hi]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            out[i] = np.median(vals)
    return out


def compute_local_z(series, blank, local_window=3):
    """Local z-score from raw series."""
    T = len(series)
    local_z = np.zeros(T)
    for t in range(T):
        if blank[t]:
            continue
        neighbors = [
            series[t + d]
            for d in range(-local_window, local_window + 1)
            if d != 0 and 0 <= t + d < T and not blank[t + d]
        ]
        if len(neighbors) >= 4:
            loc_mean = np.mean(neighbors)
            loc_std = np.std(neighbors)
            local_z[t] = (series[t] - loc_mean) / loc_std if loc_std > 0 else 0
    return local_z


def compute_detrended_local_z(series, blank, trend_window=11, local_window=3):
    """Detrend with rolling median, then compute local z-score on residuals."""
    trend = rolling_median(series, trend_window)
    residuals = series - trend
    residuals[blank] = np.nan

    T = len(residuals)
    local_z = np.zeros(T)
    for t in range(T):
        if blank[t] or np.isnan(residuals[t]):
            continue
        neighbors = [
            residuals[t + d]
            for d in range(-local_window, local_window + 1)
            if d != 0 and 0 <= t + d < T and not np.isnan(residuals[t + d])
        ]
        if len(neighbors) >= 4:
            loc_mean = np.mean(neighbors)
            loc_std = np.std(neighbors)
            local_z[t] = (residuals[t] - loc_mean) / loc_std if loc_std > 0 else 0
    return local_z, trend, residuals


if __name__ == "__main__":
    for fov_name, info in test_fovs.items():
        csv_path = run_dir / "per_fov_analysis" / fov_name / "hf_ratio_qc.csv"
        df = pd.read_csv(csv_path)

        hf = df["hf_ratio"].values.astype(float)
        blank = np.isnan(hf)

        # Current method
        lz_current = compute_local_z(hf, blank, LOCAL_WINDOW)
        outliers_current = np.where((lz_current < -THRESHOLD) & ~blank)[0]

        # Test different trend windows
        for tw in [7, 11, 15, 21]:
            lz_detrend, trend, resid = compute_detrended_local_z(
                hf, blank, trend_window=tw, local_window=LOCAL_WINDOW
            )
            outliers_detrend = np.where((lz_detrend < -THRESHOLD) & ~blank)[0]

            print(f"\n{'='*60}")
            print(f"{fov_name} | trend_window={tw}")
            print(f"  Current outliers: {outliers_current.tolist()}")
            print(f"  Detrended outliers: {outliers_detrend.tolist()}")

            # Check known blur
            for t in info["known_blur"]:
                print(f"  Known blur t={t}: current_lz={lz_current[t]:+.2f}, "
                      f"detrend_lz={lz_detrend[t]:+.2f}")
            # Check false positives
            for t in info["false_pos"]:
                print(f"  False pos t={t}: current_lz={lz_current[t]:+.2f}, "
                      f"detrend_lz={lz_detrend[t]:+.2f}")

        # Plot: best trend_window=11
        tw = 11
        lz_detrend, trend, resid = compute_detrended_local_z(
            hf, blank, trend_window=tw, local_window=LOCAL_WINDOW
        )
        outliers_detrend = np.where((lz_detrend < -THRESHOLD) & ~blank)[0]

        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
        t_axis = np.arange(len(hf))
        valid = ~blank

        # Left: current method
        axes[0, 0].plot(t_axis[valid], hf[valid], ".-", alpha=0.7)
        axes[0, 0].set_ylabel("HF ratio")
        axes[0, 0].set_title(f"{fov_name} — Current method")

        axes[1, 0].plot(t_axis[valid], lz_current[valid], ".-", alpha=0.7, color="purple")
        axes[1, 0].axhline(-THRESHOLD, color="red", ls="--", alpha=0.7)
        for t in outliers_current:
            axes[1, 0].axvline(t, color="red", alpha=0.3)
        for t in info["known_blur"]:
            axes[1, 0].axvline(t, color="green", alpha=0.5, ls=":")
        axes[1, 0].set_ylabel("Local z-score")
        axes[1, 0].set_title(f"Outliers: {outliers_current.tolist()}")

        # Right: detrended
        axes[0, 1].plot(t_axis[valid], hf[valid], ".-", alpha=0.4, label="raw")
        axes[0, 1].plot(t_axis[valid], trend[valid], "-", color="orange", lw=2, label=f"trend (w={tw})")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].set_ylabel("HF ratio")
        axes[0, 1].set_title(f"{fov_name} — Detrended (window={tw})")

        axes[1, 1].plot(t_axis[valid], resid[valid], ".-", alpha=0.7, color="teal")
        axes[1, 1].set_ylabel("Residual")
        axes[1, 1].set_title("HF ratio - trend")

        axes[2, 1].plot(t_axis[valid], lz_detrend[valid], ".-", alpha=0.7, color="purple")
        axes[2, 1].axhline(-THRESHOLD, color="red", ls="--", alpha=0.7)
        for t in outliers_detrend:
            axes[2, 1].axvline(t, color="red", alpha=0.3)
        for t in info["known_blur"]:
            axes[2, 1].axvline(t, color="green", alpha=0.5, ls=":")
        axes[2, 1].set_ylabel("Local z-score")
        axes[2, 1].set_xlabel("Timepoint")
        axes[2, 1].set_title(f"Outliers: {outliers_detrend.tolist()}")

        axes[2, 0].set_visible(False)

        plt.tight_layout()
        out_path = Path(f"/hpc/mydata/taylla.theodoro/repo/biahub_dev/scripts/hf_detrend_{fov_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved: {out_path}")
