"""Standalone QC report generator for dynacell preprocessing runs.

Generates a self-contained HTML report with embedded plots.
Each FOV gets a section with all plots in a grid layout.

Usage:
    python dynacell_qc_report.py /path/to/run_dir
"""

import base64
import io
import sys
from datetime import datetime
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _normalize(img: np.ndarray) -> np.ndarray:
    valid = img[img != 0]
    if len(valid) == 0:
        return np.zeros_like(img, dtype=np.float32)
    vmin, vmax = np.nanpercentile(valid, [1, 99])
    if vmax == vmin:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - vmin) / (vmax - vmin), 0, 1).astype(np.float32)


def _img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};">'


def _png_to_base64(png_path: Path) -> str | None:
    """Read a PNG file and return base64-encoded string, or None if missing."""
    if not png_path.exists():
        return None
    return base64.b64encode(png_path.read_bytes()).decode("utf-8")


def _get_blank_frames(fov_plots_dir: Path) -> set[int]:
    """Read blank frame timepoints from drop_list.csv."""
    drop_csv = fov_plots_dir / "drop_list.csv"
    if not drop_csv.exists() or drop_csv.stat().st_size == 0:
        return set()
    drop_df = pd.read_csv(drop_csv)
    blank_t = drop_df[drop_df["reason"].str.contains("blank", case=False, na=False)]["t"].values
    return set(int(t) for t in blank_t)


# ---------------------------------------------------------------------------
# Plot renderers — each returns a base64 string or None
# ---------------------------------------------------------------------------

def _plot_crop_overlay(
    output_zarr: Path | None,
    fov_key: str,
    overlay_channels: list[str] | None = None,
) -> str | None:
    if output_zarr is None:
        return None
    try:
        from iohub import open_ome_zarr
    except ImportError:
        return None

    pos_path = output_zarr / fov_key
    if not pos_path.exists():
        return None

    with open_ome_zarr(pos_path) as ds:
        arr = ds.data
        _T, C, Z, Y, X = arr.shape
        z_mid = Z // 2
        channel_names = list(ds.channel_names) if hasattr(ds, "channel_names") else [
            f"ch{c}" for c in range(C)
        ]

        # Filter to requested channels
        if overlay_channels is not None:
            indices = [i for i, n in enumerate(channel_names) if n in overlay_channels]
            if not indices:
                indices = list(range(C))
        else:
            indices = list(range(C))

        channel_names = [channel_names[i] for i in indices]
        slices = [np.asarray(arr[0, i, z_mid, :, :]) for i in indices]

    # Color assignment: LF -> gray, LS -> green, red, cyan
    colors = []
    for name in channel_names:
        if name.startswith(("Phase", "Retardance", "BF")):
            colors.append(np.array([0.7, 0.7, 0.7]))
        elif not colors or all(np.allclose(c, [0.7, 0.7, 0.7]) for c in colors):
            colors.append(np.array([0.0, 1.0, 0.0]))
        elif not any(np.allclose(c, [1.0, 0.0, 0.0]) for c in colors):
            colors.append(np.array([1.0, 0.0, 0.0]))
        else:
            colors.append(np.array([0.0, 1.0, 1.0]))
    if len(colors) != len(channel_names):
        cycle = [[0.7, 0.7, 0.7], [0.7, 0.7, 0.7], [0, 1, 0], [1, 0, 0], [0, 1, 1]]
        colors = [np.array(cycle[min(c, len(cycle) - 1)]) for c in range(len(channel_names))]

    rgb = np.zeros((Y, X, 3), dtype=np.float32)
    for slc, color in zip(slices, colors):
        rgb += _normalize(slc)[:, :, None] * np.array(color)[None, None, :]
    rgb = np.clip(rgb, 0, 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb, origin="upper", aspect="equal")
    ax.set_title(f"Crop overlay (t=0, z={z_mid})", fontsize=10)
    ax.tick_params(labelsize=7)

    def _cn(col):
        if np.allclose(col, [0.7, 0.7, 0.7]): return "gray"
        if np.allclose(col, [0, 1, 0]): return "green"
        if np.allclose(col, [1, 0, 0]): return "red"
        return "cyan"
    ax.set_xlabel("  ".join(f"{n}:{_cn(c)}" for n, c in zip(channel_names, colors)), fontsize=6)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_z_focus(fov_plots_dir: Path) -> str | None:
    z_csv = fov_plots_dir / "z_focus.csv"
    if not z_csv.exists():
        return None

    z_all = pd.read_csv(z_csv, index_col=0)["z_focus"].values.astype(float)
    t_all = np.arange(len(z_all))

    # Exclude blank frames
    blank_set = _get_blank_frames(fov_plots_dir)
    valid_mask = np.array([i not in blank_set for i in range(len(z_all))])
    t = t_all[valid_mask]
    z_arr = z_all[valid_mask]

    if len(z_arr) == 0:
        return None

    mu, sigma = np.mean(z_arr), np.std(z_arr)
    upper, lower = mu + 2.5 * sigma, mu - 2.5 * sigma
    outlier_mask = (z_arr > upper) | (z_arr < lower)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, z_arr, "tab:blue", alpha=0.7, linewidth=0.8)
    ax.axhline(mu, color="orange", linestyle=":", linewidth=1)
    ax.fill_between(t, mu - sigma, mu + sigma, color="orange", alpha=0.12)
    ax.fill_between(t, lower, upper, color="red", alpha=0.05)
    ax.axhline(np.median(z_arr), color="green", linestyle="--", linewidth=0.8)
    if outlier_mask.any():
        ax.scatter(t[outlier_mask], z_arr[outlier_mask], color="red", s=15, zorder=5)

    n_blank = len(blank_set)
    title = f"Z focus | mean={mu:.1f}  std={sigma:.1f}  outliers={outlier_mask.sum()}"
    if n_blank > 0:
        title += f"  (excl. {n_blank} blank)"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("t", fontsize=8)
    ax.set_ylabel("Z focus", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _parse_bbox_value(value) -> list[int]:
    bbox_str = str(value).strip()
    bbox_str = re.sub(r"np\.int\d+\(([-+]?\d+)\)", r"\1", bbox_str)
    parts = [part.strip() for part in bbox_str.strip("[]()").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox value: {value!r}")
    return [int(part) for part in parts]


def _plot_bbox(fov_plots_dir: Path) -> str | None:
    bbox_csv = fov_plots_dir / "per_t_bboxes.csv"
    if not bbox_csv.exists():
        return None

    per_t_all = pd.read_csv(bbox_csv, index_col=0).values
    t_all = np.arange(per_t_all.shape[0])

    # Exclude blank frames
    blank_set = _get_blank_frames(fov_plots_dir)
    valid_mask = np.array([i not in blank_set for i in range(len(t_all))])
    t = t_all[valid_mask]
    per_t = per_t_all[valid_mask]

    if len(t) == 0:
        return None

    height_t = per_t[:, 1] - per_t[:, 0] + 1
    width_t = per_t[:, 3] - per_t[:, 2] + 1

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, height_t, "tab:purple", alpha=0.7, linewidth=0.8, label="Height (Y)")
    ax.plot(t, width_t, "tab:green", alpha=0.7, linewidth=0.8, label="Width (X)")

    summary_csv = fov_plots_dir / "fov_summary.csv"
    if summary_csv.exists():
        bbox_str = pd.read_csv(summary_csv)["bbox"].iloc[0]
        bbox = _parse_bbox_value(bbox_str)
        ax.axhline(bbox[1] - bbox[0] + 1, color="tab:purple", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(bbox[3] - bbox[2] + 1, color="tab:green", linestyle="--", linewidth=0.8, alpha=0.5)

    title = "Bbox height/width over time"
    if blank_set:
        title += f" (excl. {len(blank_set)} blank)"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("t", fontsize=8)
    ax.set_ylabel("pixels", fontsize=8)
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_intensity(fov_plots_dir: Path) -> str | None:
    csv = fov_plots_dir / "max_intensities.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)
    channels = [c for c in df.columns if c != "t"]
    cmap = {"arr0_ch0": "gray", "arr0_ch1": "silver",
            "arr1_ch0": "green", "arr1_ch1": "red", "arr1_ch2": "cyan"}

    # Exclude blank frames
    blank_set = _get_blank_frames(fov_plots_dir)
    valid_mask = ~df["t"].isin(blank_set)
    df = df[valid_mask]
    t = df["t"].values

    if len(t) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7, 3))
    for ch in channels:
        ax.plot(t, df[ch].values, alpha=0.7, linewidth=0.8, label=ch, color=cmap.get(ch))

    title = "Per-channel max intensity"
    if blank_set:
        title += f" (excl. {len(blank_set)} blank)"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("t", fontsize=8)
    ax.set_ylabel("Max intensity", fontsize=8)
    ax.legend(fontsize=6, loc="best")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_registration_qc(fov_plots_dir: Path) -> str | None:
    csv = fov_plots_dir / "registration_qc.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)

    # Exclude blank frames
    blank_set = _get_blank_frames(fov_plots_dir)
    if blank_set:
        df = df[~df["t"].isin(blank_set)]

    t = df["t"].values
    pearson = df["pearson_corr"].values
    shift_y = df["pcc_shift_y"].values
    shift_x = df["pcc_shift_x"].values
    pcc_err = df["pcc_error"].values

    valid = ~np.isnan(pearson)
    if not valid.any():
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    mu = np.nanmean(pearson[valid])
    sigma = np.nanstd(pearson[valid])
    ax1.plot(t[valid], pearson[valid], "tab:blue", alpha=0.7, linewidth=0.8, label="Pearson r")
    ax1.axhline(mu, color="orange", linestyle=":", linewidth=1)
    ax1.fill_between(t[valid], mu - sigma, mu + sigma, color="orange", alpha=0.12)
    ax1.set_ylabel("Pearson r", fontsize=8)
    title = f"Registration QC | Pearson mean={mu:.3f}"
    if blank_set:
        title += f" (excl. {len(blank_set)} blank)"
    ax1.set_title(title, fontsize=9)
    ax1.legend(fontsize=6)
    ax1.tick_params(labelsize=7)

    valid_s = ~np.isnan(shift_y)
    if valid_s.any():
        ax2.plot(t[valid_s], shift_y[valid_s], "tab:blue", alpha=0.7, linewidth=0.8, label="shift Y")
        ax2.plot(t[valid_s], shift_x[valid_s], "tab:green", alpha=0.7, linewidth=0.8, label="shift X")
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax2b = ax2.twinx()
        valid_e = ~np.isnan(pcc_err)
        if valid_e.any():
            ax2b.plot(t[valid_e], pcc_err[valid_e], "tab:red", alpha=0.4, linewidth=0.6, label="error")
            ax2b.set_ylabel("PCC error", fontsize=7, color="tab:red")
            ax2b.tick_params(axis="y", labelcolor="tab:red", labelsize=6)
            ax2b.legend(fontsize=5, loc="upper right")
    ax2.set_xlabel("t", fontsize=8)
    ax2.set_ylabel("PCC shift (px)", fontsize=8)
    ax2.legend(fontsize=6)
    ax2.tick_params(labelsize=7)

    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_fov_registration_qc(fov_plots_dir: Path) -> str | None:
    """Render per-FOV LF–LS Pearson correlation from fov_registration_qc.csv."""
    csv = fov_plots_dir / "fov_registration_qc.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)

    # Exclude blank frames
    blank_set = _get_blank_frames(fov_plots_dir)
    if blank_set:
        df = df[~df["t"].isin(blank_set)]

    t = df["t"].values
    pearson = df["pearson_corr"].values
    valid = ~np.isnan(pearson)
    if not valid.any():
        return None

    fig, ax = plt.subplots(figsize=(7, 2.5))
    mu = np.nanmean(pearson[valid])
    sigma = np.nanstd(pearson[valid])
    ax.plot(t[valid], pearson[valid], ".-", markersize=2, linewidth=0.6, color="tab:blue")
    ax.axhline(mu, color="orange", linestyle="--", linewidth=0.8, label=f"mean={mu:.4f}")
    ax.fill_between(t[valid], mu - sigma, mu + sigma, color="orange", alpha=0.12)
    ax.set_xlabel("t", fontsize=8)
    ax.set_ylabel("Pearson r (LF vs LS)", fontsize=8)
    title = f"Registration QC | mean={mu:.4f} \u00b1 {sigma:.4f}"
    if blank_set:
        title += f" (excl. {len(blank_set)} blank)"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_laplacian_qc(fov_plots_dir: Path) -> str | None:
    csv = fov_plots_dir / "laplacian_qc.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)

    # Exclude blank frames
    blank_set = _get_blank_frames(fov_plots_dir)
    if blank_set:
        df = df[~df["t"].isin(blank_set)]

    if len(df) == 0:
        return None

    t = df["t"].values
    lap_var = df["lap3d_var"].values

    mu, sigma = np.mean(lap_var), np.std(lap_var)
    lower = mu - 2.5 * sigma

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, lap_var, "tab:blue", alpha=0.7, linewidth=0.8, label="Laplacian var")
    ax.axhline(mu, color="orange", linestyle=":", linewidth=1, label=f"mean={mu:.2f}")
    ax.fill_between(t, mu - sigma, mu + sigma, color="orange", alpha=0.12, label=f"1 std")
    ax.axhline(lower, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label=f"2.5 std={lower:.2f}")

    # Mark low-sharpness outliers
    outlier_mask = lap_var < lower
    if outlier_mask.any():
        ax.scatter(t[outlier_mask], lap_var[outlier_mask], color="red", s=15, zorder=5,
                   label=f"blur (n={outlier_mask.sum()})")

    title = f"Laplacian variance (sharpness) | mean={mu:.2f}"
    if blank_set:
        title += f" (excl. {len(blank_set)} blank)"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("t", fontsize=8)
    ax.set_ylabel("3D Laplacian var", fontsize=8)
    ax.legend(fontsize=6, loc="best")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_hf_ratio_qc(fov_plots_dir: Path) -> str | None:
    """Render per-FOV HF ratio blur QC from hf_ratio_qc.csv."""
    csv = fov_plots_dir / "hf_ratio_qc.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)

    blank_set = _get_blank_frames(fov_plots_dir)
    if blank_set:
        df = df[~df["t"].isin(blank_set)]

    if len(df) == 0:
        return None

    t = df["t"].values
    hf_ratio = df["hf_ratio"].values
    hf_local_z = df["hf_local_z"].values if "hf_local_z" in df.columns else df.get("local_z", pd.Series(np.zeros(len(df)))).values
    ent_local_z = df["ent_local_z"].values if "ent_local_z" in df.columns else None
    is_outlier = df["is_outlier"].values.astype(bool) if "is_outlier" in df.columns else np.zeros(len(df), dtype=bool)

    valid = ~np.isnan(hf_ratio)
    if not valid.any():
        return None

    mu = np.nanmean(hf_ratio[valid])
    has_ent = ent_local_z is not None

    n_panels = 3 if has_ent else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 2.5 * n_panels), sharex=True)

    ax1 = axes[0]
    ax1.plot(t[valid], hf_ratio[valid], "tab:blue", alpha=0.7, linewidth=0.8)
    ax1.axhline(mu, color="orange", linestyle=":", linewidth=1, label=f"mean={mu:.6f}")
    if is_outlier.any():
        ax1.scatter(t[is_outlier], hf_ratio[is_outlier], color="red", s=15, zorder=5,
                    label=f"blur (n={is_outlier.sum()})")
    method = "HF+entropy" if has_ent else "HF-only"
    title = f"HF ratio blur QC ({method}) | mean={mu:.6f}  outliers={is_outlier.sum()}"
    if blank_set:
        title += f" (excl. {len(blank_set)} blank)"
    ax1.set_title(title, fontsize=9)
    ax1.set_ylabel("HF energy ratio (multi-Z)", fontsize=8)
    ax1.legend(fontsize=6)
    ax1.tick_params(labelsize=7)

    ax2 = axes[1]
    valid_lz = ~np.isnan(hf_local_z)
    if valid_lz.any():
        ax2.plot(t[valid_lz], hf_local_z[valid_lz], "tab:blue", alpha=0.7, linewidth=0.8, label="HF local z")
        ax2.axhline(-3.0, color="tab:blue", linestyle="--", linewidth=0.5, alpha=0.6, label="HF thresh (-3.0)")
        if is_outlier.any():
            ax2.scatter(t[is_outlier], hf_local_z[is_outlier], color="red", s=15, zorder=5)
    if has_ent:
        ax2r = ax2.twinx()
        valid_ent = ~np.isnan(ent_local_z)
        if valid_ent.any():
            ax2r.plot(t[valid_ent], ent_local_z[valid_ent], "tab:orange", alpha=0.7, linewidth=0.8, label="Ent local z")
            ax2r.axhline(+2.0, color="tab:orange", linestyle="--", linewidth=0.5, alpha=0.6, label="Ent thresh (+2.0)")
        ax2r.set_ylabel("Entropy local z", fontsize=8, color="tab:orange")
        ax2r.tick_params(labelsize=7)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="lower left")
    else:
        ax2.legend(fontsize=6)
    ax2.set_ylabel("HF local z-score", fontsize=8, color="tab:blue")
    ax2.tick_params(labelsize=7)

    if has_ent:
        ax3 = axes[2]
        combined = np.zeros(len(t), dtype=float)
        combined[is_outlier] = 1.0
        ax3.fill_between(t, combined, alpha=0.3, color="red", label="FLAGGED")
        for idx in t[is_outlier]:
            ax3.axvline(idx, color="red", alpha=0.5)
        ax3.set_xlabel("t", fontsize=8)
        ax3.set_ylabel("Flagged", fontsize=8)
        ax3.set_title(f"Outliers: {t[is_outlier].tolist()}", fontsize=8)
        ax3.legend(fontsize=6)
        ax3.tick_params(labelsize=7)
    else:
        axes[-1].set_xlabel("t", fontsize=8)

    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_entropy_qc(fov_plots_dir: Path) -> str | None:
    csv = fov_plots_dir / "entropy_qc.csv"
    if not csv.exists():
        return None

    df = pd.read_csv(csv)

    # Exclude blank frames
    blank_set = _get_blank_frames(fov_plots_dir)
    if blank_set:
        df = df[~df["t"].isin(blank_set)]

    if len(df) == 0:
        return None

    t = df["t"].values
    entropy = df["entropy"].values
    is_outlier = df["is_outlier"].values.astype(bool) if "is_outlier" in df.columns else np.zeros(len(df), dtype=bool)

    mu, sigma = np.mean(entropy), np.std(entropy)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, entropy, "tab:blue", alpha=0.7, linewidth=0.8, label="Entropy")
    ax.axhline(mu, color="orange", linestyle=":", linewidth=1, label=f"mean={mu:.2f}")
    ax.fill_between(t, mu - sigma, mu + sigma, color="orange", alpha=0.12, label="1 std")

    if is_outlier.any():
        ax.scatter(t[is_outlier], entropy[is_outlier], color="red", s=15, zorder=5,
                   label=f"outlier (n={is_outlier.sum()})")

    # Plot local_z on twin axis if available
    if "local_z" in df.columns:
        ax2 = ax.twinx()
        local_z = df["local_z"].values
        valid_lz = ~np.isnan(local_z)
        if valid_lz.any():
            ax2.plot(t[valid_lz], local_z[valid_lz], "tab:red", alpha=0.3, linewidth=0.6, label="local z-score")
            ax2.axhline(2.5, color="red", linestyle="--", linewidth=0.5, alpha=0.4)
            ax2.axhline(-2.5, color="red", linestyle="--", linewidth=0.5, alpha=0.4)
            ax2.set_ylabel("Local z-score", fontsize=7, color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=6)
            ax2.legend(fontsize=5, loc="upper right")

    title = f"Shannon entropy | mean={mu:.2f}  outliers={is_outlier.sum()}"
    if blank_set:
        title += f" (excl. {len(blank_set)} blank)"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("t", fontsize=8)
    ax.set_ylabel("Entropy", fontsize=8)
    ax.legend(fontsize=6, loc="best")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_dust_qc(run_dir: Path) -> str | None:
    """Embed the pre-generated dust_qc.png if it exists."""
    dust_png = run_dir / "dust_qc.png"
    if not dust_png.exists():
        return None
    with open(dust_png, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _plot_bleach_qc(run_dir: Path) -> str | None:
    """Embed the pre-generated bleach_qc.png if it exists."""
    bleach_png = run_dir / "bleach_qc.png"
    if not bleach_png.exists():
        return None
    with open(bleach_png, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _plot_cross_fov_z_focus(run_dir: Path) -> str | None:
    csv = run_dir / "z_focus_all_fovs.csv"
    if not csv.exists():
        return None
    z_df = pd.read_csv(csv, index_col=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in z_df.columns:
        ax.plot(z_df.index, z_df[col], alpha=0.5, linewidth=0.8, label=col)
    ax.set_xlabel("Time point", fontsize=9)
    ax.set_ylabel("Z focus index", fontsize=9)
    ax.set_title(f"Z focus across all FOVs ({len(z_df.columns)} FOVs)", fontsize=11)
    ax.legend(fontsize=5, loc="best", ncol=3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

CSS = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: #fafafa;
    color: #333;
}
h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 10px; color: #2c3e50; }
h2 { border-bottom: 2px solid #3498db; padding-bottom: 6px; color: #2c3e50; margin-top: 40px; }
h3 { color: #2c3e50; margin-top: 30px; }
.meta { color: #666; font-size: 0.95em; margin-bottom: 20px; }
.meta span { display: inline-block; margin-right: 25px; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 0.85em;
}
th, td {
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: center;
}
th { background: #2c3e50; color: white; font-weight: 600; }
tr:nth-child(even) { background: #f2f2f2; }
tr:hover { background: #e8f4fd; }
.fov-section {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    margin: 25px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.fov-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 10px;
}
.fov-grid img { width: 100%; height: auto; border-radius: 4px; display: block; }
.fov-grid .plot-cell { text-align: center; background: #fafafa; border-radius: 4px; padding: 4px; }
.no-data { color: #999; font-style: italic; text-align: center; padding: 40px; }
.summary-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 15px;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8em;
    font-weight: 600;
}
.badge-ok { background: #d4edda; color: #155724; }
.badge-warn { background: #fff3cd; color: #856404; }
"""


def _drop_table_html(run_dir: Path, global_summary_df: pd.DataFrame) -> str:
    drop_csv = run_dir / "drop_list_all_fovs.csv"
    if not drop_csv.exists():
        return '<p class="no-data">No drop list found.</p>'

    drop_df = pd.read_csv(drop_csv)
    if len(drop_df) == 0:
        all_fovs = sorted(global_summary_df["fov"].apply(lambda f: "_".join(f.split("/"))).tolist())
        rows = "".join(
            f'<tr><td>{f}</td><td>0</td><td><span class="badge badge-ok">none</span></td></tr>'
            for f in all_fovs
        )
        return f"<table><tr><th>FOV</th><th>Dropped</th><th>Reasons</th></tr>{rows}</table>"

    drop_counts = drop_df.groupby("fov").agg(
        n_dropped=("t", "count"),
        reasons=("reason", lambda x: ", ".join(sorted(set("; ".join(x).split("; "))))),
    ).reset_index()

    all_fovs = set(global_summary_df["fov"].apply(lambda f: "_".join(f.split("/"))))
    zero_fovs = all_fovs - set(drop_counts["fov"])
    if zero_fovs:
        zero_rows = pd.DataFrame({"fov": sorted(zero_fovs), "n_dropped": 0, "reasons": ""})
        drop_counts = pd.concat([drop_counts, zero_rows], ignore_index=True)
    drop_counts = drop_counts.sort_values("fov").reset_index(drop=True)

    rows = ""
    for _, row in drop_counts.iterrows():
        n = int(row["n_dropped"])
        badge = f'<span class="badge badge-warn">{row["reasons"]}</span>' if n > 0 else '<span class="badge badge-ok">none</span>'
        rows += f"<tr><td>{row['fov']}</td><td>{n}</td><td>{badge}</td></tr>\n"

    return f"<table><tr><th>FOV</th><th>Dropped</th><th>Reasons</th></tr>{rows}</table>"


def _dimensions_table_html(global_summary_df: pd.DataFrame) -> str:
    cols = [c for c in ["fov", "T_total", "T_out", "Y_crop", "X_crop", "bbox"]
            if c in global_summary_df.columns]
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for _, row in global_summary_df.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in cols)
        rows += f"<tr>{cells}</tr>\n"
    return f"<table><tr>{header}</tr>{rows}</table>"


def _fov_summary_html(fov_plots_dir: Path) -> str:
    """Small summary stats for FOV header."""
    parts = []
    summary_csv = fov_plots_dir / "fov_summary.csv"
    if summary_csv.exists():
        s = pd.read_csv(summary_csv).iloc[0]
        parts.append(f"T_out={s.get('T_out', '?')}")
        parts.append(f"Y={s.get('Y_crop', '?')} X={s.get('X_crop', '?')}")

    drop_csv = fov_plots_dir / "drop_list.csv"
    if drop_csv.exists() and drop_csv.stat().st_size > 0:
        n_drop = len(pd.read_csv(drop_csv))
        parts.append(f"dropped={n_drop}")
    else:
        parts.append("dropped=0")

    return " &nbsp;|&nbsp; ".join(parts)


def generate_dataset_report(
    run_dir: Path,
    overlay_channels: list[str] | None = None,
) -> Path:
    """Generate a self-contained HTML QC report.

    Parameters
    ----------
    run_dir : Path
        Run directory containing plots/, global_summary.csv, etc.

    Returns
    -------
    Path to the generated HTML file.
    """
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    html_path = run_dir / "dataset_report.html"

    global_csv = run_dir / "global_summary.csv"
    if not global_csv.exists():
        raise FileNotFoundError(f"global_summary.csv not found in {run_dir}")
    global_summary_df = pd.read_csv(global_csv)

    zarr_candidates = list(run_dir.glob("*.zarr"))
    output_zarr = zarr_candidates[0] if zarr_candidates else None

    fov_dirs = sorted([d for d in plots_dir.iterdir() if d.is_dir()]) if plots_dir.exists() else []

    dataset_name = run_dir.parent.parent.name if run_dir.parent.name == "dynacell" else run_dir.name
    run_id = run_dir.name
    n_fovs = len(fov_dirs)

    print(f"Generating QC report for {run_dir}")
    print(f"  FOVs: {n_fovs}")
    print(f"  Output zarr: {output_zarr}")

    # --- Build HTML ---
    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dynacell QC Report - {dataset_name}</title>
<style>{CSS}</style>
</head>
<body>

<h1>Dynacell QC Report</h1>
<div class="meta">
    <span><b>Dataset:</b> {dataset_name}</span>
    <span><b>Run:</b> {run_id}</span>
    <span><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
    <span><b>FOVs:</b> {n_fovs}</span>
"""]

    if "T_total" in global_summary_df.columns:
        html_parts.append(f'    <span><b>T total:</b> {global_summary_df["T_total"].iloc[0]}</span>')
    if "T_out" in global_summary_df.columns:
        html_parts.append(f'    <span><b>T out (min):</b> {global_summary_df["T_out"].min()}</span>')
    if "Y_crop" in global_summary_df.columns:
        html_parts.append(
            f'    <span><b>Crop:</b> Y={global_summary_df["Y_crop"].min()} '
            f'X={global_summary_df["X_crop"].min()}</span>'
        )

    html_parts.append("</div>")

    # --- Cross-FOV summary ---
    html_parts.append("<h2>Cross-FOV Summary</h2>")

    z_img = _plot_cross_fov_z_focus(run_dir)
    if z_img:
        html_parts.append(f'<div class="summary-grid">{_img_tag(z_img, "100%")}</div>')

    dust_img = _plot_dust_qc(run_dir)
    if dust_img:
        html_parts.append("<h3>Dust QC</h3>")
        html_parts.append(f'<div class="summary-grid">{_img_tag(dust_img, "100%")}</div>')

    bleach_img = _plot_bleach_qc(run_dir)
    if bleach_img:
        html_parts.append("<h3>Bleach QC</h3>")
        html_parts.append(f'<div class="summary-grid">{_img_tag(bleach_img, "100%")}</div>')

    # All-FOV summary plots (generated during stage 1)
    for _title, _fname in [
        ("Laplacian QC — All FOVs", "laplacian_all_fovs.png"),
        ("Entropy QC — All FOVs", "entropy_all_fovs.png"),
        ("HF Ratio QC — All FOVs", "hf_ratio_all_fovs.png"),
        ("Registration PCC — All FOVs", "registration_pcc_all_fovs.png"),
        ("Blur Detection (HF + Entropy) — All FOVs", "blur_detection_all_fovs.png"),
        ("Drop Correlation — All FOVs", "drop_correlation_all_fovs.png"),
    ]:
        _b64 = _png_to_base64(run_dir / _fname)
        if _b64:
            html_parts.append(f"<h3>{_title}</h3>")
            html_parts.append(f'<div class="summary-grid">{_img_tag(_b64, "100%")}</div>')

    html_parts.append("<h3>Drop Counts</h3>")
    html_parts.append(_drop_table_html(run_dir, global_summary_df))

    html_parts.append("<h3>Per-FOV Dimensions</h3>")
    html_parts.append(_dimensions_table_html(global_summary_df))

    # --- Per-FOV sections ---
    html_parts.append("<h2>Per-FOV Analysis</h2>")

    for fov_dir in fov_dirs:
        fov_name = fov_dir.name
        fov_key = "/".join(fov_name.split("_"))
        print(f"  Rendering FOV: {fov_name}")

        summary_info = _fov_summary_html(fov_dir)

        html_parts.append(f"""
<div class="fov-section">
<h3>{fov_name}</h3>
<div class="meta">{summary_info}</div>
<div class="fov-grid">
""")

        # All plots as grid cells (CSS auto-flows 2 per row)
        for plot_fn, plot_args in [
            (_plot_crop_overlay, (output_zarr, fov_key, overlay_channels)),
            (_plot_z_focus, (fov_dir,)),
            (_plot_bbox, (fov_dir,)),
            (_plot_laplacian_qc, (fov_dir,)),
            (_plot_entropy_qc, (fov_dir,)),
            (_plot_hf_ratio_qc, (fov_dir,)),
            (_plot_fov_registration_qc, (fov_dir,)),
            (_plot_registration_qc, (fov_dir,)),
        ]:
            img = plot_fn(*plot_args)
            if img:
                html_parts.append(f'<div class="plot-cell">{_img_tag(img)}</div>')

        html_parts.append("</div></div>")  # close fov-grid, fov-section

    html_parts.append("</body></html>")

    html_path.write_text("\n".join(html_parts))
    print(f"Report saved to {html_path}")

    # Annotations CSV (preserve if exists — the main one is created by _qualify_fovs)
    annotations_path = run_dir / "annotations.csv"
    if not annotations_path.exists():
        rows = [{"fov": d.name, "status": 0, "Well-map": "", "comments": ""}
                for d in fov_dirs]
        # Add dataset row for general comments
        rows.append({"fov": "dataset", "status": 0, "Well-map": "", "comments": ""})
        pd.DataFrame(rows, columns=["fov", "status", "Well-map", "comments"]).to_csv(
            annotations_path, index=False,
        )
        print(f"Annotations template saved to {annotations_path}")
    else:
        print(f"Annotations CSV already exists at {annotations_path} (preserved)")

    return html_path


# ---------------------------------------------------------------------------
# Annotated report — report 1 + user comments from annotations.csv
# ---------------------------------------------------------------------------

ANNOTATED_CSS_EXTRA = """
.annotation-box {
    background: #fffde7;
    border-left: 4px solid #f9a825;
    padding: 10px 15px;
    margin: 10px 0;
    font-size: 0.9em;
    border-radius: 0 4px 4px 0;
}
.annotation-box b { color: #e65100; }
.dataset-comment {
    background: #e3f2fd;
    border-left: 4px solid #1565c0;
    padding: 12px 18px;
    margin: 15px 0;
    font-size: 0.95em;
    border-radius: 0 4px 4px 0;
}
.dataset-comment b { color: #0d47a1; }
.status-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8em;
    font-weight: 600;
}
.status-checked { background: #c8e6c9; color: #1b5e20; }
.status-unfit { background: #ffcdd2; color: #b71c1c; }
.status-pending { background: #fff9c4; color: #f57f17; }
"""

_STATUS_LABELS = {
    1: ('<span class="status-badge status-checked">checked</span>', "checked"),
    -1: ('<span class="status-badge status-unfit">unfit</span>', "unfit"),
    0: ('<span class="status-badge status-pending">pending</span>', "pending"),
}


def generate_annotated_report(
    run_dir: Path,
    overlay_channels: list[str] | None = None,
) -> Path:
    """Generate a second QC report that includes user annotations from annotations.csv.

    This builds on the original report and adds:
    - Dataset-level comments (from fov="dataset" row)
    - Per-FOV status badges and user comments
    - An annotations summary table

    Parameters
    ----------
    run_dir : Path
        Run directory containing plots/, global_summary.csv, annotations.csv.

    Returns
    -------
    Path to the generated annotated HTML file.
    """
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    html_path = run_dir / "dataset_report_annotated.html"

    global_csv = run_dir / "global_summary.csv"
    if not global_csv.exists():
        raise FileNotFoundError(f"global_summary.csv not found in {run_dir}")
    global_summary_df = pd.read_csv(global_csv)

    annotations_path = run_dir / "annotations.csv"
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"annotations.csv not found in {run_dir}. "
            f"Run generate_dataset_report first, then fill in annotations."
        )
    annotations_df = pd.read_csv(annotations_path)
    # Build lookup: fov_name -> {status, Well-map, comments}
    ann_map = {}
    dataset_comment = ""
    for _, row in annotations_df.iterrows():
        fov = str(row["fov"]).strip()
        if fov == "dataset":
            dataset_comment = str(row.get("comments", "")).strip()
            if dataset_comment == "nan":
                dataset_comment = ""
        else:
            ann_map[fov] = {
                "status": int(row.get("status", 0)),
                "well_map": str(row.get("Well-map", "")).strip(),
                "comments": str(row.get("comments", "")).strip(),
            }
            if ann_map[fov]["comments"] == "nan":
                ann_map[fov]["comments"] = ""
            if ann_map[fov]["well_map"] == "nan":
                ann_map[fov]["well_map"] = ""

    zarr_candidates = list(run_dir.glob("*.zarr"))
    output_zarr = zarr_candidates[0] if zarr_candidates else None

    fov_dirs = sorted([d for d in plots_dir.iterdir() if d.is_dir()]) if plots_dir.exists() else []

    dataset_name = run_dir.parent.parent.name if run_dir.parent.name == "dynacell" else run_dir.name
    run_id = run_dir.name
    n_fovs = len(fov_dirs)

    print(f"Generating annotated QC report for {run_dir}")
    print(f"  FOVs: {n_fovs}, annotations: {len(ann_map)}")

    # --- Build HTML ---
    combined_css = CSS + ANNOTATED_CSS_EXTRA
    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dynacell QC Report (Annotated) - {dataset_name}</title>
<style>{combined_css}</style>
</head>
<body>

<h1>Dynacell QC Report (Annotated)</h1>
<div class="meta">
    <span><b>Dataset:</b> {dataset_name}</span>
    <span><b>Run:</b> {run_id}</span>
    <span><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
    <span><b>FOVs:</b> {n_fovs}</span>
"""]

    if "T_total" in global_summary_df.columns:
        html_parts.append(f'    <span><b>T total:</b> {global_summary_df["T_total"].iloc[0]}</span>')
    if "T_out" in global_summary_df.columns:
        html_parts.append(f'    <span><b>T out (min):</b> {global_summary_df["T_out"].min()}</span>')
    if "Y_crop" in global_summary_df.columns:
        html_parts.append(
            f'    <span><b>Crop:</b> Y={global_summary_df["Y_crop"].min()} '
            f'X={global_summary_df["X_crop"].min()}</span>'
        )
    html_parts.append("</div>")

    # Dataset-level comment
    if dataset_comment:
        html_parts.append(
            f'<div class="dataset-comment"><b>Dataset comment:</b> {dataset_comment}</div>'
        )

    # --- Annotations summary table ---
    html_parts.append("<h2>Annotations Summary</h2>")
    ann_header = "<tr><th>FOV</th><th>Status</th><th>Well-map</th><th>Comments</th></tr>"
    ann_rows = ""
    for fov_name in sorted(ann_map.keys()):
        info = ann_map[fov_name]
        badge_html, _ = _STATUS_LABELS.get(info["status"], _STATUS_LABELS[0])
        ann_rows += (
            f'<tr><td>{fov_name}</td><td>{badge_html}</td>'
            f'<td>{info["well_map"]}</td><td>{info["comments"]}</td></tr>\n'
        )
    html_parts.append(f"<table>{ann_header}{ann_rows}</table>")

    # --- Cross-FOV summary ---
    html_parts.append("<h2>Cross-FOV Summary</h2>")

    z_img = _plot_cross_fov_z_focus(run_dir)
    if z_img:
        html_parts.append(f'<div class="summary-grid">{_img_tag(z_img, "100%")}</div>')

    dust_img = _plot_dust_qc(run_dir)
    if dust_img:
        html_parts.append("<h3>Dust QC</h3>")
        html_parts.append(f'<div class="summary-grid">{_img_tag(dust_img, "100%")}</div>')

    bleach_img = _plot_bleach_qc(run_dir)
    if bleach_img:
        html_parts.append("<h3>Bleach QC</h3>")
        html_parts.append(f'<div class="summary-grid">{_img_tag(bleach_img, "100%")}</div>')

    # All-FOV summary plots (generated during stage 1)
    for _title, _fname in [
        ("Laplacian QC — All FOVs", "laplacian_all_fovs.png"),
        ("Entropy QC — All FOVs", "entropy_all_fovs.png"),
        ("HF Ratio QC — All FOVs", "hf_ratio_all_fovs.png"),
        ("Registration PCC — All FOVs", "registration_pcc_all_fovs.png"),
        ("Blur Detection (HF + Entropy) — All FOVs", "blur_detection_all_fovs.png"),
        ("Drop Correlation — All FOVs", "drop_correlation_all_fovs.png"),
    ]:
        _b64 = _png_to_base64(run_dir / _fname)
        if _b64:
            html_parts.append(f"<h3>{_title}</h3>")
            html_parts.append(f'<div class="summary-grid">{_img_tag(_b64, "100%")}</div>')

    html_parts.append("<h3>Drop Counts</h3>")
    html_parts.append(_drop_table_html(run_dir, global_summary_df))

    html_parts.append("<h3>Per-FOV Dimensions</h3>")
    html_parts.append(_dimensions_table_html(global_summary_df))

    # --- Per-FOV sections ---
    html_parts.append("<h2>Per-FOV Analysis</h2>")

    for fov_dir in fov_dirs:
        fov_name = fov_dir.name
        fov_key = "/".join(fov_name.split("_"))
        print(f"  Rendering FOV: {fov_name}")

        summary_info = _fov_summary_html(fov_dir)

        # Annotation info for this FOV
        ann_info = ann_map.get(fov_name, {"status": 0, "well_map": "", "comments": ""})
        badge_html, _ = _STATUS_LABELS.get(ann_info["status"], _STATUS_LABELS[0])

        html_parts.append(f"""
<div class="fov-section">
<h3>{fov_name} {badge_html}</h3>
<div class="meta">{summary_info}</div>
""")

        # Show annotation comment if present
        if ann_info["comments"]:
            html_parts.append(
                f'<div class="annotation-box"><b>Comment:</b> {ann_info["comments"]}</div>'
            )
        if ann_info["well_map"]:
            html_parts.append(
                f'<div class="annotation-box"><b>Well-map:</b> {ann_info["well_map"]}</div>'
            )

        html_parts.append('<div class="fov-grid">')

        # All plots as grid cells (CSS auto-flows 2 per row)
        for plot_fn, plot_args in [
            (_plot_crop_overlay, (output_zarr, fov_key, overlay_channels)),
            (_plot_z_focus, (fov_dir,)),
            (_plot_bbox, (fov_dir,)),
            (_plot_laplacian_qc, (fov_dir,)),
            (_plot_entropy_qc, (fov_dir,)),
            (_plot_hf_ratio_qc, (fov_dir,)),
            (_plot_fov_registration_qc, (fov_dir,)),
            (_plot_registration_qc, (fov_dir,)),
        ]:
            img = plot_fn(*plot_args)
            if img:
                html_parts.append(f'<div class="plot-cell">{_img_tag(img)}</div>')

        html_parts.append("</div></div>")  # close fov-grid, fov-section

    html_parts.append("</body></html>")

    html_path.write_text("\n".join(html_parts))
    print(f"Annotated report saved to {html_path}")
    return html_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dynacell_qc_report.py /path/to/run_dir [--annotated]")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    annotated = "--annotated" in sys.argv
    if annotated:
        html_path = generate_annotated_report(run_dir)
    else:
        html_path = generate_dataset_report(run_dir)
    print(f"Report saved to {html_path}")
