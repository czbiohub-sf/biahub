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

    z_arr = pd.read_csv(z_csv, index_col=0)["z_focus"].values.astype(float)
    t = np.arange(len(z_arr))
    mu, sigma = np.mean(z_arr), np.std(z_arr)
    upper, lower = mu + 2.5 * sigma, mu - 2.5 * sigma
    outliers = np.where((z_arr > upper) | (z_arr < lower))[0]

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, z_arr, "tab:blue", alpha=0.7, linewidth=0.8)
    ax.axhline(mu, color="orange", linestyle=":", linewidth=1)
    ax.fill_between(t, mu - sigma, mu + sigma, color="orange", alpha=0.12)
    ax.fill_between(t, lower, upper, color="red", alpha=0.05)
    ax.axhline(np.median(z_arr), color="green", linestyle="--", linewidth=0.8)
    if len(outliers) > 0:
        ax.scatter(outliers, z_arr[outliers], color="red", s=15, zorder=5)

    # Blank frames
    drop_csv = fov_plots_dir / "drop_list.csv"
    if drop_csv.exists() and drop_csv.stat().st_size > 0:
        drop_df = pd.read_csv(drop_csv)
        blank_t = drop_df[drop_df["reason"].str.contains("blank", case=False, na=False)]["t"].values
        for bt in blank_t:
            ax.axvspan(bt - 0.5, bt + 0.5, color="gray", alpha=0.15)

    ax.set_title(f"Z focus | mean={mu:.1f}  std={sigma:.1f}  outliers={len(outliers)}", fontsize=9)
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

    per_t = pd.read_csv(bbox_csv, index_col=0).values
    t = np.arange(per_t.shape[0])
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

    ax.set_title("Bbox height/width over time", fontsize=9)
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
    t = df["t"].values
    channels = [c for c in df.columns if c != "t"]
    cmap = {"arr0_ch0": "gray", "arr0_ch1": "silver",
            "arr1_ch0": "green", "arr1_ch1": "red", "arr1_ch2": "cyan"}

    fig, ax = plt.subplots(figsize=(7, 3))
    for ch in channels:
        ax.plot(t, df[ch].values, alpha=0.7, linewidth=0.8, label=ch, color=cmap.get(ch))

    drop_csv = fov_plots_dir / "drop_list.csv"
    if drop_csv.exists() and drop_csv.stat().st_size > 0:
        drop_df = pd.read_csv(drop_csv)
        blank_t = drop_df[drop_df["reason"].str.contains("blank", case=False, na=False)]["t"].values
        for bt in blank_t:
            ax.axvspan(bt - 0.5, bt + 0.5, color="gray", alpha=0.15)

    ax.set_title("Per-channel max intensity", fontsize=9)
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
    t = df["t"].values
    pearson = df["pearson_corr"].values
    shift_y = df["pcc_shift_y"].values
    shift_x = df["pcc_shift_x"].values
    pcc_err = df["pcc_error"].values

    valid = ~np.isnan(pearson)
    if not valid.any():
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    mu = np.nanmean(pearson)
    sigma = np.nanstd(pearson)
    ax1.plot(t[valid], pearson[valid], "tab:blue", alpha=0.7, linewidth=0.8, label="Pearson r")
    ax1.axhline(mu, color="orange", linestyle=":", linewidth=1)
    ax1.fill_between(t, mu - sigma, mu + sigma, color="orange", alpha=0.12)
    ax1.set_ylabel("Pearson r", fontsize=8)
    ax1.set_title(f"Registration QC | Pearson mean={mu:.3f}", fontsize=9)
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
    gap: 15px;
    margin-top: 10px;
}
.fov-grid img { width: 100%; border-radius: 4px; }
.fov-grid .plot-cell { text-align: center; }
.fov-grid .plot-cell.wide { grid-column: 1 / -1; }
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

        # Crop overlay
        overlay_img = _plot_crop_overlay(output_zarr, fov_key, overlay_channels)
        if overlay_img:
            html_parts.append(f'<div class="plot-cell">{_img_tag(overlay_img)}</div>')

        # Z-focus
        zf_img = _plot_z_focus(fov_dir)
        if zf_img:
            html_parts.append(f'<div class="plot-cell">{_img_tag(zf_img)}</div>')

        # Bbox
        bbox_img = _plot_bbox(fov_dir)
        if bbox_img:
            html_parts.append(f'<div class="plot-cell">{_img_tag(bbox_img)}</div>')

        # Intensity
        int_img = _plot_intensity(fov_dir)
        if int_img:
            html_parts.append(f'<div class="plot-cell">{_img_tag(int_img)}</div>')

        # Registration QC (full width if present)
        reg_img = _plot_registration_qc(fov_dir)
        if reg_img:
            html_parts.append(f'<div class="plot-cell wide">{_img_tag(reg_img)}</div>')

        html_parts.append("</div></div>")  # close fov-grid, fov-section

    html_parts.append("</body></html>")

    html_path.write_text("\n".join(html_parts))
    print(f"Report saved to {html_path}")

    # Annotations CSV (preserve if exists)
    annotations_path = run_dir / "annotations.csv"
    if not annotations_path.exists():
        pd.DataFrame({
            "fov": [d.name for d in fov_dirs],
            "status": "pending",
            "comments": "",
        }).to_csv(annotations_path, index=False)
        print(f"Annotations template saved to {annotations_path}")
    else:
        print(f"Annotations CSV already exists at {annotations_path} (preserved)")

    return html_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dynacell_qc_report.py /path/to/run_dir")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    html_path = generate_dataset_report(run_dir)
    print(f"Report saved to {html_path}")
