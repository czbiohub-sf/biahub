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
# Plot renderers — only for plots that need live rendering (not pre-generated)
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


def _parse_bbox_value(value) -> list[int]:
    bbox_str = str(value).strip()
    bbox_str = re.sub(r"np\.int\d+\(([-+]?\d+)\)", r"\1", bbox_str)
    parts = [part.strip() for part in bbox_str.strip("[]()").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox value: {value!r}")
    return [int(part) for part in parts]


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

CSS = """
:root {
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --border: #dee2e6;
    --heading: #212529;
    --subheading: #495057;
    --text: #333333;
    --text-muted: #6c757d;
    --accent: #0d6efd;
    --accent-light: #e7f1ff;
    --success: #198754;
    --success-bg: #d1e7dd;
    --warning: #cc8a00;
    --warning-bg: #fff3cd;
    --danger: #dc3545;
}

* { box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Helvetica Neue', Arial, sans-serif;
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 32px;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}

h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--heading);
    border-bottom: 3px solid var(--accent);
    padding-bottom: 12px;
    margin-bottom: 16px;
}
h2 {
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--heading);
    border-bottom: 2px solid var(--border);
    padding-bottom: 8px;
    margin-top: 48px;
    margin-bottom: 16px;
}
h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--subheading);
    margin-top: 24px;
    margin-bottom: 8px;
}

.meta {
    color: var(--text-muted);
    font-size: 0.9em;
    margin-bottom: 24px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px 24px;
}
.meta span { white-space: nowrap; }
.meta b { color: var(--text); }

/* --- Tables --- */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 0.85em;
    background: var(--card-bg);
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
th, td {
    border: 1px solid var(--border);
    padding: 8px 12px;
    text-align: center;
}
th {
    background: var(--heading);
    color: white;
    font-weight: 600;
    font-size: 0.9em;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
tr:nth-child(even) { background: #f8f9fa; }
tr:hover { background: var(--accent-light); }

/* --- FOV sections --- */
.fov-section {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
.fov-section h3 {
    margin-top: 0;
    font-size: 1.15rem;
    color: var(--heading);
}
.fov-meta {
    color: var(--text-muted);
    font-size: 0.85em;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #eee;
}

/* --- Plot grid --- */
.fov-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 12px;
}
.fov-grid img {
    width: 100%;
    height: auto;
    border-radius: 6px;
    display: block;
    border: 1px solid #eee;
}
.plot-cell {
    text-align: center;
    background: #fafbfc;
    border-radius: 6px;
    padding: 6px;
}
.plot-cell-wide {
    grid-column: 1 / -1;
    text-align: center;
    background: #fafbfc;
    border-radius: 6px;
    padding: 6px;
}
.plot-label {
    font-size: 0.75em;
    color: var(--text-muted);
    margin-top: 4px;
}

/* --- Summary section --- */
.summary-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 16px;
}
.summary-grid img {
    width: 100%;
    border-radius: 6px;
    border: 1px solid #eee;
}

/* --- Badges --- */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
}
.badge-ok { background: var(--success-bg); color: var(--success); }
.badge-warn { background: var(--warning-bg); color: var(--warning); }
.badge-disqualified { background: #ffcdd2; color: #b71c1c; }

/* --- TOC navigation --- */
.toc {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
.toc h3 {
    margin-top: 0;
    margin-bottom: 8px;
    font-size: 0.95rem;
}
.toc-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 12px;
}
.toc-grid a {
    color: var(--accent);
    text-decoration: none;
    font-size: 0.82em;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--accent-light);
    transition: background 0.15s;
}
.toc-grid a:hover {
    background: #cfe2ff;
    text-decoration: underline;
}

.no-data { color: var(--text-muted); font-style: italic; text-align: center; padding: 40px; }
"""

# Per-FOV plots to embed (label, filename)
_PER_FOV_PLOTS = [
    ("Crop overlay (t=0)", "overlap_t0_overlay.png"),
    ("Z focus", "z_focus.png"),
    ("Bbox over time", "bbox_over_time.png"),
    ("Laplacian QC", "laplacian_qc.png"),
    ("Entropy QC", "entropy_qc.png"),
    ("HF ratio QC", "hf_ratio_qc.png"),
    ("FRC QC", "frc_qc.png"),
    ("Max intensity QC", "max_intensity_qc.png"),
    ("FOV registration QC", "fov_registration_qc.png"),
    ("Bleach QC", "bleach_qc.png"),
]

# All-FOV summary plots to embed (title, filename)
_ALL_FOV_PLOTS = [
    ("Z Focus — All FOVs", "z_focus_all_fovs.png"),
    ("Laplacian QC — All FOVs", "laplacian_all_fovs.png"),
    ("Entropy QC — All FOVs", "entropy_all_fovs.png"),
    ("HF Ratio QC — All FOVs", "hf_ratio_all_fovs.png"),
    ("FRC QC — All FOVs", "frc_all_fovs.png"),
    ("Max Intensity QC — All FOVs", "max_intensity_all_fovs.png"),
    ("Registration PCC — All FOVs", "registration_pcc_all_fovs.png"),
    ("Drop Correlation — All FOVs", "drop_correlation_all_fovs.png"),
    ("Outlier Correlation — All FOVs", "outlier_correlation_all_fovs.png"),
    ("FOV × Metric Heatmap", "outlier_heatmap_fov_metric.png"),
    ("Jaccard Co-occurrence Matrix", "outlier_cooccurrence_matrix.png"),
    ("Temporal Outlier Density", "outlier_temporal_density.png"),
    ("Spearman Correlation Matrix", "metric_correlation_matrix.png"),
]


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

    Embeds pre-generated per-FOV PNGs and all-FOV summary PNGs
    for consistent styling. Only the crop overlay is rendered live.

    Parameters
    ----------
    run_dir : Path
        Run directory containing per_fov_analysis/, global_summary.csv, etc.

    Returns
    -------
    Path to the generated HTML file.
    """
    run_dir = Path(run_dir)
    plots_dir = run_dir / "per_fov_analysis"
    html_path = run_dir / "dataset_report.html"

    global_csv = run_dir / "global_summary.csv"
    if not global_csv.exists():
        raise FileNotFoundError(f"global_summary.csv not found in {run_dir}")
    global_summary_df = pd.read_csv(global_csv)

    zarr_candidates = [z for z in run_dir.glob("*.zarr") if z.name != "dust_mask.zarr"]
    output_zarr = zarr_candidates[0] if zarr_candidates else None

    fov_dirs = sorted([d for d in plots_dir.iterdir() if d.is_dir()]) if plots_dir.exists() else []

    # Build set of qualified FOVs (present in global_summary)
    qualified_fovs = set()
    if "fov" in global_summary_df.columns:
        qualified_fovs = {
            "_".join(f.split("/")) for f in global_summary_df["fov"]
        }

    dataset_name = run_dir.parent.parent.name if run_dir.parent.name == "dynacell" else run_dir.name
    run_id = run_dir.name
    n_fovs = len(fov_dirs)

    print(f"Generating QC report for {run_dir}")
    print(f"  FOVs: {n_fovs} ({len(qualified_fovs)} qualified)")
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

    # Embed pre-generated all-FOV summary PNGs
    for _title, _fname in _ALL_FOV_PLOTS:
        _b64 = _png_to_base64(run_dir / _fname)
        if _b64:
            html_parts.append(f"<h3>{_title}</h3>")
            html_parts.append(f'<div class="summary-grid">{_img_tag(_b64, "100%")}</div>')

    # Dust and bleach summary PNGs (generated during global QC)
    for _title, _fname in [
        ("Dust QC", "dust_qc.png"),
        ("Bleach QC (all FOVs)", "bleach_qc.png"),
    ]:
        _b64 = _png_to_base64(run_dir / _fname)
        if _b64:
            html_parts.append(f"<h3>{_title}</h3>")
            html_parts.append(f'<div class="summary-grid">{_img_tag(_b64, "100%")}</div>')

    html_parts.append("<h3>Drop Counts</h3>")
    html_parts.append(_drop_table_html(run_dir, global_summary_df))

    html_parts.append("<h3>Per-FOV Dimensions</h3>")
    html_parts.append(_dimensions_table_html(global_summary_df))

    # --- Table of Contents ---
    html_parts.append("<h2>Per-FOV Analysis</h2>")
    html_parts.append('<div class="toc"><h3>Jump to FOV</h3><div class="toc-grid">')
    for fov_dir in fov_dirs:
        fov_name = fov_dir.name
        html_parts.append(f'<a href="#fov-{fov_name}">{fov_name}</a>')
    html_parts.append("</div></div>")

    # --- Per-FOV sections ---
    for fov_dir in fov_dirs:
        fov_name = fov_dir.name
        fov_key = "/".join(fov_name.split("_"))
        is_qualified = fov_name in qualified_fovs
        print(f"  Rendering FOV: {fov_name}{'  [disqualified]' if not is_qualified else ''}")

        summary_info = _fov_summary_html(fov_dir)
        badge = "" if is_qualified else ' <span class="badge badge-disqualified">disqualified</span>'

        html_parts.append(f"""
<div class="fov-section" id="fov-{fov_name}">
<h3>{fov_name}{badge}</h3>
<div class="fov-meta">{summary_info}</div>
<div class="fov-grid">
""")

        # Crop overlay (rendered live from zarr — only for qualified FOVs)
        if is_qualified:
            crop_img = _plot_crop_overlay(output_zarr, fov_key, overlay_channels)
            if crop_img:
                html_parts.append(
                    f'<div class="plot-cell">{_img_tag(crop_img)}'
                    f'<div class="plot-label">Crop overlay (t=0)</div></div>'
                )

        # Embed pre-generated per-FOV PNGs
        for label, fname in _PER_FOV_PLOTS:
            b64 = _png_to_base64(fov_dir / fname)
            if b64:
                html_parts.append(
                    f'<div class="plot-cell">{_img_tag(b64)}'
                    f'<div class="plot-label">{label}</div></div>'
                )

        html_parts.append("</div></div>")  # close fov-grid, fov-section

    html_parts.append("</body></html>")

    html_path.write_text("\n".join(html_parts))
    print(f"Report saved to {html_path}")

    # Annotations CSV — preserve if exists (created by _qualify_fovs)
    annotations_path = run_dir / "annotations.csv"
    if not annotations_path.exists():
        print(f"Annotations CSV not found at {annotations_path} "
              f"(will be created by the main pipeline)")
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
        Run directory containing per_fov_analysis/, global_summary.csv, annotations.csv.

    Returns
    -------
    Path to the generated annotated HTML file.
    """
    run_dir = Path(run_dir)
    plots_dir = run_dir / "per_fov_analysis"
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
    # Build per-FOV summary from per-timepoint annotations
    ann_map = {}
    dataset_comment = ""
    for fov_name, grp in annotations_df.groupby("fov"):
        fov_name = str(fov_name).strip()
        n_blank = int(grp["blank"].sum()) if "blank" in grp.columns else 0
        n_bad_reg = int(grp["bad_reg"].sum()) if "bad_reg" in grp.columns else 0
        n_oof = int(grp["out_of_focus"].sum()) if "out_of_focus" in grp.columns else 0
        n_total = len(grp)
        # Collect non-empty comments
        comments_list = [
            str(c).strip() for c in grp["comments"] if str(c).strip() not in ("", "nan")
        ] if "comments" in grp.columns else []
        summary_parts = []
        if n_blank:
            summary_parts.append(f"blank={n_blank}")
        if n_bad_reg:
            summary_parts.append(f"bad_reg={n_bad_reg}")
        if n_oof:
            summary_parts.append(f"oof={n_oof}")
        auto_comment = ", ".join(summary_parts)
        user_comment = "; ".join(comments_list) if comments_list else ""
        full_comment = "; ".join(filter(None, [auto_comment, user_comment]))
        ann_map[fov_name] = {
            "status": 0,
            "well_map": "",
            "comments": full_comment,
        }

    zarr_candidates = [z for z in run_dir.glob("*.zarr") if z.name != "dust_mask.zarr"]
    output_zarr = zarr_candidates[0] if zarr_candidates else None

    fov_dirs = sorted([d for d in plots_dir.iterdir() if d.is_dir()]) if plots_dir.exists() else []

    # Build set of qualified FOVs (present in global_summary)
    qualified_fovs_ann = set()
    if "fov" in global_summary_df.columns:
        qualified_fovs_ann = {
            "_".join(f.split("/")) for f in global_summary_df["fov"]
        }

    dataset_name = run_dir.parent.parent.name if run_dir.parent.name == "dynacell" else run_dir.name
    run_id = run_dir.name
    n_fovs = len(fov_dirs)

    print(f"Generating annotated QC report for {run_dir}")
    print(f"  FOVs: {n_fovs} ({len(qualified_fovs_ann)} qualified), annotations: {len(ann_map)}")

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

    # Dataset comment
    if dataset_comment:
        html_parts.append(
            f'<div class="dataset-comment"><b>Dataset comment:</b> {dataset_comment}</div>'
        )

    # --- Annotations summary table ---
    html_parts.append("<h2>Annotations Summary</h2>")
    ann_rows = ""
    for fov_name in sorted(ann_map.keys()):
        a = ann_map[fov_name]
        status_html, _ = _STATUS_LABELS.get(a["status"], _STATUS_LABELS[0])
        comment = a["comments"] or "-"
        ann_rows += (
            f"<tr><td>{fov_name}</td><td>{status_html}</td>"
            f"<td style='text-align:left'>{comment}</td></tr>\n"
        )
    html_parts.append(
        f"<table><tr><th>FOV</th><th>Status</th><th>Comments</th></tr>{ann_rows}</table>"
    )

    # --- Cross-FOV summary ---
    html_parts.append("<h2>Cross-FOV Summary</h2>")

    for _title, _fname in _ALL_FOV_PLOTS:
        _b64 = _png_to_base64(run_dir / _fname)
        if _b64:
            html_parts.append(f"<h3>{_title}</h3>")
            html_parts.append(f'<div class="summary-grid">{_img_tag(_b64, "100%")}</div>')

    for _title, _fname in [
        ("Dust QC", "dust_qc.png"),
        ("Bleach QC (all FOVs)", "bleach_qc.png"),
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
    html_parts.append('<div class="toc"><h3>Jump to FOV</h3><div class="toc-grid">')
    for fov_dir in fov_dirs:
        fov_name = fov_dir.name
        html_parts.append(f'<a href="#ann-fov-{fov_name}">{fov_name}</a>')
    html_parts.append("</div></div>")

    for fov_dir in fov_dirs:
        fov_name = fov_dir.name
        fov_key = "/".join(fov_name.split("_"))
        is_qualified = fov_name in qualified_fovs_ann
        print(f"  Rendering FOV: {fov_name}{'  [disqualified]' if not is_qualified else ''}")

        summary_info = _fov_summary_html(fov_dir)
        ann = ann_map.get(fov_name, {"status": 0, "comments": ""})
        status_html, _ = _STATUS_LABELS.get(ann["status"], _STATUS_LABELS[0])
        dq_badge = ' <span class="badge badge-disqualified">disqualified</span>' if not is_qualified else ""

        html_parts.append(f"""
<div class="fov-section" id="ann-fov-{fov_name}">
<h3>{fov_name} {status_html}{dq_badge}</h3>
<div class="fov-meta">{summary_info}</div>
""")

        if ann.get("comments"):
            html_parts.append(
                f'<div class="annotation-box"><b>Notes:</b> {ann["comments"]}</div>'
            )

        html_parts.append('<div class="fov-grid">')

        # Crop overlay (rendered live — only for qualified FOVs)
        if is_qualified:
            crop_img = _plot_crop_overlay(output_zarr, fov_key, overlay_channels)
            if crop_img:
                html_parts.append(
                    f'<div class="plot-cell">{_img_tag(crop_img)}'
                    f'<div class="plot-label">Crop overlay (t=0)</div></div>'
                )

        for label, fname in _PER_FOV_PLOTS:
            b64 = _png_to_base64(fov_dir / fname)
            if b64:
                html_parts.append(
                    f'<div class="plot-cell">{_img_tag(b64)}'
                    f'<div class="plot-label">{label}</div></div>'
                )

        html_parts.append("</div></div>")  # close fov-grid, fov-section

    html_parts.append("</body></html>")

    html_path.write_text("\n".join(html_parts))
    print(f"Annotated report saved to {html_path}")
    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dynacell_qc_report.py <run_dir> [--annotated]")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    annotated = "--annotated" in sys.argv

    if annotated:
        generate_annotated_report(run_dir)
    else:
        generate_dataset_report(run_dir)
