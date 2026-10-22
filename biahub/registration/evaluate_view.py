"""Views of a registration evaluation: a figure per record, a video, a napari check.

The reference shows green and the moving volume magenta, so where they agree is white.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biahub.registration.evaluate import Evaluation

# Arrows are drawn this many times longer than the residual, up to this share of the image.
ARROW_SHARE = 0.08


def _normalise(image: np.ndarray) -> np.ndarray:
    # the warp's empty padding is not background: leave it out of the limits
    data = image[image != 0]
    low, high = np.percentile(data if data.size else image, [1, 99.8])
    return np.clip((image - low) / (high - low + 1e-9), 0, 1)


def overlay(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return an RGB image: `moving` magenta, `reference` green (white where both)."""
    m, r = _normalise(moving), _normalise(reference)
    return np.stack([m, r, m], axis=-1)


def _arrows(measure: dict | None, views: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return residual positions (small-image pixels, yx) and displacements (µm, yx)."""
    if measure is None or not measure.get("residuals", {}).get("position_um"):
        return np.zeros((0, 2)), np.zeros((0, 2))
    um = np.asarray(views["ref_um"], float)[1:]
    factor = views["downsample"]
    full_px = np.asarray(measure["residuals"]["position_um"])[:, 1:] / um
    small_px = (full_px - (factor - 1) / 2) / factor
    return small_px, np.asarray(measure["residuals"]["diff_um"])[:, 1:]


def _arrow_scale(measures: list, views: dict) -> float:
    """Return one exaggeration for every panel: the longest arrow spans `ARROW_SHARE`."""
    um_per_px = np.asarray(views["ref_um"], float)[1] * views["downsample"]
    longest = max(
        [np.linalg.norm(_arrows(m, views)[1], axis=1).max(initial=0.0) for m in measures],
        default=0.0,
    )
    if longest == 0:
        return 1.0
    return ARROW_SHARE * max(views["reference"].shape) * um_per_px / longest


def _summary(measures: dict) -> str:
    parts = []
    if measures["points"] is not None and measures["points"]["stats"]["n"]:
        p = measures["points"]
        parts.append(f"points {p['n_matched']}: {p['stats']['median_um']:.2f} µm")
    if measures["tiles"] is not None and measures["tiles"]["stats"]["n"]:
        t = measures["tiles"]
        parts.append(f"tiles {t['n_tiles']}: {t['stats']['median_um']:.2f} µm")
    if measures["z_um"] is not None:
        parts.append(f"z {measures['z_um']:+.2f} µm")
    return " | ".join(parts) or "no residuals measured"


def _panel(ax, image, views, measures, scale, label):
    ax.imshow(image)
    um_per_px = np.asarray(views["ref_um"], float)[1] * views["downsample"]
    for key, color in (("tiles", "yellow"), ("points", "cyan")):
        where, diff = _arrows(measures[key], views)
        if len(where):
            ax.quiver(
                where[:, 1],
                where[:, 0],
                diff[:, 1] * scale / um_per_px,
                -diff[:, 0] * scale / um_per_px,
                color=color,
                angles="uv",
                scale_units="xy",
                scale=1,
                width=0.004,
                label=key,
            )
    ax.set_title(f"{label}\n{_summary(measures)}", fontsize=9)
    ax.set_axis_off()


def _z_panel(ax, record: Evaluation):
    labels, values, colors = [], [], []
    for name, measures, color in (
        ("unregistered", record.baseline, "0.6"),
        ("registered", _measures_of(record), "tab:blue"),
    ):
        crops = measures["z_crops"]
        if crops is not None:
            for crop, v in crops["crops_um"].items():
                if v is not None:
                    labels.append(f"{name}\n{crop}")
                    values.append(v[0])
                    colors.append(color)
        p = measures["points"]
        if p is not None and p["stats"]["n"]:
            labels.append(f"{name}\npoints")
            values.append(p["stats"]["median_zyx_um"][0])
            colors.append("tab:cyan" if name == "registered" else "0.3")
    if values:
        ax.barh(range(len(values)), values, color=colors)
        ax.scatter(values, range(len(values)), color=colors, edgecolor="k", zorder=3)  # zeros
        ax.set_yticks(range(len(values)), labels, fontsize=6)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("z residual (µm)")
    z_from = record.z_from or "no z measure"
    z = "n/a" if record.z_um is None else f"{record.z_um:+.2f} µm"
    ax.set_title(f"z residual: {z}\n(from {z_from})", fontsize=9)


def _measures_of(record: Evaluation) -> dict:
    return {k: getattr(record, k) for k in ("points", "tiles", "z_crops", "z_um", "z_from")}


def plot_evaluation(record: Evaluation, path: str | Path | None = None, title: str = ""):
    """Return (and save to `path`) a figure of one evaluation, unregistered vs registered.

    Two overlays of the max projections with the residuals as arrows (tiles yellow, points
    cyan; one exaggeration for both, printed), and the z residuals per measure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = record.projections
    if views is None:
        raise ValueError(
            "the record has no projections (a stored record): run evaluate() to plot it"
        )
    registered = _measures_of(record)
    scale = _arrow_scale(
        [
            record.baseline["tiles"],
            record.baseline["points"],
            registered["tiles"],
            registered["points"],
        ],
        views,
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4), width_ratios=[1, 1, 0.7])
    _panel(
        axes[0],
        overlay(views["unregistered"], views["reference"]),
        views,
        record.baseline,
        scale,
        "unregistered",
    )
    _panel(
        axes[1],
        overlay(views["registered"], views["reference"]),
        views,
        registered,
        scale,
        "registered",
    )
    _z_panel(axes[2], record)
    g = record.geometry
    figure.suptitle(
        f"{title}  transform: scale zyx {np.round(g['scale_zyx'], 4).tolist()}, "
        f"rotation yx {g['rotation_yx_deg']:.2f}°, shear {g['shear']:.3f} | "
        f"arrows x{scale:.0f}, moving magenta on reference green",
        fontsize=10,
    )
    figure.tight_layout()
    if path is not None:
        figure.savefig(path, dpi=120)
    return figure
