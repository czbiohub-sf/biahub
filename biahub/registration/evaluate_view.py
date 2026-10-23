"""Views of a registration evaluation: a figure per record, a video, a napari check.

The reference shows green and the moving volume magenta, so where they agree is white.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biahub.core.transform import Transform
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
    """Return residual positions and displacements, yx, in small-image pixels.

    Each axis converts with its own voxel size, so an anisotropic grid draws true lengths.
    """
    if measure is None or not measure.get("residuals", {}).get("position_um"):
        return np.zeros((0, 2)), np.zeros((0, 2))
    factor = views["downsample"]
    px_um = np.asarray(views["ref_um"], float)[1:]  # one full-size pixel, y and x
    full_px = np.asarray(measure["residuals"]["position_um"])[:, 1:] / px_um
    step = np.asarray(measure["residuals"]["diff_um"])[:, 1:] / (px_um * factor)
    return (full_px - (factor - 1) / 2) / factor, step


def _arrow_scale(measures: list, views: dict) -> float:
    """Return one exaggeration for every panel: the longest arrow spans `ARROW_SHARE`."""
    longest = max(
        [np.linalg.norm(_arrows(m, views)[1], axis=1).max(initial=0.0) for m in measures],
        default=0.0,
    )
    if longest == 0:
        return 1.0
    return ARROW_SHARE * max(views["reference"].shape) / longest


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
    for key, color in (("tiles", "yellow"), ("points", "cyan")):
        where, step = _arrows(measures[key], views)
        if len(where):
            ax.quiver(
                where[:, 1],
                where[:, 0],
                step[:, 1] * scale,
                -step[:, 0] * scale,
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


# --- video -----------------------------------------------------------------------------

# Frames the max projections are held for before the z scan (2 s at the default rate).
HOLD_FRAMES = 16


def _corner_pair(transform, mov, ref, size: int, origin_yx: tuple = (0, 0)) -> tuple:
    """Return (warped moving, reference) at full resolution in one yx window of `ref`.

    Only the window is warped: the transform is shifted so the window's corner is the
    output's origin.
    """
    y0, x0 = origin_yx
    ref_crop = np.asarray(ref, dtype=np.float32)[:, y0 : y0 + size, x0 : x0 + size]
    to_window = Transform.from_translation([0.0, -y0, -x0])
    warped = (to_window @ transform).apply(
        np.asarray(mov, dtype=np.float32), reference=ref_crop
    )
    return warped, ref_crop


def _limits(volume: np.ndarray) -> tuple[float, float]:
    data = volume[volume != 0]
    low, high = np.percentile(data if data.size else volume, [1, 99.8])
    return float(low), float(high)


def _rgb(moving, reference, mov_limits, ref_limits) -> np.ndarray:
    def scale(image, limits):
        return np.clip((image - limits[0]) / (limits[1] - limits[0] + 1e-9), 0, 1)

    m, r = scale(moving, mov_limits), scale(reference, ref_limits)
    return np.stack([m, r, m], axis=-1)


def animate_evaluation(
    mov,
    ref,
    transforms: dict,
    path: str | Path,
    title: str = "",
    zoom_px: int = 384,
    fps: int = 8,
) -> Path:
    """Write a video comparing transforms: max projections, then a scan through z.

    One column per entry of `transforms` (name -> forward Transform, e.g. unregistered,
    a hand registration, the skill's), moving magenta on reference green: the full field
    (averaged down) and the top-left corner at full resolution, where a scale error shows
    most. Brightness is fixed per volume across frames. Without ffmpeg the frames are
    written as PNGs in a folder next to `path`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from biahub.registration.evaluate import display_volumes

    path = Path(path)
    columns = []
    for name, transform in transforms.items():
        warped, small_ref, _factor = display_volumes(transform, mov, ref)
        corner, ref_corner = _corner_pair(transform, mov, ref, zoom_px)
        columns.append((name, warped, small_ref, corner, ref_corner))
    mov_limits = _limits(columns[0][1].max(axis=0))
    ref_limits = _limits(columns[0][2].max(axis=0))
    n_z = columns[0][2].shape[0]
    steps = [("max projection", None)] * HOLD_FRAMES + [(f"z = {z}", z) for z in range(n_z)]

    def frame(label, z):
        # 400 px per column by 832: already multiples of 16, so no padding shows
        figure, axes = plt.subplots(
            2, len(columns), figsize=(4 * len(columns), 8.32), dpi=100, squeeze=False
        )
        for k, (name, warped, small_ref, corner, ref_corner) in enumerate(columns):
            for row, (m, r, what) in enumerate(
                ((warped, small_ref, "full field"), (corner, ref_corner, "top-left corner"))
            ):
                pick = (lambda v: v.max(axis=0)) if z is None else (lambda v: v[z])
                axes[row, k].imshow(_rgb(pick(m), pick(r), mov_limits, ref_limits))
                axes[row, k].set_title(f"{name}: {what}", fontsize=10)
                axes[row, k].set_axis_off()
        figure.suptitle(f"{title}  {label}  (moving magenta, reference green)", fontsize=11)
        figure.tight_layout()
        figure.canvas.draw()
        image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        plt.close(figure)
        # the encoder wants sides in multiples of 16
        pad = [(0, -image.shape[0] % 16), (0, -image.shape[1] % 16), (0, 0)]
        return np.pad(image, pad)

    try:
        import imageio_ffmpeg

        first = frame(*steps[0])
        writer = imageio_ffmpeg.write_frames(
            str(path), (first.shape[1], first.shape[0]), fps=fps, macro_block_size=16
        )
        writer.send(None)
        writer.send(first.tobytes())
        for label, z in steps[1:]:
            writer.send(frame(label, z).tobytes())
        writer.close()
        return path
    except (ImportError, RuntimeError) as error:
        frames_dir = path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for k, (label, z) in enumerate(steps):
            plt.imsave(frames_dir / f"{k:04d}.png", frame(label, z))
        print(f"no video encoder ({error}): frames written to {frames_dir}")
        return frames_dir


# --- the visual check in napari --------------------------------------------------------


def _build_check(mov, ref, transform, ref_um, phase=None, full_resolution=False, show=True):
    """Return the napari viewer `visual_check` opens."""
    try:
        import napari
    except ImportError as error:
        raise ImportError(
            "the visual check needs napari: pip install 'biahub[gui]'"
        ) from error
    from biahub.registration.evaluate import _shrink_yx, display_volumes

    if full_resolution:
        reference = np.nan_to_num(np.asarray(ref, np.float32))
        moving = np.nan_to_num(np.asarray(mov, np.float32))
        registered = transform.apply(moving, reference=reference)
        original = Transform.identity(3).apply(moving, reference=reference)
        downsample = 1
    else:
        registered, reference, downsample = display_volumes(transform, mov, ref)
        original, _ref, _factor = display_volumes(Transform.identity(3), mov, ref)
    scale = (ref_um[0], ref_um[1] * downsample, ref_um[2] * downsample)

    viewer = napari.Viewer(title="registration check", show=show)
    if phase is not None:
        phase = np.nan_to_num(np.asarray(phase, np.float32))
        phase = phase if downsample == 1 else _shrink_yx(phase, downsample)
        # auto contrast: a few bright or dark outliers would leave min-to-max flat grey
        limits = np.percentile(phase, [0.5, 99.5]).tolist()
        viewer.add_image(
            phase, name="phase", colormap="gray", scale=scale, contrast_limits=limits
        )
    viewer.add_image(
        reference, name="reference", colormap="green", blending="additive", scale=scale
    )
    viewer.add_image(
        original,
        name="moving (original)",
        colormap="magenta",
        blending="additive",
        scale=scale,
        visible=False,
    )
    viewer.add_image(
        registered,
        name="moving (registered)",
        colormap="magenta",
        blending="additive",
        scale=scale,
    )
    return viewer


def visual_check(mov, ref, transform, ref_um, phase=None, full_resolution=False) -> None:
    """Open napari to look at a registration; returns when the window is closed.

    The reference shows green and the moving volume magenta, as it was ("moving
    (original)", hidden at first) and registered (white where they agree); `phase`, on the
    reference grid, shows grey underneath when given. The judgement is asked afterwards
    (the registration skill asks whether it is good). Views are averaged down in yx
    unless `full_resolution` (one full warp).
    """
    import napari

    _build_check(mov, ref, transform, ref_um, phase, full_resolution)
    napari.run()
