"""Shared utilities for assembling and annotating MP4 movies.

These helpers back the video-tiling tools (``grid_per_well`` and
``grid_movies``). They are deliberately free of any napari/OME-Zarr dependency
so they can be imported in headless jobs.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Box = Tuple[int, int, int, int]  # (y0, y1, x0, x1)

# Default font shipped with matplotlib/Pillow; falls back to the bitmap font.
_DEFAULT_FONT = "DejaVuSans.ttf"


# -----------------------------------------------------------------------------
# Frame geometry helpers
# -----------------------------------------------------------------------------
def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    """Return an ``(H, W, 3)`` array regardless of input layout."""
    if frame.ndim == 2:
        return np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return np.repeat(frame, 3, axis=2)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return frame[..., :3]
    return frame


def get_crop_box_nonblack(img: np.ndarray) -> Optional[Box]:
    """Tight bounding box (y0, y1, x0, x1) around the non-black pixels.

    Returns ``None`` if the image is entirely black.
    """
    rgb = img[..., :3]
    rows = np.where(np.any(rgb != 0, axis=(1, 2)))[0]
    cols = np.where(np.any(rgb != 0, axis=(0, 2)))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def apply_box(img: np.ndarray, box: Box) -> np.ndarray:
    """Crop ``img`` to the given (y0, y1, x0, x1) box."""
    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]


def center_crop_to(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Center-crop ``img`` to ``(target_h, target_w)``."""
    th, tw = target_hw
    h, w = img.shape[:2]
    y0 = max(0, (h - th) // 2)
    x0 = max(0, (w - tw) // 2)
    return img[y0 : y0 + th, x0 : x0 + tw]


def fit_to(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Center-crop, then resize as a fallback, so the result is exactly target."""
    th, tw = target_hw
    out = center_crop_to(img, target_hw)
    if out.shape[:2] != (th, tw):
        out = np.array(Image.fromarray(out).resize((tw, th)))
    return out


def rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate an image by ``degrees`` (no-op when 0), expanding the canvas."""
    if not degrees:
        return img
    return np.array(Image.fromarray(img).rotate(degrees, expand=True))


# -----------------------------------------------------------------------------
# Text annotation helpers
# -----------------------------------------------------------------------------
@lru_cache(maxsize=32)
def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load (and cache) a TrueType font, falling back to Pillow's default.

    Cached because constructing a font is relatively expensive and we draw
    text on every tile of every frame.
    """
    try:
        return ImageFont.truetype(_DEFAULT_FONT, size)
    except IOError:
        return ImageFont.load_default()


def format_hhmm(elapsed_min: float) -> str:
    """Format an elapsed time in minutes as ``HH:MM (hh:mm)``."""
    total = int(round(elapsed_min))
    h, m = divmod(total, 60)
    return f"{h:02d}:{m:02d} (hh:mm)"


def draw_label(img: Image.Image, text: str, font_size: int, margin: int = 10) -> None:
    """Draw a boxed label in the top-left corner (in place)."""
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle(
        [margin - 4, margin - 4, margin + bbox[2] + 4, margin + bbox[3] + 4],
        fill=(0, 0, 0),
    )
    draw.text((margin, margin), text, font=font, fill=(255, 255, 255))


def draw_timestamp(
    img: Image.Image, text: str, font_size: int, margin: int = 10
) -> None:
    """Draw a boxed timestamp in the bottom-right corner (in place)."""
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    w, h = img.size
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = w - tw - margin
    ty = h - th - margin
    draw.rectangle([tx - 5, ty - 5, tx + tw + 5, ty + th + 5], fill=(0, 0, 0))
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255))


def open_writer(out_file, fps: float, quality: int = 8):
    """Open an H.264 MP4 writer with web-friendly (yuv420p) output.

    ``macro_block_size=1`` disables imageio's silent resize-to-multiple-of-16;
    callers are responsible for passing even frame dimensions (required by
    yuv420p).
    """
    return imageio.get_writer(
        out_file,
        fps=fps,
        codec="libx264",
        quality=quality,
        macro_block_size=1,
        pixelformat="yuv420p",
    )


# -----------------------------------------------------------------------------
# Tiling engine
# -----------------------------------------------------------------------------
@dataclass
class TileSpec:
    """One tile in the output grid.

    A ``path`` of ``None`` produces a perpetual black tile (used to pad a grid
    that has fewer videos than cells); blank tiles never end the movie.
    """

    path: Optional[Path] = None
    legend: Optional[str] = None
    crop_box: Optional[Box] = None  # None => auto-detect non-black box
    rotation: float = 0
    frame_step: int = 1  # keep every Nth source frame
    time_sampling_min: Optional[float] = None  # minutes between source frames
    legend_font_size: int = 20
    timestamp_font_size: int = 16


@dataclass
class _OpenTile:
    spec: TileSpec
    reader: Optional[object] = None
    crop_box: Optional[Box] = None
    first: Optional[np.ndarray] = None
    kept: int = 0  # number of frames kept (post frame_step) so far
    iterator: Optional[object] = None


def _iter_step(reader, step: int):
    """Yield every ``step``-th frame from ``reader``."""
    for i, frame in enumerate(reader):
        if i % step == 0:
            yield frame


def _prepare_first_frame(tile: _OpenTile) -> np.ndarray:
    """Read, crop and rotate the first frame; record the crop box."""
    f0 = next(iter(tile.reader), None)
    if f0 is None:
        raise ValueError(f"Empty video: {tile.spec.path}")
    f0 = ensure_rgb(f0)
    box = (
        tile.spec.crop_box
        or get_crop_box_nonblack(f0)
        or (0, f0.shape[0], 0, f0.shape[1])
    )
    tile.crop_box = tuple(box)
    f0 = apply_box(f0, tile.crop_box)
    f0 = rotate(f0, tile.spec.rotation)
    return f0


def tile_videos(
    tiles: List[TileSpec],
    grid: Tuple[int, int],
    out_file,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    quality: int = 8,
) -> int:
    """Tile several videos into a single grid movie with optional annotations.

    Parameters
    ----------
    tiles : list of TileSpec
        Tile specs in row-major order. Must fit within ``grid``.
    grid : (rows, cols)
        Grid layout. ``rows * cols`` must be >= ``len(tiles)``.
    out_file : path-like
        Destination ``.mp4``.
    fps : float, optional
        Output frame rate. Defaults to the fps of the first readable source.
    max_frames : int, optional
        Stop after writing this many frames.
    quality : int, default 8
        imageio/ffmpeg quality (0-10).

    Returns
    -------
    int
        Number of frames written.
    """
    rows, cols = grid
    grid_size = rows * cols
    if len(tiles) > grid_size:
        raise ValueError(
            f"Grid {grid} has {grid_size} cells but got {len(tiles)} tiles"
        )

    # Pad with blank tiles so the grid is always full.
    tiles = list(tiles) + [TileSpec() for _ in range(grid_size - len(tiles))]

    # --- open readers and read first frames -----------------------------------
    open_tiles: List[_OpenTile] = []
    src_fps = None
    for spec in tiles:
        ot = _OpenTile(spec=spec)
        if spec.path is not None:
            try:
                ot.reader = imageio.get_reader(spec.path)
                if src_fps is None:
                    src_fps = ot.reader.get_meta_data().get("fps", None)
                ot.first = _prepare_first_frame(ot)
            except Exception as e:  # noqa: BLE001 - report and degrade to blank
                print(
                    f"  !! Failed to open {getattr(spec.path, 'name', spec.path)} -> {e}"
                )
                if ot.reader is not None:
                    ot.reader.close()
                ot.reader = None
                ot.first = None
        open_tiles.append(ot)

    real_firsts = [ot.first for ot in open_tiles if ot.first is not None]
    if not real_firsts:
        raise RuntimeError("No videos could be opened successfully.")

    # Target tile size = smallest common size, rounded down to even dims so the
    # assembled canvas is yuv420p-compatible.
    tile_h = min(f.shape[0] for f in real_firsts)
    tile_w = min(f.shape[1] for f in real_firsts)
    tile_h -= tile_h % 2
    tile_w -= tile_w % 2
    target_hw = (tile_h, tile_w)
    out_h, out_w = rows * tile_h, cols * tile_w

    fps = fps or src_fps or 30

    # Build per-tile frame iterators.
    for ot in open_tiles:
        if ot.reader is not None:
            ot.iterator = _iter_step(ot.reader, max(1, ot.spec.frame_step))

    out_file = Path(out_file)
    writer = open_writer(out_file, fps=fps, quality=quality)
    print(
        f"  >> Writing {out_file.name} | grid={rows}x{cols} | "
        f"size={out_w}x{out_h} | fps={fps}"
    )

    frames_written = 0
    try:
        while True:
            if max_frames is not None and frames_written >= max_frames:
                break

            frames = []
            exhausted = False
            for ot in open_tiles:
                if ot.iterator is None:  # blank/failed tile
                    frames.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
                    continue

                frame = next(ot.iterator, None)
                if frame is None:
                    exhausted = True
                    break

                frame = ensure_rgb(frame)
                frame = apply_box(frame, ot.crop_box)
                frame = rotate(frame, ot.spec.rotation)
                frame = fit_to(frame, target_hw)

                pil_tile = Image.fromarray(frame)
                if ot.spec.legend:
                    draw_label(pil_tile, ot.spec.legend, ot.spec.legend_font_size)
                if ot.spec.time_sampling_min is not None:
                    elapsed = ot.kept * ot.spec.frame_step * ot.spec.time_sampling_min
                    draw_timestamp(
                        pil_tile, format_hhmm(elapsed), ot.spec.timestamp_font_size
                    )

                ot.kept += 1
                frames.append(np.array(pil_tile))

            if exhausted:
                break

            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            for idx, frame in enumerate(frames):
                r, c = divmod(idx, cols)
                y0, x0 = r * tile_h, c * tile_w
                canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = frame

            writer.append_data(canvas)
            frames_written += 1

        print(f"  << Done {out_file.name}: {frames_written} frames written")
    finally:
        for ot in open_tiles:
            if ot.reader is not None:
                try:
                    ot.reader.close()
                except Exception:
                    pass
        try:
            writer.close()
        except Exception:
            pass

    return frames_written
