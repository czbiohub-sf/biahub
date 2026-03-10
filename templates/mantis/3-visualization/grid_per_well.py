from pathlib import Path
import imageio
import numpy as np
from tqdm import tqdm
import yaml
import os
from PIL import Image, ImageDraw, ImageFont
from math import ceil

# --- CONFIG ---
def load_config(path="mov_well_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_well_id(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse well id from filename: {stem}")
    return parts[1]

def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return np.repeat(frame, 3, axis=2)
    return frame

def get_crop_box_nonblack(img: np.ndarray):
    rgb = img[..., :3]
    rows = np.where(np.any(rgb != 0, axis=(1, 2)))[0]
    cols = np.where(np.any(rgb != 0, axis=(0, 2)))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

def apply_box(img: np.ndarray, box):
    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]

def center_crop_to(img: np.ndarray, target_hw):
    th, tw = target_hw
    h, w = img.shape[:2]
    y0 = max(0, (h - th) // 2)
    x0 = max(0, (w - tw) // 2)
    return img[y0:y0 + th, x0:x0 + tw]

def iter_frames(reader):
    for f in reader:
        yield f

def mov_wells(config=None):
    config = load_config() if config is None else config

    MOV_PATH = Path(config["mov_path"])
    OUTPUT_PATH = Path(config["output_path"])
    GRID = tuple(config.get("grid", [2, 2]))
    TIME_SAMPLING_MIN = config.get("time_sampling_min", None)
    ROTATION = config.get("rotation", 0)
    MAX_FRAMES = config.get("max_frames", None)
    FRAME_STEP = config.get("frame_step", 1)
    TIMESTAMP_FONT_SIZE = config.get("timestamp_font_size", 24)
    LEGEND_FONT_SIZE = config.get("legend_font_size", 20)
    ADD_LEGEND = config.get("add_legend", True)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    files = sorted(MOV_PATH.glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No .mp4 files found in {MOV_PATH}")

    # Group by well
    by_well = {}
    for f in files:
        by_well.setdefault(get_well_id(f.stem), []).append(f)

    print(f"Found wells: {sorted(by_well.keys())}")

    for well_id, all_files in sorted(by_well.items()):
        all_files = sorted(all_files)
        grid_size = GRID[0] * GRID[1]
        num_groups = ceil(len(all_files) / grid_size)

        print(f"\n[Well {well_id}] {len(all_files)} file(s) → {num_groups} output video(s)")

        for group_idx in range(num_groups):
            group_files = all_files[group_idx * grid_size : (group_idx + 1) * grid_size]

            readers, metas, processed_firsts, crop_boxes = [], [], [], []

            for f in group_files:
                try:
                    legend = f.stem
                    r = imageio.get_reader(f)
                    m = r.get_meta_data()
                    it = iter(r)
                    f0 = next(it, None)
                    if f0 is None:
                        raise ValueError(f"Empty video: {f.name}")
                    f0 = ensure_rgb(f0)

                    box = get_crop_box_nonblack(f0)
                    if box is None:
                        box = (0, f0.shape[0], 0, f0.shape[1])
                    crop_boxes.append(box)

                    f0 = apply_box(f0, box)
                    if ROTATION:
                        f0 = np.array(Image.fromarray(f0).rotate(ROTATION, expand=True))
                
                    readers.append(r)
                    metas.append(m)
                    processed_firsts.append(f0)
                except Exception as e:
                    print(f"  !! Failed to open {f.name} -> {e}")

            while len(readers) < grid_size:
                blank = np.zeros_like(processed_firsts[0])
                readers.append(None)
                metas.append({"fps": metas[0]["fps"]})
                processed_firsts.append(blank)
                crop_boxes.append((0, blank.shape[0], 0, blank.shape[1]))

            fps = metas[0].get("fps", 30)

            hs = [img.shape[0] for img in processed_firsts]
            ws = [img.shape[1] for img in processed_firsts]
            target_hw = (min(hs), min(ws))
            tile_h, tile_w = target_hw
            processed_firsts = [center_crop_to(img, target_hw) for img in processed_firsts]

            rows, cols = GRID
            out_h, out_w = rows * tile_h, cols * tile_w

            out_file = OUTPUT_PATH / f"{well_id}_{group_idx}.mp4"
            writer = imageio.get_writer(out_file, fps=fps)
            print(f"  >> Writing: {out_file.name} | grid={rows}x{cols} | size={out_w}x{out_h} | fps={fps} | rotation={ROTATION}")

            try:
                stream_iters = [
                    iter_frames(r) if r is not None else None
                    for r in zip(readers)
                ]
                frame_counts = [0] * len(readers)
                frame_index = 0
                frames_written = 0

                with tqdm(desc=f"  Frames [{well_id}]", unit="f") as pbar:
                    while True:
                        if MAX_FRAMES is not None and frame_index >= MAX_FRAMES:
                            break

                        frames = []
                        exhausted = False
                        for i, it in enumerate(stream_iters):
                            if it is None:
                                blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                                frames.append(blank)
                                continue
                            try:
                                f = next(it, None)
                            except StopIteration:
                                f = None
                            if f is None:
                                exhausted = True
                                break

                            f = ensure_rgb(f)
                            f = apply_box(f, crop_boxes[i])
                            if ROTATION:
                                f = np.array(Image.fromarray(f).rotate(ROTATION, expand=True))
                            f = center_crop_to(f, target_hw)
                            if f.shape[:2] != target_hw:
                                f = np.array(Image.fromarray(f).resize((tile_w, tile_h)))

                            pil_tile = Image.fromarray(f)

                            if TIME_SAMPLING_MIN is not None:
                                draw = ImageDraw.Draw(pil_tile)
                                try:
                                    font = ImageFont.truetype("DejaVuSans.ttf", TIMESTAMP_FONT_SIZE)
                                except IOError:
                                    font = ImageFont.load_default()

                                elapsed = frame_index * TIME_SAMPLING_MIN
                                h, m = elapsed // 60, elapsed % 60
                                timestamp = f"{h:02d}:{m:02d} (hh:mm)"
                                bbox = draw.textbbox((0, 0), timestamp, font=font)
                                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                                tx = tile_w - tw - 10
                                ty = tile_h - th - 10
                                draw.rectangle([tx - 5, ty - 5, tx + tw + 5, ty + th + 5], fill=(0, 0, 0))
                                draw.text((tx, ty), timestamp, font=font, fill=(255, 255, 255))
                            if ADD_LEGEND:
                                draw = ImageDraw.Draw(pil_tile)
                                try:
                                    font_legend = ImageFont.truetype("DejaVuSans.ttf", LEGEND_FONT_SIZE)
                                except IOError:
                                    font_legend = ImageFont.load_default()
                                bbox = draw.textbbox((0, 0), legend, font=font_legend)
                                draw.rectangle([10 - 4, 10 - 4, 10 + bbox[2] + 4, 10 + bbox[3] + 4], fill=(0, 0, 0))
                                draw.text((10, 10), legend, font=font_legend, fill=(255, 255, 255))


                            frames.append(np.array(pil_tile))

                        if exhausted:
                            break

                        if frame_index % FRAME_STEP == 0:
                            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                            for idx, frame in enumerate(frames[:grid_size]):
                                r = idx // cols
                                c = idx % cols
                                y0, x0 = r * tile_h, c * tile_w
                                canvas[y0:y0 + tile_h, x0:x0 + tile_w] = frame

                            writer.append_data(canvas)
                            pbar.update(1)
                            frames_written += 1

                        frame_index += 1

                print(f"  << Done {out_file.name}: {frames_written} frames written (from {frame_index} read).")
            finally:
                for r in readers:
                    if r is not None:
                        try: r.close()
                        except Exception: pass
                try: writer.close()
                except Exception: pass
def main():
    config = load_config()
    mov_wells(config)

if __name__ == "__main__":
    main()