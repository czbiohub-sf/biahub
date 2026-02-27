from pathlib import Path
import imageio
import numpy as np
from tqdm import tqdm
import yaml
from PIL import Image, ImageDraw, ImageFont
from math import ceil
from copy import deepcopy


def load_config(path="movs_combine_channels_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_unique_output_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    suffix = 2
    while True:
        new_path = base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")
        if not new_path.exists():
            return new_path
        suffix += 1

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


def iter_step(reader, step):
    for i, frame in enumerate(reader):
        if i % step == 0:
            yield frame
def mov_combine_channels(config=None):
    
    config = load_config() if config is None else config

    OUTPUT_PATH = Path(config["output_dir"])
    GRID = tuple(config.get("grid", [2, 2]))
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    mov_sets = config.get("movs", [])
    max_total_frames = config.get("max_frames", None)

    if not mov_sets:
        raise ValueError("'movs' list is required in config when tiling multiple videos together.")

    files = []
    legends = []
    legend_fonts = []
    timestamp_fonts = []
    rotations = []
    frame_intervals = []
    frame_steps = []

    for mov in mov_sets:
        path = Path(mov["mov_path"])
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        files.append(path)
        legends.append(mov.get("legend", None))
        legend_fonts.append(mov.get("legend_font_size", 20))
        timestamp_fonts.append(mov.get("timestamp_font_size", 16))
        rotations.append(mov.get("rotation", 0))
        frame_intervals.append(mov.get("frame_time_sampling_min", 1))
        frame_steps.append(mov.get("frame_step", 1))

    grid_size = GRID[0] * GRID[1]
    if len(files) > grid_size:
        raise ValueError(f"Grid size {GRID} is too small for {len(files)} videos")

    readers, metas, processed_firsts, crop_boxes = [], [], [], []
    for i, f in enumerate(files):
        try:
            r = imageio.get_reader(f)
            m = r.get_meta_data()
            f0 = next(iter(r), None)
            if f0 is None:
                raise ValueError(f"Empty video: {f.name}")
            f0 = ensure_rgb(f0)
            if "crop_box" in mov_sets[i]:
                box = tuple(mov_sets[i]["crop_box"])
            else:
                box = get_crop_box_nonblack(f0) or (0, f0.shape[0], 0, f0.shape[1])
            crop_boxes.append(box)
            f0 = apply_box(f0, box)
            if rotations[i]:
                f0 = np.array(Image.fromarray(f0).rotate(rotations[i], expand=True))
            readers.append(r)
            metas.append(m)
            processed_firsts.append(f0)
        except Exception as e:
            print(f"  !! Failed to open {f.name} -> {e}")

    if not processed_firsts:
        raise RuntimeError("No videos could be opened successfully.")

    while len(readers) < grid_size:
        blank = np.zeros_like(processed_firsts[0])
        readers.append(None)
        metas.append({"fps": metas[0]["fps"]})
        processed_firsts.append(blank)
        crop_boxes.append((0, blank.shape[0], 0, blank.shape[1]))
        legends.append(None)
        legend_fonts.append(20)
        timestamp_fonts.append(16)
        rotations.append(0)
        frame_intervals.append(mov.get("frame_time_sampling_min", 1))
        frame_steps.append(1)

    fps = metas[0].get("fps", 30)

    hs = [img.shape[0] for img in processed_firsts]
    ws = [img.shape[1] for img in processed_firsts]
    target_hw = (min(hs), min(ws))
    tile_h, tile_w = target_hw
    processed_firsts = [center_crop_to(img, target_hw) for img in processed_firsts]

    rows, cols = GRID
    out_h, out_w = rows * tile_h, cols * tile_w

    out_file = get_unique_output_path(OUTPUT_PATH / config["output_filename"])
    writer = imageio.get_writer(out_file, fps=fps)
    print(f"  >> Writing: {config['output_filename']} | grid={rows}x{cols} | size={out_w}x{out_h} | fps={fps}")

    try:
        stream_iters = [
            iter_step(r, frame_steps[i]) if r is not None else None
            for i, r in enumerate(readers)
        ]
        frame_counts = [0] * len(readers)
        frames_written = 0

        with tqdm(desc=f"  Frames [{config['output_filename']}]") as pbar:
            while True:
           
                if max_total_frames is not None and frames_written >= max_total_frames:
                    break

                frames = []
                exhausted = False
                for i, it in enumerate(stream_iters):
                    if it is None:
                        frames.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
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
                    if rotations[i]:
                        f = np.array(Image.fromarray(f).rotate(rotations[i], expand=True))
                    f = center_crop_to(f, target_hw)
                    if f.shape[:2] != target_hw:
                        f = np.array(Image.fromarray(f).resize((tile_w, tile_h)))

                    pil_tile = Image.fromarray(f)

                    if legends[i]:
                        draw = ImageDraw.Draw(pil_tile)
                        try:
                            font_legend = ImageFont.truetype("DejaVuSans.ttf", legend_fonts[i])
                        except IOError:
                            font_legend = ImageFont.load_default()
                        bbox = draw.textbbox((0, 0), legends[i], font=font_legend)
                        draw.rectangle([10 - 4, 10 - 4, 10 + bbox[2] + 4, 10 + bbox[3] + 4], fill=(0, 0, 0))
                        draw.text((10, 10), legends[i], font=font_legend, fill=(255, 255, 255))

                    if frame_intervals[i]:
                        draw = ImageDraw.Draw(pil_tile)
                        try:
                            font_time = ImageFont.truetype("DejaVuSans.ttf", timestamp_fonts[i])
                        except IOError:
                            font_time = ImageFont.load_default()
                        elapsed = frame_counts[i] * frame_intervals[i] * frame_steps[i]
                        h, m = divmod(elapsed, 60)
                        timestamp = f"{h:02d}:{m:02d} (hh:mm)"
                        bbox = draw.textbbox((0, 0), timestamp, font=font_time)
                        tx = tile_w - bbox[2] - 10
                        ty = tile_h - bbox[3] - 10
                        draw.rectangle([tx - 5, ty - 5, tx + bbox[2] + 5, ty + bbox[3] + 5], fill=(0, 0, 0))
                        draw.text((tx, ty), timestamp, font=font_time, fill=(255, 255, 255))

                    frame_counts[i] += 1
                    frames.append(np.array(pil_tile))

                if exhausted:
                    break

                canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                for idx, frame in enumerate(frames[:grid_size]):
                    r = idx // cols
                    c = idx % cols
                    y0, x0 = r * tile_h, c * tile_w
                    canvas[y0:y0 + tile_h, x0:x0 + tile_w] = frame

                writer.append_data(canvas)
                pbar.update(1)
                frames_written += 1

        print(f"  << Done {config['output_filename']}: {frames_written} frames written.")
    
    finally:
        for r in readers:
            if r is not None:
                try: r.close()
                except Exception: pass
        try: writer.close()
        except Exception: pass

if __name__ == "__main__":
    config = load_config()
    
    fov_list = config.get("fov_list", None)
    if fov_list is not None:
        for fov in tqdm(fov_list, desc="Processing FOVs"): 
            config_copy = deepcopy(config)
            config_copy["output_filename"] = config_copy["output_filename"].replace("0_2_002000", fov.replace("/", "_"))
            for mov in config_copy["movs"]:
                mov["mov_path"] = mov["mov_path"].replace("0_2_002000", fov.replace("/", "_"))
                mov["legend"] = mov["legend"].replace("0/2/002000", fov.replace("/", "_"))
            
            mov_combine_channels(config_copy)
       
    else:   
        mov_combine_channels(config)
 