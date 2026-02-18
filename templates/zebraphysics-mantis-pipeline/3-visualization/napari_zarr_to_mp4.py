# %%
from pathlib import Path
import imageio
import napari
import numpy as np
from iohub import open_ome_zarr
from tqdm import tqdm
import os
from natsort import natsorted
import glob
import yaml

# %%
def load_config(path="mov_fov_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# %%
def main():
    config = load_config()

    input_path = config["input_path"]
    output_path = Path(config["output_path"])/"fov"
    fps = config.get("fps", 3)
    Z_inx = config.get("z_index", None)
    channel_config = config.get("channels", [])

    # Prepare output dir
    os.makedirs(output_path, exist_ok=True)

    # Find Zarr positions
    input_positions_dir_path = [Path(p) for p in natsorted(glob.glob(str(input_path)))]

    for position_path in tqdm(input_positions_dir_path):
        print("Processing:", position_path)

        fov = "_".join(position_path.parts[-3:])
        output_file = output_path / f"{fov}.mp4"
        if output_file.exists():
            print(f"Skipping {fov} because it already exists")
            continue

        with open_ome_zarr(position_path) as dataset:
            print("Reading data")
            all_channel_names = dataset.channel_names
            scale = dataset.scale[-2:]
            arr_list = []
            contrast_limits = []
            colormaps = []

            for ch in channel_config:
                ch_name = ch["name"]
                if ch_name not in all_channel_names:
                    print(f"[WARNING] Channel '{ch_name}' not found in {position_path.name}. Skipping FOV.")
                    arr_list = []
                    break

                ch_idx = all_channel_names.index(ch_name)
                if Z_inx==None:
                    arr = dataset.data.dask_array()[:, ch_idx, :, :, :].max(axis=1)
                else:
                    arr = dataset.data.dask_array()[:, ch_idx, Z_inx, :, :]
                arr_list.append(arr)
                contrast_limits.append(ch["contrast_limits"])
                colormaps.append(ch.get("colormap", None))

            if not arr_list:
                continue

            T, _, _ = arr_list[0].shape
            print("Shape:", arr_list[0].shape)
            print("Scale:", scale)

        # Napari viewer
        viewer = napari.Viewer()
        kwargs = dict(scale=scale, blending="additive")

        for arr, ch, clim, cmap in zip(arr_list, channel_config, contrast_limits, colormaps):
            viewer.add_image(arr, name=ch["name"], **kwargs, contrast_limits=clim, colormap=cmap)

        viewer.dims.set_point(0, 0)
        viewer.window.show_fullscreen = True

        # Write video
        writer = imageio.get_writer(output_file, fps=fps)
        for t in tqdm(range(T), desc=f"Rendering {fov}"):
            viewer.dims.set_point(0, t)
            screenshot = viewer.screenshot(canvas_only=True)
            writer.append_data(np.array(screenshot))

        writer.close()
        viewer.close()

# %%
if __name__ == "__main__":
    main()
# %%