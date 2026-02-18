# %%
import os
from pathlib import Path

import imageio
import napari
import numpy as np
import pandas as pd
from iohub import open_ome_zarr
from tqdm import tqdm
from ultrack.reader.napari_reader import read_csv


def remove_black_border(screenshot):
    # Convert the screenshot to a numpy array if it's not already
    screenshot = np.array(screenshot)

    # Identify rows and columns that are completely black
    non_black_row_indices = np.where(np.any(screenshot[:, :, :3] != 0, axis=(1, 2)))[0]
    non_black_col_indices = np.where(np.any(screenshot[:, :, :3] != 0, axis=(0, 2)))[0]

    # Crop the screenshot to the non-black rows and columns
    cropped_screenshot = screenshot[
        non_black_row_indices.min() : non_black_row_indices.max() + 1,
        non_black_col_indices.min() : non_black_col_indices.max() + 1,
    ]

    return cropped_screenshot


# %%
def main() -> None:

    # %%
    # NOTE modify these paths to point to the correct zarr files
    root = Path("/hpc/projects/organelle_phenotyping/2024_11_26_A549_ZIKA-sensor_ZIKV/")
    im_path = root / "2-assemble/2024_11_26_A549_ZIKA-sensor_ZIKV.zarr"
    track_path = (
        root
        / "1-preprocess/label-free/3-track/2024_11_26_A549_ZIKA-sensor_ZIKV_max_cropped.zarr"
    )

    output_path = root / "1-preprocess/label-free/3-track/mov_max/fov"
    os.makedirs(output_path, exist_ok=True)

    z_slicing = slice(21, 50)
    blank_frames_path = root / "1-preprocess/label-free/3-track/blank_frames.csv"
    blank_frames_df = pd.read_csv(blank_frames_path, index_col=0)
    keys = blank_frames_df["key"].to_list()

    for key in tqdm(keys):
        w1, w2, f = key.split("_")
        key = f"{w1}/{w2}/{f}"
        print(key)

        im_ds = open_ome_zarr(im_path)

        scale = im_ds[key].scale[-2:]
        # Cropping the FOV to some cells

        im_arr = im_ds[key].data[:, 0, z_slicing.start, :, :]
        print("img shape:", im_arr.shape)
        nuc_arr = im_ds[key].data[:, 1, z_slicing, :, :].max(axis=1)
        print("nuc shape:", nuc_arr.shape)
        mem_arr = im_ds[key].data[:, 2, z_slicing, :, :].max(axis=1)
        print("mem shape:", mem_arr.shape)

        track_seg = open_ome_zarr(track_path)
        segments = track_seg[key].data[:, 0, 0, :, :]

        print("labels shape:", segments.shape)

        # %%
        viewer = napari.Viewer()
        kwargs = dict(scale=scale, blending="additive")

        # %%
        viewer.add_image(im_arr, name="phase", **kwargs)
        viewer.add_image(nuc_arr, name="VS nuclei", colormap="green", **kwargs)
        viewer.add_image(mem_arr, name="VS membrane", colormap="magenta", **kwargs)
        viewer.add_labels(segments, name="Segments", **kwargs)

        df_path = track_path / f"{w1}" / f"{w2}" / f"{f}" / f"tracks_{w1}_{w2}_{f}.csv"
        track_df, kwargs, *_ = read_csv(df_path)
        graph = kwargs["graph"]
        # viewer.add_labels(old_segments, name="Old_Segments", scale=scale)
        viewer.add_tracks(
            track_df[["track_id", "t", "y", "x"]],
            graph=graph,
            name="Tracks",
            scale=scale,
            colormap="hsv",
            blending="opaque",
        )
        viewer.dims.set_point(0, 0)

        # viewer.window.qt_viewer.viewer_window_fullscreen = True

        # napari.run()

        # Define the path to save the mp4 file
        output_path_mov = output_path / f"tracking_{w1}_{w2}_{f}.mp4"
        # Set up the video writer with imageio
        writer = imageio.get_writer(output_path_mov, fps=3)  # Adjust fps as needed
        # Get the number of time points (assuming all layers have the same shape)
        n_timepoints = im_arr.shape[0]
        # Iterate through each time point
        for t in range(n_timepoints):
            # Update the viewer to the current time point for each layer
            viewer.dims.set_point(0, t)
            # for layer, ld in zip(viewer.layers, layers_data):
            # layer.data[t] = ld[t]
            # Capture the screenshot with all layers combined
            screenshot = viewer.screenshot(canvas_only=True)

            # screenshot_corrected = remove_black_border(screenshot)
            writer.append_data(np.array(screenshot))
        # Close the writer to finalize the file
        writer.close()
        viewer.close()


# %%
if __name__ == "__main__":
    main()

# %%
