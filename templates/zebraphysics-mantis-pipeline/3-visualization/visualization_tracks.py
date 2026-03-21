# %%
import argparse
from pathlib import Path

import napari
import numpy as np
from iohub import open_ome_zarr
from skimage.morphology import disk, erosion
from ultrack.reader.napari_reader import read_csv
from waveorder.focus import focus_from_transverse_band

NA_DET = 1.35
LAMBDA_ILL = 0.500


def get_segments_bourder(segments, T, erosion_radius=4):
    # Create an empty array for contours (same shape as labels)
    border_labels = np.zeros_like(segments, dtype=int)
    erosion_kernel = disk(erosion_radius)
    # Erode and subtract to find the borders
    for t in range(T):
        for label in np.unique(segments[t]):
            if label == 0:
                continue  # Skip background

            # Create a binary mask for this label
            binary_mask = segments[t] == label

            # Erode the binary mask
            eroded_mask = erosion(binary_mask, erosion_kernel)

            # Subtract: Original - Eroded = Borders
            border_mask = binary_mask ^ eroded_mask  # XOR to get the difference

            # Store only the border pixels with the original label value
            border_labels[t][border_mask] = label
    return border_labels


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process command-line arguments.")
    parser.add_argument(
        "root",
        type=str,
        help="Dataset dirpath, example: /hpc/projects/intracellular_dashboard/organelle_dynamics/2024_10_31_A549_SEC61_ZIKV_DENV",
    )
    parser.add_argument("fov", type=str, help="FOV key, example: B/1/000000")
    parser.add_argument(
        "z_slices", type=str, help="Z slices to visualize, example: 21,50"
    )
    args = parser.parse_args()

    # Get the dataset paths
    root = Path(args.root)
    dataset = str(root).split("/")[-1]
    im_path = root / f"2-assemble/{dataset}.zarr"
    track_path = root / f"1-preprocess/label-free/3-track/{dataset}_cropped.zarr"

    print("Dataset:", dataset)
    print("Im Path:", im_path)
    print("Track Path:", track_path)

    # Get the FOV key and z slices
    key = args.fov
    print("FOV key:", key)

    z_start, z_end = map(int, args.z_slices.split(","))
    z_slices_vs = slice(z_start, z_end)

    print("Z slices:", z_slices_vs)

    # Load the data
    print("Loading channels...")
    im_ds = open_ome_zarr(im_path)
    channel_names = im_ds[key].channel_names
    im_arr = im_ds[key].data[:, channel_names.index("Phase3D"), :, :, :]
    nuc_arr = im_ds[key].data[:, channel_names.index("nuclei_prediction"), :, :, :]
    mem_arr = im_ds[key].data[:, channel_names.index("membrane_prediction"), :, :, :]

    _, _, _, Y_scale, X_scale = im_ds[key].scale
    yx_scale = [Y_scale, X_scale]
    T, C, Z, Y, X = im_ds[key].data.shape
    print("Data shape:", T, C, Z, Y, X)

    print("Computing focal plane...")
    # Get the focal plane for the phase image
    z_focal_plane = focus_from_transverse_band(
        im_arr[0, :, :, :],
        NA_det=NA_DET,
        lambda_ill=LAMBDA_ILL,
        pixel_size=X_scale,
    )

    print("Phase Focal Plane:", z_focal_plane)

    # Get 3D slices for the visualization (T, Y, X)
    im_arr_focus = im_arr[:, z_focal_plane, :, :]
    nuc_arr_focus = nuc_arr[:, z_slices_vs, :, :].max(axis=1)
    mem_arr_focus = mem_arr[:, z_slices_vs, :, :].max(axis=1)

    print("Loading track labels...")
    track_seg = open_ome_zarr(track_path)
    segments = track_seg[key].data[:, 0, 0, :, :]

    # Get the border labels for visualization of the tracks
    border_labels = get_segments_bourder(segments, T)

    # Get the contrast limits for the phase image
    vmin, vmax = np.percentile(im_arr_focus, (2, 98))

    print("Opening napari viewer...")
    viewer = napari.Viewer()
    kwargs = dict(scale=yx_scale, blending="additive")
    viewer.add_image(nuc_arr_focus, name="VS nuclei", colormap="green", **kwargs)
    viewer.add_image(im_arr_focus, name="Phase", **kwargs, contrast_limits=(vmin, vmax))

    viewer.add_image(mem_arr_focus, name="VS membrane", colormap="magenta", **kwargs)
    viewer.add_labels(border_labels, name="Track Labels", **kwargs)

    # Get track graph
    print("Loading track graph...")
    w1, w2, f = key.split("/")
    df_path = track_path / f"{w1}" / f"{w2}" / f"{f}" / f"tracks_{w1}_{w2}_{f}.csv"
    track_df, kwargs, *_ = read_csv(df_path)
    graph = kwargs["graph"]
    viewer.add_tracks(
        track_df[["track_id", "t", "y", "x"]],
        graph=graph,
        name="Tracks",
        scale=yx_scale,
        colormap="hsv",
        blending="opaque",
    )
    viewer.dims.set_point(0, 0)


    viewer.grid.enabled = True
    viewer.grid.shape = (1, 2)

    napari.run()


if __name__ == "__main__":
    main()
