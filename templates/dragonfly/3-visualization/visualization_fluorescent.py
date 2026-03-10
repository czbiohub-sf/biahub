# %%
import argparse
from pathlib import Path

import napari
import numpy as np
from iohub import open_ome_zarr


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process command-line arguments.")
    parser.add_argument(
        "root",
        type=str,
        help="Dataset dirpath, example: /hpc/projects/intracellular_dashboard/organelle_dynamics/2024_10_31_A549_SEC61_ZIKV_DENV",
    )
    parser.add_argument("fov", type=str, help="FOV key, example: B/1/000000")

    args = parser.parse_args()

    # Get the dataset paths
    root = Path(args.root)
    dataset = str(root).split("/")[-1]
    im_path = root / f"2-assemble/{dataset}.zarr"

    print("Dataset:", dataset)
    print("Im Path:", im_path)

    # Get the FOV key and z slices
    key = args.fov

    print("FOV key:", key)

    # Load the data
    print("Loading channels...")
    im_ds = open_ome_zarr(im_path)
    # get channel names
    channel_names = im_ds[key].channel_names
    print("Channel names:", channel_names)

    im_arr = im_ds[key].data[:, channel_names.index("Phase3D"), :, :, :]
    gfp_arr = im_ds[key].data[:, channel_names.index("GFP EX488 EM525-45"), :, :, :]
    mCherry_arr = im_ds[key].data[
        :, channel_names.index("mCherry EX561 EM600-37"), :, :, :
    ]

    _, _, Z_scale, Y_scale, X_scale = im_ds[key].scale
    zyx_scale = [Z_scale, Y_scale, X_scale]
    T, C, Z, Y, X = im_ds[key].data.shape
    print("Data shape:", T, C, Z, Y, X)

    print("Getting contrast limits...")
    # Get the contrast limits for each channel
    phase_vmin, phase_vmax = np.percentile(im_arr, (1, 98))
    print("Phase:", phase_vmin, phase_vmax)

    gfp_vmin = max(0, np.percentile(gfp_arr, 80))
    gfp_vmax = np.percentile(gfp_arr, 99.8)
    print("GFP:", gfp_vmin, gfp_vmax)

    mCherry_vmin = max(0, np.percentile(mCherry_arr, 80))
    mCherry_vmax = np.percentile(mCherry_arr, 99.8)
    print("mCherry:", mCherry_vmin, mCherry_vmax)

    print("Opening napari viewer...")
    viewer = napari.Viewer()
    kwargs = dict(scale=zyx_scale, blending="additive")

    viewer.add_image(
        gfp_arr,
        name="GFP EX488 EM525-45",
        **kwargs,
        colormap="green",
        contrast_limits=(gfp_vmin, gfp_vmax),
    )
    viewer.add_image(
        im_arr, name="Phase", **kwargs, contrast_limits=(phase_vmin, phase_vmax)
    )

    viewer.add_image(
        mCherry_arr,
        name="mCherry EX561 EM600-37",
        **kwargs,
        colormap="magenta",
        contrast_limits=(mCherry_vmin, mCherry_vmax),
    )

    viewer.dims.set_point(0, 0)
    viewer.grid.enabled = True
    viewer.grid.shape = (1, 2)

    napari.run()


if __name__ == "__main__":
    main()
