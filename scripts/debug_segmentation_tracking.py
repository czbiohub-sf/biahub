#
#  %%
import os
import uuid
from pathlib import Path

# from cellpose import models, core
import napari
import numpy as np
import pandas as pd
from iohub import open_ome_zarr

# import cupy as cp
# import cucim.skimage.morphology as morph
from numpy.typing import ArrayLike
from rich import print
from scipy.ndimage import binary_erosion, gaussian_filter, median_filter
from skimage import morphology as morph

# from cupyx.scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist, match_histograms, rescale_intensity
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, remove_small_objects
from tqdm import tqdm
from ultrack import MainConfig, Tracker
from ultrack.core.tracker import TrackerStatus
from ultrack.imgproc import (
    Cellpose,
    detect_foreground,
    inverted_edt,
    normalize,
    robust_invert,
)
from ultrack.utils import estimate_parameters_from_labels
from ultrack.utils.array import array_apply, create_zarr

# def detect_foreground_mem(mem: ArrayLike, threshold: float, disk=(7, 3)) -> np.ndarray:
#     mem = cp.asarray(mem)
#     fg = cp.zeros_like(mem, dtype=bool)
#     big_disk = morph.disk(disk[0])
#     small_disk = morph.disk(disk[1])
#     for t in tqdm(range(mem.shape[0]), "Processing foreground"):
#         mask = morph.closing(mem[t], big_disk) > threshold
#         mask = morph.remove_small_objects(mask, 500)
#         mask = morph.remove_small_holes(mask, 500)
#         mask = morph.erosion(mask, small_disk)
#         fg[t] = mask
#     return fg.get()


def detect_empty_frames(arr):
    empty_frames_idx = []
    # Iterate over the first dimension of the array, assuming it's the frame index
    for f in range(arr.shape[0]):
        # Use np.sum for potential speed improvement with NumPy's optimized operations
        if np.sum(arr[f]) == 0.0:
            empty_frames_idx.append(f)
    return empty_frames_idx


def mem_nuc_contor(
    nuc_arr,
    mem_arr,
):
    contourn = (np.array(mem_arr) + (1 - np.array(nuc_arr))) / 2
    return contourn


def mem_nuc_edt_contour(nuc_arr, mem_arr, edt_arr):
    contourn = (np.array(mem_arr) + np.array(edt_arr) + (1 - np.array(nuc_arr))) / 3
    return contourn


def mem_edt_contour(mem_arr, edt_arr):
    contourn = (np.array(mem_arr) + np.array(edt_arr)) / 2
    return contourn

from scipy.ndimage import gaussian_filter, distance_transform_edt

def generate_gradient_map(nuc_arr_norm, mem_arr_norm):
    # Distance transform from membrane (inverted so distance increases toward center)
    #mem_thresh = mem_arr_norm > 0.5  # You may adjust this threshold
    #mem_edt = distance_transform_edt(~mem_thresh)

    # Smooth nuclei to emphasize centers
    smoothed_nuc = gaussian_filter(nuc_arr_norm, sigma=1.5)

    # Normalize both
    #mem_edt_norm = (mem_edt - mem_edt.min()) / (mem_edt.max() - mem_edt.min())
    smoothed_nuc_norm = (smoothed_nuc - smoothed_nuc.min()) / (smoothed_nuc.max() - smoothed_nuc.min())

    # Combine and mask

   # gradient = (mem_edt_norm + smoothed_nuc_norm) / 2

    return smoothed_nuc_norm.astype(np.float32)


from scipy.ndimage import gaussian_filter
import numpy as np

def nuclei_split_contourn(nuc_pred: np.ndarray, sigma_center=1.0, sigma_border=4.0) -> np.ndarray:
    """
    Generate a contourn image from nuclei prediction that helps split touching nuclei.
    
    Parameters:
        nuc_pred (ndarray): Normalized or raw nuclei prediction.
        sigma_center (float): Mild smoothing to keep nuclei peaks.
        sigma_border (float): Strong smoothing to model general region intensity.
    
    Returns:
        ndarray: Gradient map with strong contrast between nucleus centers and borders.
    """
    # Smooth for peak centers
    center_smooth = gaussian_filter(nuc_pred, sigma=sigma_center)

    # Smooth more broadly to model general nuclei area
    broad_smooth = gaussian_filter(nuc_pred, sigma=sigma_border)

    # Subtract to emphasize centers and suppress edges
    contour = center_smooth - broad_smooth
    contour = np.clip(contour, a_min=0, a_max=None)

    # Normalize
    contour = (contour - contour.min()) / (contour.max() - contour.min())

    return contour.astype(np.float32)

# %%
def main() -> None:
    pass

    # %%
    # NOTE modify these paths to point to the correct zarr files

    ### CHANGE THIS VALUE
    dataset = "2024_10_31_A549_SEC61_ZIKV_DENV"

    root = Path(f"/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/{dataset}")
    im_path = root / f"1-preprocess/label-free/2-stabilize/phase/{dataset}.zarr/B/4/000000"
    vs_path = root / f"1-preprocess/label-free/2-stabilize/virtual-stain/{dataset}.zarr/B/4/000000"

    output_path = root / "1-preprocess/label-free/3-track/test_tracking"

    ### CHANGE THIS VALUE
    key = "B_1_000000"
    track_output_path = output_path / "track_v1_config" / key

    os.makedirs(track_output_path, exist_ok=True)
    key = key.replace("_", "/")

    print(f"Processing key: {key}")

    ### CHANGE THIS VALUE

    with open_ome_zarr(vs_path) as vs_ds:
        nuc_arr = np.asarray(vs_ds.data[:, 0, :, :, :]) # take nuclei prediction channel
        mem_arr = np.asarray(vs_ds.data[:, 1, :, :, :]) # take membrane prediction channel
        shape = nuc_arr.shape
    z_max = shape[1]
    n_slices = max(3, z_max // 2)    
    if n_slices % 2 == 0:
        n_slices += 1  # make it odd to center cleanly
    z_center = z_max // 2
    half_window = n_slices // 2

    z_start = max(0, z_center - half_window)
    z_end = min(z_max, z_center + half_window + 1)  # +1 because slice end is exclusive
    z_slicing = slice(z_start, z_end)
    print(f"Z center: {z_center}")
    print(f"Z slices: {z_slicing}")


    with open_ome_zarr(im_path) as im_ds:
        im_arr = np.asarray(im_ds.data[:, 0, z_center,:,:]) # take phase channel
        scale = im_ds.scale[-2:]

    nuc_arr = nuc_arr[:, z_slicing, :, :].mean(axis=1)
    mem_arr = mem_arr[:, z_slicing, :, :].mean(axis=1)
# %%
    viewer = napari.Viewer()
    kwargs = dict(scale=scale, blending="additive")
    viewer.add_image(im_arr, name="Original", **kwargs)
    viewer.add_image(nuc_arr, name="VS nuclei", colormap="green", **kwargs)
    viewer.add_image(mem_arr, name="VS membrane", colormap="magenta", **kwargs)
# %%
    ### CHANGE THE FUNCTION AS NEEDED

    nuc_arr_norm = create_zarr(shape=nuc_arr.shape, dtype=float)
    array_apply(nuc_arr, out_array=nuc_arr_norm, func=normalize, gamma=0.7)

    mem_arr_norm = create_zarr(shape=mem_arr.shape, dtype=float)
    array_apply(mem_arr, out_array=mem_arr_norm, func=normalize, gamma=0.7)

    empty_frames_idx = detect_empty_frames(im_arr)
    for idx in empty_frames_idx:

        nuc_arr_norm[idx, :, :] = nuc_arr_norm[idx - 1, :, :]
        mem_arr_norm[idx, :, :] = mem_arr_norm[idx - 1, :, :]
    print(f"Empty frames: {empty_frames_idx}")
# %%
    viewer = napari.Viewer()
    kwargs = dict(scale=scale, blending="additive")
    viewer.add_image(im_arr, name="Original", **kwargs)
    viewer.add_image(nuc_arr_norm, name="VS nuclei", colormap="green", **kwargs)
    viewer.add_image(mem_arr_norm, name="VS membrane", colormap="magenta", **kwargs)
    # viewer.add_image(contourn_arr, name="Contourn", colormap="magma", **kwargs)
    # viewer.add_labels(fg_arr, name="Foreground", **kwargs)
# %%

    ### CHANGE THIS FUNCTION AS NEEDED
    import numpy as np
    from cellpose import models, core, io, plot
    from pathlib import Path
    import matplotlib.pyplot as plt


    io.logger_setup() # run this to get printing of progress

    #Check if colab notebook instance has GPU access
    if core.use_gpu()==False:
        raise ImportError("No GPU access, change your runtime")



    model = models.CellposeModel(gpu=True)
    cellpose = Cellpose(model_type = 'nuclei')

    flow_threshold = 0.4
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0

    masks, flows, styles = model.eval(img_selected_channels, batch_size=32, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                    normalize={"tile_norm_blocksize": tile_norm_blocksize})

    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, img_selected_channels, masks, flows[0])
    plt.tight_layout()
    plt.show()

   # fg_arr = array_apply(nuc_arr_norm, func = cellpose, out_zarr_kwargs={'shape': nuc_arr_norm.shape, 'dtype': np.uint32})
    fg_arr = array_apply(
        nuc_arr_norm,
        func=detect_foreground,
        sigma=90,
        out_zarr_kwargs={'shape': nuc_arr_norm.shape, 'dtype': bool},
    )
    #%%
    viewer = napari.Viewer()
    kwargs = dict(scale=scale, blending="additive")
    viewer.add_image(im_arr, name="Original", **kwargs)
    viewer.add_image(nuc_arr_norm, name="VS nuclei", colormap="green", **kwargs)
    viewer.add_image(mem_arr_norm, name="VS membrane", colormap="magenta", **kwargs)
    viewer.add_labels(fg_arr.astype(np.uint8), name="Foreground", **kwargs)
# %%
    ### CHANGE THIS FUNCTION AS NEEDED
    # contourn_arr = array_apply(fg_arr, func=inverted_edt)
    # contourn_arr = array_apply(
    #     nuc_arr_norm, mem_arr_norm,
    #     func=mem_nuc_contor,
    #     out_zarr_kwargs={'shape': nuc_arr_norm.shape, 'dtype': np.float32})
    # contourn_arr = array_apply(
    #     nuc_arr_norm,
    #     mem_arr_norm,
    #     func=generate_gradient_map,
    #     out_zarr_kwargs={'shape': mem_arr_norm.shape, 'dtype': np.float32},
    # )

    contourn_arr = array_apply(
        nuc_arr_norm,
        func=nuclei_split_contourn,
        out_zarr_kwargs={'shape': mem_arr_norm.shape, 'dtype': np.float32},
    )

# %%   
    viewer = napari.Viewer()
    kwargs = dict(scale=scale, blending="additive")
    viewer.add_image(im_arr, name="Original", **kwargs)
    viewer.add_image(nuc_arr_norm, name="VS nuclei", colormap="green", **kwargs)
    viewer.add_image(mem_arr_norm, name="VS membrane", colormap="magenta", **kwargs)
    viewer.add_image(contourn_arr, name="Contourn", colormap="magma", **kwargs)
    viewer.add_labels(fg_arr.astype(np.uint8), name="Foreground", **kwargs)
#%%
    ## CHANGE THE CONFIG AS NEEDED

    # Ultrack config
    cfg = MainConfig()
    cfg.data_config.working_dir = track_output_path

    cfg.segmentation_config.min_area = 2_800
    cfg.segmentation_config.max_area = 80_000
    cfg.segmentation_config.n_workers = 10
    cfg.segmentation_config.min_frontier = 0.5
    cfg.segmentation_config.max_noise = 0

    cfg.linking_config.n_workers = 10
    cfg.linking_config.max_distance = 15
    cfg.linking_config.distance_weight = -0.0001
    cfg.linking_config.max_neighbors = 3

    cfg.tracking_config.n_threads = 14
    cfg.tracking_config.disappear_weight = -0.0001
    cfg.tracking_config.appear_weight = -0.001
    cfg.tracking_config.division_weight = -0.0001
    cfg.tracking_config.bias = 0.0001

    print(cfg)

    tracker = Tracker(cfg)
    tracker.track(
        detection=fg_arr,
        edges=contourn_arr,
        scale=scale,
        overwrite=True,
    )


    tracks_df, graph = tracker.to_tracks_layer()
    tracks_path = track_output_path / "tracks.csv"
    tracks_df.to_csv(tracks_path, index=False)
    segments = tracker.to_zarr(
        tracks_df=tracks_df,
        store_or_path=track_output_path / "track.zarr",
        overwrite=True,
    )

    viewer = napari.Viewer()
    kwargs = dict(scale=scale, blending="additive")
    viewer.add_image(im_arr, name="Original", **kwargs)
    viewer.add_image(nuc_arr_norm, name="VS nuclei", colormap="green", **kwargs)
    viewer.add_image(mem_arr, name="VS membrane", colormap="magenta", **kwargs)
    viewer.add_image(contourn_arr, name="Contourn", colormap="magma", **kwargs)
    viewer.add_labels(fg_arr.astype(np.uint8), name="Foreground", **kwargs)
    viewer.add_labels(segments, name="Segments", scale=scale)
    viewer.add_tracks(
        tracks_df[["track_id", "t", "y", "x"]],
        graph=graph,
        name="Tracks",
        scale=scale,
        colormap="hsv",
        blending="opaque",
    )
    # %%
    # Save the foreground
    import zarr
    import toml

    with open(track_output_path/"config.toml", mode="w") as f:
        toml.dump(cfg.dict(by_alias=True), f)

    z = zarr.array(fg_arr, dtype="uint32")
    zarr.save("track_output_path/foreground.zarr", z)

    z = zarr.array(contourn_arr, dtype="float32")
    zarr.save("track_output_path/contourn_arr.zarr", z)


    # %%
    

    ## UNCOMMENT IF YOU WANT TO SAVE FOREGROUND AND CONTOURN

    # napari.run()


# %%
if __name__ == "__main__":
    main()