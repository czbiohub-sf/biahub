#%%
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

import click
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iohub.ngff import open_ome_zarr
from numpy.typing import ArrayLike
from pystackreg import StackReg
from scipy.fftpack import next_fast_len
from tqdm import tqdm
from waveorder.focus import focus_from_transverse_band
import napari
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
NA_DET = 1.35
LAMBDA_ILL = 0.500
center_crop_xy = [800, 800]
radius_um = 2.0
#%%
with open_ome_zarr(
        '/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/1-preprocess/label-free/1-virtual-stain/2024_11_21_A549_TOMM20_DENV.zarr/B/1/000000'
    ) as vs_ds:
        T,C,Z,Y,X = vs_ds.data.dask_array().shape
#%%

def plot_imgs_3d(img_ref, img_mov, corrected_img, output_path, title):
    if img_ref.ndim == 3:
        img_ref_xy = img_ref[img_ref.shape[0] // 2, :, :]
        img_ref_xz = img_ref[:, img_ref.shape[1] // 2, :]
        img_ref_yz = img_ref[:, :, img_ref.shape[2] // 2]
    else:
        raise ValueError("Reference image must be 3D")

    if img_mov.ndim == 3:
        img_mov_xy = img_mov[img_mov.shape[0] // 2, :, :]
        img_mov_xz = img_mov[:, img_mov.shape[1] // 2, :]
        img_mov_yz = img_mov[:, :, img_mov.shape[2] // 2]
    else:
        raise ValueError("Moving image must be 3D")

    if corrected_img.ndim == 3:
        corrected_img_xy = corrected_img[corrected_img.shape[0] // 2, :, :]
        corrected_img_xz = corrected_img[:, corrected_img.shape[1] // 2, :]
        corrected_img_yz = corrected_img[:, :, corrected_img.shape[2] // 2]
    else:
        raise ValueError("Corrected image must be 3D")

    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    # Row 0: XY planes
    axs[0, 0].imshow(img_ref_xy, cmap="gray")
    axs[0, 0].set_title("Ref: XY")
    axs[0, 1].imshow(img_mov_xy, cmap="gray")
    axs[0, 1].set_title("Mov: XY")
    axs[0, 2].imshow(corrected_img_xy, cmap="gray")
    axs[0, 2].set_title("Reg: XY")

    # Row 1: XZ planes
    axs[1, 0].imshow(img_ref_xz, cmap="gray")
    axs[1, 0].set_title("Ref: XZ")
    axs[1, 1].imshow(img_mov_xz, cmap="gray")
    axs[1, 1].set_title("Mov: XZ")
    axs[1, 2].imshow(corrected_img_xz, cmap="gray")
    axs[1, 2].set_title("Reg: XZ")

    # Row 2: YZ planes
    axs[2, 0].imshow(img_ref_yz, cmap="gray")
    axs[2, 0].set_title("Ref: YZ")
    axs[2, 1].imshow(img_mov_yz, cmap="gray")
    axs[2, 1].set_title("Mov: YZ")
    axs[2, 2].imshow(corrected_img_yz, cmap="gray")
    axs[2, 2].set_title("Reg: YZ")

 # Show grid and axes for all subplots
    for ax in axs.flat:
        ax.grid(True, linestyle='--', linewidth=0.5, color='white')
        ax.tick_params(color='white', labelcolor='black')
        ax.set_xticks(np.linspace(0, ax.images[0].get_array().shape[1], 5))
        ax.set_yticks(np.linspace(0, ax.images[0].get_array().shape[0], 5))


    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    #plt.savefig(output_path)
    plt.show()

#%%
focus_z_image = []
focus_z_cropped_image = []
focus_z_filtered_image = []
focus_z_cropped_filtered_image = []
for t in range(20):
    print(f"t={t}")
    # --- Load virtual-stain data ---


    with open_ome_zarr(
        '/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/1-preprocess/label-free/1-virtual-stain/2024_11_21_A549_TOMM20_DENV.zarr/B/1/000000'
    ) as vs_ds:
        nuc_arr = np.asarray(vs_ds.data.dask_array()[t, 0, :, :, :])
    
    min_val = np.min(nuc_arr)
    max_val = np.max(nuc_arr)
    print(f"min_val: {min_val}")
    print(f"max_val: {max_val}")
    # viewer = napari.Viewer()
    # viewer.add_image(nuc_arr, name="nuc_arr")
    # napari.run()
    
    threshold = threshold_otsu(nuc_arr)
    nuc_arr_mask = nuc_arr > threshold

    print(f"Otsu threshold: {threshold:.4f}")

    # --- Load reconstruction to get pixel sizes ---
    with open_ome_zarr(
        '/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/1-preprocess/label-free/0-reconstruct/2024_11_21_A549_TOMM20_DENV.zarr/B/1/000000'
    ) as ds:
        image = np.asarray(ds.data.dask_array()[t, 0, :, :, :])
        Z, Y, X = image.shape
        _, _, z_spacing, y_spacing, x_spacing = ds.scale  # unpack pixel spacing
        pixel_size = x_spacing
        print(f"Pixel size (Z, Y, X): {pixel_size}")


    # --- Physically correct 3D dilation using distance transform ---
    radius_um = 2.0  # desired physical dilation radius (µm)

    dist = distance_transform_edt(~nuc_arr_mask, sampling=pixel_size)
    dilated_nuc_arr_mask = dist <= radius_um

    image_cropped = image[
            :,
            Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2,
            X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2,
        ] 
    # --- Focus estimation ---
    z_idx_focus_z_image = focus_from_transverse_band(
        image,
        NA_det=NA_DET,
        lambda_ill=LAMBDA_ILL,
        pixel_size=pixel_size,
    )
    focus_z_image.append(z_idx_focus_z_image)

    z_idx_focus_z_cropped_image = focus_from_transverse_band(
        image_cropped,
        NA_det=NA_DET,
        lambda_ill=LAMBDA_ILL,
        pixel_size=pixel_size,
    )
    focus_z_cropped_image.append(z_idx_focus_z_cropped_image)
        # --- Apply mask ---
    filtered_image = image * dilated_nuc_arr_mask

    filtered_image_cropped = filtered_image[
            :,
            Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2,
            X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2,
        ] 
    # --- Focus estimation ---
    z_idx_focus_z_filtered_image = focus_from_transverse_band(
        filtered_image,
        NA_det=NA_DET,
        lambda_ill=LAMBDA_ILL,
        pixel_size=pixel_size,
    )

    z_idx_focus_z_cropped_filtered_image = focus_from_transverse_band(
        filtered_image_cropped,
        NA_det=NA_DET,
        lambda_ill=LAMBDA_ILL,
        pixel_size=pixel_size,
    )
    focus_z_filtered_image.append(z_idx_focus_z_filtered_image)
    focus_z_cropped_filtered_image.append(z_idx_focus_z_cropped_filtered_image)

    print(f"t={t} focus slice index: {z_idx_focus_z_image}")
    print(f"t={t} cropped focus slice index: {z_idx_focus_z_cropped_image}")
    print(f"t={t} filtered focus slice index: {z_idx_focus_z_filtered_image}")
    print(f"t={t} cropped filtered focus slice index: {z_idx_focus_z_cropped_filtered_image}")
    

#%%
tforms = []
for focus_idx in [focus_z_image, focus_z_cropped_image, focus_z_filtered_image, focus_z_cropped_filtered_image]:
    z_focus_shift = [np.eye(4)]
    z_val = focus_idx[0]
    for z_val_next in focus_idx[1:]:
        shift = np.eye(4)
        shift[0, 3] = z_val_next - z_val
        z_focus_shift.append(shift)
    transform = np.array(z_focus_shift)
    tforms.append(transform)
    print(f"transform: {transform}")

# apply transforms to images
from skimage.transform import warp
transformed_image = warp(image, tforms[-1], mode='constant', cval=0.0)
viewer = napari.Viewer()
viewer.add_image(image, name="image")
viewer.add_image(transformed_image, name="transformed_image")
napari.run()


#%%
import ants
from biahub.register import convert_transform_to_ants



stabilized_image = np.zeros((20, Z, Y, X))
image = np.zeros((20, Z, Y, X))

tform = tforms[-1]
i = 0
for t in range(21):
    with open_ome_zarr(
    '/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/1-preprocess/label-free/0-reconstruct/2024_11_21_A549_TOMM20_DENV.zarr/B/1/000000') as ds:
        img_t = np.asarray(ds.data.dask_array()[t, 0, :, :, :])

    image[i] = img_t
    target_ants = ants.from_numpy(img_t.astype(np.float32))
    zyx_data = np.nan_to_num(img_t, nan=0)
    zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
    
    t_form_ants = convert_transform_to_ants(tform[t])
    stabilized_image[i] = t_form_ants.apply_to_image(
        zyx_data_ants, reference=target_ants, interpolation="linear"
    ).numpy()
    i += 1

#%%
import ants
from biahub.register import convert_transform_to_ants
from biahub.estimate_stabilization import phase_cross_corr_padding, phase_cross_corr
def get_tform_from_pcc(
    target: ArrayLike,
    source: ArrayLike,
    function_type: Literal["custom_padding", "custom"] = "custom",
    normalization: Optional[Literal["magnitude", "classic"]] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[ArrayLike, Tuple[int, int, int]]:
    """
    Get the transformation matrix from phase cross correlation.

    Parameters
    ----------
    t : int
        Time index.
    source_channel_tzyx : da.Array
        Source channel data.
    target_channel_tzyx : da.Array
        Target channel data.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    ArrayLike
        Transformation matrix.
    """


   
    shift, corr = phase_cross_corr_padding(
        target, source, normalization=normalization, output_path=output_path
    )
    if verbose:
        click.echo(f"Time {t}: shift (dz,dy,dx) = {shift[0]}, {shift[1]}, {shift[2]}")

    dz, dy, dx = shift

    transform = np.eye(4)
    transform[2, 3] = dx
    transform[1, 3] = dy
    transform[0, 3] = dz
    if verbose:
        click.echo(f"transform: {transform}")

    return transform, shift, corr
  
def apply_stabilization_transform_custom(
    zyx_data: ArrayLike,
    transform: ArrayLike,
):
    tx_shifts = convert_transform_to_ants(transform)

    zyx_data = np.nan_to_num(zyx_data, nan=0)
    zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
    stabilized_zyx = tx_shifts.apply_to_image(
        zyx_data_ants,interpolation="linear"
    ).numpy()
    return stabilized_zyx

def get_z_tform_from_focus(focus_idx_ref, focus_idx_mov):
    print(f"focus_idx_ref: {focus_idx_ref}, focus_idx_mov: {focus_idx_mov}")
    transform = np.eye(4)
    transform[0, 3] =  focus_idx_mov - focus_idx_ref 
    return transform
#%%
t_ref = 0
t_mov = 1

with open_ome_zarr(
        '/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/1-virtual-stain/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr/C/1/000000'
    ) as vs_ds:
        nuc_arr_ref = np.asarray(vs_ds.data.dask_array()[t_ref, 0, :,  Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2, X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2])
        nuc_arr_mov = np.asarray(vs_ds.data.dask_array()[t_mov, 0, :,  Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2, X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2])



with open_ome_zarr(
        '/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/0-reconstruct/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr/C/1/000000'
    ) as ds:
        image_ref = np.asarray(ds.data.dask_array()[t_ref, 0, :, Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2, X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2])
        image_mov = np.asarray(ds.data.dask_array()[t_mov, 0, :, Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2, X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2])
        _, _, z_spacing, y_spacing, x_spacing = ds.scale  # unpack pixel spacing
        pixel_size = x_spacing
        print(f"Pixel size (Z, Y, X): {pixel_size}")

#%%
from scipy.ndimage import shift

#image_mov = shift(image_ref, (10,0,0), mode='constant', cval=0.0)
#nuc_arr_mov = shift(nuc_arr_ref, (10,0,0), mode='constant', cval=0.0)
print(f"t_ref={t_ref}, t_mov={t_mov}")
# --- Load virtual-stain data ---
threshold_ref = threshold_otsu(nuc_arr_ref)
nuc_arr_mask_ref = nuc_arr_ref > threshold_ref

threshold_mov = threshold_otsu(nuc_arr_mov)
nuc_arr_mask_mov = nuc_arr_mov > threshold_mov

print(f"Otsu threshold ref: {threshold_ref:.4f}")
print(f"Otsu threshold mov: {threshold_mov:.4f}")

# --- Physically correct 3D dilation using distance transform ---
radius_um = 2.0  # desired physical dilation radius (µm)

dist_ref = distance_transform_edt(~nuc_arr_mask_ref, sampling=pixel_size)
dilated_nuc_arr_mask_ref = dist_ref <= radius_um

dist_mov = distance_transform_edt(~nuc_arr_mask_mov, sampling=pixel_size)
dilated_nuc_arr_mask_mov = dist_mov <= radius_um

filtered_image_ref = image_ref * dilated_nuc_arr_mask_ref
filtered_image_mov = image_mov * dilated_nuc_arr_mask_mov


z_idx_focus_z_image_ref = focus_from_transverse_band(
    image_ref,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"z_idx_focus_z_image_ref: {z_idx_focus_z_image_ref}")
z_idx_focus_z_image_mov = focus_from_transverse_band(
    image_mov,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"z_idx_focus_z_image_mov: {z_idx_focus_z_image_mov}")

transform = get_z_tform_from_focus(z_idx_focus_z_image_ref, z_idx_focus_z_image_mov)
# inverse_transform = np.linalg.inv(transform)
print(f"transform: {transform}")
#print(f"inverse_transform: {inverse_transform}")

# # --- Focus estimation ---

# transform, shift, _ = get_tform_from_pcc(
#     target=image[t-1],
#     source=image[t],
#     verbose=True,
#     function_type="custom",
# )
print(f"transform: {transform}")
print(f"shift: {shift}")

image_shifted = apply_stabilization_transform_custom(
    image_mov,
    transform,
)

z_idx_focus_z_filtered_image_ref = focus_from_transverse_band(
    filtered_image_ref,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"z_idx_focus_z_filtered_image_ref: {z_idx_focus_z_filtered_image_ref}")
z_idx_focus_z_filtered_image_mov = focus_from_transverse_band(
    filtered_image_mov,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"z_idx_focus_z_filtered_image_mov: {z_idx_focus_z_filtered_image_mov}")

transform_filtered = get_z_tform_from_focus(z_idx_focus_z_filtered_image_ref, z_idx_focus_z_filtered_image_mov)
print(f"transform_filtered: {transform_filtered}")

# transform_filtered, shift_filtered, _ = get_tform_from_pcc(
#     target=filtered_image_ref,
#     source=filtered_image,
#     verbose=True,
#     function_type="custom",
# )
# print(f"transform_filtered: {transform_filtered}")
# print(f"shift_filtered: {shift_filtered}")


filtered_image_shifted = apply_stabilization_transform_custom(
    image_mov,
    transform_filtered,
)

z_idx_focus_z_image_shifted = focus_from_transverse_band(
    image_shifted,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"z_idx_focus_z_image_shifted: {z_idx_focus_z_image_shifted}")

filtered_image_shifted_masked = apply_stabilization_transform_custom(
    filtered_image_mov,
    transform_filtered,
)
z_idx_focus_z_filtered_image_shifted = focus_from_transverse_band(
    filtered_image_shifted_masked,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"z_idx_focus_z_filtered_image_shifted: {z_idx_focus_z_filtered_image_shifted}")


#plot_imgs_3d(image_ref, image_mov, image_shifted, "image_shifted.png", "image_shifted")
#plot_imgs_3d(image_ref, image_mov, filtered_image_shifted, "filtered_image_shifted.png", "filtered_image_shifted")

viewer = napari.Viewer()
viewer.add_image(image_ref, name="image_ref")
viewer.add_image(image_mov, name="image_mov")
viewer.add_image(image_shifted, name="image_shifted")
viewer.add_image(filtered_image_shifted, name="filtered_image_shifted")
napari.run()


   #%%
from scipy.ndimage import shift
shifted_skimage = shift(image[t], shift_filtered, mode='constant', cval=0.0)

viewer = napari.Viewer()
viewer.add_image(image[t-1], name="image_ref")
viewer.add_image(image[t], name="image_mov")
viewer.add_image(image_shifted, name="image_shifted")
viewer.add_image(filtered_image_shifted, name="filtered_image_shifted")
viewer.add_image(shifted_skimage, name="shifted_skimage")
napari.run()

  

#%%
with open_ome_zarr(
    '/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/1-virtual-stain/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr/C/1/000000'
) as vs_ds:
    nuc_arr_ref = np.asarray(vs_ds.data.dask_array()[0, 0, :, :, :])
    nuc_arr_mov = np.asarray(vs_ds.data.dask_array()[43, 0, :, :, :])

threshold_ref = threshold_otsu(nuc_arr_ref)
nuc_arr_mask_ref = nuc_arr_ref > threshold_ref

threshold_mov = threshold_otsu(nuc_arr_mov)
nuc_arr_mask_mov = nuc_arr_mov > threshold_mov

print(f"Otsu threshold ref: {threshold_ref:.4f}")
print(f"Otsu threshold mov: {threshold_mov:.4f}")

# --- Load reconstruction to get pixel sizes ---
with open_ome_zarr(
    '/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/0-reconstruct/2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr/C/1/000000'
) as ds:
    image_ref = np.asarray(ds.data.dask_array()[0, 0, :, :, :])
    image_mov = np.asarray(ds.data.dask_array()[43, 0, :, :, :])
    Z, Y, X = image_ref.shape
    _, _, z_spacing, y_spacing, x_spacing = ds.scale  # unpack pixel spacing
    pixel_size = x_spacing
    print(f"Pixel size (Z, Y, X): {pixel_size}")


# --- Physically correct 3D dilation using distance transform ---
radius_um = 2.0  # desired physical dilation radius (µm)

dist_ref = distance_transform_edt(~nuc_arr_mask_ref, sampling=pixel_size)
dilated_nuc_arr_mask_ref = dist_ref <= radius_um


dist_mov = distance_transform_edt(~nuc_arr_mask_mov, sampling=pixel_size)
dilated_nuc_arr_mask_mov = dist_mov <= radius_um

#--- Crop images ---
image_cropped_ref = image_ref[
        :,
        Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2,
        X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2,
    ] 

image_cropped_mov = image_mov[
        :,
        Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2,
        X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2,
    ] 
# --- Focus estimation ---
z_idx_ref = focus_from_transverse_band(
    image_ref,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=0 focus slice index: {z_idx_ref}")

z_idx_mov = focus_from_transverse_band(
    image_mov,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=43 focus slice index: {z_idx_mov}")

z_idx_cropped_ref = focus_from_transverse_band(
    image_cropped_ref,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=0 cropped focus slice index: {z_idx_cropped_ref}")

z_idx_cropped_mov = focus_from_transverse_band(
    image_cropped_mov,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=43 cropped focus slice index: {z_idx_cropped_mov}")

    # --- Apply mask ---
filtered_image_ref = image_ref * dilated_nuc_arr_mask_ref

filtered_image_cropped_ref = filtered_image_ref[
        :,
        Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2,
        X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2,
    ] 


filtered_image_mov = image_mov * dilated_nuc_arr_mask_mov
filtered_image_cropped_mov = filtered_image_mov[
        :,
        Y // 2 - center_crop_xy[1] // 2 : Y // 2 + center_crop_xy[1] // 2,
        X // 2 - center_crop_xy[0] // 2 : X // 2 + center_crop_xy[0] // 2,
    ] 
# --- Focus estimation ---


z_idx_filtered_ref = focus_from_transverse_band(
    filtered_image_ref,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=0 filtered focus slice index: {z_idx_filtered_ref}")


z_idx_filtered_mov = focus_from_transverse_band(
    filtered_image_mov,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=43 filtered focus slice index: {z_idx_filtered_mov}")

z_idx_filtered_cropped_ref = focus_from_transverse_band(
    filtered_image_cropped_ref,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=0 filtered cropped focus slice index: {z_idx_filtered_cropped_ref}")


z_idx_filtered_cropped_mov = focus_from_transverse_band(
    filtered_image_cropped_mov,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=43 filtered cropped focus slice index: {z_idx_filtered_cropped_mov}")



viewer = napari.Viewer()
viewer.add_image(image_ref, name="image_ref")
viewer.add_image(image_mov, name="image_mov")
viewer.add_image(image_cropped_ref, name="image_cropped_ref")
viewer.add_image(image_cropped_mov, name="image_cropped_mov")
viewer.add_image(filtered_image_ref, name="filtered_image_ref")
viewer.add_image(filtered_image_mov, name="filtered_image_mov")
viewer.add_image(filtered_image_cropped_ref, name="filtered_image_cropped_ref")
viewer.add_image(filtered_image_cropped_mov, name="filtered_image_cropped_mov")
napari.run()

#%%

from biahub.estimate_stabilization import phase_cross_corr_padding

shift_image, corr_image= phase_cross_corr_padding(
    image_ref,
    image_mov,
)

print(f"shift_image: {shift_image}")
print(f"corr_image: {corr_image}")


shift_image_cropped, corr_image_cropped = phase_cross_corr_padding(
    image_cropped_ref,
    image_cropped_mov,
)

print(f"shift_image_cropped: {shift_image_cropped}")
print(f"corr_image_cropped: {corr_image_cropped}")


shift_filtered_image, corr_filtered_image = phase_cross_corr_padding(
    filtered_image_ref,
    filtered_image_mov,
)

print(f"shift_filtered_image: {shift_filtered_image}")
print(f"corr_filtered_image: {corr_filtered_image}")


shift_filtered_image_cropped, corr_filtered_image_cropped = phase_cross_corr_padding(
    filtered_image_cropped_ref,
    filtered_image_cropped_mov,
)

print(f"shift_filtered_image_cropped: {shift_filtered_image_cropped}")
print(f"corr_filtered_image_cropped: {corr_filtered_image_cropped}")

#%%
print(f"shift_image: {shift_image}")


print(f"shift_image_cropped: {shift_image_cropped}")

print(f"shift_filtered_image: {shift_filtered_image}")


print(f"shift_filtered_image_cropped: {shift_filtered_image_cropped}")
#%%
from scipy.ndimage import shift

shifted_image_mov = shift(image_mov, shift_image, mode='constant', cval=0.0)

shifted_image_cropped_mov = shift(image_cropped_mov, shift_image_cropped, mode='constant', cval=0.0)

shifted_filtered_image_mov = shift(filtered_image_mov, shift_filtered_image, mode='constant', cval=0.0)

shifted_filtered_image_cropped_mov = shift(filtered_image_cropped_mov, shift_filtered_image_cropped, mode='constant', cval=0.0)

viewer = napari.Viewer()
viewer.add_image(image_ref, name="image_ref")
viewer.add_image(shifted_image_mov, name="shifted_image_mov")
viewer.add_image(shifted_image_cropped_mov, name="shifted_image_cropped_mov")
viewer.add_image(shifted_filtered_image_mov, name="shifted_filtered_image_mov")
viewer.add_image(shifted_filtered_image_cropped_mov, name="shifted_filtered_image_cropped_mov")
napari.run()
#%%


z_idx_mov_shifted = focus_from_transverse_band(
    shifted_image_mov,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=43 focus slice index: {z_idx_mov_shifted}")


z_idx_cropped_mov_shifted = focus_from_transverse_band(
    shifted_image_cropped_mov,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=43 cropped focus slice index: {z_idx_cropped_mov_shifted}")

    # --- Apply mask ---

# --- Focus estimation ---



z_idx_filtered_mov_shifted = focus_from_transverse_band(
    shifted_filtered_image_mov,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=43 filtered focus slice index: {z_idx_filtered_mov_shifted}")


z_idx_filtered_cropped_mov_shifted = focus_from_transverse_band(
    shifted_filtered_image_cropped_mov,
    NA_det=NA_DET,
    lambda_ill=LAMBDA_ILL,
    pixel_size=pixel_size,
)
print(f"t=43 filtered cropped focus slice index: {z_idx_filtered_cropped_mov_shifted}")



# %%
