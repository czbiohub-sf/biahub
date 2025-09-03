# %%
import ants
import numpy as np
from pathlib import Path
from iohub import open_ome_zarr
from biahub.register import convert_transform_to_ants
import napari
from biahub.estimate_registration import (
    filter_matches,
    detect_bead_peaks,
    get_matches_from_beads, 
    estimate_transform)
from biahub.settings import EstimateRegistrationSettings
import numpy as np


# %%%

dataset = '2024_11_21_A549_TOMM20_DENV'
fov = 'C/1/000000'
root_path = Path(f'/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/{dataset}/1-preprocess/')
t_idx = 76
lf_data_path = root_path / f"label-free/0-reconstruct/{dataset}.zarr" / fov
ls_data_path = root_path / f"light-sheet/raw/0-deskew/{dataset}.zarr" / fov

visualize = True

# %%
config_dict = {
    "target_channel_name": "Phase3D",
    "source_channel_name": "GFP EX488 EM525-45",
    "beads_match_settings": {
        "algorithm": "hungarian",
        "source_peaks_settings": {
            "threshold_abs": 110,
            "nms_distance": 16,
            "min_distance": 0,
            "block_size": [8, 8, 8]
        },
        "target_peaks_settings": {
            "threshold_abs": 0.8,
            "nms_distance": 16,
            "min_distance": 0,
            "block_size": [8, 8, 8]
        },
        "match_descriptor_settings": {
            "distance_metric": "euclidean",
            "max_ratio": 1,
            "cross_check": True
        },
        "hungarian_match_settings": {
            "distance_metric": "euclidean",
            "cost_threshold": 0.05,
            "cross_check": True,
            "max_ratio": 0.95,
            "edge_graph_settings": {
                "method": "knn",
                "k": 10
            },
            "cost_matrix_settings": {
                "normalize": False,
                "weights": {
                    "dist": 0.2,
                    "edge_angle": 1,
                    "edge_length": 1,
                    "pca_dir": 0,
                    "pca_aniso": 0,
                    "edge_descriptor": 0
                }
            }
        },
        "filter_distance_threshold": 0.95,
        "filter_angle_threshold": 30,
    },
    "affine_transform_settings": {
        "transform_type": "similarity",
        "approx_transform": [
            [1, 0, 0, 0],
            [0, 0, -1.288, 1960],
            [0, 1.288, 0, -460],
            [0.0, 0.0, 0.0, 1.0]
        ]
    },

    "verbose": True
}
config = EstimateRegistrationSettings(**config_dict)

# %% Load data
with open_ome_zarr(lf_data_path) as target_ds:
    target_channel_name = target_ds.channel_names
    target_channel_index = target_ds.channel_names.index(config.target_channel_name)
    target_data = np.asarray(target_ds.data[t_idx, target_channel_index]) # take phase channel
    target_scale = target_ds.scale

with open_ome_zarr(ls_data_path) as source_ds:
    source_channel_name = source_ds.channel_names
    source_channel_index = source_channel_name.index(config.source_channel_name)
    source_data = np.asarray(source_ds.data[t_idx, source_channel_index]) # take mCherry channel or the GFP channel (depending where the beads are)
    source_scale = source_ds.scale

# Register LS data with approx tranform
source_data_ants = ants.from_numpy(source_data)
target_data_ants = ants.from_numpy(target_data)

source_data_reg_ants = convert_transform_to_ants(np.asarray(config.affine_transform_settings.approx_transform)).apply_to_image(
    source_data_ants, reference=target_data_ants
)
source_data_reg = source_data_reg_ants.numpy()


# %%
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(target_data, name='LF')
    viewer.add_image(source_data_reg, name='LS')

# %% Detect peaks in LS data

source_peaks, target_peaks = detect_bead_peaks(
    source_data_reg,
    target_data,
    config.beads_match_settings.source_peaks_settings,
    config.beads_match_settings.target_peaks_settings,
    verbose=True,
    filter_dirty_peaks=False
)

#%%
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(source_data_reg, name='LS')
    viewer.add_points(
        source_peaks, name='peaks local max LS', size=20, symbol='disc', face_color='magenta'
    )
# %%
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(target_data, name='LF')
    viewer.add_points(
        target_peaks, name='peaks local max LF', size=20, symbol='disc', face_color='green'
    )

# %%
matches = get_matches_from_beads(
    source_peaks,
    target_peaks,
    config.beads_match_settings,
    verbose=True
)


# %%
matches = filter_matches(
    matches,
    source_peaks,
    target_peaks,
    angle_threshold=config.beads_match_settings.filter_angle_threshold,
    distance_threshold=config.beads_match_settings.filter_distance_threshold,
    verbose=True
)
# %%
if visualize:
    # visualize matches
    viewer = napari.Viewer()
    viewer.add_image(target_data, name='LF', contrast_limits=(0.5, 1.0),blending='additive')
    viewer.add_points(
        target_peaks, name='LF peaks', size=12, symbol='ring', face_color='yellow',blending='additive'
    )
    viewer.add_image(source_data_reg, name='LS', contrast_limits=(110, 230), blending='additive', colormap='green')
    viewer.add_points(
        source_peaks, name='LS peaks', size=12, symbol='ring', face_color='red',blending='additive'
    )
    # Project in 3D to be able to view the lines
    viewer.add_shapes(
        data=[np.asarray([source_peaks[m[0]], target_peaks[m[1]]]) for m in matches],
        shape_type='line',
        edge_width=5,
        blending='additive',
    )
    viewer.dims.ndisplay = 3


# %% Register LS data using compount transform

tform = estimate_transform(
    matches,
    source_peaks,
    target_peaks,
    config.affine_transform_settings,
    verbose=True
)


compount_tform = np.asarray(config.affine_transform_settings.approx_transform) @ tform.inverse.params
compount_tform_ants = convert_transform_to_ants(compount_tform)
source_data_reg_2 = compount_tform_ants.apply_to_image(
    source_data_ants, reference=target_data_ants
).numpy()

if visualize:
    viewer = napari.Viewer()
    viewer.add_image(target_data, name='LF', contrast_limits=(-0.5, 1.0))
    viewer.add_image(
    source_data_reg,
    name='LS approx registered',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='green'
    )
    viewer.add_image(
        source_data_reg_2,
        name='LS registered',
        contrast_limits=(110, 230),
        blending='additive',
        colormap='magenta'
    )

# %%

fov_cell = 'C/2/000000'
root_path = Path(f'/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/{dataset}/1-preprocess/')
lf_data_sample_path = root_path / f"label-free/0-reconstruct/{dataset}.zarr" / fov_cell
ls_data_sample_path = root_path / f"light-sheet/raw/0-deskew/{dataset}.zarr" / fov_cell

with open_ome_zarr(lf_data_sample_path) as target_ds:
    target_channel_name = target_ds.channel_names
    target_channel_index = target_ds.channel_names.index(config.target_channel_name)
    target_sample_data = np.asarray(target_ds.data[t_idx, target_channel_index]) # take phase channel
    target_scale = target_ds.scale

with open_ome_zarr(ls_data_sample_path) as source_ds:
    source_channel_name = source_ds.channel_names
    source_channel_index = source_channel_name.index(config.source_channel_name)
    source_sample_data = np.asarray(source_ds.data[t_idx, source_channel_index]) # take mCherry channel or the GFP channel (depending where the beads are)
    source_scale = source_ds.scale

# Register LS data with approx tranform
source_data_sample_ants = ants.from_numpy(source_sample_data)
target_data_sample_ants = ants.from_numpy(target_sample_data)

source_data_sample_reg_ants = convert_transform_to_ants(np.asarray(config.affine_transform_settings.approx_transform)).apply_to_image(
    source_data_sample_ants, reference=target_data_sample_ants
)
source_data_sample_reg = source_data_sample_reg_ants.numpy()

source_data_sample_reg_2 = compount_tform_ants.apply_to_image(
    source_data_sample_ants, reference=target_data_sample_ants
).numpy()

if visualize:
    viewer = napari.Viewer()
    viewer.add_image(target_sample_data, name='LF', contrast_limits=(-0.5, 1.0))
    viewer.add_image(
    source_data_sample_reg,
    name='LS approx registered',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='green'
    )
    viewer.add_image(
        source_data_sample_reg_2,
        name='LS registered',
        contrast_limits=(110, 230),
        blending='additive',
        colormap='magenta'
    )

# %%
