# %%
import ants
import numpy as np
from pathlib import Path
from iohub import open_ome_zarr
from biahub.characterize_psf import detect_peaks
from biahub.register import convert_transform_to_ants
import napari
from skimage.transform import AffineTransform
from skimage.feature import match_descriptors
from biahub.estimate_registration import (
    build_edge_graph,
    compute_cost_matrix,
    match_hungarian_global_cost,
    filter_matches,
    detect_bead_peaks,
    get_matches_from_beads, 
    estimate_transform)
from biahub.settings import EstimateStabilizationSettings
import numpy as np


# %%%

dataset = '2025_05_21_A549_MAP1LC3B_RPL36_GFP_sensor_ZIKV_DENV'
fov = 'A/3/000000'
root_path = Path(f'/hpc/projects/intracellular_dashboard/viral-sensor/{dataset}/1-preprocess/')

t_target = 0
t_source = 10
lf_data_path = root_path / f"label-free/0-reconstruct/{dataset}.zarr" / fov
visualize = True

# %% Load data
with open_ome_zarr(lf_data_path) as target_ds:
    target_channel_name = target_ds.channel_names
    target_channel_index = target_ds.channel_names.index(config['target_channel_name'])
    target_data = np.asarray(target_ds.data[t_target, target_channel_index]) # take phase channel
    target_scale = target_ds.scale

with open_ome_zarr(lf_data_path) as source_ds:
    source_channel_name = source_ds.channel_names
    source_channel_index = source_channel_name.index(config['source_channel_name'])
    source_data = np.asarray(source_ds.data[t_source, source_channel_index]) # take mCherry channel or the GFP channel (depending where the beads are)
    source_scale = source_ds.scale

#%%
## If stabilizing LF beads, peaks threshold_abs == 0.8, if LS beads, peaks threshold_abs == 110

config_dict = {
    "stabilization_estimation_channel": "Phase3D",
    "stabilization_channels": ["Phase3D"],
    "stabilization_type": "xyz",
    "stabilization_method": "beads",
    "beads_match_settings": {
        "algorithm": "hungarian",
        "t_reference": "first",
        "source_peaks_settings": {
            "threshold_abs": 0.8,
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
            "cost_threshold": 0.1,
            "cross_check": True,
            "max_ratio": 1,
            "edge_graph_settings": {
                "method": "knn",
                "k": 5
            },
            "cost_matrix_settings": {
                "normalize": False,
                "weights": {
                    "dist": 0.5,
                    "edge_angle": 1.0,
                    "edge_length": 1.0,
                    "pca_dir": 0.0,
                    "pca_aniso": 0.0,
                    "edge_descriptor": 0.0
                }
            }
        },
        "filter_distance_threshold": 0.95,
        "filter_angle_threshold": 0
    },
    "affine_transform_settings": {
        "transform_type": "euclidean",
    },

    "verbose": True
}
config = EstimateStabilizationSettings(**config_dict)


# %%
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(target_data, name='Target data')
    viewer.add_image(source_data, name='Source data')

# %% Detect peaks in LS data

source_peaks, target_peaks = detect_bead_peaks(
    source_data,
    target_data,
    config.beads_match_settings.source_peaks_settings,
    config.beads_match_settings.target_peaks_settings,
    verbose=True,
    filter_dirty_peaks=True
)

#%%
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(source_data, name='Source data')
    viewer.add_points(
        source_peaks, name='peaks local max Source', size=20, symbol='disc', face_color='magenta'
    )
# %%
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(target_data, name='Target data')
    viewer.add_points(
        target_peaks, name='peaks local max Target', size=20, symbol='disc', face_color='green'
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
    viewer.add_image(target_data, name='Target data', contrast_limits=(0.5, 1.0),blending='additive')
    viewer.add_points(
        target_peaks, name='Target peaks', size=12, symbol='ring', face_color='yellow',blending='additive'
    )
    viewer.add_image(source_data, name='Source data', contrast_limits=(110, 230), blending='additive', colormap='green')
    viewer.add_points(
        source_peaks, name='Source peaks', size=12, symbol='ring', face_color='red',blending='additive'
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

compount_tform = tform.inverse.params
source_data_ants = ants.from_numpy(source_data)
target_data_ants = ants.from_numpy(target_data)
compount_tform_ants = convert_transform_to_ants(compount_tform)
source_data_reg_2 = compount_tform_ants.apply_to_image(
    source_data_ants, reference=target_data_ants
).numpy()

if visualize:
    viewer = napari.Viewer()
    viewer.add_image(target_data, name='Target data', contrast_limits=(-0.5, 1.0))
    viewer.add_image(
    source_data,
    name='Source data',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='green'
    )
    viewer.add_image(
        source_data_reg_2,
        name='Source data registered',
        contrast_limits=(110, 230),
        blending='additive',
        colormap='magenta'
    )

# %%
