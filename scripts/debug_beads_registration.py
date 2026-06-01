"""
Debug script for beads-based registration.

Interactive notebook-style script for developing and debugging the beads
registration pipeline. Walks through each step:
1. Load data (label-free reference + light-sheet moving volumes)
2. Apply approximate transform and visualize alignment
3. Detect bead peaks in both channels
4. Match beads and visualize correspondences
5. Estimate correction transform from matches
6. Run the full iterative optimization pipeline (optimize_transform)

Usage: Run cell-by-cell in an IDE with interactive Python support (e.g. VSCode).
Set ``visualize = True`` to open napari viewers at each stage.
"""

# %% Imports
import ants
import numpy as np
from pathlib import Path
from iohub import open_ome_zarr
from biahub.core.transform import Transform
import napari
from biahub.settings import EstimateRegistrationSettings
from biahub.registration.beads import (
    transform_from_matches,
    matches_from_beads,
    peaks_from_beads,
    optimize_transform,
)

# %% Dataset configuration
dataset = '2024_12_11_A549_LAMP1_DENV'
fov = 'C/1/000000'
root_path = Path(f'/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}/1-preprocess/')
t_idx = 3
lf_data_path = root_path / f"label-free/0-reconstruct/{dataset}.zarr" / fov
ls_data_path = root_path / f"light-sheet/raw/0-deskew/{dataset}.zarr" / fov

visualize = True

# %% Registration settings
config_dict = {
    "target_channel_name": "Phase3D",
    "source_channel_name": "GFP EX488 EM525-45",
    "beads_match_settings": { 
        "algorithm": "hungarian",
        "qc_settings": {
            "iterations": 2,
            "score_threshold": 0.40,
            "score_centroid_mask_radius": 6
        },
        "filter_matches_settings": {
            "min_distance_quantile": 0.05,
            "max_distance_quantile": 0.99,
            "angle_threshold": 0,
            "direction_threshold": 50
        },
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
        "hungarian_match_settings": {
            "distance_metric": "euclidean",
            "cost_threshold": 0.05,
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
                    "edge_angle": 1,
                    "edge_length": 1.0,
                    "pca_dir": 0.0,
                    "pca_aniso": 0.0,
                    "edge_descriptor": 0
                }
            }
        },
    },
    "affine_transform_settings": {
        "use_prev_t_transform": True,
        "transform_type": "affine",
        "compute_approx_transform": False,
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

# %% Load reference (label-free) and moving (light-sheet) volumes for a single timepoint
with open_ome_zarr(lf_data_path) as ref_ds:
    ref_channel_name = ref_ds.channel_names
    ref_channel_index = ref_ds.channel_names.index(config.target_channel_name)
    ref_data = np.asarray(ref_ds.data[t_idx, ref_channel_index])
    ref_scale = ref_ds.scale

with open_ome_zarr(ls_data_path) as mov_ds:
    mov_channel_name = mov_ds.channel_names
    mov_channel_index = mov_channel_name.index(config.source_channel_name)
    mov_data = np.asarray(mov_ds.data[t_idx, mov_channel_index])
    mov_scale = mov_ds.scale

# Convert to ANTs images (reused throughout the script)
mov_data_ants = ants.from_numpy(mov_data)
ref_data_ants = ants.from_numpy(ref_data)

# %% Compute approximate transform from voxel sizes (if not provided in config)
if config.affine_transform_settings.compute_approx_transform:
    from biahub.registration.utils import get_aprox_transform

    approx_transform = get_aprox_transform(
        mov_shape=mov_data.shape[-3:],
        ref_shape=ref_data.shape[-3:],
        pre_affine_90degree_rotation=-1,
        pre_affine_fliplr=False,
        verbose=True,
        ref_voxel_size=ref_scale,
        mov_voxel_size=mov_scale,
    )
    config.affine_transform_settings.approx_transform = approx_transform.to_list()
# %% Apply approximate transform to moving volume and visualize overlay
initial_transform = Transform(
    matrix=np.asarray(config.affine_transform_settings.approx_transform)
)
mov_data_reg_ants = initial_transform.to_ants().apply_to_image(
    mov_data_ants, reference=ref_data_ants
)
mov_data_reg = mov_data_reg_ants.numpy()

# %%
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(ref_data, name='LF (reference)')
    viewer.add_image(mov_data_reg, name='LS (approx registered)')

# %% Detect bead peaks in both the approximately registered moving and the reference volumes
mov_peaks, ref_peaks = peaks_from_beads(
    mov_data_reg,
    ref_data,
    config.beads_match_settings.source_peaks_settings,
    config.beads_match_settings.target_peaks_settings,
    verbose=True,
)


# %% Visualize detected moving peaks overlaid on the registered LS volume
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(mov_data_reg, name='LS (approx registered)')
    viewer.add_points(
        mov_peaks, name='LS peaks', size=20, symbol='disc', face_color='magenta'
    )

# %% Visualize detected reference peaks overlaid on the LF volume
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(ref_data, name='LF (reference)')
    viewer.add_points(
        ref_peaks, name='LF peaks', size=20, symbol='disc', face_color='green'
    )

# %% Match beads between moving and reference peak sets
matches = matches_from_beads(
    mov_peaks,
    ref_peaks,
    config.beads_match_settings,
    verbose=True
)

# %% Visualize matched bead pairs as 3D lines connecting corresponding peaks
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(ref_data, name='LF', contrast_limits=(0.5, 1.0), blending='additive')
    viewer.add_points(
        ref_peaks, name='LF peaks', size=12, symbol='ring',
        face_color='yellow', blending='additive'
    )
    viewer.add_image(
        mov_data_reg, name='LS', contrast_limits=(110, 230),
        blending='additive', colormap='green'
    )
    viewer.add_points(
        mov_peaks, name='LS peaks', size=12, symbol='ring',
        face_color='red', blending='additive'
    )
    viewer.add_shapes(
        data=[np.asarray([mov_peaks[m[0]], ref_peaks[m[1]]]) for m in matches],
        shape_type='line',
        edge_width=5,
        blending='additive',
    )
    viewer.dims.ndisplay = 3

# %% Estimate correction transform from matches and compose with approx transform
fwd_transform, inv_transform = transform_from_matches(
    matches=matches,
    mov_peaks=mov_peaks,
    ref_peaks=ref_peaks,
    affine_transform_settings=config.affine_transform_settings,
    ndim=mov_data_reg.ndim,
)

# Compose: apply approx transform first, then the bead-based correction
composed_transform = initial_transform @ inv_transform
mov_data_reg_2 = composed_transform.to_ants().apply_to_image(
    mov_data_ants, reference=ref_data_ants
).numpy()

# %% Compare approx-only vs bead-corrected registration
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(ref_data, name='LF (reference)', contrast_limits=(-0.5, 1.0))
    viewer.add_image(
        mov_data_reg,
        name='LS approx registered',
        contrast_limits=(110, 230),
        blending='additive',
        colormap='green'
    )
    viewer.add_image(
        mov_data_reg_2,
        name='LS bead-corrected',
        contrast_limits=(110, 230),
        blending='additive',
        colormap='magenta'
    )

# %% Inspect matches and composed transform
matches
# %%
composed_transform
# %% Run the full iterative optimization pipeline (detect -> match -> correct -> score)
optimized_transform, quality_score_optimized = optimize_transform(
    transform=initial_transform,
    mov=mov_data,
    ref=ref_data,
    beads_match_settings=config.beads_match_settings,
    affine_transform_settings=config.affine_transform_settings,
    verbose=True,
    debug=True,
)
print(f"Optimized transform:\n{optimized_transform}")
print(f"Quality score: {quality_score_optimized:.4f}")
# %%