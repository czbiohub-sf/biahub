# %%
import ants
import numpy as np
from pathlib import Path
from iohub import open_ome_zarr
from biahub.core.transform import convert_transform_to_ants
import napari
from biahub.settings import EstimateRegistrationSettings
import numpy as np

from biahub.registration.beads import transform_from_matches, matches_from_beads, peaks_from_beads


# %%%

dataset = '2024_12_03_A549_LAMP1_ZIKV_DENV'
fov = 'C/1/000000'
root_path = Path(f'/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}/1-preprocess/')
t_idx = 0
lf_data_path = root_path / f"label-free/0-reconstruct/{dataset}.zarr" / fov
ls_data_path = root_path / f"light-sheet/raw/0-deskew/{dataset}.zarr" / fov

visualize = True

# %%
config_dict = {
    "target_channel_name": "Phase3D",
    "source_channel_name": "mCherry EX561 EM600-37",
    "beads_match_settings": { 
        "algorithm": "hungarian",
        "qc_settings": {
            "iterations": 2,
            "score_threshold": 0.40,
            "score_centroid_mask_radius": 6
        },
        "filter_matches_settings": {
            "min_distance_quantile": 0.0,
            "max_distance_quantile": 0.0,
            "angle_threshold": 0,
            "direction_threshold": 00
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
            "max_ratio": 0.95,
            "edge_graph_settings": {
                "method": "knn",
                "k": 10
            },
            "cost_matrix_settings": {
                "normalize": False,
                "weights": {
                    "dist": 1.0,
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
with open_ome_zarr(lf_data_path) as ref_ds:
    ref_channel_name = ref_ds.channel_names
    ref_channel_index = ref_ds.channel_names.index(config.target_channel_name)
    ref_data = np.asarray(ref_ds.data[t_idx, ref_channel_index]) # take phase channel
    ref_scale = ref_ds.scale

with open_ome_zarr(ls_data_path) as mov_ds:
    mov_channel_name = mov_ds.channel_names
    mov_channel_index = mov_channel_name.index(config.source_channel_name)
    mov_data = np.asarray(mov_ds.data[t_idx, mov_channel_index]) # take mCherry channel or the GFP channel (depending where the beads are)
    mov_scale = mov_ds.scale

# Register LS data with approx tranform
mov_data_ants = ants.from_numpy(mov_data)
ref_data_ants = ants.from_numpy(ref_data)

mov_data_reg_ants = convert_transform_to_ants(np.asarray(config.affine_transform_settings.approx_transform)).apply_to_image(
    mov_data_ants, reference=ref_data_ants
)
mov_data_reg = mov_data_reg_ants.numpy()

# qc, measure matches vectors directions, lenght, angle .. statiscs, then evaluate the mean, std, min, max, etc.
# to determin filter
# %%
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(ref_data, name='LF')
    viewer.add_image(mov_data_reg, name='LS')

# %% Detect peaks in LS data

mov_peaks, ref_peaks = peaks_from_beads(
    mov_data_reg,
    ref_data,
    config.beads_match_settings.source_peaks_settings,
    config.beads_match_settings.target_peaks_settings,
    verbose=True,
)

#%%
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(mov_data_reg, name='LS')
    viewer.add_points(
        mov_peaks, name='peaks local max LS', size=20, symbol='disc', face_color='magenta'
    )
# %%
if visualize:
    viewer = napari.Viewer()
    viewer.add_image(ref_data, name='LF')
    viewer.add_points(
        ref_peaks, name='peaks local max LF', size=20, symbol='disc', face_color='green'
    )

# %%

matches = matches_from_beads(
    mov_peaks,
    ref_peaks,
    config.beads_match_settings,
    verbose=True
)

# %%
if visualize:
    # visualize matches
    viewer = napari.Viewer()
    viewer.add_image(ref_data, name='LF', contrast_limits=(0.5, 1.0),blending='additive')
    viewer.add_points(
        ref_peaks, name='LF peaks', size=12, symbol='ring', face_color='yellow',blending='additive'
    )
    viewer.add_image(mov_data_reg, name='LS', contrast_limits=(110, 230), blending='additive', colormap='green')
    viewer.add_points(
        mov_peaks, name='LS peaks', size=12, symbol='ring', face_color='red',blending='additive'
    )
    # Project in 3D to be able to view the lines
    viewer.add_shapes(
        data=[np.asarray([mov_peaks[m[0]], ref_peaks[m[1]]]) for m in matches],
        shape_type='line',
        edge_width=5,
        blending='additive',
    )
    viewer.dims.ndisplay = 3


# %% Register LS data using compount transform
from biahub.core.transform import Transform
initial_transform = Transform(
        matrix=np.asarray(config.affine_transform_settings.approx_transform)
    )

fwd_transform, inv_transform = transform_from_matches(
    matches=matches,
    mov_peaks=mov_peaks,
    ref_peaks=ref_peaks,
    affine_transform_settings=config.affine_transform_settings,
    ndim=mov_data_reg.ndim,
)

composed_transform = initial_transform @ inv_transform
mov_data_reg_2 = composed_transform.to_ants().apply_to_image(
    mov_data_ants, reference=ref_data_ants
).numpy()

if visualize:
    viewer = napari.Viewer()
    viewer.add_image(ref_data, name='LF', contrast_limits=(-0.5, 1.0))
    viewer.add_image(
    mov_data_reg,
    name='LS approx registered',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='green'
    )
    viewer.add_image(
        mov_data_reg_2,
        name='LS registered',
        contrast_limits=(110, 230),
        blending='additive',
        colormap='magenta'
    )

# %%
matches
# %%
composed_transform.matrix.tolist()
# %%
print(composed_transform)
