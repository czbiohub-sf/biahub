# %%
import ants
import numpy as np
from pathlib import Path
from iohub import open_ome_zarr
from biahub.analysis.analyze_psf import detect_peaks
from biahub.analysis.register import convert_transform_to_ants
import napari
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform
from skimage.feature import match_descriptors
from biahub.cli.estimate_registration import _knn_edges, _compute_cost_matrix, match_hungarian

import numpy as np

#%%
path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_01_22_A549_G3BP1_ZIKV_DENV/1-preprocess/label-free/0-reconstruct/2025_01_22_A549_G3BP1_ZIKV_DENV.zarr/B/1/000000")

# read dataset, zero t=1, save a new dataset with
with open_ome_zarr(path) as dataset:
    data = np.asarray(dataset.data[1, 0]) # take phase channel
    scale = dataset.scale
# save as empty frame




#%%



dataset = '2025_05_01_A549_DENV_sensor_DENV'
fov = 'C/1/000000'
root_path = Path(f'/hpc/projects/intracellular_dashboard/viral-sensor/2025_05_01_A549_DENV_sensor_DENV')
t_idx = 26
match_algorithm = 'hungarian' # 'hungarian' or 'match_descriptors'

lf_data_path = Path(f'/hpc/projects/intracellular_dashboard/viral-sensor/2025_05_01_A549_DENV_sensor_DENV/1-preprocess/light-sheet/raw/1-register/debug/2025_05_01_A549_DENV_sensor_DENV.zarr/C/1/000000')
ls_data_path = Path(f'/hpc/projects/intracellular_dashboard/viral-sensor/2025_05_01_A549_DENV_sensor_DENV/1-preprocess/light-sheet/raw/1-register/debug/fluorecent/2025_05_01_A549_DENV_sensor_DENV.zarr/C/1/000000')
approx_tform = np.asarray(
    [
        [1, 0, 0, 0],
        [0, 0, -1.288, 1960],
        [0, 1.288, 0, -460],
        [0, 0, 0, 1]
    ]
)

# %% Load data
with open_ome_zarr(lf_data_path) as lf_ds:
    lf_data = np.asarray(lf_ds.data[t_idx, 0]) # take phase channel
    lf_scale = lf_ds.scale

with open_ome_zarr(ls_data_path) as ls_ds:
    ls_data = np.asarray(ls_ds.data[t_idx, 0]) # take mCherry channel or the GFP channel (depending where the beads are)
    ls_scale = ls_ds.scale

# Register LS data with approx tranform
ls_data_ants = ants.from_numpy(ls_data)
lf_data_ants = ants.from_numpy(lf_data)

ls_data_reg_ants = convert_transform_to_ants(approx_tform).apply_to_image(
    ls_data_ants, reference=lf_data_ants
)
ls_data_reg = ls_data_reg_ants.numpy()

# %%
viewer = napari.Viewer()
viewer.add_image(lf_data, name='LF')
viewer.add_image(ls_data_reg, name='LS')

# %% Detect peaks in LS data
ls_peaks = detect_peaks(
    ls_data_reg,
    block_size=[8, 8, 8],
    threshold_abs=110,
    nms_distance=16,
    min_distance=0,
    verbose=True
)

# %% Detect peaks in LF data
lf_peaks = detect_peaks(
    lf_data,
    block_size=[8, 8, 8],
    threshold_abs=0.8,
    nms_distance=16,
    min_distance=0,
    verbose=True
)
#%%
viewer = napari.Viewer()
viewer.add_image(ls_data_reg, name='LS')
viewer.add_points(
    ls_peaks, name='peaks local max', size=12, symbol='ring', edge_color='yellow'
)

#%%
viewer = napari.Viewer()
viewer.add_image(lf_data, name='LF')
viewer.add_points(
    lf_peaks, name='peaks local max', size=12, symbol='ring', edge_color='yellow'
)
#%%
# Find matching peaks in the two datasets
if match_algorithm == 'match_descriptor':
    matches = match_descriptors(ls_peaks, lf_peaks, metric='euclidean',max_ratio=0.6,cross_check=True)
else:
    source_edges = _knn_edges(ls_peaks, k=5)
    target_edges = _knn_edges(lf_peaks, k=5)

    # Step 1: A → B
    C_ab = _compute_cost_matrix(ls_peaks, lf_peaks, source_edges, target_edges)
    matches_ab = match_hungarian(C_ab, cost_threshold=np.quantile(C_ab, 0.10), max_ratio = 1)
    
    # Step 2: B → A (swap arguments)
    C_ba = _compute_cost_matrix(
        lf_peaks,
        ls_peaks,
        target_edges,
        source_edges,
        distance_metric='euclidean',
    )
    matches_ba = match_hungarian(C_ba, cost_threshold=np.quantile(C_ba, 0.10), max_ratio = 1)
    # Step 3: Invert matches_ba to compare

    reverse_map = {(j, i) for i, j in matches_ba}

    # Step 4: Keep only symmetric matches
    matches = np.array([
        [i, j] for i, j in matches_ab if (i, j) in reverse_map
    ])
    len(matches)
# Exclude top 5% of distances as outliers
dist = np.linalg.norm(ls_peaks[matches[:, 0]] - lf_peaks[matches[:, 1]], axis=1)
matches = matches[dist<np.quantile(dist, 0.95), :]
len(matches)

#%%
# Calculate vectors between matches
vectors = lf_peaks[matches[:, 1]] - ls_peaks[matches[:, 0]]

# Compute angles in radians relative to the x-axis
angles = np.arctan2(vectors[:, 1], vectors[:, 0])

# Convert to degrees for easier interpretation
angles_deg = np.degrees(angles)

# Create a histogram of angles
bins = np.linspace(-180, 180, 36)  # 10-degree bins
hist, bin_edges = np.histogram(angles_deg, bins=bins)

# Find the dominant bin
dominant_bin_index = np.argmax(hist)
dominant_angle = (bin_edges[dominant_bin_index] + bin_edges[dominant_bin_index + 1]) / 2

# Filter matches within ±45 degrees of the dominant direction
threshold = 30
filtered_indices = np.where(
    np.abs(angles_deg - dominant_angle) <= threshold
)[0]
matches = matches[filtered_indices]

# %%
viewer = napari.Viewer()
viewer.add_image(lf_data, name='LF', contrast_limits=(0.5, 1.0),blending='additive')
viewer.add_points(
    lf_peaks, name='LF peaks', size=12, symbol='ring', edge_color='yellow',blending='additive'
)
viewer.add_image(ls_data_reg, name='LS', contrast_limits=(110, 230), blending='additive', colormap='green')
viewer.add_points(
    ls_peaks, name='LS peaks', size=12, symbol='ring', edge_color='red',blending='additive'
)

# Project in 3D to be able to view the lines
viewer.add_shapes(
    data=[np.asarray([ls_peaks[m[0]], lf_peaks[m[1]]]) for m in matches],
    shape_type='line',
    edge_width=5,
    blending='additive',
)
viewer.dims.ndisplay = 3


# %% Register LS data using compount transform
tform = AffineTransform(dimensionality=3) # Affine transform performs better than Euclidean
tform.estimate(ls_peaks[matches[:, 0]], lf_peaks[matches[:, 1]])

compount_tform = approx_tform @ tform.inverse.params
compount_tform_ants = convert_transform_to_ants(compount_tform)
ls_data_reg_2 = compount_tform_ants.apply_to_image(
    ls_data_ants, reference=lf_data_ants
).numpy()

viewer = napari.Viewer()
viewer.add_image(lf_data, name='LF', contrast_limits=(-0.5, 1.0))
viewer.add_image(
    ls_data_reg,
    name='LS approx registered',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='green'
)
viewer.add_image(
    ls_data_reg_2,
    name='LS registered',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='magenta'
)


# %%

reg_transform_9 = np.asarray(
    [
        [1.06865836, -0.00599757, 0.00012818, -21.32906809],
        [-0.25638638, 0.02707905, -1.28278347, 2026.21559108],
        [-0.16449603, 1.28123386, 0.04360918, -514.35471224],
        [0., 0., 0., 1.]
    ]
)
# %%
R_ref_LS2LF= reg_transform_9
T_LF_bad2ref = np.asarray(
    [
        [0.99999582, -0.00289274, 0.00047204, 3.03527011],
        [1.18541358e-06, 0.99999989, 0.00047204, 0.46286015],
        [-0.00289274, -0.00047204, 0.9999957, 3.03527011],
        [0., 0., 0., 1.]
    ]
)

T_LS_bad2ref = np.asarray(
    [
        [0.99864702, -0.03937632, -0.03396529, 3.45095229],
        [-0.03943379, 0.99922166, -0.00102356, 2.46622107],
        [-0.03389855, 0.00236155, 0.99942249, 6.65830089],
        [0., 0., 0., 1.]
    ]
)
R_bad_LS2LF = np.linalg.inv(T_LF_bad2ref) @ R_ref_LS2LF @ T_LS_bad2ref

# %%
compount_tform = reg_transform_9
compount_tform_ants = convert_transform_to_ants(compount_tform)
ls_data_reg_2 = compount_tform_ants.apply_to_image(
    ls_data_ants, reference=lf_data_ants
).numpy()

viewer = napari.Viewer()
viewer.add_image(lf_data, name='LF', contrast_limits=(-0.5, 1.0))
viewer.add_image(
    ls_data_reg,
    name='LS approx registered',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='green'
)
viewer.add_image(
    ls_data_reg_2,
    name='LS registered',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='magenta'
)

# %%
t_stab = 5
t_ref = 9
root_path = Path(f'/hpc/projects/intracellular_dashboard/viral-sensor/2025_05_01_A549_DENV_sensor_DENV')

data_path = root_path / "1-preprocess/label-free/0-reconstruct" /f"{dataset}.zarr/{fov}"

approx_tform = np.eye(4)
match_algorithm = 'hungarian' # 'hungarian' or 'match_descriptors'

# %% Load data
with open_ome_zarr(data_path) as lf_ds:
    ref_data = np.asarray(lf_ds.data[t_ref][0]).astype(np.float32)
    mov_data = np.asarray(lf_ds.data[t_stab][0]).astype(np.float32)

# %%
viewer = napari.Viewer()
viewer.add_image(ref_data, name='Reference')
viewer.add_image(mov_data, name='Moving')
ref_data.shape
# %% Detect peaks in LS data
target_peaks = detect_peaks(
    ref_data,
    block_size=[8, 8, 8],
    threshold_abs=0.8,
    nms_distance=16,
    min_distance=0,
    verbose=True
)
# %%
viewer = napari.Viewer()
viewer.add_image(ref_data, name='reference')
viewer.add_points(
    target_peaks, name='peaks local max', size=12, symbol='ring', edge_color='red'
)

# %% Detect peaks in LF data
source_peaks = detect_peaks(
    mov_data,
    block_size=[8, 8, 8],
    threshold_abs=0.8,
    nms_distance=16,
    min_distance=0,
    verbose=True
)
#%%

viewer = napari.Viewer()
viewer.add_image(mov_data, name='moving')
viewer.add_points(
    source_peaks, name='peaks local max', size=12, symbol='ring', edge_color='yellow'
)

# %% Find matching peaks in the two datasets
if match_algorithm == 'match_descriptor':

    matches = match_descriptors(source_peaks, target_peaks, metric='correlation',max_ratio=1,cross_check=True)
else:
    source_edges = _knn_edges(source_peaks, k=5)
    target_edges = _knn_edges(target_peaks, k=5)

    C_ab = _compute_cost_matrix(source_peaks, target_peaks, source_edges, target_edges)
    matches_ab = match_hungarian(C_ab, cost_threshold=np.quantile(C_ab, 0.10))

    # Step 2: B → A (swap arguments)
    C_ba = _compute_cost_matrix(
        target_peaks,
        source_peaks,
        target_edges,
        source_edges,
        distance_metric='euclidean',
    )
    matches_ba = match_hungarian(C_ba, cost_threshold=np.quantile(C_ba, 0.10))

    # Step 3: Invert matches_ba to compare
    reverse_map = {(j, i) for i, j in matches_ba}

    # Step 4: Keep only symmetric matches
    matches = np.array([[i, j] for i, j in matches_ab if (i, j) in reverse_map])

# Exclude top 5% of distances as outliers
dist = np.linalg.norm(source_peaks[matches[:, 0]] - target_peaks[matches[:, 1]], axis=1)
matches = matches[dist<np.quantile(dist, 0.95), :]
len(matches)
# %%
# Calculate vectors between matches
vectors = target_peaks[matches[:, 1]] - source_peaks[matches[:, 0]]

# Compute angles in radians relative to the x-axis
angles = np.arctan2(vectors[:, 1], vectors[:, 0])

# Convert to degrees for easier interpretation
angles_deg = np.degrees(angles)

# Create a histogram of angles
bins = np.linspace(-180, 180, 36)  # 10-degree bins
hist, bin_edges = np.histogram(angles_deg, bins=bins)

# Find the dominant bin
dominant_bin_index = np.argmax(hist)
dominant_angle = (bin_edges[dominant_bin_index] + bin_edges[dominant_bin_index + 1]) / 2

# Filter matches within ±45 degrees of the dominant direction
threshold = 0
filtered_indices = np.where(
    np.abs(angles_deg - dominant_angle) <= threshold
)[0]
matches = matches[filtered_indices]

# %%
viewer = napari.Viewer()
viewer.add_image(ref_data, name='LF', contrast_limits=(0.5, 1.0),blending='additive')
viewer.add_points(
    target_peaks, name='LF peaks', size=12, symbol='ring', edge_color='red',blending='additive'
)
viewer.add_image(mov_data, name='LS', contrast_limits=(110, 230), blending='additive', colormap='green')
viewer.add_points(
    source_peaks, name='LS peaks', size=12, symbol='ring', edge_color='yellow',blending='additive'
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
tform = EuclideanTransform(dimensionality=3) # Affine transform performs better than Euclidean
tform.estimate(source_peaks[matches[:, 0]], target_peaks[matches[:, 1]])

ref_data_ants = ants.from_numpy(ref_data)
mov_data_ants = ants.from_numpy(mov_data)

compount_tform = approx_tform @ tform.inverse.params
compount_tform_ants = convert_transform_to_ants(compount_tform)
mov_data_reg = compount_tform_ants.apply_to_image(
    mov_data_ants, reference=ref_data_ants
).numpy()

viewer = napari.Viewer()
viewer.add_image(ref_data, name='Target', contrast_limits=(-0.5, 1.0))
viewer.add_image(
    mov_data,
    name='Source orig',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='green'
)
viewer.add_image(
    mov_data_reg,
    name='Source registered',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='magenta'
)

