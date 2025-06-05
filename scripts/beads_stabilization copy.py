# %%
import ants
import numpy as np
from pathlib import Path
from iohub import open_ome_zarr
from biahub.analysis.analyze_psf import detect_peaks
from biahub.analysis.register import convert_transform_to_ants
import napari
from skimage.transform import EuclideanTransform
from skimage.feature import match_descriptors
from biahub.cli.estimate_registration import _knn_edges, _compute_cost_matrix, match_hungarian
# %%
dataset = '2024_10_31_A549_SEC61_ZIKV_DENV'
fov = 'C/1/000000'
root_path = Path(f'/hpc/projects/intracellular_dashboard/organelle_dynamics/V2/')
t_idx = 25
t_ref = 0

data_path = root_path/ dataset / "1-preprocess/label-free/0-reconstruct" /f"{dataset}.zarr/{fov}"

approx_tform = np.eye(4)
match_algorithm = 'hungarian' # 'hungarian' or 'match_descriptors'

# %% Load data
with open_ome_zarr(data_path) as lf_ds:
    ref_data = np.asarray(lf_ds.data[0][0]).astype(np.float32)
    mov_data = np.asarray(lf_ds.data[t_idx][0]).astype(np.float32)

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
threshold = 60
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
viewer.add_image(ref_data, name='reference data', contrast_limits=(-0.5, 1.0))
viewer.add_image(
    mov_data,
    name='LS',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='green'
)
viewer.add_image(
    mov_data_reg,
    name='LS registered',
    contrast_limits=(110, 230),
    blending='additive',
    colormap='magenta'
)

# %%