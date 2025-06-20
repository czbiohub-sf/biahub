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
from skimage.filters import threshold_otsu
import pandas as pd
import numpy as np

# Enhanced Graph-Based Cost Matrix for Bead Matching
# Enhanced Graph-Based Cost Matrix for Bead Matching
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter

# Enhanced Graph-Based Cost Matrix for Bead Matching (with plotting + evaluation utilities)
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter
import itertools
from sklearn.neighbors import radius_neighbors_graph

def build_edge_graph(points, mode="knn", k=10, radius=30.0):
    """
    Build a set of edges for a graph based on a given strategy.

    Parameters:
    - points (ndarray): (N, 3) array of 3D point coordinates.
    - mode (str): "knn", "radius", or "full".
    - k (int): Number of neighbors if mode == "knn".
    - radius (float): Distance threshold if mode == "radius".

    Returns:
    - List of (i, j) index pairs representing edges.
    """
    n = len(points)
    if mode == "knn":
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(points)
        _, indices = nbrs.kneighbors(points)
        edges = [(i, j) for i in range(n) for j in indices[i] if i != j]

    elif mode == "radius":
        graph = radius_neighbors_graph(points, radius=radius, mode='connectivity', include_self=False)
        edges = [(i, j) for i in range(n) for j in graph[i].nonzero()[1]]

    elif mode == "full":
        edges = [(i, j) for i in range(n) for j in range(n) if i != j]

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'knn', 'radius', or 'full'.")

    return edges


def plot_feature_costs_with_matches(feature_costs, matches):
    n_features = len(feature_costs)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
    if n_features == 1:
        axes = [axes]
    for ax, (name, matrix) in zip(axes, feature_costs.items()):
        ax.imshow(matrix, cmap='viridis', aspect='auto')
        for i, j in matches:
            ax.plot(j, i, 'ro', markersize=3)
        ax.set_title(f"{name}")
        ax.set_xlabel("Target index")
        ax.set_ylabel("Source index")
    plt.tight_layout()
    plt.show()


def evaluate_matches(matches, source_peaks, target_peaks):
    if len(matches) == 0:
        return {"num_matches": 0, "mean_dist": np.nan, "max_dist": np.nan}
    dists = np.linalg.norm(source_peaks[matches[:, 0]] - target_peaks[matches[:, 1]], axis=1)
    return {
        "num_matches": len(matches),
        "mean_dist": float(np.mean(dists)),
        "max_dist": float(np.max(dists))
    }

def grid_search_weights(
    source_peaks, target_peaks,
    source_edges, target_edges,
    source_image=None, target_image=None,
    base_weights=None,
    variable_weights=None,
    match_kwargs=None,
    top_k=5
):
    if base_weights is None:
        base_weights = {k: 0 for k in [
            "dist", "edge_angle", "edge_length", "pca_dir",
            "pca_aniso", "patch_ncc", "overlap", "edge_descriptor"]}

    if match_kwargs is None:
        match_kwargs = {}

    # Cartesian product of variable weight combinations
    keys = list(variable_weights.keys())
    weight_ranges = list(variable_weights.values())
    combos = list(itertools.product(*weight_ranges))

    results = []
    for combo in combos:
        weights = {**base_weights}  # Copy base
        for k, v in zip(keys, combo):
            weights[k] = v

        try:
            C = compute_cost_matrix_all_features(
                source_peaks, target_peaks,
                source_edges, target_edges,
                source_image=source_image,
                target_image=target_image,
                weights=weights,
                return_feature_matrices=False
            )

            from biahub.cli.estimate_registration import match_hungarian
            matches = match_hungarian(C, **match_kwargs)
            eval_metrics = evaluate_matches(matches, source_peaks, target_peaks)
            results.append((weights.copy(), eval_metrics))
        except Exception as e:
            print(f"Skipped combo {combo} due to error: {e}")

    # Sort by num_matches (descending) and then mean distance (ascending)
    results.sort(key=lambda x: (-x[1]['num_matches'], x[1]['mean_dist']))
    return results[:top_k]


def compute_local_pca_features(points, k=10):
    """Compute dominant direction and anisotropy for each point using PCA."""
    directions = np.zeros((len(points), 3))
    anisotropy = np.zeros(len(points))

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(points))).fit(points)
    _, indices = nbrs.kneighbors(points)

    for i, neighbors in enumerate(indices):
        local_points = points[neighbors[1:]].astype(np.float32)  # exclude self
        local_points -= local_points.mean(axis=0)
        _, S, Vt = svd(local_points, full_matrices=False)
        directions[i] = Vt[0]  # Vt[0] is the principal direction (shape (3,))
        anisotropy[i] = S[0] / (S[2] + 1e-5)
    return directions, anisotropy


def compute_patch_similarity_matrix(source_peaks, target_peaks, source_img, target_img, patch_size=11):
    """Compute negative normalized cross-correlation as a cost matrix."""
    half = patch_size // 2
    def extract_patch(img, center):
        z, y, x = map(int, center)
        return img[
            max(z - half, 0): z + half + 1,
            max(y - half, 0): y + half + 1,
            max(x - half, 0): x + half + 1
        ]

    C = np.zeros((len(source_peaks), len(target_peaks)))
    for i, sp in enumerate(source_peaks):
        patch_s = extract_patch(source_img, sp)
        for j, tp in enumerate(target_peaks):
            patch_t = extract_patch(target_img, tp)
            if patch_s.shape == patch_t.shape:
                patch_s = (patch_s - patch_s.mean()) / (patch_s.std() + 1e-5)
                patch_t = (patch_t - patch_t.mean()) / (patch_t.std() + 1e-5)
                ncc = np.mean(patch_s * patch_t)
                C[i, j] = 1 - ncc
            else:
                C[i, j] = 1.0
    return C


def compute_overlap_score_matrix(matches, source_edges, target_edges, n_source, n_target):
    match_set = set(map(tuple, matches))
    score = np.ones((n_source, n_target))
    for i, j in matches:
        s_neighbors = [b for a, b in source_edges if a == i]
        t_neighbors = [b for a, b in target_edges if a == j]
        matched_neighbors = sum((ni, nj) in match_set for ni in s_neighbors for nj in t_neighbors)
        total_possible = max(1, len(s_neighbors))
        score[i, j] = 1.0 - (matched_neighbors / total_possible)
    return score


def compute_edge_descriptors(points, edges):
    n = len(points)
    desc = np.zeros((n, 4))
    for i in range(n):
        neighbors = [j for a, j in edges if a == i]
        if not neighbors:
            continue
        vectors = points[neighbors] - points[i]
        lengths = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        desc[i, 0] = np.mean(lengths)
        desc[i, 1] = np.std(lengths)
        desc[i, 2] = np.mean(angles)
        desc[i, 3] = np.std(angles)
    return desc


def compute_cost_matrix_all_features(
    source_peaks,
    target_peaks,
    source_edges,
    target_edges,
    source_image=None,
    target_image=None,
    patch_size=11,
    weights=None,
    k_neighbors=10,
    normalize=True,
    return_feature_matrices=False
):
    if weights is None:
        weights = {
            "dist": 1.0,
            "edge_angle": 1.0,
            "edge_length": 1.0,
            "pca_dir": 0.5,
            "pca_aniso": 0.5,
            "patch_ncc": 1.0,
            "overlap": 0.5,
            "edge_descriptor": 0.5
        }

    n, m = len(source_peaks), len(target_peaks)
    C_total = np.zeros((n, m))
    feature_costs = {}

    if weights.get("dist", 0):
        C = cdist(source_peaks, target_peaks)
        if normalize: C /= C.max()
        feature_costs["dist"] = C
        C_total += weights["dist"] * C

    def compute_edge_attrs(points, edges):
        angles, lengths = {}, {}
        for i, j in edges:
            v = points[j] - points[i]
            angles[(i, j)] = angles[(j, i)] = np.arctan2(v[1], v[0])
            lengths[(i, j)] = lengths[(j, i)] = np.linalg.norm(v)
        return angles, lengths

    sa, sl = compute_edge_attrs(source_peaks, source_edges)
    ta, tl = compute_edge_attrs(target_peaks, target_edges)

    def edge_cost(attr_s, attr_t, default=1.0):
        M = np.full((n, m), default)
        for i in range(n):
            for j in range(m):
                s_n = [b for a, b in source_edges if a == i]
                t_n = [b for a, b in target_edges if a == j]
                common = min(len(s_n), len(t_n))
                diffs = []
                for k in range(common):
                    se = (i, s_n[k])
                    te = (j, t_n[k])
                    if se in attr_s and te in attr_t:
                        diff = np.abs(attr_s[se] - attr_t[te])
                        diffs.append(diff)
                if diffs:
                    M[i, j] = np.mean(diffs)
        return M

    if weights.get("edge_angle", 0):
        C = edge_cost(sa, ta, default=np.pi)
        if normalize: C /= np.pi
        feature_costs["edge_angle"] = C
        C_total += weights["edge_angle"] * C

    if weights.get("edge_length", 0):
        C = edge_cost(sl, tl)
        if normalize: C /= C.max()
        feature_costs["edge_length"] = C
        C_total += weights["edge_length"] * C

    dirs_s, aniso_s = compute_local_pca_features(source_peaks, k=k_neighbors)
    dirs_t, aniso_t = compute_local_pca_features(target_peaks, k=k_neighbors)

    if weights.get("pca_dir", 0):
        dot = np.clip(np.dot(dirs_s, dirs_t.T), -1.0, 1.0)
        C = 1 - np.abs(dot)
        if normalize: C /= C.max()
        feature_costs["pca_dir"] = C
        C_total += weights["pca_dir"] * C

    if weights.get("pca_aniso", 0):
        C = np.abs(aniso_s[:, None] - aniso_t[None, :])
        if normalize: C /= C.max()
        feature_costs["pca_aniso"] = C
        C_total += weights["pca_aniso"] * C

    if weights.get("patch_ncc", 0) and source_image is not None:
        C = compute_patch_similarity_matrix(source_peaks, target_peaks, source_image, target_image, patch_size)
        if normalize: C /= C.max()
        feature_costs["patch_ncc"] = C
        C_total += weights["patch_ncc"] * C

    if weights.get("overlap", 0):
        dummy_matches = [(i, j) for i in range(n) for j in range(m)]
        C = compute_overlap_score_matrix(dummy_matches, source_edges, target_edges, n, m)
        if normalize: C /= C.max()
        feature_costs["overlap"] = C
        C_total += weights["overlap"] * C

    if weights.get("edge_descriptor", 0):
        desc_s = compute_edge_descriptors(source_peaks, source_edges)
        desc_t = compute_edge_descriptors(target_peaks, target_edges)
        C = cdist(desc_s, desc_t)
        if normalize: C /= C.max()
        feature_costs["edge_descriptor"] = C
        C_total += weights["edge_descriptor"] * C

    if return_feature_matrices:
        return C_total, feature_costs
    return C_total


def plot_feature_costs(feature_costs):
    """Plot each cost matrix from feature_costs as a heatmap."""
    n_features = len(feature_costs)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
    if n_features == 1:
        axes = [axes]
    for ax, (name, matrix) in zip(axes, feature_costs.items()):
        ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_title(name)
        ax.set_xlabel('Target index')
        ax.set_ylabel('Source index')
    plt.tight_layout()
    plt.show()


def plot_feature_costs_with_matches(feature_costs, matches):
    """
    Plot each feature's cost matrix with overlaid red dots for final symmetric matches.

    Parameters:
    - feature_costs: dict[str, np.ndarray]
    - matches: np.ndarray of shape (N, 2), with source-target indices
    """
    n_features = len(feature_costs)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
    if n_features == 1:
        axes = [axes]
    for ax, (name, matrix) in zip(axes, feature_costs.items()):
        ax.imshow(matrix, cmap='viridis', aspect='auto')
        for i, j in matches:
            ax.plot(j, i, 'ro', markersize=3)  # (target, source)
        ax.set_title(f"{name}")
        ax.set_xlabel("Target index")
        ax.set_ylabel("Source index")
    plt.tight_layout()
    plt.show()

# %%
dataset = '2025_05_01_A549_DENV_sensor_DENV'
fov = 'C/1/000000'
root_path = Path(f'/hpc/projects/intracellular_dashboard/viral-sensor/2025_05_01_A549_DENV_sensor_DENV')
t_idx = 7
match_algorithm = 'hungarian' # 'hungarian' or 'match_descriptors'

lf_data_path = root_path / '1-preprocess/label-free/0-reconstruct' / f'{dataset}.zarr' / fov
ls_data_path = root_path / '1-preprocess/light-sheet/raw/0-deskew/' /  f'{dataset}.zarr' / fov

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
def mask_to_nonzero_region(target_img: np.ndarray, reference_img: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    Zero out `target_img` outside the nonzero region of `reference_img` (plus padding),
    preserving the original shape.
    
    Parameters
    ----------
    target_img : np.ndarray
        Image to be masked (e.g., LF).
    reference_img : np.ndarray
        Image defining the region of interest (e.g., registered LS).
    padding : int
        Padding in voxels around the nonzero region.
        
    Returns
    -------
    masked_img : np.ndarray
        Same shape as input, but with values outside region masked to zero.
    """
    assert target_img.shape == reference_img.shape, "Shape mismatch between target and reference images."

    nonzero_coords = np.argwhere(reference_img > 0)
    min_coords = np.maximum(nonzero_coords.min(axis=0) - padding, 0)
    max_coords = np.minimum(nonzero_coords.max(axis=0) + padding + 1, reference_img.shape)

    # Create binary mask
    mask = np.zeros_like(reference_img, dtype=bool)
    mask_slices = tuple(slice(start, end) for start, end in zip(min_coords, max_coords))
    mask[mask_slices] = True

    # Apply mask to target
    masked_img = np.where(mask, target_img, 0)
    return masked_img


lf_data_mask = mask_to_nonzero_region(lf_data, ls_data_reg, padding=100)

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
# %%
# filter peaks of specific z
def exclude_peaks_on_z(peaks, z_target):
    return peaks[peaks[:, 0] != z_target]

for z in range(0, 44):
    print(f'Excluding peaks on z={z}')
    lf_peaks = exclude_peaks_on_z(lf_peaks, z)
print(f'Number of LF peaks: {len(lf_peaks)}')
#%%
viewer = napari.Viewer()
viewer.add_image(ls_data_reg, name='LS')
viewer.add_points(
    ls_peaks, name='peaks local max', size=12, symbol='ring', edge_color='red'
)


#%%
viewer = napari.Viewer()
viewer.add_image(lf_data, name='LF')
viewer.add_points(
    lf_peaks, name='peaks local max', size=12, symbol='ring', edge_color='yellow'
)

# %%
source_edges = _knn_edges(ls_peaks, k=5)
target_edges = _knn_edges(lf_peaks, k=5)


#source_edges = build_edge_graph(ls_peaks, mode="radius", radius=100.0)
#target_edges = build_edge_graph(lf_peaks, mode="radius", radius=100.0)

# %%

base_weights = {
    "dist": 0.5,
    "edge_length": 0.0,
    "pca_dir": 0.5,
    "pca_aniso": 0.2,       # varied below
    "patch_ncc": 0.0,
    "overlap": 0.1,         # varied below
    "edge_descriptor": 0.5
}


variable_weights = {
    "edge_angle": [0.0,0.2],
    "overlap": [0.0, 0.2, 0.4, 0.5,1],
    "pca_aniso": [0.0, 0.2, 0.4, 0.5,1]
}



results = grid_search_weights(
    ls_peaks, lf_peaks,
    source_edges, target_edges,
    source_image=ls_data_reg, target_image=lf_data,
    variable_weights=variable_weights,
    match_kwargs={"cost_threshold": 0.2, "max_ratio": 1},
    top_k=10
)

for w, r in results:
    print(f"Weights: {w}, Matches: {r['num_matches']}, Mean Dist: {r['mean_dist']:.2f}")

#%%
#%%
# Find matching peaks in the two datasets
if match_algorithm == 'match_descriptor':
    matches = match_descriptors(ls_peaks,
                                 lf_peaks, metric='euclidean',max_ratio=0.8,cross_check=True)
else:

    weights = {
        "dist": 0.1,
        "edge_angle": 0.2,
        "edge_length": 0,
        "pca_dir": 0.5,
        "pca_aniso": 0.2,
        "patch_ncc": 0,
        "overlap": 0.2,
        "edge_descriptor": 0.5
    }
    C_ab, feature_costs_ab = compute_cost_matrix_all_features(
        ls_peaks, lf_peaks,
        source_edges, target_edges,
        source_image=ls_data_reg, target_image=lf_data,
        weights=weights,
        return_feature_matrices=True
    )
    C_ab = np.nan_to_num(C_ab, nan=1.0, posinf=1.0, neginf=1.0)
    plot_feature_costs(feature_costs_ab)
    # Step 1: A → B
    # C_ab = _compute_cost_matrix(
    #     ls_peaks,
    #     lf_peaks,
    #     source_edges,
    #     target_edges,
    #     distance_metric='cosine',
#)
    matches_ab = match_hungarian(C_ab, cost_threshold=np.quantile(C_ab, 0.3), max_ratio=1)
    
    C_ba, feature_costs_ba = compute_cost_matrix_all_features(
        lf_peaks, ls_peaks,
        target_edges, source_edges,
        source_image=lf_data, target_image=ls_data_reg,
        weights=weights,
        return_feature_matrices=True
    )
    C_ab = np.nan_to_num(C_ab, nan=1.0, posinf=1.0, neginf=1.0)
    plot_feature_costs(feature_costs_ba)
    # Step 2: B → A (swap arguments)
    # C_ba = _compute_cost_matrix(
    #     lf_peaks,
    #     ls_peaks,
    #     target_edges,
    #     source_edges,
    #     distance_metric='cosine',

    # )
    matches_ba = match_hungarian(C_ba, cost_threshold=np.quantile(C_ba, 0.3), max_ratio=1)
    # Step 3: Invert matches_ba to compare

    reverse_map = {(j, i) for i, j in matches_ba}

    # Step 4: Keep only symmetric matches
    matches = np.array([
        [i, j] for i, j in matches_ab if (i, j) in reverse_map
    ])
    plot_feature_costs_with_matches(feature_costs_ab, matches)
    len(matches)
    # Exclude top 5% of distances as outliers
dist = np.linalg.norm(ls_peaks[matches[:, 0]] - lf_peaks[matches[:, 1]], axis=1)
matches = matches[dist<np.quantile(dist, 0.95), :]
len(matches)

# %%
print("C_ba shape:", C_ba.shape)
print("NaNs:", np.isnan(C_ba).sum(), "Infs:", np.isinf(C_ba).sum())


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
print(f"Filtered matches: {len(matches)}")


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
matches
# %%

compount_tform
# %%
tform.inverse.params

# %%
