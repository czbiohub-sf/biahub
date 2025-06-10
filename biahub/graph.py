from typing import List, Tuple

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


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
        "max_dist": float(np.max(dists)),
    }


def grid_search_weights(
    source_peaks,
    target_peaks,
    source_edges,
    target_edges,
    source_image=None,
    target_image=None,
    base_weights=None,
    variable_weights=None,
    match_kwargs=None,
    top_k=5,
):
    if base_weights is None:
        base_weights = {
            k: 0
            for k in [
                "dist",
                "edge_angle",
                "edge_length",
                "pca_dir",
                "pca_aniso",
                "patch_ncc",
                "overlap",
                "edge_descriptor",
            ]
        }

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
                source_peaks,
                target_peaks,
                source_edges,
                target_edges,
                source_image=source_image,
                target_image=target_image,
                weights=weights,
                return_feature_matrices=False,
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


def compute_patch_similarity_matrix(
    source_peaks, target_peaks, source_img, target_img, patch_size=11
):
    """Compute negative normalized cross-correlation as a cost matrix."""
    half = patch_size // 2

    def extract_patch(img, center):
        z, y, x = map(int, center)
        return img[
            max(z - half, 0) : z + half + 1,
            max(y - half, 0) : y + half + 1,
            max(x - half, 0) : x + half + 1,
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
        matched_neighbors = sum(
            (ni, nj) in match_set for ni in s_neighbors for nj in t_neighbors
        )
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
    return_feature_matrices=False,
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
            "edge_descriptor": 0.5,
        }

    n, m = len(source_peaks), len(target_peaks)
    C_total = np.zeros((n, m))
    feature_costs = {}

    if weights.get("dist", 0):
        C = cdist(source_peaks, target_peaks)
        if normalize:
            C /= C.max()
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
        if normalize:
            C /= np.pi
        feature_costs["edge_angle"] = C
        C_total += weights["edge_angle"] * C

    if weights.get("edge_length", 0):
        C = edge_cost(sl, tl)
        if normalize:
            C /= C.max()
        feature_costs["edge_length"] = C
        C_total += weights["edge_length"] * C

    dirs_s, aniso_s = compute_local_pca_features(source_peaks, k=k_neighbors)
    dirs_t, aniso_t = compute_local_pca_features(target_peaks, k=k_neighbors)

    if weights.get("pca_dir", 0):
        dot = np.clip(np.dot(dirs_s, dirs_t.T), -1.0, 1.0)
        C = 1 - np.abs(dot)
        if normalize:
            C /= C.max()
        feature_costs["pca_dir"] = C
        C_total += weights["pca_dir"] * C

    if weights.get("pca_aniso", 0):
        C = np.abs(aniso_s[:, None] - aniso_t[None, :])
        if normalize:
            C /= C.max()
        feature_costs["pca_aniso"] = C
        C_total += weights["pca_aniso"] * C

    if weights.get("patch_ncc", 0) and source_image is not None:
        C = compute_patch_similarity_matrix(
            source_peaks, target_peaks, source_image, target_image, patch_size
        )
        if normalize:
            C /= C.max()
        feature_costs["patch_ncc"] = C
        C_total += weights["patch_ncc"] * C

    if weights.get("overlap", 0):
        dummy_matches = [(i, j) for i in range(n) for j in range(m)]
        C = compute_overlap_score_matrix(dummy_matches, source_edges, target_edges, n, m)
        if normalize:
            C /= C.max()
        feature_costs["overlap"] = C
        C_total += weights["overlap"] * C

    if weights.get("edge_descriptor", 0):
        desc_s = compute_edge_descriptors(source_peaks, source_edges)
        desc_t = compute_edge_descriptors(target_peaks, target_edges)
        C = cdist(desc_s, desc_t)
        if normalize:
            C /= C.max()
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
        graph = radius_neighbors_graph(
            points, radius=radius, mode='connectivity', include_self=False
        )
        edges = [(i, j) for i in range(n) for j in graph[i].nonzero()[1]]

    elif mode == "full":
        edges = [(i, j) for i in range(n) for j in range(n) if i != j]

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'knn', 'radius', or 'full'.")

    return edges


def match_hungarian(
    C,
    cost_threshold=1e5,
    dummy_cost=1e6,
    max_ratio=None,
):
    """
    Match points in two sets using the Hungarian algorithm.

    Parameters:
    - C (ndarray): Cost matrix of shape (n_A, n_B).
    - cost_threshold (float): Maximum cost to consider a valid match.
    - dummy_cost (float): Cost assigned to dummy nodes (must be > cost_threshold).
    - max_ratio (float, optional): Maximum allowed ratio between best and second-best cost.

    Returns:
    - matches (ndarray): Array of shape (N_matches, 2) with valid (A_idx, B_idx) pairs.
    """
    n_A, n_B = C.shape
    n = max(n_A, n_B)

    # Pad cost matrix to square shape
    C_padded = np.full((n, n), fill_value=dummy_cost)
    C_padded[:n_A, :n_B] = C

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(C_padded)

    matches = []
    for i, j in zip(row_ind, col_ind):
        if i >= n_A or j >= n_B:
            continue  # matched with dummy
        if C[i, j] >= cost_threshold:
            continue  # too costly

        if max_ratio is not None:
            # Find second-best match for i
            costs_i = C[i, :]
            sorted_costs = np.sort(costs_i)
            if len(sorted_costs) > 1:
                second_best = sorted_costs[1]
                ratio = C[i, j] / (second_best + 1e-10)  # avoid division by zero
                if ratio > max_ratio:
                    continue  # reject if not sufficiently better
            # else (only one candidate) => accept by default

        matches.append((i, j))

    return np.array(matches)


def cost_matrix(
    source_peaks,
    target_peaks,
    source_edges,
    target_edges,
    distance_metric='euclidean',
    distance_weight=0.5,
    nodes_angle_weight=1.0,
    nodes_distance_weight=1.0,
):
    """
    Compute a cost matrix for matching peaks between two graphs based on:
    - Euclidean or other distance between peaks
    - Consistency in edge distances
    - Consistency in edge angles

    Parameters:
    - source_peaks (ndarray): (n, 2) array of source node coordinates.
    - target_peaks (ndarray): (m, 2) array of target node coordinates.
    - source_edges (list of tuple): List of edges (i, j) in source graph.
    - target_edges (list of tuple): List of edges (i, j) in target graph.
    - distance_metric (str): Metric for direct point-to-point distances.
    - distance_weight (float): Weight for point distance cost.
    - nodes_angle_weight (float): Weight for angular consistency cost.
    - nodes_distance_weight (float): Weight for local edge distance cost.

    Returns:
    - ndarray: Cost matrix of shape (n, m).
    """
    n, m = len(source_peaks), len(target_peaks)

    def compute_edge_attributes(peaks, edges):
        distances = {}
        angles = {}
        for i, j in edges:
            vec = peaks[j] - peaks[i]
            d = np.linalg.norm(vec)
            angle = np.arctan2(vec[1], vec[0])
            distances[(i, j)] = distances[(j, i)] = d
            angles[(i, j)] = angles[(j, i)] = angle
        return distances, angles

    def local_edge_costs(
        source_edges, target_edges, source_attrs, target_attrs, attr='distance', default=1e6
    ):
        cost_matrix = np.full((n, m), default)
        for i in range(n):
            s_neighbors = [j for a, j in source_edges if a == i]
            for j in range(m):
                t_neighbors = [k for a, k in target_edges if a == j]
                common_len = min(len(s_neighbors), len(t_neighbors))
                diffs = []
                for k in range(common_len):
                    s_edge = (i, s_neighbors[k])
                    t_edge = (j, t_neighbors[k])
                    if s_edge in source_attrs and t_edge in target_attrs:
                        v1 = source_attrs[s_edge]
                        v2 = target_attrs[t_edge]
                        if attr == 'angle':
                            diff = np.abs(v1 - v2)
                        else:  # distance
                            diff = np.abs(v1 - v2)
                        diffs.append(diff)
                cost_matrix[i, j] = np.mean(diffs) if diffs else default
        return cost_matrix

    # Compute direct point-wise distance
    C_dist = cdist(source_peaks, target_peaks, metric=distance_metric)

    # Compute edge distances and angles
    source_dists, source_angles = compute_edge_attributes(source_peaks, source_edges)
    target_dists, target_angles = compute_edge_attributes(target_peaks, target_edges)

    # Compute local consistency costs
    C_dist_node = local_edge_costs(
        source_edges, target_edges, source_dists, target_dists, attr='distance', default=1e6
    )
    C_angle_node = local_edge_costs(
        source_edges, target_edges, source_angles, target_angles, attr='angle', default=np.pi
    )

    # Combine all costs
    C_total = (
        distance_weight * C_dist
        + nodes_angle_weight * C_angle_node
        + nodes_distance_weight * C_dist_node
    )

    return C_total


def _knn_edges(points, k=5):
    """
    Find k-nearest neighbors for each point in a set of points.

    Parameters:
    - points (ndarray): (n, 2) array of point coordinates.
    - k (int): Number of nearest neighbors to find.

    Returns:
    - edges (list of tuples): List of edges (i, j) where i and j are indices of points.
    """
    n_points = len(points)
    if n_points <= 1:
        return []

    k_eff = min(k, n_points - 1)  # Prevent k >= n
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(points)  # +1 includes self
    _, indices = nbrs.kneighbors(points)

    edges = [(i, j) for i, neighbors in enumerate(indices) for j in neighbors if i != j]
    return edges
