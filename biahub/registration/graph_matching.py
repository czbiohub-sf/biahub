from typing import Literal

import click
import numpy as np

from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

from biahub.settings import (
    HungarianMatchSettings,
)


def get_local_pca_features(
    points: ArrayLike,
    edges: list[tuple[int, int]],
) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute dominant direction and anisotropy for each point using PCA,
    using neighborhoods defined by existing graph edges.

    Parameters
    ----------
    points : ArrayLike
        (n, 2) array of points.
    edges : list[tuple[int, int]]
        List of edges (i, j) in the graph.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        - directions : (n, 3) array of dominant directions.
        - anisotropy : (n,) array of anisotropy.

    Notes
    -----
    The PCA features are computed as the dominant direction and anisotropy of the local neighborhood of each point.
    The direction is the first principal component of the local neighborhood.
    The anisotropy is the ratio of the first to third principal component of the local neighborhood.
    """
    n = len(points)
    directions = np.zeros((n, 3))
    anisotropy = np.zeros(n)

    # Build neighbor list from edges
    from collections import defaultdict

    neighbor_map = defaultdict(list)
    for i, j in edges:
        neighbor_map[i].append(j)

    for i in range(n):
        neighbors = neighbor_map[i]
        if not neighbors:
            directions[i] = np.nan
            anisotropy[i] = np.nan
            continue

        local_points = points[neighbors].astype(np.float32)
        local_points -= local_points.mean(axis=0)
        _, S, Vt = np.linalg.svd(local_points, full_matrices=False)

        directions[i] = Vt[0] if Vt.shape[0] > 0 else np.zeros(3)
        anisotropy[i] = S[0] / (S[2] + 1e-5) if len(S) >= 3 else 0.0

    return directions, anisotropy


def get_edge_descriptors(
    points: ArrayLike,
    edges: list[tuple[int, int]],
) -> ArrayLike:
    """
    Compute edge descriptors for a set of points.

    Parameters
    ----------
    points : ArrayLike
        (n, 2) array of points.
    edges : list[tuple[int, int]]
        List of edges (i, j) in the graph.

    Returns
    -------
    ArrayLike
        (n, 4) array of edge descriptors.
        Each row contains:
        - mean length
        - std length
        - mean angle
        - std angle

    Notes
    -----
    The edge descriptors are computed as the mean and standard deviation of the lengths and angles of the edges.
    """
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


def get_edge_attrs(
    points: ArrayLike,
    edges: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """
    Compute edge distances and angles for a set of points.

    Parameters
    ----------
    points : ArrayLike
        (n, 2) array of points.
    edges : list[tuple[int, int]]
        List of edges (i, j) in the graph.

    Returns
    -------
    tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]
        - distances : dict[tuple[int, int], float]
        - angles : dict[tuple[int, int], float]

    """
    distances, angles = {}, {}
    for i, j in edges:
        vec = points[j] - points[i]
        d = np.linalg.norm(vec)
        angle = np.arctan2(vec[1], vec[0])
        distances[(i, j)] = distances[(j, i)] = d
        angles[(i, j)] = angles[(j, i)] = angle
    return distances, angles


def match_hungarian_local_cost(
    i: int,
    j: int,
    mov_neighbors: list[int],
    ref_neighbors: list[int],
    mov_attrs: dict[tuple[int, int], float],
    ref_attrs: dict[tuple[int, int], float],
    default_cost: float,
) -> float:
    """
    Match neighbor edges between two graphs using the Hungarian algorithm for local cost estimation.
    The cost is the mean of the absolute differences between the moving and reference edge attributes.

    Parameters
    ----------
    i : int
        Index of the moving edge.
    j : int
        Index of the reference edge.
    mov_neighbors : list[int]
        List of moving neighbors.
    ref_neighbors : list[int]
        List of reference neighbors.
    mov_attrs : dict[tuple[int, int], float]
        Dictionary of moving edge attributes.
    ref_attrs : dict[tuple[int, int], float]
        Dictionary of reference edge attributes.
    """
    C = np.full((len(mov_neighbors), len(ref_neighbors)), default_cost)

    # compute cost matrix
    for ii, mov_neighbor in enumerate(mov_neighbors):
        # get reference neighbors
        for jj, ref_neighbor in enumerate(ref_neighbors):
            mov_edge = (i, mov_neighbor)
            ref_edge = (j, ref_neighbor)
            if mov_edge in mov_attrs and ref_edge in ref_attrs:
                C[ii, jj] = abs(mov_attrs[mov_edge] - ref_attrs[ref_edge])

    # use hungarian algorithm to find the best match
    row_ind, col_ind = linear_sum_assignment(C)
    # get the mean of the matched costs
    matched_costs = C[row_ind, col_ind]
    # return the mean of the matched costs

    return matched_costs.mean() if len(matched_costs) > 0 else default_cost


def compute_edge_consistency_cost(
    n: int,
    m: int,
    mov_attrs: dict[tuple[int, int], float],
    ref_attrs: dict[tuple[int, int], float],
    mov_edges: list[tuple[int, int]],
    ref_edges: list[tuple[int, int]],
    default: float = 1e6,
    hungarian: bool = True,
) -> ArrayLike:
    """
    Compute the cost matrix for matching edges between two graphs.

    Parameters
    ----------
    n : int
        Number of moving edges.
    m : int
        Number of reference edges.
    mov_attrs : dict[tuple[int, int], float]
        Dictionary of moving edge attributes.
    ref_attrs : dict[tuple[int, int], float]
        Dictionary of reference edge attributes.
    mov_edges : list[tuple[int, int]]
        List of edges (i, j) in moving graph.
    ref_edges : list[tuple[int, int]]
        List of edges (i, j) in reference graph.
    default : float
        Default value for the cost matrix.
    hungarian : bool
        Whether to use the Hungarian algorithm for local cost estimation.
        If False, the cost matrix is computed as the mean of the absolute differences between the moving and reference edge attributes.
        If True, the cost matrix is computed as the mean of the absolute differences between the moving and reference edge attributes using the Hungarian algorithm.

    Returns
    -------
    ArrayLike
        Cost matrix of shape (n, m).

    Notes
    -----
    The cost matrix is computed as the mean of the absolute differences between the moving and reference edge attributes.
    """
    cost_matrix = np.full((n, m), default)
    for i in range(n):
        # get moving neighbors
        mov_neighbors = [j for a, j in mov_edges if a == i]
        for j in range(m):
            # get reference neighbors
            ref_neighbors = [k for a, k in ref_edges if a == j]
            if hungarian:
                # hungarian algorithm based cost estimation
                cost_matrix[i, j] = match_hungarian_local_cost(
                    i, j, mov_neighbors, ref_neighbors, mov_attrs, ref_attrs, default
                )
            else:
                # position based cost estimation (mean of the absolute differences between the source and target edge attributes)
                common_len = min(len(mov_neighbors), len(ref_neighbors))
                diffs = []
                for k in range(common_len):
                    mov_edge = (i, mov_neighbors[k])
                    ref_edge = (j, ref_neighbors[k])
                    if mov_edge in mov_attrs and ref_edge in ref_attrs:
                        v1 = mov_attrs[mov_edge]
                        v2 = ref_attrs[ref_edge]
                        diff = np.abs(v1 - v2)
                        diffs.append(diff)
                cost_matrix[i, j] = np.mean(diffs) if diffs else default

    return cost_matrix


def compute_cost_matrix(
    mov_points: ArrayLike,
    ref_points: ArrayLike,
    mov_edges: list[tuple[int, int]],
    ref_edges: list[tuple[int, int]],
    weights: dict[str, float] = None,
    distance_metric: str = 'euclidean',
    normalize: bool = False,
) -> ArrayLike:
    """
    Compute a cost matrix for matching two graphs based on:
    - Euclidean or other distance between points
    - Consistency in edge distances
    - Consistency in edge angles
    - PCA features of the edges
    - Edge descriptors

    Parameters
    ----------
    mov_points : ArrayLike
        (n, 2) array of moving points.
    ref_points : ArrayLike
        (m, 2) array of reference points.
    mov_edges : list[tuple[int, int]]
        List of edges (i, j) in moving graph.
    ref_edges : list[tuple[int, int]]
        List of edges (i, j) in reference graph.
    weights : dict[str, float]
        Weights for different cost components.
    distance_metric : str
        Metric for direct point-to-point distances.
    normalize : bool
        Whether to normalize the cost matrix.

    Notes
    -----
    The cost matrix is computed as the sum of the weighted costs for each component.
    The weights are defined in the `weights` parameter.
    The default weights are:
    - dist: 0.5
    - edge_angle: 1.0
    - edge_length: 1.0
    - pca_dir: 0.0
    - pca_aniso: 0.0
    - edge_descriptor: 0.0

    Returns
    -------
    ArrayLike
        Cost matrix of shape (n, m).
    """
    n, m = len(mov_points), len(ref_points)
    C_total = np.zeros((n, m))

    # --- Default weights ---
    default_weights = {
        "dist": 0.5,
        "edge_angle": 1.0,
        "edge_length": 1.0,
        "pca_dir": 0.0,
        "pca_aniso": 0.0,
        "edge_descriptor": 0.0,
    }
    if weights is None:
        weights = default_weights
    else:
        weights = {**default_weights, **weights}  # override defaults

    # --- Base distance cost ---
    if weights["dist"] > 0:
        C_dist = cdist(mov_points, ref_points, metric=distance_metric)
        if normalize:
            C_dist /= C_dist.max()
        C_total += weights["dist"] * C_dist

    # --- Edge angle and length costs ---
    mov_dists, mov_angles = get_edge_attrs(mov_points, mov_edges)
    ref_dists, ref_angles = get_edge_attrs(ref_points, ref_edges)

    if weights["edge_length"] > 0:
        C_edge_len = compute_edge_consistency_cost(
            n=n,
            m=m,
            mov_attrs=mov_dists,
            ref_attrs=ref_dists,
            mov_edges=mov_edges,
            ref_edges=ref_edges,
            default=1e6,
        )
        if normalize:
            C_edge_len /= C_edge_len.max()
        C_total += weights["edge_length"] * C_edge_len

    if weights["edge_angle"] > 0:
        C_edge_ang = compute_edge_consistency_cost(
            n=n,
            m=m,
            mov_attrs=mov_angles,
            ref_attrs=ref_angles,
            mov_edges=mov_edges,
            ref_edges=ref_edges,
            default=np.pi,
        )
        if normalize:
            C_edge_ang /= np.pi
        C_total += weights["edge_angle"] * C_edge_ang

    # --- PCA features ---
    if weights["pca_dir"] > 0 or weights["pca_aniso"] > 0:
        mov_dirs, mov_aniso = get_local_pca_features(mov_points, mov_edges)
        ref_dirs, ref_aniso = get_local_pca_features(ref_points, ref_edges)

        if weights["pca_dir"] > 0:
            dot = np.clip(np.dot(mov_dirs, ref_dirs.T), -1.0, 1.0)
            C_dir = 1 - np.abs(dot)
            if normalize:
                C_dir /= C_dir.max()
            C_total += weights["pca_dir"] * C_dir

        if weights["pca_aniso"] > 0:
            C_aniso = np.abs(mov_aniso[:, None] - ref_aniso[None, :])
            if normalize:
                C_aniso /= C_aniso.max()
            C_total += weights["pca_aniso"] * C_aniso
    # --- Edge descriptors ---
    if weights["edge_descriptor"] > 0:
        mov_desc = get_edge_descriptors(mov_points, mov_edges)
        ref_desc = get_edge_descriptors(ref_points, ref_edges)
        C_desc = cdist(mov_desc, ref_desc)
        if normalize:
            C_desc /= C_desc.max()
        C_total += weights["edge_descriptor"] * C_desc

    return C_total


def build_edge_graph(
    points: ArrayLike,
    mode: Literal["knn", "radius", "full"] = "knn",
    k: int = 5,
    radius: float = 30.0,
) -> list[tuple[int, int]]:
    """
    Build a set of edges for a graph based on a given strategy.

    Parameters
    ----------
    points : ArrayLike
        (N, 3) array of 3D point coordinates.
    mode : Literal["knn", "radius", "full"]
        Mode for building the edge graph.
    k : int
        Number of neighbors if mode == "knn".
    radius : float
        Distance threshold if mode == "radius".

    Returns
    -------
    list[tuple[int, int]]
        List of (i, j) index pairs representing edges.
    """
    n = len(points)
    if n <= 1:
        return []

    if mode == "knn":
        k_eff = min(k + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k_eff).fit(points)
        _, indices = nbrs.kneighbors(points)
        edges = [(i, j) for i in range(n) for j in indices[i] if i != j]

    elif mode == "radius":
        graph = radius_neighbors_graph(
            points, radius=radius, mode='connectivity', include_self=False
        )
        if graph.nnz == 0:
            return []
        edges = [(i, j) for i in range(n) for j in graph[i].nonzero()[1]]

    elif mode == "full":
        edges = [(i, j) for i in range(n) for j in range(n) if i != j]

    return edges


def match_hungarian_global_cost(
    C: ArrayLike,
    cost_threshold: float = 1e5,
    dummy_cost: float = 1e6,
    max_ratio: float = None,
) -> ArrayLike:
    """
    Runs Hungarian matching with padding for unequal-sized graphs,
    optionally applying max_ratio filtering similar to match_descriptors.

    Parameters
    ----------
    C : ArrayLike
        Cost matrix of shape (n_A, n_B).
    cost_threshold : float
        Maximum cost to consider a valid match.
    dummy_cost : float
        Cost assigned to dummy nodes (must be > cost_threshold).
    max_ratio : float, optional
        Maximum allowed ratio between best and second-best cost.

    Returns
    -------
    ArrayLike
        Array of shape (N_matches, 2) with valid (A_idx, B_idx) pairs.
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


def get_matches_from_hungarian(
    mov_points: ArrayLike,
    ref_points: ArrayLike,
    hungarian_settings: HungarianMatchSettings,
    verbose: bool = False,
) -> ArrayLike:
    """
    Get matches from beads using the hungarian algorithm.
    Parameters
    ----------
    mov_points : ArrayLike
        (n, 2) array of moving points.
    ref_points : ArrayLike
        (m, 2) array of reference points.
    hungarian_settings : HungarianMatchSettings
        Settings for the hungarian match.
    verbose : bool
        If True, prints detailed logs during the process.

    Returns
    -------
    ArrayLike
        (n, 2) array of matches.
    """
    cost_settings = hungarian_settings.cost_matrix_settings
    edge_settings = hungarian_settings.edge_graph_settings
    mov_edges = build_edge_graph(
        mov_points, mode=edge_settings.method, k=edge_settings.k, radius=edge_settings.radius
    )
    ref_edges = build_edge_graph(
        ref_points, mode=edge_settings.method, k=edge_settings.k, radius=edge_settings.radius
    )

    if hungarian_settings.cross_check:
        # Step 1: A → B
        C_ab = compute_cost_matrix(
            mov_points,
            ref_points,
            mov_edges,
            ref_edges,
            weights=cost_settings.weights,
            distance_metric=hungarian_settings.distance_metric,
            normalize=cost_settings.normalize,
        )

        matches_ab = match_hungarian_global_cost(
            C_ab,
            cost_threshold=np.quantile(C_ab, hungarian_settings.cost_threshold),
            max_ratio=hungarian_settings.max_ratio,
        )

        # Step 2: B → A (swap arguments)
        C_ba = compute_cost_matrix(
            ref_points,
            mov_points,
            ref_edges,
            mov_edges,
            weights=cost_settings.weights,
            distance_metric=hungarian_settings.distance_metric,
            normalize=cost_settings.normalize,
        )

        matches_ba = match_hungarian_global_cost(
            C_ba,
            cost_threshold=np.quantile(C_ba, hungarian_settings.cost_threshold),
            max_ratio=hungarian_settings.max_ratio,
        )

        # Step 3: Invert matches_ba to compare
        reverse_map = {(j, i) for i, j in matches_ba}

        # Step 4: Keep only symmetric matches
        matches = np.array([[i, j] for i, j in matches_ab if (i, j) in reverse_map])
    else:
        # without cross-check

        C = compute_cost_matrix(
            mov_points,
            ref_points,
            mov_edges,
            ref_edges,
            weights=cost_settings.weights,
            distance_metric=hungarian_settings.distance_metric,
            normalize=cost_settings.normalize,
        )

        matches = match_hungarian_global_cost(
            C,
            cost_threshold=np.quantile(C, hungarian_settings.cost_threshold),
            max_ratio=hungarian_settings.max_ratio,
        )
    return matches


def filter_matches(
    matches: ArrayLike,
    mov_points: ArrayLike,
    ref_points: ArrayLike,
    angle_threshold: float = 30,
    min_distance_threshold: float = 0.01,
    max_distance_threshold: float = 0.95,
    verbose: bool = False,
) -> ArrayLike:
    """
    Filter matches based on angle and distance thresholds.

    Parameters
    ----------
    matches : ArrayLike
        (n, 2) array of matches.
    mov_points : ArrayLike
        (n, 2) array of moving points.
    ref_points : ArrayLike
        (n, 2) array of reference points.
    angle_threshold : float
        Maximum allowed deviation from dominant angle (degrees).
    min_distance_threshold : float
        Lower quantile cutoff for distance filtering (e.g. 0.05 keeps matches above 5th percentile).
    max_distance_threshold : float
        Upper quantile cutoff for distance filtering (e.g. 0.95 keeps matches below 95th percentile).
    verbose : bool
        If True, prints detailed logs.

    Returns
    -------
    ArrayLike
        (n, 2) array of filtered matches.
    """
    # --- Distance filtering ---
    if min_distance_threshold is not None or max_distance_threshold is not None:
        dist = np.linalg.norm(
            mov_points[matches[:, 0]] - ref_points[matches[:, 1]], axis=1
        )

        low = np.quantile(dist, min_distance_threshold)
        high = np.quantile(dist, max_distance_threshold)

        if verbose:
            click.echo(
                f"Filtering matches with distance quantiles: [{min_distance_threshold}, {max_distance_threshold}]"
            )
            click.echo(f"Distance range: [{low:.3f}, {high:.3f}]")

        keep = (dist >= low) & (dist <= high)
        matches = matches[keep]

        if verbose:
            click.echo(f"Total matches after distance filtering: {len(matches)}")

    # --- Angle filtering ---
    if angle_threshold:
        vectors = ref_points[matches[:, 1]] - mov_points[matches[:, 0]]
        angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles_deg = np.degrees(angles_rad)

        bins = np.linspace(-180, 180, 36)
        hist, bin_edges = np.histogram(angles_deg, bins=bins)
        dominant_bin_index = np.argmax(hist)
        dominant_angle = (
            bin_edges[dominant_bin_index] + bin_edges[dominant_bin_index + 1]
        ) / 2

        filtered_indices = np.where(np.abs(angles_deg - dominant_angle) <= angle_threshold)[0]
        matches = matches[filtered_indices]

        if verbose:
            click.echo(f"Total matches after angle filtering: {len(matches)}")

    return matches
