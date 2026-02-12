from collections import defaultdict
from functools import cached_property
from typing import Literal, Optional

import numpy as np

from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import match_descriptors
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

# ============================================================
# GRAPH CLASS
# ============================================================


class Graph:
    """
    Geometric graph for 2D/3D point registration with local feature extraction.

    Parameters
    ----------
    nodes : NDArray[np.floating]
        (N, D) array of nodes (points)
    edges : list[tuple[int, int]]
        List of edges (pairs of node indices)
    mode : Literal["knn", "radius", "full"], default="knn"
        Mode for building edges:
        - "knn": k-nearest neighbors
        - "radius": radius neighbors
        - "full": all-to-all
    k : int, default=5
        Number of nearest neighbors for k-nearest neighbors mode
    radius : float, default=30.0
        Radius for radius neighbors mode

    Examples
    --------
    >>> # Build a graph from nodes with k-nearest neighbors
    >>> graph = Graph.from_nodes(nodes, mode='knn', k=5)

    >>> # Build a graph from nodes with radius neighbors
    >>> graph = Graph.from_nodes(nodes, mode='radius', radius=30.0)

    >>> # Build a graph from nodes with all-to-all edges
    >>> graph = Graph.from_nodes(nodes, mode='full')
    """

    def __init__(
        self,
        nodes: NDArray[np.floating],
        edges: list[tuple[int, int]],
    ):
        self.nodes = np.asarray(nodes, dtype=np.float32)
        self._edges = edges

        if self.nodes.ndim != 2:
            raise ValueError(f"nodes must be 2D array, got shape {self.nodes.shape}")
        if self.dim not in (2, 3):
            raise ValueError(f"nodes must be 2D or 3D points, got dim={self.dim}")

    @classmethod
    def from_nodes(
        cls,
        nodes: NDArray[np.floating],
        mode: Literal["knn", "radius", "full"] = "knn",
        k: int = 5,
        radius: float = 30.0,
    ) -> "Graph":
        """Build a graph from nodes with automatic edge construction."""
        edges = cls._build_edges(nodes, mode=mode, k=k, radius=radius)
        return cls(nodes, edges)

    @staticmethod
    def _build_edges(
        points: NDArray[np.floating],
        mode: Literal["knn", "radius", "full"] = "knn",
        k: int = 5,
        radius: float = 30.0,
    ) -> list[tuple[int, int]]:
        """Build edges for a point set using various strategies."""
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

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return edges

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def dim(self) -> int:
        return self.nodes.shape[1]

    @property
    def edges(self) -> list[tuple[int, int]]:
        return self._edges

    @cached_property
    def neighbor_map(self) -> dict[int, list[int]]:
        """Adjacency list representation."""
        neighbors = defaultdict(list)
        for i, j in self._edges:
            neighbors[i].append(j)
        return dict(neighbors)

    @cached_property
    def edge_distances(self) -> dict[tuple[int, int], float]:
        """Distance for each edge (bidirectional)."""
        distances = {}
        for i, j in self._edges:
            vec = self.nodes[j] - self.nodes[i]
            d = float(np.linalg.norm(vec))
            distances[(i, j)] = distances[(j, i)] = d
        return distances

    @cached_property
    def edge_angles(self) -> dict[tuple[int, int], float]:
        """Angle for each edge in radians (2D only)."""
        if self.dim != 2:
            return {}

        angles = {}
        for i, j in self._edges:
            vec = self.nodes[j] - self.nodes[i]
            angle = float(np.arctan2(vec[1], vec[0]))
            angles[(i, j)] = angles[(j, i)] = angle
        return angles

    @cached_property
    def edge_descriptors(self) -> NDArray[np.floating]:
        """
        Local edge statistics for each node.

        Returns (N, 4): [mean_length, std_length, mean_angle, std_angle]
        """
        desc = np.zeros((self.n_nodes, 4), dtype=np.float32)

        for i in range(self.n_nodes):
            neighbors = self.neighbor_map.get(i, [])
            if not neighbors:
                continue

            lengths = np.array([self.edge_distances[(i, j)] for j in neighbors])
            desc[i, 0] = np.mean(lengths)
            desc[i, 1] = np.std(lengths)

            if self.dim == 2 and self.edge_angles:
                angles = np.array([self.edge_angles[(i, j)] for j in neighbors])
                desc[i, 2] = np.mean(angles)
                desc[i, 3] = np.std(angles)

        return desc

    @cached_property
    def pca_features(self) -> tuple[NDArray, NDArray]:
        """
        PCA-based features for local neighborhoods.

        Returns
        -------
        directions : (N, D) array of dominant directions
        anisotropy : (N,) array of anisotropy ratios
        """
        n = self.n_nodes
        d = self.dim
        directions = np.zeros((n, d), dtype=np.float32)
        anisotropy = np.zeros(n, dtype=np.float32)

        for i in range(n):
            neighbors = self.neighbor_map.get(i, [])
            if not neighbors:
                directions[i] = np.nan
                anisotropy[i] = np.nan
                continue

            local_points = self.nodes[neighbors].copy()
            local_points -= local_points.mean(axis=0)

            _, S, Vt = np.linalg.svd(local_points, full_matrices=False)

            directions[i] = Vt[0] if Vt.shape[0] > 0 else np.zeros(d)
            anisotropy[i] = S[0] / (S[-1] + 1e-5) if len(S) >= 2 else 0.0

        return directions, anisotropy

    def get_neighbors(self, node_idx: int) -> list[int]:
        """Get neighbor indices for a specific node."""
        return self.neighbor_map.get(node_idx, [])

    def __repr__(self) -> str:
        return f"Graph(n_nodes={self.n_nodes}, n_edges={len(self.edges)}, dim={self.dim})"


# ============================================================
# GRAPH MATCHER
# ============================================================


class GraphMatcher:
    """
    Matches nodes between two geometric graphs for point registration.

    Supports two matching algorithms:
    - 'hungarian': Graph-based matching using cost matrix + Hungarian algorithm
    - 'descriptor': Feature-based matching using skimage.feature.match_descriptors

    Parameters
    ----------
    algorithm : {'hungarian', 'descriptor'}, default='hungarian'
        Matching algorithm to use
    weights : dict[str, float], optional
        Weights for cost components (hungarian only):
        - 'dist': Direct position distance (default: 0.5)
        - 'edge_length': Local edge length consistency (default: 1.0)
        - 'edge_angle': Local edge angle consistency (default: 1.0)
        - 'pca_dir': PCA direction similarity (default: 0.0)
        - 'pca_aniso': PCA anisotropy similarity (default: 0.0)
        - 'edge_descriptor': Edge descriptor distance (default: 0.0)
    distance_metric : str, default='euclidean'
        Metric for position distances
    normalize : bool, default=False
        Whether to normalize each cost component to [0, 1]
    cost_threshold : float, default=0.9
        Quantile threshold for accepting matches (0.0-1.0)
    cross_check : bool, default=False
        Whether to require bidirectional consistency
    max_ratio : float, optional
        Maximum ratio between best and second-best match (Lowe's ratio test)
    metric : str, default='euclidean'
        Distance metric for descriptor matching (descriptor algorithm only)
    verbose : bool, default=False
        Print matching statistics

    Examples
    --------
    >>> # Hungarian matching (graph-based)
    >>> matcher = GraphMatcher(
    ...     algorithm='hungarian',
    ...     weights={'dist': 0.5, 'edge_length': 1.0},
    ...     cross_check=True
    ... )
    >>> matches = matcher.match(moving_graph, ref_graph)

    >>> # Descriptor matching (feature-based)
    >>> matcher = GraphMatcher(
    ...     algorithm='descriptor',
    ...     cross_check=True,
    ...     max_ratio=0.8
    ... )
    >>> matches = matcher.match(moving_graph, ref_graph)
    """

    def __init__(
        self,
        algorithm: Literal['hungarian', 'descriptor'] = 'hungarian',
        weights: Optional[dict[str, float]] = None,
        distance_metric: str = 'euclidean',
        normalize: bool = False,
        cost_threshold: float = 0.9,
        cross_check: bool = False,
        max_ratio: Optional[float] = None,
        metric: str = 'euclidean',  # for descriptor matching
        verbose: bool = False,
    ):
        self.algorithm = algorithm

        # Hungarian-specific parameters
        default_weights = {
            "dist": 0.5,
            "edge_length": 1.0,
            "edge_angle": 1.0,
            "pca_dir": 0.0,
            "pca_aniso": 0.0,
            "edge_descriptor": 0.0,
        }
        self.weights = {**default_weights, **(weights or {})}
        self.distance_metric = distance_metric
        self.normalize = normalize
        self.cost_threshold = cost_threshold

        # Common parameters
        self.cross_check = cross_check
        self.max_ratio = max_ratio
        self.verbose = verbose

        # Descriptor matching parameters
        self.metric = metric

    def match(
        self,
        moving: Graph,
        reference: Graph,
        verbose: Optional[bool] = None,
    ) -> NDArray[np.integer]:
        """
        Find correspondences between two graphs.

        Parameters
        ----------
        moving : Graph
            Moving/source graph
        reference : Graph
            Reference/target graph
        verbose : bool, optional
            Override instance verbose setting

        Returns
        -------
        NDArray
            (N_matches, 2) array of [moving_idx, reference_idx] pairs
        """
        verbose = verbose if verbose is not None else self.verbose

        # Validate
        if moving.dim != reference.dim:
            raise ValueError(
                f"Dimension mismatch: moving={moving.dim}D, reference={reference.dim}D"
            )

        if moving.n_nodes == 0 or reference.n_nodes == 0:
            if verbose:
                print("Warning: One or both graphs are empty")
            return np.array([]).reshape(0, 2).astype(np.int32)

        # Dispatch to appropriate algorithm
        if self.algorithm == 'hungarian':
            return self._match_hungarian(moving, reference, verbose)
        elif self.algorithm == 'descriptor':
            return self._match_descriptor(moving, reference, verbose)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    # ============================================================
    # HUNGARIAN MATCHING
    # ============================================================

    def _match_hungarian(
        self,
        moving: Graph,
        reference: Graph,
        verbose: bool,
    ) -> NDArray[np.integer]:
        """Hungarian algorithm matching."""
        if self.cross_check:
            return self._match_hungarian_cross_check(moving, reference, verbose)
        else:
            return self._match_hungarian_single(moving, reference, verbose)

    def _match_hungarian_single(
        self,
        moving: Graph,
        reference: Graph,
        verbose: bool,
    ) -> NDArray[np.integer]:
        """Hungarian matching in one direction."""
        C = self.compute_cost_matrix(moving, reference)
        matches = self._solve_assignment(C, verbose)
        return matches

    def _match_hungarian_cross_check(
        self,
        moving: Graph,
        reference: Graph,
        verbose: bool,
    ) -> NDArray[np.integer]:
        """Hungarian matching with bidirectional consistency."""
        if verbose:
            print("Computing forward matches (A → B)...")

        C_ab = self.compute_cost_matrix(moving, reference)
        matches_ab = self._solve_assignment(C_ab, False)

        if verbose:
            print(f"Forward: {len(matches_ab)} matches")
            print("Computing backward matches (B → A)...")

        C_ba = self.compute_cost_matrix(reference, moving)
        matches_ba = self._solve_assignment(C_ba, False)

        if verbose:
            print(f"Backward: {len(matches_ba)} matches")

        # Keep only symmetric matches
        reverse_map = {(j, i) for i, j in matches_ba}
        matches = np.array(
            [[i, j] for i, j in matches_ab if (i, j) in reverse_map], dtype=np.int32
        )

        if verbose:
            print(f"Cross-check: {len(matches)} symmetric matches")

        return matches

    def compute_cost_matrix(
        self,
        moving: Graph,
        reference: Graph,
    ) -> NDArray[np.floating]:
        """
        Compute full cost matrix between two graphs.

        Parameters
        ----------
        moving : Graph
            Moving graph with N nodes
        reference : Graph
            Reference graph with M nodes

        Returns
        -------
        NDArray
            (N, M) cost matrix where C[i,j] = cost of matching moving[i] to reference[j]
        """
        n, m = moving.n_nodes, reference.n_nodes
        C_total = np.zeros((n, m), dtype=np.float32)
        w = self.weights

        # Position distance
        if w["dist"] > 0:
            C_dist = cdist(moving.nodes, reference.nodes, metric=self.distance_metric)
            if self.normalize:
                max_val = C_dist.max()
                if max_val > 0:
                    C_dist = C_dist / max_val
            C_total += w["dist"] * C_dist

        # Edge consistency
        if w["edge_length"] > 0:
            C_edge_len = self._compute_edge_consistency_cost(
                moving, reference, attr_type='distance', default_cost=1e6
            )
            if self.normalize:
                max_val = C_edge_len.max()
                if max_val > 0:
                    C_edge_len = C_edge_len / max_val
            C_total += w["edge_length"] * C_edge_len

        if w["edge_angle"] > 0 and moving.dim == 2:
            C_edge_ang = self._compute_edge_consistency_cost(
                moving, reference, attr_type='angle', default_cost=np.pi
            )
            if self.normalize:
                C_edge_ang = C_edge_ang / np.pi
            C_total += w["edge_angle"] * C_edge_ang

        # PCA features
        if w["pca_dir"] > 0 or w["pca_aniso"] > 0:
            mov_dirs, mov_aniso = moving.pca_features
            ref_dirs, ref_aniso = reference.pca_features

            if w["pca_dir"] > 0:
                dot = np.clip(np.dot(mov_dirs, ref_dirs.T), -1.0, 1.0)
                C_dir = 1 - np.abs(dot)
                if self.normalize:
                    max_val = C_dir.max()
                    if max_val > 0:
                        C_dir = C_dir / max_val
                C_total += w["pca_dir"] * C_dir

            if w["pca_aniso"] > 0:
                C_aniso = np.abs(mov_aniso[:, None] - ref_aniso[None, :])
                if self.normalize:
                    max_val = C_aniso.max()
                    if max_val > 0:
                        C_aniso = C_aniso / max_val
                C_total += w["pca_aniso"] * C_aniso

        # Edge descriptors
        if w["edge_descriptor"] > 0:
            mov_desc = moving.edge_descriptors
            ref_desc = reference.edge_descriptors
            C_desc = cdist(mov_desc, ref_desc)
            if self.normalize:
                max_val = C_desc.max()
                if max_val > 0:
                    C_desc = C_desc / max_val
            C_total += w["edge_descriptor"] * C_desc

        return C_total

    def _compute_edge_consistency_cost(
        self,
        moving: Graph,
        reference: Graph,
        attr_type: str,
        default_cost: float,
    ) -> NDArray[np.floating]:
        """Compute cost based on local edge attribute consistency."""
        n, m = moving.n_nodes, reference.n_nodes
        cost_matrix = np.full((n, m), default_cost, dtype=np.float32)

        if attr_type == 'distance':
            mov_attrs = moving.edge_distances
            ref_attrs = reference.edge_distances
        elif attr_type == 'angle':
            mov_attrs = moving.edge_angles
            ref_attrs = reference.edge_angles
            if not mov_attrs or not ref_attrs:
                return cost_matrix
        else:
            raise ValueError(f"Unknown attr_type: {attr_type}")

        mov_neighbors = moving.neighbor_map
        ref_neighbors = reference.neighbor_map

        for i in range(n):
            i_neighbors = mov_neighbors.get(i, [])
            if not i_neighbors:
                continue

            for j in range(m):
                j_neighbors = ref_neighbors.get(j, [])
                if not j_neighbors:
                    continue

                C_local = np.full(
                    (len(i_neighbors), len(j_neighbors)), default_cost, dtype=np.float32
                )

                for ii, ni in enumerate(i_neighbors):
                    for jj, nj in enumerate(j_neighbors):
                        mov_edge = (i, ni)
                        ref_edge = (j, nj)

                        if mov_edge in mov_attrs and ref_edge in ref_attrs:
                            C_local[ii, jj] = abs(mov_attrs[mov_edge] - ref_attrs[ref_edge])

                row_ind, col_ind = linear_sum_assignment(C_local)
                matched_costs = C_local[row_ind, col_ind]
                cost_matrix[i, j] = (
                    matched_costs.mean() if len(matched_costs) > 0 else default_cost
                )

        return cost_matrix

    def _solve_assignment(
        self,
        C: NDArray[np.floating],
        verbose: bool,
    ) -> NDArray[np.integer]:
        """Solve assignment problem with padding for unequal sizes."""
        n_A, n_B = C.shape
        n = max(n_A, n_B)

        # Pad to square
        dummy_cost = 1e6
        C_padded = np.full((n, n), dummy_cost, dtype=np.float32)
        C_padded[:n_A, :n_B] = C

        # Solve
        row_ind, col_ind = linear_sum_assignment(C_padded)

        # Filter matches
        cost_thresh = np.quantile(C, self.cost_threshold)
        matches = []

        for i, j in zip(row_ind, col_ind):
            if i >= n_A or j >= n_B:
                continue

            if C[i, j] >= cost_thresh:
                continue

            if self.max_ratio is not None:
                costs_i = C[i, :]
                sorted_costs = np.sort(costs_i)
                if len(sorted_costs) > 1:
                    second_best = sorted_costs[1]
                    ratio = C[i, j] / (second_best + 1e-10)
                    if ratio > self.max_ratio:
                        continue

            matches.append((i, j))

        if verbose:
            print(f"Found {len(matches)} matches (cost_threshold={cost_thresh:.3f})")

        return np.array(matches, dtype=np.int32).reshape(-1, 2)

    # ============================================================
    # DESCRIPTOR MATCHING
    # ============================================================

    def _match_descriptor(
        self,
        moving: Graph,
        reference: Graph,
        verbose: bool,
    ) -> NDArray[np.integer]:
        """
        Feature-based matching using edge descriptors.

        Uses skimage.feature.match_descriptors with edge_descriptors
        as the feature vectors.
        """
        # Get descriptors
        mov_desc = moving.nodes
        ref_desc = reference.nodes

        if verbose:
            print(
                f"Matching {mov_desc.shape[0]} moving descriptors to {ref_desc.shape[0]} reference descriptors"
            )

        # Use skimage's match_descriptors
        matches = match_descriptors(
            mov_desc,
            ref_desc,
            metric=self.metric,
            cross_check=self.cross_check,
            max_ratio=self.max_ratio if self.max_ratio is not None else 1.0,
        )

        if verbose:
            print(f"Found {len(matches)} descriptor matches")

        return matches.astype(np.int32)

    def filter_matches(
        self,
        matches: NDArray[np.integer],
        moving: Graph,
        reference: Graph,
        angle_threshold: Optional[float] = None,
        direction_threshold: Optional[float] = None,
        min_distance_quantile: float = 0.01,
        max_distance_quantile: float = 0.95,
        verbose: Optional[bool] = None,
    ) -> NDArray[np.integer]:
        """
        Filter matches based on geometric consistency.

        Parameters
        ----------
        matches : NDArray
            (N, 2) array of matches
        moving : Graph
            Moving graph
        reference : Graph
            Reference graph
        angle_threshold : float, optional
            Maximum deviation from dominant angle (degrees, 2D only).
            If None, skip 2D angle filtering.
        direction_threshold : float, optional
            Maximum angular deviation from dominant direction (degrees, 2D/3D).
            Uses dot product between normalized vectors.
            If None, skip direction filtering.
        min_distance_quantile : float
            Lower quantile cutoff for distances
        max_distance_quantile : float
            Upper quantile cutoff for distances
        verbose : bool, optional
            Override instance verbose setting

        Returns
        -------
        NDArray
            (K, 2) filtered matches where K <= N
        """
        verbose = verbose if verbose is not None else self.verbose

        if len(matches) == 0:
            return matches

        # Distance filtering
        if min_distance_quantile is not None or max_distance_quantile is not None:
            dist = np.linalg.norm(
                moving.nodes[matches[:, 0]] - reference.nodes[matches[:, 1]], axis=1
            )

            low = np.quantile(dist, min_distance_quantile)
            high = np.quantile(dist, max_distance_quantile)

            if verbose:
                print(
                    f"Distance filtering: quantiles [{min_distance_quantile}, {max_distance_quantile}]"
                )
                print(f"Distance range: [{low:.3f}, {high:.3f}]")

            keep = (dist >= low) & (dist <= high)
            matches = matches[keep]

            if verbose:
                print(f"Matches after distance filtering: {len(matches)}")

        # Direction filtering (2D/3D) - NEW
        if direction_threshold is not None:
            vectors = reference.nodes[matches[:, 1]] - moving.nodes[matches[:, 0]]

            # Normalize vectors
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            unit_vectors = vectors / (norms + 1e-10)

            # Find dominant direction using circular/spherical mean
            mean_direction = unit_vectors.mean(axis=0)
            mean_direction = mean_direction / (np.linalg.norm(mean_direction) + 1e-10)

            # Compute angular deviation from dominant direction
            dot_products = np.clip(unit_vectors @ mean_direction, -1.0, 1.0)
            angles_rad = np.arccos(dot_products)
            angles_deg = np.degrees(angles_rad)

            keep = angles_deg <= direction_threshold
            matches = matches[keep]

            if verbose:
                print(f"Dominant direction: {mean_direction}")
                print(f"Direction threshold: {direction_threshold}°")
                print(f"Matches after direction filtering: {len(matches)}")

        # Angle filtering (2D only, legacy)
        if angle_threshold is not None and moving.dim == 2:
            vectors = reference.nodes[matches[:, 1]] - moving.nodes[matches[:, 0]]
            angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
            angles_deg = np.degrees(angles_rad)

            bins = np.linspace(-180, 180, 36)
            hist, bin_edges = np.histogram(angles_deg, bins=bins)
            dominant_bin_index = np.argmax(hist)
            dominant_angle = (
                bin_edges[dominant_bin_index] + bin_edges[dominant_bin_index + 1]
            ) / 2

            keep = np.abs(angles_deg - dominant_angle) <= angle_threshold
            matches = matches[keep]

            if verbose:
                print(f"Dominant angle: {dominant_angle:.1f}°")
                print(f"Matches after 2D angle filtering: {len(matches)}")

        return matches
