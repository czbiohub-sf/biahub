"""Graphs of nodes: positions, edges (kNN, radius, all) and per-node descriptors.

A node's descriptors summarize its neighbourhood (edge lengths and angles, PCA of its
neighbours), so matching nodes on them compares their local graphs, not coordinates.
"""

from collections import defaultdict
from functools import cached_property
from typing import Literal

import numpy as np

from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph


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
    >>> graph = Graph.from_nodes(nodes, mode="knn", k=5)

    >>> # Build a graph from nodes with radius neighbors
    >>> graph = Graph.from_nodes(nodes, mode="radius", radius=30.0)

    >>> # Build a graph from nodes with all-to-all edges
    >>> graph = Graph.from_nodes(nodes, mode="full")
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
                points, radius=radius, mode="connectivity", include_self=False
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
