"""Costs between nodes: what makes a pairing good (unary) or two pairings agree (pairwise).

Unary costs give an (N, M) matrix for a linear assignment; the pairwise affinity scores
pairs of candidate pairings by how well their distances agree (graph matching).
"""

import numpy as np

from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from biahub.core.nodes.graph import Graph

# Spectral affinity entries kept: candidate pairs whose distances agree within this many
# sigmas (beyond it the Gaussian weight is < 1.5e-8, numerically irrelevant).
SPECTRAL_CUTOFF_SIGMAS = 6.0
# Largest sparse affinity the spectral matcher builds. Construction holds int64 rows and
# columns and float64 values (24 bytes per entry), then the CSR copy, so the peak is about
# twice that: ~10 GB at this cap (200 x 200 beads measured 3 GB).
SPECTRAL_MAX_NONZEROS = 200_000_000


class MatchingTooLargeError(ValueError):
    """The candidate set is too large for the requested matching algorithm."""


def spectral_affinity(
    d_mov: NDArray, d_ref: NDArray, sigma: float, cutoff_sigmas: float = SPECTRAL_CUTOFF_SIGMAS
) -> csr_matrix:
    """Sparse Leordeanu-Hebert affinity between candidate correspondences.

    Candidate a = (i, j) is index i * n_ref + j. Entry (a, b) = exp(-(d_mov[i, i'] -
    d_ref[j, j'])^2 / 2 sigma^2) for i != i' and j != j' (a point takes one partner, so
    candidates sharing a row or a column never reinforce one another), kept only where
    the distances agree within `cutoff_sigmas`.
    """
    n_m, n_r = len(d_mov), len(d_ref)
    off_diagonal = ~np.eye(n_r, dtype=bool)
    ref_j, ref_j2 = np.nonzero(off_diagonal)
    ref_d = d_ref[off_diagonal]
    order = np.argsort(ref_d, kind="stable")
    ref_j, ref_j2, ref_d = ref_j[order], ref_j2[order], ref_d[order]
    window = cutoff_sigmas * sigma

    mov_i, mov_i2 = np.nonzero(~np.eye(n_m, dtype=bool))
    mov_d = d_mov[mov_i, mov_i2]
    lo = np.searchsorted(ref_d, mov_d - window, side="left")
    hi = np.searchsorted(ref_d, mov_d + window, side="right")
    nnz = int((hi - lo).sum())
    if nnz > SPECTRAL_MAX_NONZEROS:
        raise MatchingTooLargeError(
            f"spectral matching would need {nnz} affinity entries ({n_m} x {n_r} nodes, "
            f"sigma={sigma}); restrict the candidate set or lower sigma"
        )
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    vals = np.empty(nnz, dtype=np.float64)
    at = 0
    for i, i2, d, a, b in zip(mov_i, mov_i2, mov_d, lo, hi, strict=True):
        if a == b:
            continue
        n = b - a
        rows[at : at + n] = i * n_r + ref_j[a:b]
        cols[at : at + n] = i2 * n_r + ref_j2[a:b]
        vals[at : at + n] = np.exp(-((ref_d[a:b] - d) ** 2) / (2 * sigma**2))
        at += n
    return csr_matrix((vals, (rows, cols)), shape=(n_m * n_r, n_m * n_r))


def edge_consistency_cost(
    moving: Graph,
    reference: Graph,
    attr_type: str,
    default_cost: float,
) -> NDArray[np.floating]:
    """Cost of pairing each moving node with each reference node by how well their edges agree.

    Each pair's cost is the mean of the best one-to-one assignment between the two nodes'
    edges (`attr_type`: their lengths, or angles in 2D); `default_cost` where either has none.
    """
    n, m = moving.n_nodes, reference.n_nodes
    cost_matrix = np.full((n, m), default_cost, dtype=np.float32)

    if attr_type == "distance":
        mov_attrs = moving.edge_distances
        ref_attrs = reference.edge_distances
    elif attr_type == "angle":
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


def weighted_cost(
    moving: Graph,
    reference: Graph,
    weights: dict[str, float],
    distance_metric: str = "euclidean",
    normalize: bool = False,
) -> NDArray[np.floating]:
    """
    Return the cost of pairing each moving node with each reference node: a weighted sum.

    `weights` per term -- dist (positions), edge_length, edge_angle (2D), pca_dir,
    pca_aniso, edge_descriptor; a term with weight 0 is skipped. With `normalize`, each
    term is scaled to [0, 1] before weighting.

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
    w = weights

    # Position distance
    if w["dist"] > 0:
        C_dist = cdist(moving.nodes, reference.nodes, metric=distance_metric)
        if normalize:
            max_val = C_dist.max()
            if max_val > 0:
                C_dist = C_dist / max_val
        C_total += w["dist"] * C_dist

    # Edge consistency
    if w["edge_length"] > 0:
        C_edge_len = edge_consistency_cost(
            moving, reference, attr_type="distance", default_cost=1e6
        )
        if normalize:
            max_val = C_edge_len.max()
            if max_val > 0:
                C_edge_len = C_edge_len / max_val
        C_total += w["edge_length"] * C_edge_len

    if w["edge_angle"] > 0 and moving.dim == 2:
        C_edge_ang = edge_consistency_cost(
            moving, reference, attr_type="angle", default_cost=np.pi
        )
        if normalize:
            C_edge_ang = C_edge_ang / np.pi
        C_total += w["edge_angle"] * C_edge_ang

    # PCA features
    if w["pca_dir"] > 0 or w["pca_aniso"] > 0:
        mov_dirs, mov_aniso = moving.pca_features
        ref_dirs, ref_aniso = reference.pca_features

        if w["pca_dir"] > 0:
            dot = np.clip(np.dot(mov_dirs, ref_dirs.T), -1.0, 1.0)
            C_dir = 1 - np.abs(dot)
            if normalize:
                max_val = C_dir.max()
                if max_val > 0:
                    C_dir = C_dir / max_val
            C_total += w["pca_dir"] * C_dir

        if w["pca_aniso"] > 0:
            C_aniso = np.abs(mov_aniso[:, None] - ref_aniso[None, :])
            if normalize:
                max_val = C_aniso.max()
                if max_val > 0:
                    C_aniso = C_aniso / max_val
            C_total += w["pca_aniso"] * C_aniso

    # Edge descriptors
    if w["edge_descriptor"] > 0:
        mov_desc = moving.edge_descriptors
        ref_desc = reference.edge_descriptors
        C_desc = cdist(mov_desc, ref_desc)
        if normalize:
            max_val = C_desc.max()
            if max_val > 0:
                C_desc = C_desc / max_val
        C_total += w["edge_descriptor"] * C_desc

    return C_total
