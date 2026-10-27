"""Solvers: from costs to pairs (or a centre) -- how the best match is found.

A linear assignment pairs nodes on unary costs (exact, Hungarian); the spectral solver
relaxes the pairwise (quadratic) problem to an eigenvector and rounds it greedily; the
medoid is the centre of one set of nodes.
"""

import click
import numpy as np

from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def linear_assignment(
    C: NDArray[np.floating],
    cost_threshold: float = 0.9,
    max_ratio: float | None = None,
    verbose: bool = False,
) -> NDArray[np.integer]:
    """Return the minimum-cost one-to-one pairs of an (N, M) cost matrix (Hungarian).

    Padded to square with a large cost; pairs at or above the `cost_threshold` quantile of
    the costs are dropped, and with `max_ratio`, pairs whose cost is not clearly below the
    row's second best (Lowe's ratio test).
    """
    n_A, n_B = C.shape
    n = max(n_A, n_B)

    # Pad to square
    dummy_cost = 1e6
    C_padded = np.full((n, n), dummy_cost, dtype=np.float32)
    C_padded[:n_A, :n_B] = C

    # Solve
    row_ind, col_ind = linear_sum_assignment(C_padded)

    # Filter matches
    cost_thresh = np.quantile(C, cost_threshold)
    matches = []

    for i, j in zip(row_ind, col_ind, strict=True):
        if i >= n_A or j >= n_B:
            continue

        if C[i, j] >= cost_thresh:
            continue

        if max_ratio is not None:
            costs_i = C[i, :]
            sorted_costs = np.sort(costs_i)
            if len(sorted_costs) > 1:
                second_best = sorted_costs[1]
                ratio = C[i, j] / (second_best + 1e-10)
                if ratio > max_ratio:
                    continue

        matches.append((i, j))

    if verbose:
        click.echo(f"Found {len(matches)} matches (cost_threshold={cost_thresh:.3f})")

    return np.array(matches, dtype=np.int32).reshape(-1, 2)


def spectral_assignment(
    affinity, n_mov: int, n_ref: int, rel_cut: float = 0.5, max_iter: int = 60
) -> NDArray[np.integer]:
    """Return one-to-one pairs from a pairwise affinity's leading eigenvector (spectral).

    Candidate a = (i, j) is index i * n_ref + j (`costs.spectral_affinity`). Power
    iteration finds the eigenvector; pairs are taken greedily in score order, each node
    once, down to `rel_cut` times the best score (never a zero score).
    """
    n_cand = n_mov * n_ref
    ii = np.repeat(np.arange(n_mov), n_ref)
    jj = np.tile(np.arange(n_ref), n_mov)
    x = np.ones(n_cand) / np.sqrt(n_cand)
    for _ in range(max_iter):
        x = affinity @ x
        norm = np.linalg.norm(x)
        if norm == 0:
            break
        x /= norm

    threshold = rel_cut * x.max()
    used_i, used_j, matches = set(), set(), []
    for idx in np.argsort(-x):
        # keep scores at the cut (rel_cut 1.0 keeps the top one), never a zero score
        if x[idx] < threshold or x[idx] <= 0:
            break
        i, j = int(ii[idx]), int(jj[idx])
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matches.append((i, j))
    return np.asarray(matches, dtype=np.int32).reshape(-1, 2)


def cross_checked(
    forward: NDArray[np.integer], backward: NDArray[np.integer]
) -> NDArray[np.integer]:
    """Return the forward pairs (i, j) that the backward pass also chose, as (j, i)."""
    reverse_map = {(j, i) for i, j in backward}
    return np.array([[i, j] for i, j in forward if (i, j) in reverse_map], dtype=np.int32)


def medoid(distances: NDArray[np.floating]) -> int:
    """Return the node with the smallest summed distance to all the others.

    `distances` is an (N, N) matrix between nodes of one set (points, or e.g. positions
    with their transforms and `Transform.displacement_to` as the distance). The first
    such node on a tie.
    """
    distances = np.asarray(distances, dtype=float)
    return int(np.argmin(distances.sum(axis=1)))
