"""Point-cloud registration primitives for sparse bead fields.

Numpy/scipy algorithms over (N, 3) ZYX peak coordinates, kept apart from the volume code
in `registration.beads` so each stage is testable on its own. Returned transforms are
homogeneous (D+1, D+1) matrices in the pull convention used by `vote_icp_register`'s
`initial_transform`: they map reference-space points to moving-space sample coordinates.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike
from scipy.spatial import cKDTree


def translation_matrix(displacement: ArrayLike) -> np.ndarray:
    matrix = np.eye(len(displacement) + 1)
    matrix[:-1, -1] = displacement
    return matrix


def fit_affine(src: ArrayLike, dst: ArrayLike) -> np.ndarray:
    """Least-squares affine C with C(src_i) ~ dst_i, as a homogeneous matrix (N >= D + 1)."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    ndim = src.shape[1]
    design = np.hstack([src, np.ones((len(src), 1))])
    coeffs, *_ = np.linalg.lstsq(design, dst, rcond=None)
    matrix = np.eye(ndim + 1)
    matrix[:ndim, :ndim] = coeffs[:ndim].T
    matrix[:ndim, ndim] = coeffs[ndim]
    return matrix


def vote_icp_register(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    initial_transform: ArrayLike,
    initial_capture_radius: float = 80.0,
    min_capture_radius: float = 10.0,
    radius_decay: float = 0.5,
    cluster_radius: float = 10.0,
    min_votes: int = 3,
    max_iterations: int = 20,
    convergence_translation: float = 0.5,
) -> tuple[np.ndarray | None, dict]:
    """Register two peak clouds by iterated displacement voting.

    An ICP whose correspondence stage is a vote: each iteration maps the moving peaks
    into reference space with the current transform, lets every mapped peak vote for
    its displacement to each reference peak within the capture radius, keeps the densest
    `cluster_radius`-ball of votes as the inlier correspondences and re-fits on them (a
    full affine with >= D + 3 pairs, a pure translation otherwise). The capture radius
    shrinks geometrically, floored at `min_capture_radius`, so early iterations have the
    reach to recover a large residual and late ones only see unambiguous neighbours.

    Unlike one-to-one matching the vote never commits a peak to a single partner, so a
    split or fuzzy detection just casts extra votes outside the winning cluster.

    `initial_transform` and the returned matrix map reference-space points to
    moving-space sample coordinates. Returns `(None, info)` when voting abstained before
    any fit; `info` holds iterations, converged, n_inliers and the final capture radius.
    """
    mov_peaks = np.asarray(mov_peaks, dtype=float)
    ref_peaks = np.asarray(ref_peaks, dtype=float)
    matrix = np.asarray(initial_transform, dtype=float)
    ref_tree = cKDTree(ref_peaks)
    ndim = mov_peaks.shape[1]

    def map_to_ref(m):
        inverse = np.linalg.inv(m)
        return mov_peaks @ inverse[:ndim, :ndim].T + inverse[:ndim, ndim]

    mapped = map_to_ref(matrix)
    radius = initial_capture_radius
    info = {"iterations": 0, "converged": False, "n_inliers": 0, "capture_radius": radius}
    fitted_once = False
    for iteration in range(1, max_iterations + 1):
        votes, pair_indices = [], []
        for i, p in enumerate(mapped):
            for j in ref_tree.query_ball_point(p, r=radius):
                votes.append(ref_peaks[j] - p)
                pair_indices.append((i, j))
        votes = np.asarray(votes)
        if len(votes) < min_votes:
            break
        vote_tree = cKDTree(votes)
        counts = np.array(
            [len(vote_tree.query_ball_point(v, r=cluster_radius)) for v in votes]
        )
        members = vote_tree.query_ball_point(votes[np.argmax(counts)], r=cluster_radius)
        inlier_mov = np.array([pair_indices[m][0] for m in members])
        inlier_ref = np.array([pair_indices[m][1] for m in members])
        # A full affine needs a margin over its degrees of freedom; with fewer inliers the
        # translation update is the only fit that cannot hallucinate.
        if len(members) >= ndim + 3:
            new_matrix = fit_affine(ref_peaks[inlier_ref], mov_peaks[inlier_mov])
        else:
            drift = votes[members].mean(axis=0)
            new_matrix = matrix @ translation_matrix(-drift)
        new_mapped = map_to_ref(new_matrix)
        step = float(np.mean(np.linalg.norm(new_mapped - mapped, axis=1)))
        matrix, mapped, fitted_once = new_matrix, new_mapped, True
        info.update(iterations=iteration, n_inliers=int(len(members)), capture_radius=radius)
        radius = max(min_capture_radius, radius * radius_decay)
        if step < convergence_translation:
            info["converged"] = True
            break
    if not fitted_once:
        return None, info
    return matrix, info


def vote_drift(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    capture_radius: float,
    cluster_radius: float,
    min_votes: int,
) -> tuple[np.ndarray | None, list[tuple[np.ndarray, np.ndarray]], int]:
    """One round of displacement voting between already-aligned peak clouds.

    Returns the mean displacement of the densest vote cluster, that cluster's
    (moving peak, reference peak) pairs, and the cluster size -- or `(None, [], 0)`
    when fewer than `min_votes` votes were cast.
    """
    mov_peaks = np.asarray(mov_peaks, dtype=float)
    ref_peaks = np.asarray(ref_peaks, dtype=float)
    ref_tree = cKDTree(ref_peaks)
    votes, pairs = [], []
    for p in mov_peaks:
        for j in ref_tree.query_ball_point(p, r=capture_radius):
            votes.append(ref_peaks[j] - p)
            pairs.append((p, ref_peaks[j]))
    if len(votes) < min_votes:
        return None, [], 0
    votes = np.asarray(votes)
    vote_tree = cKDTree(votes)
    counts = np.array([len(vote_tree.query_ball_point(v, r=cluster_radius)) for v in votes])
    members = vote_tree.query_ball_point(votes[np.argmax(counts)], r=cluster_radius)
    return votes[members].mean(axis=0), [pairs[m] for m in members], int(counts.max())
