"""Matchers: a cost and a solver, named -- two graphs in, (moving, reference) index pairs out.

"hungarian" = the weighted unary cost + linear assignment; "spectral" = the pairwise
affinity + the spectral solver; "descriptor" = scikit-image's descriptor matching.
`filter_matches` then drops pairs inconsistent with the dominant displacement.
"""

from typing import Literal

import click
import numpy as np

from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from skimage.feature import match_descriptors

from biahub.core.nodes.costs import spectral_affinity, weighted_cost
from biahub.core.nodes.graph import Graph
from biahub.core.nodes.solvers import cross_checked, linear_assignment, spectral_assignment


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
    ...     algorithm="hungarian", weights={"dist": 0.5, "edge_length": 1.0}, cross_check=True
    ... )
    >>> matches = matcher.match(moving_graph, ref_graph)

    >>> # Descriptor matching (feature-based)
    >>> matcher = GraphMatcher(algorithm="descriptor", cross_check=True, max_ratio=0.8)
    >>> matches = matcher.match(moving_graph, ref_graph)
    """

    def __init__(
        self,
        algorithm: Literal["hungarian", "descriptor", "spectral"] = "hungarian",
        weights: dict[str, float] | None = None,
        distance_metric: str = "euclidean",
        normalize: bool = False,
        cost_threshold: float = 0.9,
        cross_check: bool = False,
        max_ratio: float | None = None,
        metric: str = "euclidean",  # for descriptor matching
        spectral_sigma: float = 3.0,
        spectral_rel_cut: float = 0.5,
        spectral_max_iter: int = 60,
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

        # Spectral matching parameters
        self.spectral_sigma = spectral_sigma
        self.spectral_rel_cut = spectral_rel_cut
        self.spectral_max_iter = spectral_max_iter

    def match(
        self,
        moving: Graph,
        reference: Graph,
        verbose: bool | None = None,
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
                click.echo("Warning: One or both graphs are empty")
            return np.array([]).reshape(0, 2).astype(np.int32)

        # Dispatch to appropriate algorithm
        if self.algorithm == "hungarian":
            return self._match_hungarian(moving, reference, verbose)
        elif self.algorithm == "descriptor":
            return self._match_descriptor(moving, reference, verbose)
        elif self.algorithm == "spectral":
            return self._match_spectral(moving, reference, verbose)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    # ============================================================
    # HUNGARIAN MATCHING
    # ============================================================

    def _match_spectral(
        self,
        moving: Graph,
        reference: Graph,
        verbose: bool,
    ) -> NDArray[np.integer]:
        """Leordeanu-Hebert spectral matching.

        Scores each candidate correspondence by how many OTHER candidates agree with it
        under rigid geometry: (i, j) and (i', j') agree when the distance between moving
        points i and i' matches the distance between reference points j and j'. Only
        relative distances enter, so the result is invariant to translation and rotation
        -- which is what lets it acquire a correspondence from a poor initial transform,
        where the position-dominated Hungarian cost pairs beads with the wrong neighbour.
        Given a good initial transform the Hungarian matcher is more precise: acquire with
        this, then refine.

        The affinity is sparse (`spectral_affinity`): only candidate pairs whose distances
        agree within a few sigma are stored, so memory follows the geometric agreement,
        not (n_mov * n_ref)^2. Raises `MatchingTooLargeError` past a fixed entry budget.
        """
        n_m, n_r = moving.n_nodes, reference.n_nodes
        n_cand = n_m * n_r
        M = spectral_affinity(
            cdist(moving.nodes, moving.nodes),
            cdist(reference.nodes, reference.nodes),
            self.spectral_sigma,
        )

        matches = spectral_assignment(
            M, n_m, n_r, self.spectral_rel_cut, self.spectral_max_iter
        )
        if verbose:
            click.echo(
                f"Spectral matching: {len(matches)} matches from {n_cand} candidates "
                f"(sigma={self.spectral_sigma}, rel_cut={self.spectral_rel_cut})"
            )
        return matches

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
        return linear_assignment(C, self.cost_threshold, self.max_ratio, verbose)

    def _match_hungarian_cross_check(
        self,
        moving: Graph,
        reference: Graph,
        verbose: bool,
    ) -> NDArray[np.integer]:
        """Hungarian matching with bidirectional consistency."""
        if verbose:
            click.echo("Computing forward matches (A → B)...")

        C_ab = self.compute_cost_matrix(moving, reference)
        matches_ab = linear_assignment(C_ab, self.cost_threshold, self.max_ratio)

        if verbose:
            click.echo(f"Forward: {len(matches_ab)} matches")
            click.echo("Computing backward matches (B → A)...")

        C_ba = self.compute_cost_matrix(reference, moving)
        matches_ba = linear_assignment(C_ba, self.cost_threshold, self.max_ratio)

        if verbose:
            click.echo(f"Backward: {len(matches_ba)} matches")

        matches = cross_checked(matches_ab, matches_ba)  # keep only symmetric matches

        if verbose:
            click.echo(f"Cross-check: {len(matches)} symmetric matches")

        return matches

    def compute_cost_matrix(
        self,
        moving: Graph,
        reference: Graph,
    ) -> NDArray[np.floating]:
        """Return the (N, M) cost matrix of this matcher's weights (`costs.weighted_cost`)."""
        return weighted_cost(
            moving, reference, self.weights, self.distance_metric, self.normalize
        )

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
            click.echo(
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
            click.echo(f"Found {len(matches)} descriptor matches")

        return matches.astype(np.int32)

    def filter_matches(
        self,
        matches: NDArray[np.integer],
        moving: Graph,
        reference: Graph,
        angle_threshold: float | None = 0,
        direction_threshold: float | None = 0,
        min_distance_quantile: float = 0.01,
        max_distance_quantile: float = 0.95,
        verbose: bool | None = None,
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
            0 or None skips 2D angle filtering.
        direction_threshold : float, optional
            Maximum angular deviation from dominant direction (degrees, 2D/3D).
            Uses dot product between normalized vectors.
            0 or None skips direction filtering.
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
        if min_distance_quantile != 0 or max_distance_quantile != 0:
            dist = np.linalg.norm(
                moving.nodes[matches[:, 0]] - reference.nodes[matches[:, 1]], axis=1
            )

            low = np.quantile(dist, min_distance_quantile)
            high = np.quantile(dist, max_distance_quantile)

            if verbose:
                click.echo(
                    f"Distance filtering: quantiles [{min_distance_quantile}, {max_distance_quantile}]"
                )
                click.echo(f"Distance range: [{low:.3f}, {high:.3f}]")

            keep = (dist >= low) & (dist <= high)
            matches = matches[keep]

            if verbose:
                click.echo(f"Matches after distance filtering: {len(matches)}")

        # Direction filtering (2D/3D) - NEW
        if direction_threshold:  # 0 or None: no direction filter
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
                click.echo(f"Dominant direction: {mean_direction}")
                click.echo(f"Direction threshold: {direction_threshold}°")
                click.echo(f"Matches after direction filtering: {len(matches)}")

        # Angle filtering (2D only, legacy)
        if angle_threshold and moving.dim == 2:  # 0 or None: no angle filter
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
                click.echo(f"Dominant angle: {dominant_angle:.1f}°")
                click.echo(f"Matches after 2D angle filtering: {len(matches)}")

        return matches
