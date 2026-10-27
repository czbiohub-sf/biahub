"""core.nodes: graphs, costs and solvers, usable on their own (arrays in, pairs out)."""

import numpy as np

from scipy.spatial.distance import cdist

from biahub.core.nodes.costs import spectral_affinity, weighted_cost
from biahub.core.nodes.graph import Graph
from biahub.core.nodes.matching import GraphMatcher
from biahub.core.nodes.solvers import cross_checked, linear_assignment, spectral_assignment


def test_linear_assignment_pairs_each_row_with_its_cheapest_free_column():
    costs = np.array([[1.0, 9.0, 9.0], [9.0, 9.0, 2.0], [9.0, 3.0, 9.0], [9.0, 9.0, 9.0]])
    pairs = linear_assignment(costs, cost_threshold=0.5)
    assert pairs.tolist() == [[0, 0], [1, 2], [2, 1]]  # row 3 only had costs at the cut


def test_cross_checked_keeps_pairs_both_directions_chose():
    forward = np.array([[0, 1], [1, 0], [2, 2]])
    backward = np.array([[1, 0], [0, 2], [2, 2]])  # (j, i): agrees on (0, 1) and (2, 2)
    assert cross_checked(forward, backward).tolist() == [[0, 1], [2, 2]]


def _clouds():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 100, (30, 3))
    mov = ref[::-1] + np.array([1.0, 2.0, -1.0])  # reversed order, shifted
    return mov, ref


def test_the_matchers_are_a_cost_and_a_solver():
    mov, ref = _clouds()
    mg, rg = Graph.from_nodes(mov, mode="knn", k=5), Graph.from_nodes(ref, mode="knn", k=5)
    hungarian = GraphMatcher(algorithm="hungarian")
    cost = weighted_cost(mg, rg, hungarian.weights)
    np.testing.assert_array_equal(
        hungarian.match(mg, rg), linear_assignment(cost, hungarian.cost_threshold)
    )
    spectral = GraphMatcher(algorithm="spectral")
    affinity = spectral_affinity(cdist(mov, mov), cdist(ref, ref), spectral.spectral_sigma)
    np.testing.assert_array_equal(
        spectral.match(mg, rg), spectral_assignment(affinity, len(mov), len(ref))
    )
    # reversed order: node i pairs with node 29 - i
    pairs = spectral.match(mg, rg)
    assert len(pairs) >= 28 and all(i + j == 29 for i, j in pairs)


def test_the_medoid_is_the_node_closest_to_all_the_others():
    from biahub.core.nodes.solvers import medoid

    points = np.array([[0.0], [1.0], [2.0], [10.0]])
    assert medoid(cdist(points, points)) == 1  # 1 + 1 + 9 beats 1 + 2 + 10 and 2 + 1 + 8
