import numpy as np

from numpy.typing import NDArray


def connectivity(points: NDArray) -> dict:
    """Build the connectivity graph between 4-connected (right / down) neighbors.

    Each input point is paired with its right (``+x``) and down (``+y``)
    neighbor when present. The traversal preserves input order, so when
    fed Hilbert-curve order the returned edges come out in Hilbert order.
    Diagonal neighbors are not considered.

    Parameters
    ----------
    points : numpy.ndarray
        Integer array of shape ``(N, 2)`` of ``(x, y)`` grid coordinates.

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        Mapping from edge index (string-typed) to a 2-element list
        ``[(x, y), (x', y')]`` representing the edge endpoints.
    """
    point_set = set(map(tuple, points))
    edges = dict()
    edge_inx = 0
    directions = np.array([(0, 1), (1, 0)])
    for x, y in points:
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if neighbor in point_set:
                edges[f"{edge_inx}"] = [(x, y), neighbor]
                edge_inx += 1
    return edges


def hilbert_index_to_xy(n, d):
    """Convert a 1D Hilbert index to 2D ``(x, y)`` coordinates on an n x n grid.

    Parameters
    ----------
    n : int
        Side length of the (square) grid. Must be a power of 2 for the
        Hilbert curve to fill the entire grid.
    d : int
        Hilbert-curve index, ``0 <= d < n * n``.

    Returns
    -------
    numpy.ndarray
        Two-element integer array ``[x, y]``.
    """
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - y, s - 1 - x
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return np.asarray([x, y])


def generate_hilbert_curve(n):
    """Generate the Hilbert-curve traversal of an ``n x n`` grid.

    Only fills the entire grid when ``n`` is a power of 2; otherwise the
    output is the prefix of the next-power-of-2 curve that fits.

    Parameters
    ----------
    n : int
        Side length of the grid.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n * n, 2)``: one ``(x, y)`` row per
        Hilbert-index in increasing order.
    """
    order = []
    for i in range(n * n):
        order.append(hilbert_index_to_xy(n, i))
    return np.asarray(order)


def hilbert_over_points(points: NDArray) -> NDArray:
    """Return the Hilbert-curve order of the given set of grid points.

    Generates the Hilbert curve over the smallest power-of-2 grid that
    contains the input bounding box, then keeps only the points present
    in ``points`` (preserving Hilbert order).

    Parameters
    ----------
    points : numpy.ndarray
        Integer array of shape ``(N, 2)`` of ``(x, y)`` grid coordinates.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(N, 2)`` with the same set of points
        reordered along the Hilbert curve.
    """
    n = int(np.max(points) + 1)
    n_full_curve = 1 if n == 0 else 1 << (n - 1).bit_length()
    hilbert_curve = generate_hilbert_curve(n_full_curve)

    order = []
    for p in hilbert_curve:
        if np.any(np.all(points == p, axis=1)):
            order.append(p)
    return np.asarray(order)
