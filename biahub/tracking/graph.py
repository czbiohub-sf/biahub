"""Turn a tracker's solution graph into tracked labels and a tracks table."""

import numpy as np
import pandas as pd

from biahub.tracking.postprocess import _merge_single_children

TRACK_COLUMNS = ["track_id", "t", "y", "x", "parent_track_id"]


def _empty_result(shape: tuple[int, ...]) -> tuple[np.ndarray, pd.DataFrame]:
    """Zero labels and a tracks table with no rows, for movies with nothing to track."""
    tracks = pd.DataFrame(
        {
            "track_id": pd.Series(dtype="int64"),
            "t": pd.Series(dtype="int64"),
            "y": pd.Series(dtype="float64"),
            "x": pd.Series(dtype="float64"),
            "parent_track_id": pd.Series(dtype="int64"),
        }
    )
    return np.zeros(shape, dtype=np.int32), tracks


def _select_every_detection(graph, keep_edges: bool = False):
    """Mark every node selected: each detection becomes its own track.

    Used when a candidate graph has no edges (one frame, or objects farther apart than
    the distance threshold), which the ILP solvers refuse. ``keep_edges`` also selects
    the edges, for a graph whose single edge is the only possible link.
    """
    import polars as pl
    import tracksdata as td

    solution = td.DEFAULT_ATTR_KEYS.SOLUTION
    graph.add_node_attr_key(solution, pl.Boolean, default_value=True)
    graph.add_edge_attr_key(solution, pl.Boolean, default_value=keep_edges)
    return graph


def _isolated_tracks(labels: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Each detection of (T, Y, X) ``labels`` as its own track, without linking."""
    import tracksdata as td

    graph = td.graph.InMemoryGraph()
    td.nodes.RegionPropsNodes().add_nodes(graph, labels=labels)
    return _solution_to_tracks(_select_every_detection(graph), labels.shape)


def _solution_to_tracks(graph, shape: tuple[int, ...]) -> tuple[np.ndarray, pd.DataFrame]:
    """Convert a solved tracksdata graph to labels (value = track_id) and ultrack-like tracks.

    The tracks have ``track_id, t, y, x, parent_track_id`` (-1 for roots), the columns
    the sibling filter and the tracks CSV use. HOCT promotes 2D movies to (T, 1, Y, X)
    and records that shape in the graph metadata, so labels are rendered in the graph's
    own shape and reshaped to ``shape``.
    """
    import tracksdata as td

    solution_key = td.DEFAULT_ATTR_KEYS.SOLUTION
    if graph.num_nodes() == 0 or (
        solution_key in graph.node_attr_keys()
        and not graph.node_attrs(attr_keys=[solution_key])[solution_key].any()
    ):
        # The solver selected nothing (an empty solution can be the cheapest one).
        return _empty_result(shape)

    metadata = getattr(graph, "metadata", None) or {}
    render_shape = tuple(metadata.get("shape", shape))
    tracks, parents, labels_view = td.functional.to_napari_format(
        graph, shape=render_shape, mask_key=td.DEFAULT_ATTR_KEYS.MASK
    )
    tracks = tracks.to_pandas().rename(columns={"tracklet_id": "track_id"})
    tracks = tracks[["track_id", "t", "y", "x"]].copy()

    def parent_of(track_id):
        parent = parents.get(int(track_id), -1)
        return int(parent[0]) if isinstance(parent, (list, tuple)) else int(parent)

    tracks["parent_track_id"] = tracks.track_id.map(parent_of)
    tracks = tracks.sort_values(["track_id", "t"]).reset_index(drop=True)
    labels = np.stack([np.asarray(labels_view[t]) for t in range(shape[0])])
    tracks, labels = _merge_single_children(tracks, labels.reshape(shape).astype(np.int32))
    return labels, tracks
