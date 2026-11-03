"""tracksdata: link fixed labels with an IoU/distance graph and an ILP.

Also turns a solved tracksdata graph into tracked labels and a tracks table, for this
method and for HOCT and Trackastra, which build tracksdata graphs too.
"""

import os

import numpy as np
import pandas as pd

from biahub.settings import TracksdataConfig
from biahub.tracking.postprocess import _merge_single_children


def _num_threads() -> int:
    """CPUs for the ILP solvers: the SLURM allocation, else this process's CPUs."""
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):  # not on macOS or Windows
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _solve_tracksdata(labels: np.ndarray, config: TracksdataConfig, pixel_size: float):
    """Build the tracksdata graph of (T, Y, X) ``labels`` and solve its ILP.

    Returns the solved graph (nodes and edges carry the ``solution`` attribute).
    """
    import polars as pl
    import tracksdata as td

    _, Y, X = labels.shape
    graph = td.graph.InMemoryGraph()
    td.nodes.RegionPropsNodes().add_nodes(graph, labels=labels)
    margin = config.border_margin_um / pixel_size
    appearance = config.appearance_weight
    if margin > 0:
        nodes = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "y", "x"])
        y, x = nodes["y"].to_numpy(), nodes["x"].to_numpy()
        is_border = ((y < margin) | (y > Y - margin) | (x < margin) | (x > X - margin)).astype(
            float
        )
        graph.add_node_attr_key("is_border", pl.Float64, default_value=0.0)
        graph.update_node_attrs(
            attrs={"is_border": is_border.tolist()},
            node_ids=nodes[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
        )
        border = td.NodeAttr("is_border")
        appearance = appearance * (1 - border) + config.border_appearance_weight * border
    td.edges.DistanceEdges(
        distance_threshold=config.distance_threshold_um / pixel_size,
        n_neighbors=config.n_neighbors,
    ).add_edges(graph)
    if graph.num_edges() == 0:
        return _select_every_detection(graph)
    td.edges.IoUEdgeAttr(output_key="iou").add_edge_attrs(graph)
    edge_weight = (
        -td.EdgeAttr("iou")
        * (td.EdgeAttr("distance") * (pixel_size / config.distance_norm_um)).exp()
    )
    td.solvers.ILPSolver(
        edge_weight=edge_weight,
        node_weight=config.node_weight,
        appearance_weight=appearance,
        disappearance_weight=config.disappearance_weight,
        division_weight=config.division_weight,
        num_threads=_num_threads(),
        return_solution=False,
    ).solve(graph)
    return graph


def run_tracksdata(
    labels: np.ndarray, config: TracksdataConfig, pixel_size: float
) -> tuple[np.ndarray, pd.DataFrame]:
    """Link fixed (T, Y, X) instance labels with tracksdata's ILP."""
    if not labels.any():
        return _empty_result(labels.shape)
    graph = _solve_tracksdata(labels, config, pixel_size)
    return _solution_to_tracks(graph, labels.shape)


# --- solved graph -> tracked labels and tracks table (shared with HOCT, Trackastra) ---

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
