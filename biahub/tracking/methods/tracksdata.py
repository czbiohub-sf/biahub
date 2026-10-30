"""tracksdata: link fixed labels with an IoU/distance graph and an ILP."""

import os

import numpy as np
import pandas as pd

from biahub.settings import TracksdataConfig
from biahub.tracking.graph import _empty_result, _select_every_detection, _solution_to_tracks


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
