"""HOCT: link fixed labels with a learned edge classifier and an ILP."""

import click
import numpy as np
import pandas as pd

from biahub.settings import HoctConfig
from biahub.tracking.graph import _empty_result, _select_every_detection, _solution_to_tracks
from biahub.tracking.postprocess import _upsample_result
from biahub.tracking.preprocess import _downsample


def _import_hoct():
    try:
        import hoct
        import hoct.features
    except ImportError as e:
        raise ImportError(
            "tracker: hoct needs HOCT, which biahub[track] installs on Python < 3.14 only "
            "(its gurobipy pin has no Python 3.14 wheels)."
        ) from e
    return hoct


def hoct_solver_config(config: HoctConfig):
    """HOCT's ILPSolverConfig: ``config.solver`` overrides on HOCT's tracking defaults.

    ILPSolverConfig needs every field when built directly, and refuses unknown keys.
    """
    _import_hoct()
    from hoct.tracking import ILPSolverConfig

    return ILPSolverConfig.model_validate(
        {**ILPSolverConfig.default().model_dump(), **config.solver}
    )


def warm_hoct_weights(model: str) -> None:
    """Download HOCT weights into the shared cache once (from --init, on the head node)."""
    hoct = _import_hoct()
    hoct.load_model(model, device="cpu")
    click.echo(f"HOCT weights ready: {model}")


def _select_sparse_links(graph):
    """Select every detection and its candidate links, one incoming link per detection.

    For graphs where HOCT could score no inference window (each holds at most one
    candidate link, so each link is the only option where it is): every node is kept,
    and each node keeps one incoming edge (the first), so no detection gets two parents.
    """
    import polars as pl
    import tracksdata as td

    keys = td.DEFAULT_ATTR_KEYS
    graph.add_node_attr_key(keys.SOLUTION, pl.Boolean, default_value=True)
    graph.add_edge_attr_key(keys.SOLUTION, pl.Boolean, default_value=False)
    edges = graph.edge_attrs(attr_keys=[keys.EDGE_ID, keys.EDGE_SOURCE, keys.EDGE_TARGET])
    keep = edges.unique(subset=[keys.EDGE_TARGET], keep="first")[keys.EDGE_ID].to_list()
    if keep:
        graph.update_edge_attrs(attrs={keys.SOLUTION: True}, edge_ids=keep)
    return graph


def run_hoct(
    labels: np.ndarray, images: np.ndarray, config: HoctConfig, pixel_size: float
) -> tuple[np.ndarray, pd.DataFrame]:
    """Link fixed (T, Y, X) instance labels with HOCT's edge model and ILP.

    With ``config.downsample`` > 1 the labels and image are shrunk first (HOCT's models
    were trained on small nuclei) and the result is mapped back to full resolution.
    """
    factor = config.downsample
    if factor == 1:
        return _run_hoct(labels, images, config, pixel_size)
    labels_small, images_small = _downsample(labels, images, factor)
    tracked, tracks = _run_hoct(labels_small, images_small, config, pixel_size * factor)
    return _upsample_result(tracked, tracks, labels_small, labels, factor)


def _run_hoct(
    labels: np.ndarray, images: np.ndarray, config: HoctConfig, pixel_size: float
) -> tuple[np.ndarray, pd.DataFrame]:
    """Link fixed (T, Y, X) instance labels with HOCT's edge model and ILP.

    ``images`` is the (T, Y, X) image the labels were segmented from. HOCT runs in pixel
    space, as its models were trained, so distances are converted to pixels.
    """
    if not labels.any():
        return _empty_result(labels.shape)

    import torch

    hoct = _import_hoct()

    # Build HOCT's candidate graph first: inference fails on a graph without edges.
    graph = hoct.features.create_graph(
        labels,
        images=images,
        distance_threshold=config.distance_threshold_um / pixel_size,
        n_neighbors=config.n_neighbors,
        delta_t=config.max_delta_t,
    )
    # HOCT 0.3.0's inference skips batches with at most one edge and then fails on an
    # empty prediction list, so sparse graphs are resolved here: no edge leaves each
    # detection on its own, a single edge is the only possible link.
    if graph.num_edges() <= 1:
        selected = _select_every_detection(graph, keep_edges=graph.num_edges() == 1)
        return _solution_to_tracks(selected, labels.shape)

    solver_config = hoct_solver_config(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = hoct.load_model(config.model, device=device)
    try:
        solution = hoct.predict(
            model, graph=graph, solver_config=solver_config, return_solution=True
        )
    except ValueError as e:
        # Every inference window held at most one candidate link, so HOCT scored none
        # (0.3.0 then fails on an empty prediction list); each link is the only option.
        if "non-empty list of Tensors" not in str(e):
            raise
        click.echo(
            "HOCT scored no inference window (each holds at most one candidate link); "
            "keeping every detection and its candidate links."
        )
        return _solution_to_tracks(_select_sparse_links(graph), labels.shape)
    return _solution_to_tracks(solution, labels.shape)
