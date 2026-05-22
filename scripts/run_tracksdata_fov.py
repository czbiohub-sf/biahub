#!/usr/bin/env python3
"""Run tracksdata + cellpose tracking for a single FOV and save CSV output."""

from __future__ import annotations

import argparse
import os
import site
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
from cellpose import models as cp_models
from iohub import open_ome_zarr
from rich import print
from tqdm import tqdm
from waveorder.focus import focus_from_transverse_band

from biahub.track import fill_empty_frames


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_channel_index(ds: object, channel_name: str | None) -> int:
    if not channel_name:
        return 0
    channel_names = getattr(ds, "channel_names", None) or []
    if channel_name in channel_names:
        return channel_names.index(channel_name)
    raise ValueError(f"Channel {channel_name!r} not found in {channel_names!r}")


def resolve_image_node(image_root: object, fov_key: str) -> object:
    if hasattr(image_root, "data"):
        return image_root
    if hasattr(image_root, "positions"):
        for key, position in image_root.positions():
            if key == fov_key:
                return position
        available = [key for key, _ in image_root.positions()]
        raise KeyError(f"FOV {fov_key!r} not found in image root; available keys: {available!r}")
    raise TypeError(f"Unsupported image root type: {type(image_root)!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fov-key", required=True, help="FOV key like A/3/000002")
    return parser


def load_tracksdata() -> object:
    extra_site = os.environ.get("TRACKSDATA_SITEPACKAGES")
    if extra_site:
        site.addsitedir(extra_site)
    import tracksdata as td  # noqa: PLC0415

    return td


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    dataset = cfg["dataset"]
    root = Path(cfg["root"])
    fov_key = args.fov_key
    fov_path = fov_key
    dataset_root = root / dataset
    image_cfg = cfg.get("image", {})
    image_path = Path(
        image_cfg.get(
            "path",
            dataset_root / "1-preprocess/2-reconstruct" / f"{dataset}.zarr" / fov_path,
        )
    )
    channel_name = image_cfg.get("channel_name")

    opt = cfg["optical"]
    cp = cfg["cellpose"]
    td_cfg = cfg["tracksdata"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{fov_key.replace('/', '_')}.csv"

    print(f"[bold]Dataset[/bold] {dataset}")
    print(f"[bold]FOV[/bold]     {fov_key}")
    print(f"[bold]Output[/bold]   {csv_path}")

    with open_ome_zarr(image_path, mode="r") as image_root:
        image_ds = resolve_image_node(image_root, fov_key)
        T, C, Z, Y, X = image_ds.data.shape
        scale = tuple(float(s) for s in image_ds.scale[-2:])
        channel_idx = resolve_channel_index(image_ds, channel_name)

        z_focus = np.zeros(T, dtype=int)
        for t in tqdm(range(T), desc="Focus"):
            zyx = np.asarray(image_ds.data[t, channel_idx, :, :, :])
            z_f = focus_from_transverse_band(
                zyx,
                NA_det=opt["NA_det"],
                lambda_ill=opt["lambda_ill"],
                pixel_size=opt["pixel_size"],
            )
            z_focus[t] = int(np.clip(z_f if z_f is not None else Z // 2, 0, Z - 1))

        z_median = int(np.median(z_focus))
        z_window = opt["z_window"]
        z_below = z_window // 3
        z_above = z_window - z_below - 1
        z_slicing = slice(max(0, z_median - z_below), min(Z, z_median + z_above + 1))

        image_dask = image_ds.data.dask_array()
        nuc_arr = image_dask[:, channel_idx, z_slicing, :, :].mean(axis=1).compute()
        im_arr = np.stack(
            [np.asarray(image_ds.data[t, channel_idx, z_focus[t], :, :]) for t in range(T)]
        )
        empty = [f for f in range(T) if np.sum(im_arr[f]) == 0.0]
        if empty:
            nuc_arr = fill_empty_frames(nuc_arr, empty)

    print("[bold]Running cellpose[/bold]")
    model = cp_models.CellposeModel(model_type=cp["model_type"], gpu=cp["gpu"])
    cellpose_labels = np.zeros((T, Y, X), dtype=np.int32)
    for t in tqdm(range(T), desc="Cellpose"):
        mask, _, _ = model.eval(
            nuc_arr[t],
            diameter=cp["diameter"],
            channels=[0, 0],
            cellprob_threshold=cp["cellprob_threshold"],
            flow_threshold=cp["flow_threshold"],
            min_size=cp["min_size"],
        )
        cellpose_labels[t] = mask

    print("[bold]Running tracksdata ILP[/bold]")
    td = load_tracksdata()
    td.options.set_options(show_progress=True)
    graph = td.graph.InMemoryGraph()
    nodes_op = td.nodes.RegionPropsNodes()
    nodes_op.add_nodes(graph, labels=cellpose_labels)

    border_margin = td_cfg.get("border_margin", 0)
    if border_margin > 0:
        node_df = graph.node_attrs(attr_keys=["y", "x"])
        node_ids = graph.node_attrs(attr_keys=["node_id"])["node_id"].to_list()
        y_vals = node_df["y"].to_numpy()
        x_vals = node_df["x"].to_numpy()
        is_border = (
            (y_vals < border_margin)
            | (y_vals > Y - border_margin)
            | (x_vals < border_margin)
            | (x_vals > X - border_margin)
        ).astype(float)
        graph.add_node_attr_key("is_border", pl.Float64, default_value=0.0)
        graph.update_node_attrs(attrs={"is_border": is_border.tolist()}, node_ids=node_ids)

    dist_thresh = td_cfg["distance_threshold"]
    td.edges.DistanceEdges(
        distance_threshold=dist_thresh,
        n_neighbors=td_cfg["n_neighbors"],
    ).add_edges(graph)

    if td_cfg.get("use_iou", True):
        td.edges.IoUEdgeAttr(output_key="iou").add_edge_attrs(graph)

    dist_norm = td_cfg.get("distance_norm", dist_thresh)
    mode = td_cfg.get("edge_weight_mode", "iou_distance")
    if mode == "iou_distance" and td_cfg.get("use_iou", True):
        edge_weight = -td.EdgeAttr("iou") * (td.EdgeAttr("distance") * (1.0 / dist_norm)).exp()
    elif mode == "iou":
        edge_weight = -td.EdgeAttr("iou")
    else:
        edge_weight = "distance"

    app_weight = td_cfg["appearance_weight"]
    if border_margin > 0:
        border_app_weight = td_cfg.get("border_appearance_weight", 1.0)
        appearance_weight = (
            app_weight * (1 - td.NodeAttr("is_border"))
            + border_app_weight * td.NodeAttr("is_border")
        )
    else:
        appearance_weight = app_weight

    solver = td.solvers.ILPSolver(
        edge_weight=edge_weight,
        node_weight=td_cfg.get("node_weight", -10.0),
        appearance_weight=appearance_weight,
        disappearance_weight=td_cfg["disappearance_weight"],
        division_weight=td_cfg["division_weight"],
        num_threads=td_cfg.get("num_threads", 4),
    )
    solver.solve(graph)

    tracks_df, track_graph, _ = td.functional.to_napari_format(
        graph, cellpose_labels.shape, mask_key="mask"
    )
    if not isinstance(tracks_df, pd.DataFrame):
        tracks_df = tracks_df.to_pandas()
    if "tracklet_id" in tracks_df.columns:
        tracks_df = tracks_df.rename(columns={"tracklet_id": "track_id"})

    tracks_df["parent_track_id"] = tracks_df["track_id"].map(track_graph).fillna(-1).astype(int)
    tracks_df["fov_name"] = fov_key
    tracks_df.to_csv(csv_path, index=False)

    print(f"[green]Saved[/green] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
