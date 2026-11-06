"""Track one FOV with a `biahub track` config and show the result in napari.

For tuning tracking parameters before running `biahub track` on a whole plate.
It reads the same YAML config as `biahub track` (see
settings/example_track_settings.yml), tracks a single position in memory and
opens napari with the inputs, the tracked labels and the tracks. Nothing is
written into the plate.

Track does not project or select Z-planes: for 2D tracking, give a position of
the projected (Z=1) plate (`biahub project`), as `biahub track` expects.

Usage:
    python scripts/debug_track.py -c track.yml -i <plate>.zarr/B/1/000000
    python scripts/debug_track.py -c track.yml -i <plate>.zarr/B/1/000000 \
        --input-images-path <plate>.zarr   # first null input_images path
"""

import argparse
import tempfile

from pathlib import Path

import napari
import numpy as np

from iohub import open_ome_zarr
from ultrack import MainConfig

from biahub.settings import TrackingSettings
from biahub.track import (
    _load_and_preprocess,
    check_input_dimensionality,
    detect_foreground_segmentation,
    link_fixed_labels,
    load_labels,
    run_ultrack,
)
from biahub.utils.config import update_model, yaml_to_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", type=Path, required=True, help="track config YAML")
    parser.add_argument(
        "-i", "--position", type=Path, required=True, help="one position: <plate>.zarr/R/C/F"
    )
    parser.add_argument(
        "--input-images-path",
        type=Path,
        default=None,
        help="fills the first null input_images path (default: the -i plate), as in biahub track",
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=None,
        help="keep the ultrack database here (default: a temporary directory, deleted)",
    )
    parser.add_argument("--n-workers", type=int, default=8, help="ultrack workers")
    return parser.parse_args()


def main():
    args = parse_args()
    position_dirpath = args.position.resolve()
    position_key = position_dirpath.parts[-3:]
    fov = "/".join(position_key)

    settings = yaml_to_model(args.config, TrackingSettings)

    # Same resolution as `biahub track`: the first null path is the primary source.
    primary_path = args.input_images_path or position_dirpath.parents[2]
    for image in settings.input_images:
        if image.path is None:
            image.path = primary_path
            break

    with open_ome_zarr(position_dirpath, mode="r") as dataset:
        T, C, Z, Y, X = dataset.data.shape
        scale = tuple(float(s) for s in dataset.scale)
    check_input_dimensionality(settings.output_mode, Z)
    track_scale = scale[-2:] if settings.output_mode == "2D" else scale[-3:]

    print(f"FOV {fov}: T={T}, C={C}, Z={Z}, Y={Y}, X={X}, scale={track_scale}")
    print(f"tracker={settings.tracker}, segmentation_method={settings.segmentation_method}")

    inputs = {}  # name -> (array, kind) shown in napari
    if settings.tracker != "ultrack":
        # tracksdata / hoct / trackastra link the segment labels as they are (2D only).
        tracking_labels, tracks_df = link_fixed_labels(
            settings.tracker,
            position_dirpath,
            settings.labels,
            settings.blank_frames_path,
            pixel_size=track_scale[-1],
            tracksdata_config=settings.tracksdata_config,
            hoct_config=settings.hoct_config,
            trackastra_config=settings.trackastra_config,
        )
        graph = None
        inputs[f"labels/{settings.labels}"] = (
            load_labels(position_dirpath, settings.labels, "2D", settings.blank_frames_path),
            "labels",
        )
    else:
        tracking_cfg = dict(settings.tracking_config)
        tracking_cfg.setdefault("segmentation_config", {})["n_workers"] = args.n_workers
        tracking_cfg.setdefault("linking_config", {})["n_workers"] = args.n_workers
        tracking_cfg = update_model(MainConfig(), tracking_cfg)

        if settings.segmentation_method == "labels":
            labels = load_labels(
                position_dirpath,
                settings.labels,
                settings.output_mode,
                settings.blank_frames_path,
            )
            detection_kwargs = {"labels": labels, "sigma": settings.labels_sigma}
            inputs[f"labels/{settings.labels}"] = (labels, "labels")
        else:
            data_dict = _load_and_preprocess(
                position_key, settings.input_images, settings.blank_frames_path
            )
            foreground, contour = detect_foreground_segmentation(data_dict)
            detection_kwargs = {"detection": foreground, "edges": contour}
            for name, array in data_dict.items():
                inputs[name] = (array, "image")
            inputs["foreground"] = (foreground, "labels")

        with tempfile.TemporaryDirectory() as tmp:
            database_path = args.database_dir or Path(tmp) / "_".join(position_key)
            database_path.mkdir(parents=True, exist_ok=True)
            tracking_labels, tracks_df, graph = run_ultrack(
                tracking_config=tracking_cfg,
                database_path=database_path,
                min_sibling_length=settings.min_sibling_length,
                scale=track_scale,
                overwrite=True,
                **detection_kwargs,
            )
            # Read the labels before the temporary database directory goes away.
            tracking_labels = np.asarray(tracking_labels)

    print(f"tracks: {tracks_df['track_id'].nunique()}, timepoints: {tracks_df['t'].nunique()}")

    viewer = napari.Viewer(title=f"debug_track {fov}")
    for name, (array, kind) in inputs.items():
        if kind == "labels":
            viewer.add_labels(np.asarray(array), name=name, scale=track_scale, visible=False)
        else:
            viewer.add_image(array, name=name, scale=track_scale, blending="additive")
    viewer.add_labels(tracking_labels, name=settings.tracked_label, scale=track_scale)
    spatial = ["y", "x"] if settings.output_mode == "2D" else ["z", "y", "x"]
    viewer.add_tracks(
        tracks_df[["track_id", "t", *spatial]],
        graph=graph,
        name="tracks",
        scale=(1, *track_scale),
        colormap="hsv",
        blending="opaque",
    )
    napari.run()


if __name__ == "__main__":
    main()
