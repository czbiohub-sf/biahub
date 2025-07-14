import os
import shutil
from pathlib import Path
from biahub.settings import TrackingSettings
import napari
from iohub import open_ome_zarr
from rich import print
from biahub.track import run_ultrack, resolve_z_slice, load_data, run_preprocessing_pipeline, fill_empty_frames_from_csv
from ultrack import MainConfig
from biahub.cli.utils import update_model


def main():

    config = {
        "z_range": [-1, -1],
        "target_channel": "nuclei_prediction",
        "mode": "2D",
        "fov": "*/*/*",
        "blank_frames_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_06_26_A549_G3BP1_ZIKV/1-preprocess/label-free/3-track/blank_frames.csv",
        "input_images": [
            {
                "path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_06_26_A549_G3BP1_ZIKV/1-preprocess/label-free/2-stabilize/virtual-stain/2025_06_26_A549_G3BP1_ZIKV.zarr",
                "channels": {
                    "nuclei_prediction": [
                        {
                            "function": "np.mean",
                            "kwargs": {"axis": 1},
                            "per_timepoint": False
                        },
                        {
                            "function": "ultrack.imgproc.normalize",
                            "kwargs": {"gamma": 0.7},
                            "per_timepoint": False
                        }
                    ],
                    "membrane_prediction": [
                        {
                            "function": "np.mean",
                            "kwargs": {"axis": 1},
                            "per_timepoint": False
                        },
                        {
                            "function": "ultrack.imgproc.normalize",
                            "kwargs": {"gamma": 0.7},
                            "per_timepoint": False
                        }
                    ]
                }
            },
            {
                "path": None,
                "channels": {
                    "foreground": [
                        {
                            "function": "ultrack.imgproc.detect_foreground",
                            "input_channels": ["nuclei_prediction"],
                            "kwargs": {"sigma": 80},
                            "per_timepoint": False
                        }
                    ],
                    "contour": [
                        {
                            "function": "biahub.cli.track.mem_nuc_contour",
                            "input_channels": ["nuclei_prediction", "membrane_prediction"],
                            "kwargs": {},
                            "per_timepoint": False
                        }
                    ]
                }
            },
        ],
        "tracking_config": {
            "segmentation_config": {
                "min_area": 2100,
                "max_area": 80000,
                "n_workers": 10,
                "min_frontier": 0.4,
                "max_noise": 0.05,
            },
            "linking_config": {
                "n_workers": 10,
                "max_distance": 15,
                "distance_weight": -0.0001,
                "max_neighbors": 3,
            },
            "tracking_config": {
                "n_threads": 14,
                "disappear_weight": -0.0001,
                "appear_weight": -0.001,
                "division_weight": -0.0001,
            },
        },
    }


    config = TrackingSettings(**config)    

    ### CHANGE THIS VALUE
    dataset = "2025_06_26_A549_G3BP1_ZIKV"
    vs_path = f"{dataset}.zarr/B/1/000000"
    root = Path(f"/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/{dataset}")
    output_dirpath = root / "1-preprocess/label-free/3-track/test_tracking"
    os.makedirs(output_dirpath, exist_ok=True)


    ### CHANGE THIS VALUE
    position_key = ("B", "1", "000000")

    blank_frames_path = config.blank_frames_path
    input_images = config.input_images

    with open_ome_zarr(input_images[0].path / Path(*position_key)) as vs_ds:
        T, C, Z, Y, X = vs_ds.data.shape
        scale = tuple(float(s) for s in vs_ds.scale[-2:])
    z_slices, Z = resolve_z_slice(config.z_range, Z)
    fov = "_".join(position_key)

    data_dict = load_data(position_key, input_images, z_slices)
    data_dict = run_preprocessing_pipeline(data_dict, input_images)
    data_dict = fill_empty_frames_from_csv(fov, data_dict, blank_frames_path)
    
    channel_names = data_dict.keys()
    print(channel_names)
    if "foreground" in channel_names and "contour" in channel_names:
        foreground_mask, contour_gradient_map = data_dict["foreground"], data_dict["contour"]
    elif "foreground_contour" in channel_names:
        foreground_mask, contour_gradient_map = data_dict["foreground_contour"]
    else:
        raise ValueError("Foreground and contour channels are required for tracking.")
    
    default_config = MainConfig()
    tracking_cfg = update_model(default_config, config.tracking_config)
    databaset_path = output_dirpath / f"{dataset}_config_tracking/{fov}"
    os.makedirs(databaset_path, exist_ok=True)

    print("Tracking...")
    tracking_labels, tracks_df, graph = run_ultrack(tracking_cfg, foreground_mask, contour_gradient_map, scale, databaset_path)
    shutil.rmtree(databaset_path)
 
    kwargs = dict(scale=scale, blending="additive")
    viewer = napari.Viewer()
    viewer.add_image(data_dict["nuclei_prediction"], name="VS nuclei Preprocessed", **kwargs, colormap="green")
    viewer.add_image(data_dict["membrane_prediction"], name="VS membrane Preprocessed", colormap="magma", **kwargs)
    viewer.add_labels(foreground_mask, name="Foreground ", **kwargs)
    viewer.add_image(contour_gradient_map, name="Contourn ", colormap="magma", **kwargs)
    viewer.add_labels(tracking_labels, name="Tracking Segments", scale=scale)
    viewer.add_tracks(
        tracks_df[["track_id", "t", "y", "x"]],
        graph=graph,
        name="Tracks",
        scale=scale,
        colormap="hsv",
        blending="opaque",
    )
    napari.run()


if __name__ == "__main__":
    main()
