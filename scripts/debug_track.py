import os
import shutil
from pathlib import Path
from biahub.settings import ProjectionSettings, TrackingSettings
import numpy as np
from iohub import open_ome_zarr
from rich import print
from biahub.project import _FLOAT_METHODS, _find_focus_per_t, project_zyx, resolve_z_indices
from biahub.track import check_input_dimensionality, load_data
from biahub.tracking.methods.ultrack import detect_foreground_segmentation, run_ultrack
from biahub.tracking.preprocess import fill_empty_frames_from_csv, run_preprocessing_pipeline
from ultrack import MainConfig
from biahub.utils.config import update_model


def find_focus(position_path, projection, pixel_size, num_workers=8):
    """Return the in-focus plane of every timepoint, as `biahub project` finds it.

    None when no channel projects around the focus. With ``per: fov`` every
    timepoint takes the median plane over T.
    """
    if not projection.uses_focus:
        return None
    focus_channel = projection.focus.channel
    with open_ome_zarr(position_path) as dataset:
        store_channels = dataset.channel_names
        T = dataset.data.shape[0]
    if focus_channel not in store_channels:
        raise ValueError(
            f"Focus channel {focus_channel!r} is not in {position_path} "
            f"({store_channels}); set focus_store_path to a store that has it."
        )
    focus_per_t = _find_focus_per_t(
        position_path,
        store_channels.index(focus_channel),
        projection.focus,
        pixel_size,
        num_workers=num_workers,
    )
    if projection.focus.per == "fov":
        focus_per_t = np.full(T, int(round(float(np.median(focus_per_t)))))
    print(f"Focus on {focus_channel}: {focus_per_t.tolist()}")
    return focus_per_t


def project_fov(position_path, channel_names, projection, focus_per_t=None):
    """Project one FOV's channels to (T, Y, X) in memory with the `biahub project` API.

    The same per-channel projection `biahub project` writes to a 2D plate, without
    writing a store: reduce every timepoint's Z-stack around ``focus_per_t`` (from
    :func:`find_focus`) where a channel projects around the focus.
    """
    missing = [name for name in channel_names if name not in projection.channels]
    if missing:
        raise ValueError(f"No projection set for channels {missing}.")
    # Like `biahub project`, every channel is float32 when any projection is mean/sum.
    methods = {plan.method for plan in projection.channels.values()}
    as_float = bool(methods & set(_FLOAT_METHODS))

    with open_ome_zarr(position_path) as dataset:
        store_channels = dataset.channel_names
        T, _, Z, _, _ = dataset.data.shape
        data_dict = {}
        for name in channel_names:
            plan = projection.channels[name]
            c = store_channels.index(name)
            frames = []
            for t in range(T):
                focus_z = None if focus_per_t is None else int(focus_per_t[t])
                z_indices = resolve_z_indices(plan, Z, focus_z)
                frames.append(project_zyx(np.asarray(dataset.data[t, c]), plan.method, z_indices))
            data = np.stack(frames)
            data_dict[name] = data.astype(np.float32, copy=False) if as_float else data
    return data_dict


def main():
    """
    Debug script for testing tracking configuration parameters.
    
    This script loads a single field of view (FOV), applies preprocessing,
    runs tracking with specified parameters, and visualizes results in Napari.
    Useful for tuning tracking parameters before running on full datasets.

    It uses the foreground + contour path of `biahub track`
    (segmentation_method: foreground_contour). Track does not select or project
    Z-planes, so for 2D tracking of a 3D store this script projects the FOV in
    memory with the `biahub project` API (see `projection` below), the way the
    mantis-v2 pipeline projects the plate before tracking. A store that is
    already projected (Z=1) is tracked as is.
    """

    # ============================================================================
    # CONFIGURATION SETUP
    # ============================================================================
    
    # Define the tracking configuration with all parameters
    config = {
        # Track a foreground mask and contour map computed from input_images
        "segmentation_method": "foreground_contour",

        # Name of the label image the tracks are written to
        "target_channel": "nuclei_prediction",

        # Tracking dimensionality: "2D" needs a Z=1 input, "3D" a Z>1 input
        "output_mode": "2D",
        
        # Field of view pattern to match in the dataset
        "fov": "*/*/*",
        
        # Path to CSV file containing blank frame information
        "blank_frames_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_06_26_A549_G3BP1_ZIKV/1-preprocess/label-free/3-track/blank_frames.csv",
        
        # Input image configurations - defines data sources and preprocessing pipelines
        "input_images": [
            {
                # First input: Load data from Zarr store.
                ### CHANGE THIS VALUE - a projected (Z=1) plate, or a 3D store that
                # this script projects with `projection` below (2D tracking).
                "path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_06_26_A549_G3BP1_ZIKV/1-preprocess/label-free/2-stabilize/virtual-stain/2025_06_26_A549_G3BP1_ZIKV.zarr",
                "channels": {
                    # Nuclei channel preprocessing pipeline
                    "nuclei_prediction": [
                        {
                            "function": "ultrack.imgproc.normalize",  # Normalize intensity
                            "kwargs": {"gamma": 0.7},
                            "per_timepoint": False
                        }
                    ],
                    # Membrane channel preprocessing pipeline
                    "membrane_prediction": [
                        {
                            "function": "ultrack.imgproc.normalize",  # Normalize intensity
                            "kwargs": {"gamma": 0.7},
                            "per_timepoint": False
                        }
                    ]
                }
            },
            {
                # Second input: Generate derived channels (no path = derived from first input)
                "path": None,
                "channels": {
                    # Foreground detection for tracking
                    "foreground": [
                        {
                            "function": "ultrack.imgproc.detect_foreground",
                            "input_channels": ["nuclei_prediction"],
                            "kwargs": {"sigma": 80},  # Gaussian blur sigma for foreground detection
                            "per_timepoint": False
                        }
                    ],
                    # Contour map for boundary refinement
                    "contour": [
                        {
                            "function": "biahub.track.mem_nuc_contour",
                            "input_channels": ["nuclei_prediction", "membrane_prediction"],
                            "kwargs": {},
                            "per_timepoint": False
                        }
                    ]
                }
            },
        ],
        
        # Ultrack tracking configuration parameters
        "tracking_config": {
            # Segmentation parameters - control object detection
            "segmentation_config": {
                "min_area": 2100,      # Minimum object area in pixels
                "max_area": 80000,     # Maximum object area in pixels
                "n_workers": 10,       # Number of parallel workers
                "min_frontier": 0.4,   # Minimum frontier threshold
                "max_noise": 0.05,     # Maximum noise threshold
            },
            # Linking parameters - control object association between frames
            "linking_config": {
                "n_workers": 10,       # Number of parallel workers
                "max_distance": 15,    # Maximum distance for linking objects
                "distance_weight": -0.0001,  # Weight for distance in cost function
                "max_neighbors": 3,    # Maximum number of neighbors to consider
            },
            # Tracking parameters - control track optimization
            "tracking_config": {
                "n_threads": 14,       # Number of optimization threads
                "disappear_weight": -0.0001,  # Cost for object disappearance
                "appear_weight": -0.001,      # Cost for object appearance
                "division_weight": -0.0001,   # Cost for cell division
            },
        },
    }

    # Convert dictionary to TrackingSettings model for validation
    config = TrackingSettings(**config)

    # Z projection of the input_images channels (a `biahub project` config), used
    # when the input is 3D and output_mode is 2D. Same window as
    # nextflow/configs/a549/project.yml: mean over 60 planes starting 20 below the
    # Phase3D in-focus plane, found per timepoint. To project without focus, use
    # e.g. {"method": "mean"} (the whole stack) and drop `focus`.
    projection = ProjectionSettings(
        channels={
            "nuclei_prediction": {"method": "mean", "z": {"center": "focus", "below": 20, "above": 40}},
            "membrane_prediction": {"method": "mean", "z": {"center": "focus", "below": 20, "above": 40}},
        },
        focus={"channel": "Phase3D", "per": "timepoint"},
    )

    ### CHANGE THIS VALUE - the plate the focus channel is read from, when it is not
    # in input_images[0] (a virtual-stain store usually has no Phase3D; the
    # assembled or label-free plate does). None reads it from input_images[0].
    focus_store_path = None

    # ============================================================================
    # DATASET AND PATH SETUP
    # ============================================================================
    
    ### CHANGE THIS VALUE - Set your dataset name here
    dataset = "2025_06_26_A549_G3BP1_ZIKV"
    
    root = Path(f"/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/{dataset}")
    
    # Output directory for tracking results
    output_dirpath = root / "1-preprocess/label-free/3-track/test_tracking"
    os.makedirs(output_dirpath, exist_ok=True)

    ### CHANGE THIS VALUE - Set the position key to track (Plate/Well/Position)
    position_key = ("B", "1", "000000")

    # Extract configuration components
    blank_frames_path = config.blank_frames_path
    input_images = config.input_images

    # ============================================================================
    # DATA LOADING AND PREPROCESSING
    # ============================================================================
    
    # Get dataset metadata (shape and scale)
    with open_ome_zarr(input_images[0].path / Path(*position_key)) as vs_ds:
        T, C, Z, Y, X = vs_ds.data.shape  # Time, Channels, Z, Y, X dimensions
        full_scale = tuple(float(s) for s in vs_ds.scale)
    
    # Track does not select Z-planes: project a 3D input for 2D tracking
    project_input = config.output_mode == "2D" and Z > 1
    if not project_input:
        check_input_dimensionality(config.output_mode, Z)
    # Physical scale used for tracking: (Y, X) in 2D, (Z, Y, X) in 3D
    scale = full_scale[-2:] if config.output_mode == "2D" else full_scale[-3:]
    fov = "/".join(position_key)  # FOV identifier, as in the blank-frames CSV

    print(f"Processing FOV: {fov}")
    print(f"Data shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    print(f"Scale: {scale}")

    # Load (and, for a 3D input, project) the data
    if project_input:
        focus_store = Path(focus_store_path) if focus_store_path else input_images[0].path
        focus_per_t = find_focus(
            focus_store / Path(*position_key), projection, pixel_size=full_scale[-1]
        )
        # Every input with a path is loaded (and projected), as in `biahub track`
        data_dict = {}
        for image in input_images:
            if image.path is None:
                continue
            print(f"Projecting data from {image.path}...")
            data_dict.update(
                project_fov(
                    image.path / Path(*position_key),
                    list(image.channels),
                    projection,
                    focus_per_t,
                )
            )
    else:
        print("Loading data...")
        data_dict = load_data(position_key, input_images)
    
    print("Running preprocessing pipeline...")
    data_dict = run_preprocessing_pipeline(data_dict, input_images)
    
    print("Filling empty frames...")
    data_dict = fill_empty_frames_from_csv(fov, data_dict, blank_frames_path)
    
    # ============================================================================
    # EXTRACT TRACKING CHANNELS
    # ============================================================================
    
    # Get available channel names after preprocessing
    channel_names = data_dict.keys()
    print(f"Available channels: {channel_names}")
    
    # Extract foreground mask and contour gradient map for tracking
    foreground_mask, contour_gradient_map = detect_foreground_segmentation(data_dict)
    
    print(f"Foreground mask shape: {foreground_mask.shape}")
    print(f"Contour gradient map shape: {contour_gradient_map.shape}")
    
    # ============================================================================
    # TRACKING EXECUTION
    # ============================================================================
    
    # Create Ultrack configuration
    default_config = MainConfig()
    tracking_cfg = update_model(default_config, config.tracking_config)
    
    # Create temporary directory for tracking database
    database_path = output_dirpath / f"{dataset}_config_tracking" / "_".join(position_key)
    os.makedirs(database_path, exist_ok=True)

    print("Running Ultrack tracking...")
    tracking_labels, tracks_df, graph = run_ultrack(
        tracking_config=tracking_cfg,
        database_path=database_path,
        min_sibling_length=config.min_sibling_length,
        detection=foreground_mask,
        edges=contour_gradient_map,
        scale=scale,
        overwrite=True,
    )
    
    # Clean up temporary tracking database
    shutil.rmtree(database_path)
    
    print("Tracking complete!")
    print(f"Number of tracks: {len(tracks_df['track_id'].unique())}")
    print(f"Number of timepoints: {len(tracks_df['t'].unique())}")
 
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    
    import napari

    # Setup visualization parameters
    kwargs = dict(scale=scale, blending="additive")
    
    # Create Napari viewer for visualization
    viewer = napari.Viewer()
    
    # Add preprocessed channels
    viewer.add_image(data_dict["nuclei_prediction"], name="VS nuclei Preprocessed", **kwargs, colormap="green")
    viewer.add_image(data_dict["membrane_prediction"], name="VS membrane Preprocessed", colormap="magma", **kwargs)
    
    # Add segmentation and tracking results
    viewer.add_labels(foreground_mask, name="Foreground", **kwargs)
    viewer.add_image(contour_gradient_map, name="Contour", colormap="magma", **kwargs)
    viewer.add_labels(tracking_labels, name="Tracking Segments", scale=scale)
    
    # Add tracks with graph visualization
    # Tracks carry a time column: (t, y, x) in 2D, (t, z, y, x) in 3D
    track_columns = ["track_id", "t", "y", "x"] if config.output_mode == "2D" else ["track_id", "t", "z", "y", "x"]
    viewer.add_tracks(
        tracks_df[track_columns],
        graph=graph,
        name="Tracks",
        scale=(1, *scale),
        colormap="hsv",
        blending="opaque",
    )
    
    # Launch Napari viewer
    napari.run()


if __name__ == "__main__":
    main()
