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
    """
    Debug script for testing tracking configuration parameters.
    
    This script loads a single field of view (FOV), applies preprocessing,
    runs tracking with specified parameters, and visualizes results in Napari.
    Useful for tuning tracking parameters before running on full datasets.
    """

    # ============================================================================
    # CONFIGURATION SETUP
    # ============================================================================
    
    # Define the tracking configuration with all parameters
    config = {
        # Z-slice range to process: [-1, -1] means use central slices
        "z_range": [-1, -1],
        
        # Target channel for tracking (used for visualization)
        "target_channel": "nuclei_prediction",
        
        # Tracking mode: "2D" or "3D"
        "mode": "2D",
        
        # Field of view pattern to match in the dataset
        "fov": "*/*/*",
        
        # Path to CSV file containing blank frame information
        "blank_frames_path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_06_26_A549_G3BP1_ZIKV/1-preprocess/label-free/3-track/blank_frames.csv",
        
        # Input image configurations - defines data sources and preprocessing pipelines
        "input_images": [
            {
                # First input: Load raw data from Zarr store
                "path": "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_06_26_A549_G3BP1_ZIKV/1-preprocess/label-free/2-stabilize/virtual-stain/2025_06_26_A549_G3BP1_ZIKV.zarr",
                "channels": {
                    # Nuclei channel preprocessing pipeline
                    "nuclei_prediction": [
                        {
                            "function": "np.mean",  # Average across Z-axis
                            "kwargs": {"axis": 1},
                            "per_timepoint": False
                        },
                        {
                            "function": "ultrack.imgproc.normalize",  # Normalize intensity
                            "kwargs": {"gamma": 0.7},
                            "per_timepoint": False
                        }
                    ],
                    # Membrane channel preprocessing pipeline
                    "membrane_prediction": [
                        {
                            "function": "np.mean",  # Average across Z-axis
                            "kwargs": {"axis": 1},
                            "per_timepoint": False
                        },
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
                            "function": "biahub.cli.track.mem_nuc_contour",
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

    # ============================================================================
    # DATASET AND PATH SETUP
    # ============================================================================
    
    ### CHANGE THIS VALUE - Set your dataset name here
    dataset = "2025_06_26_A549_G3BP1_ZIKV"
    
    # Construct paths for virtual staining data
    vs_path = f"{dataset}.zarr/B/1/000000"
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
        scale = tuple(float(s) for s in vs_ds.scale[-2:])  # Physical scale (Y, X)
    
    # Resolve which Z-slices to use for tracking
    z_slices, Z = resolve_z_slice(config.z_range, Z)
    fov = "_".join(position_key)  # Create FOV identifier string

    print(f"Processing FOV: {fov}")
    print(f"Data shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    print(f"Scale: {scale}")
    print(f"Z-slices: {z_slices}")

    # Load and preprocess the data
    print("Loading data...")
    data_dict = load_data(position_key, input_images, z_slices)
    
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
    if "foreground" in channel_names and "contour" in channel_names:
        foreground_mask, contour_gradient_map = data_dict["foreground"], data_dict["contour"]
    elif "foreground_contour" in channel_names:
        foreground_mask, contour_gradient_map = data_dict["foreground_contour"]
    else:
        raise ValueError("Foreground and contour channels are required for tracking.")
    
    print(f"Foreground mask shape: {foreground_mask.shape}")
    print(f"Contour gradient map shape: {contour_gradient_map.shape}")
    
    # ============================================================================
    # TRACKING EXECUTION
    # ============================================================================
    
    # Create Ultrack configuration
    default_config = MainConfig()
    tracking_cfg = update_model(default_config, config.tracking_config)
    
    # Create temporary directory for tracking database
    databaset_path = output_dirpath / f"{dataset}_config_tracking/{fov}"
    os.makedirs(databaset_path, exist_ok=True)

    print("Running Ultrack tracking...")
    tracking_labels, tracks_df, graph = run_ultrack(tracking_cfg, foreground_mask, contour_gradient_map, scale, databaset_path)
    
    # Clean up temporary tracking database
    shutil.rmtree(databaset_path)
    
    print(f"Tracking complete!")
    print(f"Number of tracks: {len(tracks_df['track_id'].unique())}")
    print(f"Number of timepoints: {len(tracks_df['t'].unique())}")
 
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    
    # Setup visualization parameters
    kwargs = dict(scale=scale, blending="additive")
    
    # Create Napari viewer for visualization
    viewer = napari.Viewer()
    
    # Add preprocessed channels
    viewer.add_image(data_dict["nuclei_prediction"], name="VS nuclei Preprocessed", **kwargs, colormap="green")
    viewer.add_image(data_dict["membrane_prediction"], name="VS membrane Preprocessed", colormap="magma", **kwargs)
    
    # Add segmentation and tracking results
    viewer.add_labels(foreground_mask, name="Foreground ", **kwargs)
    viewer.add_image(contour_gradient_map, name="Contourn ", colormap="magma", **kwargs)
    viewer.add_labels(tracking_labels, name="Tracking Segments", scale=scale)
    
    # Add tracks with graph visualization
    viewer.add_tracks(
        tracks_df[["track_id", "t", "y", "x"]],
        graph=graph,
        name="Tracks",
        scale=scale,
        colormap="hsv",
        blending="opaque",
    )
    
    # Launch Napari viewer
    napari.run()


if __name__ == "__main__":
    main()
