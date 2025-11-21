# %%
from pathlib import Path
import imageio
import napari
import numpy as np
from iohub import open_ome_zarr
from tqdm import tqdm
import os
from natsort import natsorted
import glob
import yaml
from ultrack.reader.napari_reader import read_csv
import click

from biahub.cli.parsing import config_filepath


def load_config(path="napari_zarr_to_mp4.yml"):
    """
    Load the YAML configuration file for video generation parameters.
    
    Parameters
    ----------
    path : str, optional
        Path to the YAML configuration file (default: "napari_zarr_to_mp4.yml")
    
    Returns
    -------
    dict
        Parsed configuration dictionary containing input paths, output settings, and channel configurations
    
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the specified path
    yaml.YAMLError
        If the configuration file contains invalid YAML syntax
    """
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    except yaml.YAMLError:
        raise yaml.YAMLError(f"Error parsing YAML configuration file: {path}")
    

def process_image(input_path, fov, verbose=False):
    """
    Process and extract image data from a single field of view (FOV).
    
    Supports max projection or single slice extraction based on configuration.
    Handles multiple channels with individual contrast limits and colormaps.
    
    Parameters
    ----------
    input_path : dict
        Configuration dictionary containing:
        - 'input_path': str, path to the Zarr dataset
        - 'channels': list of dict, channel configurations with 'name', 'contrast_limits', and optional 'colormap'
        - 'z_index': int or None, specific Z-slice to extract (None for max projection)
        - 'time_index': int or None, specific time point to extract (None for all timepoints)
    fov : str
        Field of view identifier (e.g., "plate/well/position")
    verbose : bool, optional
        Enable detailed logging output (default: False)
    
    Returns
    -------
    dict or None
        Dictionary containing:
        - 'channel_names': list of str, processed channel names
        - 'arr_data': list of arrays, processed image data for each channel
        - 'contrast_limits': list of tuples, contrast limits for each channel
        - 'colormaps': list of str, colormap names for each channel
        - 'scale': tuple, pixel scaling factors (Y, X)
        Returns None if FOV does not exist
    
    Notes
    -----
    - z_index and time_index cannot be used simultaneously
    - If a channel is not found, it will be skipped with a warning
    """
    if verbose:
        click.echo(f"\n[IMAGE] Processing FOV: {fov}")
        click.echo(f"        Source: {input_path['input_path']}")
    
    channel_config = input_path.get("channels", [])
    z_index = input_path.get("z_index", None)
    time_index = input_path.get("time_index", None)
    
    data_dict = {"channel_names": [], "arr_data": [], "contrast_limits": [], "colormaps": [], "scale": None}
    
    with open_ome_zarr(Path(input_path["input_path"])/fov) as dataset:
        if verbose:
            click.echo(f"        Dataset shape (T,C,Z,Y,X): {dataset.data.shape}")
        T, C, Z, Y, X = dataset.data.shape
        if not (Path(input_path["input_path"])/fov).exists():
            click.echo(f"⚠️  Skipping FOV '{fov}' - path does not exist: {input_path['input_path']}")
            return
    
        all_channel_names = dataset.channel_names
        if verbose:
            click.echo(f"        Available channels: {', '.join(all_channel_names)}")

        data_dict["scale"] = dataset.scale[-2:]
        if verbose:
            click.echo(f"        Pixel scale (Y,X): {data_dict['scale']}")

        for ch in channel_config:
            ch_name = ch["name"]
            
            if verbose:
                click.echo(f"        → Processing channel '{ch_name}'")
                click.echo(f"          Contrast limits: {ch['contrast_limits']}")
                click.echo(f"          Colormap: {ch.get('colormap', 'gray')}")

            if ch_name not in all_channel_names:
                click.echo(f"⚠️  Channel '{ch_name}' not found in {fov}. Available: {', '.join(all_channel_names)}")
                data_dict["arr_data"] = []
                continue
            ch_idx = all_channel_names.index(ch_name)

            if z_index is None and time_index is None:
                if verbose:
                    click.echo(f"          Projection mode: Z-axis max projection")
                arr = dataset.data.dask_array()[:, ch_idx, :, :, :].max(axis=1)

            elif z_index is not None and time_index is None:
                if verbose:
                    click.echo(f"          Extraction mode: Z-slice {z_index}")
                arr = dataset.data.dask_array()[:, ch_idx, z_index, :, :]

            elif z_index is None and time_index is not None:
                if verbose:
                    click.echo(f"          Extraction mode: Timepoint {time_index}")
                arr = dataset.data.dask_array()[time_index, ch_idx, :, :, :]

            elif z_index is not None and time_index is not None:
                raise ValueError("Cannot specify both z_index and time_index simultaneously")
            else:
                raise ValueError("Invalid combination of z_index and time_index")
            
            if verbose:
                click.echo(f"          Output shape: {arr.shape}")

            data_dict["arr_data"].append(arr)
            data_dict["channel_names"].append(ch["name"])
            data_dict["contrast_limits"].append(ch["contrast_limits"])
            data_dict["colormaps"].append(ch.get("colormap", None))

        return data_dict


def process_label(input_path, fov, verbose=False):
    """
    Process and extract 2D label/segmentation data from a single field of view.
    
    Parameters
    ----------
    input_path : dict
        Configuration dictionary containing:
        - 'input_path': str, path to the Zarr dataset
        - 'channels': list of dict, channel configurations with 'name'
        - 'z_index': int or None, specific Z-slice to extract (None for max projection)
        - 'time_index': int or None, specific time point to extract (None for all timepoints)
    fov : str
        Field of view identifier (e.g., "plate/well/position")
    verbose : bool, optional
        Enable detailed logging output (default: False)
    
    Returns
    -------
    dict or None
        Dictionary containing:
        - 'channel_names': list of str, processed channel names
        - 'arr_data': list of arrays, processed label data for each channel
        - 'scale': tuple, pixel scaling factors (Y, X)
        Returns None if FOV does not exist
    
    Notes
    -----
    - z_index and time_index cannot be used simultaneously
    - Labels are typically integer arrays representing segmented objects
    """
    if verbose:
        click.echo(f"\n[LABEL] Processing FOV: {fov}")
        click.echo(f"        Source: {input_path['input_path']}")
    
    channel_config = input_path.get("channels", [])
    data_dict = {"channel_names": [], "arr_data": [], "scale": None}
    z_index = input_path.get("z_index", None)
    time_index = input_path.get("time_index", None)

    with open_ome_zarr(Path(input_path["input_path"])/fov) as dataset:
        if verbose:
            click.echo(f"        Dataset shape (T,C,Z,Y,X): {dataset.data.shape}")
        T, C, Z, Y, X = dataset.data.shape

        if not (Path(input_path["input_path"])/fov).exists():
            click.echo(f"⚠️  Skipping FOV '{fov}' - path does not exist: {input_path['input_path']}")
            return

        all_channel_names = dataset.channel_names
        if verbose:
            click.echo(f"        Available channels: {', '.join(all_channel_names)}")

        data_dict["scale"] = dataset.scale[-2:]
        if verbose:
            click.echo(f"        Pixel scale (Y,X): {data_dict['scale']}")

        for ch in channel_config:
            ch_name = ch["name"]
            
            if verbose:
                click.echo(f"        → Processing label channel '{ch_name}'")

            if ch_name not in all_channel_names:
                click.echo(f"⚠️  Channel '{ch_name}' not found in {fov}. Available: {', '.join(all_channel_names)}")
                data_dict["arr_data"] = []
                continue     
            ch_idx = all_channel_names.index(ch_name)

            if time_index is None and z_index is None:
                if verbose:
                    click.echo(f"          Projection mode: Z-axis max projection")
                arr = dataset.data.dask_array()[:, ch_idx, :, :, :].max(axis=1)

            elif time_index is not None and z_index is None:
                if verbose:
                    click.echo(f"          Extraction mode: Timepoint {time_index}")
                arr = dataset.data.dask_array()[time_index, ch_idx, :, :, :]

            elif time_index is None and z_index is not None:
                if verbose:
                    click.echo(f"          Extraction mode: Z-slice {z_index}")
                arr = dataset.data.dask_array()[:, ch_idx, z_index, :, :]

            elif time_index is not None and z_index is not None:
                raise ValueError("Cannot specify both time_index and z_index simultaneously")
            else:
                raise ValueError("Invalid combination of time_index and z_index")
            
            if verbose:
                click.echo(f"          Output shape: {arr.shape}")

            data_dict["channel_names"].append(ch["name"])
            data_dict["arr_data"].append(arr)

        return data_dict


def process_tracks(input_path, fov, verbose=False):
    """
    Load and process Ultrack tracking data from CSV files.
    
    Parameters
    ----------
    input_path : dict
        Configuration dictionary containing 'input_path' key with path to tracking data
    fov : str
        Field of view identifier used to locate the tracks CSV file
    verbose : bool, optional
        Enable detailed logging output (default: False)
    
    Returns
    -------
    tuple
        (track_df, graph) where:
        - track_df: pandas.DataFrame, tracking data with columns [track_id, t, y, x, ...]
        - graph: dict, graph structure representing parent-child relationships between tracks
    
    Notes
    -----
    Expects CSV files named as 'tracks_{fov}.csv' where slashes in FOV are replaced with underscores
    """
    if verbose:
        click.echo(f"\n[TRACKS] Processing FOV: {fov}")
        click.echo(f"         Source: {input_path['input_path']}")

    df_path = Path(input_path["input_path"]) / fov / f"tracks_{fov.replace('/', '_')}.csv"
    
    if verbose:
        click.echo(f"         Loading tracks from: {df_path}")
    
    track_df, kwargs, *_ = read_csv(df_path)
    graph = kwargs["graph"]
    
    if verbose:
        click.echo(f"         Loaded {len(track_df)} track points across {track_df['track_id'].nunique()} unique tracks")
    
    return track_df, graph


def get_unique_output_filename(output_path, fov, verbose=False):
    """
    Generate a unique output filename to avoid overwriting existing files.
    
    Parameters
    ----------
    output_path : Path
        Directory where the output file will be saved
    fov : str
        Field of view identifier used as the base filename
    verbose : bool, optional
        Enable detailed logging output (default: False)
    
    Returns
    -------
    Path
        Unique output file path with .mp4 extension
    
    Notes
    -----
    If a file already exists, appends an incremental counter (e.g., _001, _002)
    """
    output_file = output_path / f"{fov.replace('/', '_')}.mp4"

    if output_file.exists():
        if verbose:
            click.echo(f"⚠️  File exists: {output_file.name} - generating unique filename")
        n = 1
        while output_file.exists():
            output_file = output_path / f"{fov.replace('/', '_')}_{n:03d}.mp4"
            n += 1
        if verbose:
            click.echo(f"✓  Using filename: {output_file.name}")
    
    return output_file


def process_fov(fov_path, input_paths, output_path, fps, verbose=False, view_napari=False):
    """
    Process a single field of view and generate an MP4 video using napari rendering.
    
    Combines multiple image channels, labels, and optional tracking data into a single
    rendered video file with configurable frame rate.
    
    Parameters
    ----------
    fov_path : Path
        Path object pointing to the FOV directory
    input_paths : list of dict
        List of input configurations, each specifying data type ('image', 'label'),
        paths, channels, and optional tracking information
    output_path : Path
        Directory where the output MP4 file will be saved
    fps : int or float
        Frames per second for the output video
    verbose : bool, optional
        Enable detailed logging output (default: False)
    debug : bool, optional
        If True, opens napari GUI instead of rendering to file (default: False)
    
    Returns
    -------
    None
    
    Notes
    -----
    - In debug mode, napari viewer stays open for interactive inspection
    - In normal mode, renders all frames to MP4 and closes automatically
    - Skips FOV if no valid channels are found
    """
    fov = "/".join(fov_path.parts[-3:])
    click.echo("\n" + "="*60)
    click.echo(f"FOV: {fov}")
    click.echo("="*60)

    output_file = get_unique_output_filename(output_path, fov, verbose)
    click.echo(f"Output file: {output_file}")

    viewer = napari.Viewer()
    
    for input_path in input_paths:
        if input_path["type"] == "image":
            data_dict = process_image(input_path, fov, verbose)

            if not data_dict["arr_data"]:
                click.echo("❌ No valid channels found. Skipping FOV.")
                return
            
            for arr, ch, clim, cmap in zip(data_dict["arr_data"], data_dict["channel_names"], 
                                           data_dict["contrast_limits"], data_dict["colormaps"]):
                viewer.add_image(arr, name=ch, scale=data_dict["scale"], 
                               blending="additive", contrast_limits=clim, colormap=cmap)
                if verbose:
                    click.echo(f"✓ Added image layer: {ch}")

        elif input_path["type"] == "label":
            data_dict = process_label(input_path, fov, verbose)
            
            if not data_dict["arr_data"]:
                click.echo("❌ No valid label channels found. Skipping FOV.")
                return
            
            for arr, ch in zip(data_dict["arr_data"], data_dict["channel_names"]):
                viewer.add_labels(arr, name=ch, scale=data_dict["scale"], blending="additive")
                if verbose:
                    click.echo(f"✓ Added label layer: {ch}")

            if input_path.get("tracks", False):
                track_df, graph = process_tracks(input_path, fov, verbose)
                viewer.add_tracks(
                    track_df[["track_id", "t", "y", "x"]],
                    graph=graph,
                    name="Tracks",
                    scale=data_dict["scale"],
                    colormap="hsv",
                    blending="opaque",
                )
                if verbose:
                    click.echo(f"✓ Added tracks layer")
    
        # Setup for fullscreen rendering
    viewer.dims.set_point(0, 0)
    viewer.window.show_fullscreen = True
    viewer.reset_view()
    if view_napari:
        click.echo("🔍 Visualization mode: Opening napari viewer for inspection")
        napari.run()
    else:


        click.echo(f"🎬 Rendering video at {fps} FPS...")
        writer = imageio.get_writer(output_file, fps=fps)
        num_frames = data_dict["arr_data"][0].shape[0]
        
        for i in tqdm(range(num_frames), desc=f"Rendering {fov}", unit="frame"):
            viewer.dims.set_point(0, i)
            screenshot = viewer.screenshot(canvas_only=True)
            writer.append_data(np.array(screenshot))

        writer.close()
        viewer.close()

        click.echo(f"✅ Completed: {output_file}")
        if verbose:
            click.echo(f"   Total frames: {num_frames}")
            click.echo(f"   Duration: {num_frames/fps:.1f} seconds")


def solve_io_parameters():
    """
    Parse configuration file and resolve input/output paths.
    
    Parameters
    ----------
    config_filepath : str
        Path to the configuration file
    
    Returns
    -------
    tuple
        (input_fov_path, input_paths, output_path, fps, verbose) where:
        - input_fov_path: list of Path objects for each FOV to process
        - input_paths: list of dicts containing input configurations
        - output_path: Path object for output directory
        - fps: float, frames per second for video rendering
        - verbose: bool, enable detailed logging
    
    Raises
    ------
    ValueError
        If input_paths is missing from config or no FOV paths are found
    """
    config = load_config(config_filepath)
    click.echo("\n" + "="*60)
    click.echo("CONFIGURATION LOADED")
    click.echo("="*60)
    click.echo(f"Config file: napari_zarr_to_mp4.yaml")

    fov = config.get("fov", "/*/*/*")
    fps = config.get("fps", 3)
    verbose = config.get("verbose", False)
    input_paths = config.get("input_paths", None)

    if input_paths is None:
        raise ValueError("❌ Configuration error: 'input_paths' is required in config file")
    
    pattern = str(config["input_paths"][0]["input_path"] + fov)
    input_fov_path = [Path(p) for p in natsorted(glob.glob(pattern))]
    
    if len(input_fov_path) == 0:
        raise ValueError(f"❌ No FOV paths found matching pattern: {pattern}\nCheck paths in config file")

    output_path = Path(config["output_path"])/"fov"
    os.makedirs(output_path, exist_ok=True)
    
    click.echo(f"FOV pattern: {fov}")
    click.echo(f"Found {len(input_fov_path)} FOV(s) to process")
    click.echo(f"Output directory: {output_path}")
    click.echo(f"Frame rate: {fps} FPS")
    click.echo(f"Verbose mode: {'ON' if verbose else 'OFF'}")
    click.echo("="*60)
    
    return input_fov_path, input_paths, output_path, fps, verbose



# %%
def napari_zarr_to_mp4(config_filepath: str, view_napari: bool):
    """
    Main execution function for batch video generation.
    
    Generates MP4 videos from Zarr datasets using napari for rendering.
    Processes multiple FOVs with configurable channel settings, contrast limits,
    colormaps, and optional tracking overlays.
    
    Configuration is loaded from 'napari_zarr_to_mp4.yaml' file.
    Skips FOVs that encounter errors and continues with remaining FOVs.

    Parameters
    ----------
    config_filepath : str
        Path to the configuration file
    view_napari : bool
        If True, open the napari viewer after processing
    Raises
    ------
    FileNotFoundError
        If configuration file is not found
    ValueError
        If configuration is invalid or no input paths are found
    """
    input_fov_path, input_paths, output_path, fps, verbose = solve_io_parameters(config_filepath)

    click.echo(f"\n🚀 Starting batch processing of {len(input_fov_path)} FOV(s)...\n")
    
    success_count = 0
    error_count = 0
    
    for fov_path in tqdm(input_fov_path, desc="Overall progress", unit="FOV"):
        try:
            process_fov(fov_path, input_paths, output_path, fps, verbose, view_napari=view_napari)
            success_count += 1
        except Exception as e:
            error_count += 1
            click.echo(f"\n❌ Error processing FOV {fov_path}: {e}")
            continue

    click.echo("\n" + "="*60)
    click.echo("BATCH PROCESSING COMPLETE")
    click.echo("="*60)
    click.echo(f"✅ Successfully processed: {success_count} FOV(s)")
    if error_count > 0:
        click.echo(f"❌ Failed: {error_count} FOV(s)")
    click.echo(f"📁 Output directory: {output_path}")
    click.echo("="*60)

@config_filepath()
@click.option(
    "--view-napari",
    "-v",
    is_flag=True,
    help="View the napari viewer after processing",
)
def napari_zarr_to_mp4_cli(config_filepath: str, view_napari: bool):
    """
    Generate MP4 videos from Zarr datasets using napari for rendering.
    
    Parameters
    ----------
    config_filepath : str
        Path to the configuration file
    view_napari : bool
        If True, open the napari viewer after processing
    """
    napari_zarr_to_mp4(config_filepath, view_napari)

if __name__ == "__main__":
    napari_zarr_to_mp4_cli()