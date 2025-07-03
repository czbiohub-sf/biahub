import warnings

from pathlib import Path

import click
import numpy as np
import submitit
import torch

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.cli import utils
from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.settings import SegmentationSettings


def segment_data(
    czyx_data: np.ndarray,
    segmentation_models: dict,
    gpu: bool = True,
) -> np.ndarray:
    """
    Segment a CZYX image using Cellpose segmentation models.

    Parameters
    ----------
    czyx_data : np.ndarray
        A CZYX image to segment
    segmentation_models : dict
        A dictionary of segmentation models to use
    gpu : bool, optional
        Whether to use a GPU for segmentation

    Returns
    -------
    np.ndarray
        A CZYX segmentation image
    """
    from cellpose import models

    if gpu:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except torch.cuda.CudaError:
            click.echo("No GPU available. Using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    click.echo(f"Using device: {device}")

    # Pre-load unique models to avoid redundant loading
    unique_models = {}
    for model_name, model_args in segmentation_models.items():
        model_path = model_args.path_to_model
        if model_path not in unique_models:
            click.echo(f"Loading model: {model_path}")
            unique_models[model_path] = models.CellposeModel(
                gpu=gpu, device=device, pretrained_model=model_path
            )

    czyx_segmentation = []

    # Process each model
    for model_name, model_args in segmentation_models.items():
        click.echo(f"Starting segmentation with model {model_name}")

        # Extract the data we need for this model 2D or 3D
        z_slice_2D = model_args.z_slice_2D
        if z_slice_2D is not None:
            czyx_data_to_segment = czyx_data[:, z_slice_2D].copy()
            z_axis = None
        else:
            czyx_data_to_segment = czyx_data.copy()
            z_axis = 1
        click.echo(f"Segmenting {model_name} with z_axis {z_axis}")
        # Apply preprocessing specific to this model
        for preproc in model_args.preprocessing:
            func = preproc.function
            kwargs = preproc.kwargs.copy()
            c_idx = preproc.channel

            # Convert list to tuple for out_range if needed
            if "out_range" in kwargs and isinstance(kwargs["out_range"], list):
                kwargs["out_range"] = tuple(kwargs["out_range"])

            click.echo(
                f"Processing with {func.__name__} with kwargs {kwargs} to channel {c_idx}"
            )
            czyx_data_to_segment[c_idx] = func(czyx_data_to_segment[c_idx], **kwargs)

        # Prepare cellpose input
        cellpose_czyx = np.zeros(
            (3, *czyx_data_to_segment.shape[1:]), dtype=czyx_data_to_segment.dtype
        )
        for i, channel in enumerate(model_args.channels):
            if channel is not None:
                cellpose_czyx[i] = czyx_data_to_segment[channel]

        # Get pre-loaded model and run segmentation
        click.echo(f"Running segmentation for {model_args.path_to_model}")
        model = unique_models[model_args.path_to_model]
        segmentation, _, _ = model.eval(
            cellpose_czyx, channel_axis=0, z_axis=z_axis, **model_args.eval_args
        )

        # Handle 2D output formatting
        if z_slice_2D is not None and isinstance(z_slice_2D, int):
            segmentation = segmentation[np.newaxis, ...]

        czyx_segmentation.append(segmentation)

        # Clean up intermediate arrays
        del cellpose_czyx, czyx_data_to_segment

    # Clean up GPU memory
    if gpu and device.type == 'cuda':
        torch.cuda.empty_cache()

    return np.stack(czyx_segmentation, axis=0)


@click.command("segment")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
def segment_cli(
    input_position_dirpaths: list[str],
    config_filepath: str,
    output_dirpath: str,
    sbatch_filepath: str | None = None,
    local: bool = False,
):
    """
    Segment a single position across T axes using the configuration file.

    >> biahub segment \
        -i ./input.zarr/*/*/* \
        -c ./segment_params.yml \
        -o ./output.zarr
    """

    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    if sbatch_filepath is not None:
        sbatch_filepath = Path(sbatch_filepath)

    # Handle single position or wildcard filepath
    output_position_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)

    # Get the deskewing parameters
    # Load the first position to infer dataset information
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        settings = yaml_to_model(config_filepath, SegmentationSettings)
        scale = input_dataset.scale
        channel_names = input_dataset.channel_names

    # Load the segmentation models with their respective configurations
    # TODO: implement logic for 2D segmentation. Have a slicing parameter
    segment_args = settings.models
    C_segment = len(segment_args)
    for model_name, model_args in segment_args.items():
        if model_args.z_slice_2D is not None and isinstance(model_args.z_slice_2D, int):
            Z = 1
        # Ensure channel names exist in the dataset
        if not all(channel in channel_names for channel in model_args.channels):
            raise ValueError(
                f"Channels {model_args.channels} not found in dataset {channel_names}"
            )
        # Channel strings to indices to be used in cellpose. Hiding this from the
        model_args.channels = [channel_names.index(channel) for channel in model_args.channels]
        # NOTE: Cellpose requires 3 channels. If the channels list is less than 3, the first channel is repeated.

        if len(model_args.channels) < 3:
            model_args.channels.extend(
                [model_args.channels[0]] * (3 - len(model_args.channels))
            )
        else:
            warnings.warn(
                f"Model {model_name} has more than 3 channels. Only the first 3 channels will be used."
            )

        click.echo(f"Segmenting {model_name} using channels {model_args.channels}")
        if (
            "anisotropy" not in model_args.eval_args
            or model_args.eval_args["anisotropy"] is None
        ):
            # Using dataset anisotropy
            model_args.eval_args["anisotropy"] = scale[-3] / scale[-1]
            click.echo(
                f"Using anisotropy from scale metadata: {model_args.eval_args['anisotropy']}"
            )
        else:
            click.echo(
                f"Using anisotropy from the config: {model_args.eval_args['anisotropy']}"
            )

        # Check if preprocessing functions exist and replace channel name with channel index
        if model_args.preprocessing is not None:
            for preproc in model_args.preprocessing:
                # Replace the channel name with the channel index
                if preproc.channel is not None:
                    preproc.channel = channel_names.index(preproc.channel)
                else:
                    raise ValueError("Channel must be specified for preprocessing functions")

    segmentation_shape = (T, C_segment, Z, Y, X)

    # Create a zarr store output to mirror the input
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[path.parts[-3:] for path in input_position_dirpaths],
        channel_names=[model_name + "_labels" for model_name in segment_args.keys()],
        shape=segmentation_shape,
        chunks=None,
        scale=scale,
    )

    # Estimate resources
    num_cpus, gb_ram_request = estimate_resources(shape=segmentation_shape, ram_multiplier=10)
    num_gpus = 1
    slurm_time = np.ceil(np.max([120, T * Z * 10])).astype(int)
    slurm_array_parallelism = 9
    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "segment",
        "slurm_gres": f"gpu:{num_gpus}",
        "slurm_mem_per_cpu": f"{gb_ram_request}G",
        "slurm_cpus_per_task": np.max([int(slurm_array_parallelism * 2), num_cpus]),
        "slurm_array_parallelism": slurm_array_parallelism,
        "slurm_time": slurm_time,
        "slurm_partition": "gpu",
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    with executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths
        ):
            jobs.append(
                executor.submit(
                    process_single_position,
                    func=segment_data,
                    input_position_path=input_position_path,
                    output_position_path=output_position_path,
                    input_channel_indices=[list(range(C))],
                    output_channel_indices=[list(range(C_segment))],
                    num_processes=np.min([slurm_array_parallelism, int(num_cpus * 0.8)]),
                    segmentation_models=segment_args,
                )
            )

    monitor_jobs(jobs, input_position_dirpaths)


if __name__ == "__main__":
    segment_cli()
