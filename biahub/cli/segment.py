from pathlib import Path

import click
import numpy as np
import submitit
import torch

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.analysis.AnalysisSettings import SegmentationSettings
from biahub.cli import utils
from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
)
from biahub.cli.utils import yaml_to_model


def segment_data(
    czyx_data: np.ndarray,
    segmentation_models: dict,
    gpu: bool = True,
) -> np.ndarray:
    from cellpose import models

    """
    Segment a CZYX image using a Cellpose segmentation model

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

    # Segmenetation in cpu or gpu
    if gpu:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except torch.cuda.CudaError:
            print("No GPU available. Using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    czyx_segmentation = []
    # Process each model in a loop
    for i, (model_name, model_args) in enumerate(segmentation_models.items()):
        print(f"Segmenting with model {model_name}")
        z_slice_2D = model_args.z_slice_2D
        czyx_data_to_segment = (
            czyx_data[:, z_slice_2D : z_slice_2D + 1] if z_slice_2D is not None else czyx_data
        )
        model = models.CellposeModel(
            model_type=model_args.path_to_model, gpu=gpu, device=device
        )
        segmentation, _, _ = model.eval(
            czyx_data_to_segment, channel_axis=0, z_axis=1, **model_args.eval_args
        )  # noqa: python-no-eval
        if z_slice_2D is not None and isinstance(z_slice_2D, int):
            segmentation = segmentation[np.newaxis, ...]
        czyx_segmentation.append(segmentation)
    czyx_segmentation = np.stack(czyx_segmentation, axis=0)

    return czyx_segmentation


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
def segment(
    input_position_dirpaths: list[str],
    config_filepath: str,
    output_dirpath: str,
    sbatch_filepath: str = None,
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

    # TODO: impelment logic for 2D segmentation. Have a slicing parameter
    segment_args = settings.models
    C_segment = len(segment_args)
    for model_name, model_args in segment_args.items():
        if model_args.z_slice_2D is not None and isinstance(model_args.z_slice_2D, int):
            Z = 1
        # Ensure channel names exist in the dataset
        if not all(channel in channel_names for channel in model_args.eval_args["channels"]):
            raise ValueError(
                f"Channels {model_args.eval_args['channels']} not found in dataset {channel_names}"
            )
        # Channel strings to indices with the cellpose offset of 1
        model_args.eval_args["channels"] = [
            channel_names.index(channel) + 1 for channel in model_args.eval_args["channels"]
        ]
        # NOTE:List of channels, either of length 2 or of length number of images by 2.
        # First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        # Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        if len(model_args.eval_args["channels"]) < 2:
            model_args.eval_args["channels"].append(0)

        click.echo(
            f"Segmenting with model {model_name} using channels {model_args.eval_args['channels']}"
        )
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

    # Estimate Resrouces
    gb_per_element = 4 / 2**30  # bytes_per_float32 / bytes_per_gb
    num_cpus = np.min([T * C, 16])
    input_memory = num_cpus * Z * Y * X * gb_per_element
    gb_ram_request = np.ceil(np.max([1, input_memory])).astype(int)
    num_gpus = 1

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "segment",
        "slurm_gres": f"gpu:{num_gpus}",
        "slurm_mem_per_cpu": f"{gb_ram_request}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 60,
        "slurm_partition": "gpu",
    }

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
                    segment_data,
                    input_position_path,
                    output_position_path,
                    input_channel_indices=[list(range(C))],
                    output_channel_indices=[list(range(C_segment))],
                    num_processes=slurm_args["slurm_cpus_per_task"],
                    segmentation_models=segment_args,
                )
            )

    monitor_jobs(jobs, input_position_dirpaths)


if __name__ == "__main__":
    segment()
