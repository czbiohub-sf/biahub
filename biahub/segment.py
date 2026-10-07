from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.settings import SegmentationSettings
from biahub.utils.cellpose import cellpose_device, load_cellpose_model
from biahub.utils.cluster import estimate_resources, get_submitit_cluster
from biahub.utils.config import yaml_to_model
from biahub.utils.ngff import get_output_paths, resolve_ome_zarr_version


@dataclass(frozen=True)
class ResolvedModel:
    """A segmentation model resolved against one input plate (indices instead of names)."""

    name: str
    pretrained_model: str
    channel_indices: list[int]
    eval_args: dict[str, Any]
    z_slice_2D: int | None
    # (function, channel index, kwargs), applied in order
    preprocessing: list[tuple[Callable, int, dict[str, Any]]] = field(default_factory=list)


def resolve_models(
    settings: SegmentationSettings,
    channel_names: list[str],
    scale: tuple[float, ...],
    z_size: int,
) -> list[ResolvedModel]:
    """Resolve each configured model against the input plate, without touching ``settings``.

    Channel names become dataset indices, 3D models get the dataset anisotropy unless the
    config sets one, and ``z_slice_2D`` is checked against the stack. ``settings`` is left
    exactly as written, so it can be recorded as provenance.
    """

    def index(name: str, what: str) -> int:
        if name not in channel_names:
            raise ValueError(
                f"{what} channel {name!r} is not in the input channels {channel_names}."
            )
        return channel_names.index(name)

    resolved = []
    for name, model in settings.models.items():
        if model.z_slice_2D is not None and model.z_slice_2D >= z_size:
            raise ValueError(
                f"Model {name}: z_slice_2D={model.z_slice_2D} is outside the input stack (Z={z_size})."
            )
        eval_args = dict(model.eval_args)
        if model.z_slice_2D is None and eval_args.get("anisotropy") is None:
            eval_args["anisotropy"] = scale[-3] / scale[-1]
        resolved.append(
            ResolvedModel(
                name=name,
                pretrained_model=model.pretrained_model,
                channel_indices=[index(c, f"Model {name}:") for c in model.channels],
                eval_args=eval_args,
                z_slice_2D=model.z_slice_2D,
                preprocessing=[
                    (
                        p.function,
                        index(p.channel, f"Model {name} preprocessing:"),
                        {
                            k: tuple(v) if k == "out_range" and isinstance(v, list) else v
                            for k, v in p.kwargs.items()
                        },
                    )
                    for p in model.preprocessing
                ],
            )
        )
    return resolved


def segment_data(
    czyx_data: np.ndarray,
    models: list[ResolvedModel],
    gpu: bool = True,
) -> np.ndarray:
    """Segment a CZYX image with each resolved cellpose model.

    Returns an array of shape (n_models, Z or 1, Y, X).
    """
    # Every job this step submits asks SLURM for a GPU, so an unusable one is a
    # broken allocation, not a reason to fall back to a ~130x slower CPU run.
    device = cellpose_device(gpu)
    click.echo(f"Using device: {device}")

    czyx_segmentation = []
    for model in models:
        click.echo(f"Segmenting with model {model.name}")
        for func, c_idx, kwargs in model.preprocessing:
            click.echo(
                f"Processing with {func.__name__} with kwargs {kwargs} to channel {c_idx}"
            )
            czyx_data[c_idx] = func(czyx_data[c_idx], **kwargs)

        # Cellpose 4 refuses a z axis for 2D processing, so a 2D model gets the (C, Y, X)
        # plane and a 3D model the (C, Z, Y, X) stack. It also ignores `channels` and keeps
        # the first 3 channels it is given, so pass only the configured ones.
        cellpose_model = load_cellpose_model(model.pretrained_model, device)
        if model.z_slice_2D is not None:
            image, z_axis = czyx_data[model.channel_indices, model.z_slice_2D], None
        else:
            image, z_axis = czyx_data[model.channel_indices], 1
        segmentation, _, _ = cellpose_model.eval(
            image, channel_axis=0, z_axis=z_axis, **model.eval_args
        )
        if model.z_slice_2D is not None:
            segmentation = segmentation[np.newaxis, ...]
        czyx_segmentation.append(segmentation)
    return np.stack(czyx_segmentation, axis=0)


@click.command("segment")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
@monitor()
def segment_cli(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str | None = None,
    local: bool = False,
    monitor: bool = True,
):
    """Segment a single position across T axes using the configuration file.

    >>> biahub segment \
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
    output_position_paths = get_output_paths(input_position_dirpaths, output_dirpath)

    # Get the deskewing parameters
    # Load the first position to infer dataset information
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        settings = yaml_to_model(config_filepath, SegmentationSettings)
        scale = input_dataset.scale
        channel_names = input_dataset.channel_names

    models = resolve_models(settings, channel_names, scale, Z)
    for m in models:
        click.echo(f"Segmenting with model {m.name} using channels {m.channel_indices}")
    C_segment = len(models)
    Z_out = 1 if models[0].z_slice_2D is not None else Z

    segmentation_shape = (T, C_segment, Z_out, Y, X)

    # Create a zarr store output to mirror the input
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[path.parts[-3:] for path in input_position_dirpaths],
        channel_names=[m.name + "_labels" for m in models],
        shape=segmentation_shape,
        chunks=None,
        scale=scale,
        version=resolve_ome_zarr_version(
            input_position_dirpaths[0], settings.output_ome_zarr_version
        ),
    )

    # Estimate resources
    _, num_cpus, gb_ram_request = estimate_resources(
        shape=segmentation_shape, ram_multiplier=20
    )
    num_gpus = 1
    slurm_time = np.ceil(np.max([80, T * 2.5])).astype(int)
    slurm_array_parallelism = 100
    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "segment",
        "slurm_gres": f"gpu:{num_gpus}",
        "slurm_mem_per_cpu": f"{gb_ram_request}G",
        "slurm_cpus_per_task": np.max([int(20 * 1.3), num_cpus]),
        "slurm_array_parallelism": slurm_array_parallelism,  # process up to 20 positions at a time
        "slurm_time": slurm_time,
        "slurm_partition": "gpu",
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    cluster = get_submitit_cluster(local)

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths, strict=True
        ):
            jobs.append(
                executor.submit(
                    process_single_position,
                    segment_data,
                    input_position_path,
                    output_position_path,
                    input_channel_indices=[list(range(C))],
                    output_channel_indices=[list(range(C_segment))],
                    num_workers=np.min([20, int(num_cpus * 0.8)]),
                    models=models,
                )
            )

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


if __name__ == "__main__":
    segment_cli()
