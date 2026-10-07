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
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    resume,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.settings import SegmentationSettings
from biahub.utils.cellpose import (
    cellpose_device,
    check_cellpose_model_name,
    load_cellpose_model,
    stage_cellpose_weights,
    warm_cellpose_weights,
)
from biahub.utils.cluster import (
    echo_resources,
    estimate_resources,
    get_submitit_cluster,
    gpu_executor_parameters,
)
from biahub.utils.config import settings_fingerprint, yaml_to_model
from biahub.utils.ngff import (
    PROVENANCE_METADATA_KEYS,
    get_output_paths,
    resolve_ome_zarr_version,
)


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


# Loaded cellpose models of this process, keyed by (pretrained_model, device). A position
# runs one segment_data call per timepoint; without the cache every frame reloaded the
# ~1.2 GB checkpoint.
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _cached_model(pretrained_model: str, device):
    key = (pretrained_model, str(device))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_cellpose_model(pretrained_model, device)
    return _MODEL_CACHE[key]


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
    # Stage weights on node-local scratch before cellpose is first imported (no-op after).
    stage_cellpose_weights()

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
        cellpose_model = _cached_model(model.pretrained_model, device)
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


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    settings: SegmentationSettings,
) -> tuple[tuple[int, int, int, int, int], list[ResolvedModel]]:
    """Create (or extend) the empty label plate and resolve the models against the input.

    create_empty_plate is idempotent: re-running with the same positions is a no-op and
    new positions are appended, so both --init and per-position workers can call it.
    The settings are recorded as provenance exactly as written (names, not indices).

    Returns the input (T, C, Z, Y, X) shape and the resolved models.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        scale = input_dataset.scale
        channel_names = input_dataset.channel_names

    models = resolve_models(settings, channel_names, scale, Z)
    Z_out = 1 if models[0].z_slice_2D is not None else Z

    input_plate = Path(input_position_dirpaths[0]).parents[2]
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[Path(p).parts[-3:] for p in input_position_dirpaths],
        channel_names=[m.name + "_labels" for m in models],
        shape=(T, len(models), Z_out, Y, X),
        dtype=np.uint32,
        scale=scale,
        version=resolve_ome_zarr_version(
            input_position_dirpaths[0], settings.output_ome_zarr_version
        ),
        metadata_sources=input_plate,
        metadata_keys=PROVENANCE_METADATA_KEYS,
        extra_metadata={"biahub-segment": settings.model_dump(mode="json")},
    )
    return (T, C, Z, Y, X), models


def segment(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
    resume: bool = False,
):
    """Segment positions with cellpose 4 models (one GPU job per position).

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to input positions, for example: "input.zarr/0/0/0", "input.zarr/0/0/[0-9]",
        or "input.zarr/*/*/*".
    config_filepath : Path
        Path to the segmentation YAML configuration file.
    output_dirpath : Path
        Path to the "output.zarr" label plate.
    sbatch_filepath : str, optional
        SBATCH filepath that contains slurm parameters to overwrite defaults.
    cluster : str, optional
        Execution cluster: 'slurm' submits to a Slurm cluster, 'local' runs jobs as
        subprocesses on this machine, 'debug' runs jobs in-process in the foreground.
    monitor : bool, optional
        Monitor of submitted SLURM jobs.
    init_only : bool, optional
        Only initialize the output store (and check/warm the models) and exit.
    resume : bool, optional
        Skip the (time, channel) units a previous attempt already finished; see
        ``iohub.ngff.utils.process_single_position``.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    if not init_only:
        # Validating the settings imports cellpose.models (eval_args are checked against
        # its signature), which fixes the weights directory. Stage first, so an in-process
        # worker (--cluster debug) reads the checkpoint from node-local scratch, not NFS.
        stage_cellpose_weights()
    settings = yaml_to_model(config_filepath, SegmentationSettings)
    if init_only:
        # Fail on a bad model name before scaffolding anything.
        for name in dict.fromkeys(m.pretrained_model for m in settings.models.values()):
            check_cellpose_model_name(name)
    (T, C, Z, Y, X), models = _init_output_plate(
        input_position_dirpaths, output_dirpath, settings
    )

    # Calibrated on a real 2D run (2026_04_28 SEC61B, 67 x 1664 x 1193, 6 input channels,
    # cpsam_v2 on one GPU): ~5 s per frame (6 min per position) and 6.1 GB peak RSS with one
    # busy CPU. RAM: the input timepoint (C volumes) plus cellpose buffers. Time: 0.2 min per
    # 2D frame per model (~2x margin); 3D cellpose runs per plane in 3 orientations and has
    # not been measured, so it keeps the earlier conservative 2.5 min per frame.
    is_2d = models[0].z_slice_2D is not None
    time_minutes, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(T, len(models), Z, Y, X),
        ram_multiplier=C + 4,
        time_multiplier=0.2 if is_2d else 2.5,
        max_num_cpus=4,
        min_time_minutes=30 if is_2d else 80,
    )
    mem_gb = num_cpus * gb_ram_per_cpu
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        # --init runs once on the head node: fail on a bad model name here, not in every
        # worker, and populate the shared weights cache once. Workers must not do this
        # (importing cellpose would fix the weights directory before staging).
        for name in dict.fromkeys(m.pretrained_model for m in models):
            warm_cellpose_weights(name)
        click.echo(f"Initialized {output_dirpath} ({len(input_position_dirpaths)} positions)")
        return

    output_position_paths = get_output_paths(input_position_dirpaths, output_dirpath)

    slurm_args = {
        "slurm_job_name": "segment",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": time_minutes,
        # cellpose runs on the GPU; the non-preemptible `gpu` partition keeps long
        # per-position jobs from being evicted. Override via --sbatch-filepath.
        "slurm_partition": "gpu",
        "slurm_gpus_per_node": 1,
        "slurm_use_srun": False,
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(
        **slurm_args, **gpu_executor_parameters(resolved_cluster, time_minutes)
    )

    click.echo("Submitting jobs...")
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
                    output_channel_indices=[list(range(len(models)))],
                    # One process per position: timepoints share the loaded model and
                    # the GPU, which is the bottleneck.
                    num_workers=1,
                    resume=resume,
                    resume_token=settings_fingerprint(settings),
                    models=models,
                )
            )

    job_ids = [job.job_id for job in jobs]
    slurm_out_path.mkdir(exist_ok=True)
    with (slurm_out_path / "submitit_jobs_ids.log").open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a DebugJob but
    # execution only happens on .wait()/.done()/.result(). On the Nextflow path
    # (--cluster debug) run each position in the foreground.
    if resolved_cluster == "debug":
        for job, path in zip(jobs, input_position_dirpaths, strict=True):
            job.wait()
            click.echo(f"Segmentation complete: {path}")
        return

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("segment")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@resume()
def segment_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    resume: bool = False,
):
    """Segment positions with cellpose 4 models configured in a YAML file.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub segment -i ./input.zarr/*/*/* -c ./segment.yml -o ./segment.zarr

    \b
    Initialize the output plate only (Nextflow init step):
    >>> biahub segment --init -i ./input.zarr/*/*/* -c ./segment.yml -o ./segment.zarr

    \b
    In-process run of a single position (Nextflow per-position worker):
    >>> biahub segment --cluster debug -i ./input.zarr/B/3/000000 -c ./segment.yml \\
        -o ./segment.zarr --resume
    """  # noqa: D301
    segment(
        input_position_dirpaths=input_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
        resume=resume,
    )


if __name__ == "__main__":
    segment_cli()
