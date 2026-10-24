from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_labels, process_single_position

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
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
    apply_sbatch_overrides,
    echo_resources,
    estimate_resources,
    get_submitit_cluster,
    local_executor_parameters,
)
from biahub.utils.config import settings_fingerprint, yaml_to_model
from biahub.utils.ngff import label_layout

if TYPE_CHECKING:
    import torch

# Provenance key on each label image: the segment settings as written.
PROVENANCE_KEY = "biahub-segment"


@dataclass(frozen=True)
class ResolvedModel:
    """A segmentation model resolved against one input plate (indices instead of names)."""

    name: str
    pretrained_model: str
    channel_indices: list[int]
    eval_args: dict[str, Any]
    is_3d: bool
    # (function, channel index, kwargs), applied in order
    preprocessing: list[tuple[Callable, int, dict[str, Any]]] = field(default_factory=list)


def resolve_models(
    settings: SegmentationSettings,
    channel_names: list[str],
    scale: tuple[float, ...],
    z_size: int,
) -> list[ResolvedModel]:
    """Resolve each configured model against the input plate, without touching ``settings``.

    Channel names become dataset indices and 3D models get the dataset anisotropy unless
    the config sets one. The models' dimensionality must match the input: 2D models need
    a Z=1 plate (see `biahub project`), 3D models a stack. ``settings`` is left exactly as
    written, so it can be recorded as provenance.
    """

    def index(name: str, what: str) -> int:
        if name not in channel_names:
            raise ValueError(
                f"{what} channel {name!r} is not in the input channels {channel_names}."
            )
        return channel_names.index(name)

    if settings.is_3d and z_size == 1:
        raise ValueError(
            "The input has one Z-plane but the models segment in 3D (eval_args.do_3D or "
            "stitch_threshold); remove those arguments to segment in 2D."
        )
    if not settings.is_3d and z_size > 1:
        raise ValueError(
            f"The input is 3D (Z={z_size}) but the models segment in 2D. Project the plate "
            "with `biahub project` and segment the projected (Z=1) plate, or set "
            "eval_args.do_3D: true to segment the stack."
        )

    resolved = []
    for name, model in settings.models.items():
        eval_args = dict(model.eval_args)
        if model.is_3d and eval_args.get("anisotropy") is None:
            eval_args["anisotropy"] = scale[-3] / scale[-1]
        resolved.append(
            ResolvedModel(
                name=name,
                pretrained_model=model.pretrained_model,
                channel_indices=[index(c, f"Model {name}:") for c in model.channels],
                eval_args=eval_args,
                is_3d=model.is_3d,
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
# runs one _segment_czyx call per timepoint; without the cache every frame reloaded the
# ~1.2 GB checkpoint.
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _cached_model(pretrained_model: str, device: "torch.device") -> Any:
    key = (pretrained_model, str(device))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_cellpose_model(pretrained_model, device)
    return _MODEL_CACHE[key]


def _segment_czyx(
    czyx_data: np.ndarray,
    models: list[ResolvedModel],
    gpu: bool = True,
) -> np.ndarray:
    """Segment a CZYX image with each resolved cellpose model.

    Returns an array of shape (n_models, Z, Y, X); Z is 1 for 2D models.
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
        # plane of the Z=1 input and a 3D model the (C, Z, Y, X) stack. It also ignores
        # `channels` and keeps the first 3 channels it is given, so pass only the
        # configured ones.
        cellpose_model = _cached_model(model.pretrained_model, device)
        if model.is_3d:
            image, z_axis = czyx_data[model.channel_indices], 1
        else:
            image, z_axis = czyx_data[model.channel_indices, 0], None
        segmentation, _, _ = cellpose_model.eval(
            image, channel_axis=0, z_axis=z_axis, **model.eval_args
        )
        # Cellpose squeezes singleton axes; restore the (Z, Y, X) shape the output
        # plate expects (Z is 1 for a 2D model).
        segmentation = np.reshape(
            segmentation, image.shape[1:] if model.is_3d else (1, *image.shape[-2:])
        )
        czyx_segmentation.append(segmentation)
    return np.stack(czyx_segmentation, axis=0)


def _check_existing_label(
    position, name: str, model_settings: dict, shape_tczyx: tuple
) -> None:
    """Refuse to write model ``name`` into a label image made with other settings.

    Segmentation strategies accumulate in a plate as differently named label images, so
    a label image is only ever rewritten by the model that made it. A different shape
    (a model renamed from 3D to 2D, say) or different recorded model settings mean
    another strategy owns the name.
    """
    if name not in position.label_names():
        return
    label = position.get_label(name)
    existing_shape = tuple(label["0"].shape)
    if existing_shape != shape_tczyx:
        raise ValueError(
            f"{position.zgroup.path} already has a label image {name!r} of shape "
            f"{existing_shape}, but a label of this image is {shape_tczyx} (T, 1, Z, Y, X). Rename the "
            "model or remove the old label image."
        )
    recorded = dict(label.zattrs).get(PROVENANCE_KEY, {}).get("models", {}).get(name)
    if recorded is not None and recorded != model_settings:
        raise ValueError(
            f"{position.zgroup.path} already has a label image {name!r} segmented with "
            f"other settings:\n  recorded: {recorded}\n  this config: {model_settings}\n"
            "Give this strategy its own model name to keep both, or remove the old "
            "label image to replace it."
        )


def _init_label_images(
    input_position_dirpaths: list[Path],
    settings: SegmentationSettings,
) -> tuple[tuple[int, int, int, int, int], list[ResolvedModel]]:
    """Create the empty ``labels/<model>`` images in the input positions.

    One label image per model, named after it, with the image's shape (its Z: one
    plane for a projected plate). Existing label images are left untouched, so both
    --init and per-position workers can call this; one made by another strategy (other
    shape or settings under the same name, see :func:`_check_existing_label`) is
    refused. The settings are recorded on each label image exactly as written (names,
    not indices).

    Returns the input (T, C, Z, Y, X) shape and the resolved models.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        scale = input_dataset.scale
        channel_names = input_dataset.channel_names

    models = resolve_models(settings, channel_names, scale, Z)
    label_names = [m.name for m in models]
    recorded = settings.model_dump(mode="json")

    for position_dirpath in input_position_dirpaths:
        with open_ome_zarr(str(position_dirpath), mode="r") as position:
            for name in label_names:
                _check_existing_label(
                    position, name, recorded["models"][name], (T, 1, Z, Y, X)
                )

    _, layout = label_layout(input_position_dirpaths[0])
    create_empty_labels(
        Path(input_position_dirpaths[0]).parents[2],
        [Path(p).parts[-3:] for p in input_position_dirpaths],
        label_names,
        label_metadata={PROVENANCE_KEY: recorded},
        **layout,
    )
    return (T, C, Z, Y, X), models


def _resume_token(settings: SegmentationSettings, model: str) -> str:
    """Fingerprint the settings and the model whose label image is being written.

    Each model has its own label image, so the model order does not matter; a changed
    setting does, and makes a resumed run recompute.
    """
    return settings_fingerprint({"settings": settings.model_dump(mode="json"), "model": model})


def _segment_position(
    input_position_path: Path,
    models: list[ResolvedModel],
    n_channels: int,
    settings: SegmentationSettings,
    resume: bool,
) -> None:
    """Segment one position into its own label images, one per model.

    Each model gets one ``process_single_position`` call writing the label image named
    after it (iohub's ``output_label``); the cellpose model cache makes the repeated
    calls free.
    """
    for model in models:
        process_single_position(
            _segment_czyx,
            input_position_path,
            input_position_path,
            input_channel_indices=[list(range(n_channels))],
            output_label=model.name,
            # One process per position: timepoints share the loaded model and
            # the GPU, which is the bottleneck.
            num_workers=1,
            resume=resume,
            resume_token=_resume_token(settings, model.name),
            models=[model],
        )


def segment(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
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
        or "input.zarr/*/*/*". The labels are written into these positions as OME-NGFF
        label images, ``labels/<model>``, which is what `biahub track` reads.
    config_filepath : Path
        Path to the segmentation YAML configuration file.
    sbatch_filepath : str, optional
        SBATCH filepath that contains slurm parameters to overwrite defaults.
    cluster : str, optional
        Execution cluster: 'slurm' submits to a Slurm cluster, 'local' runs jobs as
        subprocesses on this machine, 'debug' runs jobs in-process in the foreground.
    monitor : bool, optional
        Monitor of submitted SLURM jobs.
    init_only : bool, optional
        Only create the empty label images (and check/warm the models) and exit.
    resume : bool, optional
        Skip the timepoints a previous attempt already finished; see
        ``iohub.ngff.utils.process_single_position``.

    Notes
    -----
    Segmentation strategies accumulate in a plate as differently named label images
    (the model names); a name is only rewritten by the model that made it (see
    :func:`_check_existing_label`). The SLURM logs go to ``./slurm_output``.
    """
    # Not $TMPDIR: submitit's folder must be visible from the compute nodes.
    slurm_out_path = Path("slurm_output").absolute()

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    if not init_only and resolved_cluster == "debug":
        # An in-process worker (Nextflow) segments in this process, and validating the
        # settings imports cellpose.models, which fixes the weights directory: stage first
        # so the checkpoint is read from node-local scratch, not NFS. Never on a submitting
        # node: the staged path would be exported to SLURM/local workers, which stage their
        # own copy inside the job (_segment_czyx).
        stage_cellpose_weights()
    settings = yaml_to_model(config_filepath, SegmentationSettings)
    if init_only:
        # Fail on a bad model name before scaffolding anything.
        for name in dict.fromkeys(m.pretrained_model for m in settings.models.values()):
            check_cellpose_model_name(name)
    (T, C, Z, Y, X), models = _init_label_images(input_position_dirpaths, settings)

    # Calibrated on a real 2D run (2026_04_28 SEC61B, 67 x 1664 x 1193, 6 input channels,
    # cpsam_v2 on one GPU): ~5 s per frame (6-10 min per position, incl. weight staging) and
    # 6.1 GB peak RSS with one busy CPU. Time: 0.2 min per 2D frame per model (~2x margin);
    # 3D cellpose runs per plane in 3 orientations and has not been measured, so it keeps the
    # earlier conservative 2.5 min per frame. CPUs and RAM (the input timepoint, C volumes,
    # plus cellpose buffers, x up to 4 CPUs) deliberately keep a margin over the 2D run,
    # which also covers the unmeasured 3D path (~20-30 GB for an A549 plate).
    is_2d = not settings.is_3d
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
        # --init runs once on the head node: populate the shared weights cache once, rather
        # than N workers racing for it. Workers must not (staging needs cellpose unimported).
        for name in dict.fromkeys(m.pretrained_model for m in models):
            warm_cellpose_weights(name)
        click.echo(
            f"Created labels/{{{', '.join(m.name for m in models)}}} in "
            f"{Path(input_position_dirpaths[0]).parents[2]} "
            f"({len(input_position_dirpaths)} positions)"
        )
        return

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
        slurm_args = apply_sbatch_overrides(slurm_args, sbatch_to_submitit(sbatch_filepath))

    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    if resolved_cluster == "local":
        # Explicit --sbatch-filepath values (e.g. '#LOCAL --timeout-min') win over
        # the computed local executor parameters.
        slurm_args = {**local_executor_parameters(time_minutes), **slurm_args}
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path in input_position_dirpaths:
            jobs.append(
                executor.submit(
                    _segment_position,
                    input_position_path,
                    models=models,
                    n_channels=C,
                    settings=settings,
                    resume=resume,
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
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@resume()
def segment_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    resume: bool = False,
):
    """Segment positions with cellpose 4 models configured in a YAML file.

    Writes one label image per model into each -i position (labels/<model>), where
    `biahub track` reads them. Strategies accumulate as differently named models; a
    name is only rewritten by the model that made it.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub segment -i ./input.zarr/*/*/* -c ./segment.yml

    \b
    Create the empty label images only (Nextflow init step):
    >>> biahub segment --init -i ./input.zarr/*/*/* -c ./segment.yml

    \b
    In-process run of a single position (Nextflow per-position worker):
    >>> biahub segment --cluster debug -i ./input.zarr/B/3/000000 -c ./segment.yml --resume
    """  # noqa: D301
    segment(
        input_position_dirpaths=input_position_dirpaths,
        config_filepath=config_filepath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
        resume=resume,
    )


if __name__ == "__main__":
    segment_cli()
