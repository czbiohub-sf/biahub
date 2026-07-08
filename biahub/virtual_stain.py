import time

from pathlib import Path
from typing import Literal

import click
import numpy as np
import submitit
import torch

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate

from biahub.cli import utils
from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    echo_resources,
    get_submitit_cluster,
    resolve_ome_zarr_version,
)


def build_predict_parser():
    """Build a jsonargparse parser that mirrors the ``viscy predict`` config.

    The model and data blocks are registered as subclass arguments of VisCy's
    own ``VSUNet`` and ``HCSDataModule`` classes, so the configuration is
    validated against VisCy's actual class signatures -- the same machinery the
    ``viscy predict`` CLI uses. biahub therefore keeps no copy of the model
    schema that could fall out of date when VisCy changes upstream.

    The biahub-specific orchestration knobs (``sliding_window_step``,
    ``device``, ``output_ome_zarr_version``) and the top-level ``ckpt_path``
    are added as plain arguments. ``data.init_args.data_path`` is intentionally
    left out of the config file -- biahub injects the position path per job.
    """
    import jsonargparse

    from cytoland.engine import VSUNet
    from viscy_data.hcs import HCSDataModule

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_subclass_arguments(VSUNet, "model")
    parser.add_subclass_arguments(HCSDataModule, "data")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--sliding_window_step", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    # Output OME-Zarr version: None preserves the input store's version.
    parser.add_argument(
        "--output_ome_zarr_version",
        type=Literal["0.4", "0.5"] | None,
        default=None,
    )
    return parser


def load_predict_config(config_filepath: Path, data_path: Path):
    """Validate a VisCy predict config and inject the per-position ``data_path``.

    Parameters
    ----------
    config_filepath : Path
        Path to the VisCy-style predict YAML config.
    data_path : Path
        Position path injected as ``data.init_args.data_path`` (biahub supplies
        this per job; the first position is used for submit-time validation).

    Returns
    -------
    tuple
        ``(parser, cfg)`` where ``cfg`` is the validated config namespace.
    """
    parser = build_predict_parser()
    cfg = parser.parse_args(
        [
            "--config",
            str(config_filepath),
            "--data.init_args.data_path",
            str(data_path),
        ]
    )
    return parser, cfg


def virtual_stain_position(
    config_filepath: Path,
    input_position_path: Path,
    output_position_path: Path,
):
    """Run cytoland virtual staining on a single position, looping over time.

    Inference is GPU-bound, so timepoints are processed serially in a ``for``
    loop rather than in parallel. Each timepoint volume is normalized with the
    config's ``NormalizeSampled`` transforms (using the precomputed statistics
    in the store) and passed through
    ``AugmentedPredictionVSUNet.predict_sliding_windows`` -- the same linear
    feathering blend used by the ``viscy predict`` ``HCSPredictionWriter``.

    Normalization and test-time augmentation are delegated to VisCy
    (``NormalizeSampled`` / ``read_norm_meta`` / ``with_rotation_tta``) so the
    behavior stays in sync with ``viscy predict``.

    Parameters
    ----------
    config_filepath : Path
        Path to the VisCy-style predict config.
    input_position_path : Path
        Input position (label-free source) to virtually stain.
    output_position_path : Path
        Output position to write predictions into.
    """
    from cytoland.engine import AugmentedPredictionVSUNet
    from monai.transforms import Compose
    from viscy_data import read_norm_meta

    position_name = "/".join(Path(input_position_path).parts[-3:])
    position_start = time.perf_counter()
    parser, cfg = load_predict_config(config_filepath, input_position_path)

    device = torch.device(
        cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    click.echo(f"[{position_name}] Starting virtual staining on device '{device}'")

    # Instantiate the VSUNet and the data module from VisCy's own classes.
    # Route the top-level ckpt_path into the model's init args so VSUNet loads
    # the weights itself: VSUNet.__init__ extracts the Lightning checkpoint's
    # "state_dict" exactly as `viscy predict` does. Delegating keeps biahub free
    # of both the model schema and any assumption about the checkpoint layout.
    cfg.model.init_args.ckpt_path = str(cfg.ckpt_path)
    instances = parser.instantiate(cfg)
    vsunet = instances.model
    vsunet.eval().to(device)
    click.echo(f"[{position_name}] Loaded checkpoint: {cfg.ckpt_path}")

    # Test-time augmentation: reuse VisCy's rotation-TTA helper, which is
    # correct for non-square FOVs. Defaults follow the config's model flags.
    use_tta = bool(getattr(cfg.model.init_args, "test_time_augmentations", False))
    tta_type = getattr(cfg.model.init_args, "tta_type", "mean")
    reduction = "median" if tta_type == "median" else "mean"
    if use_tta:
        predictor = AugmentedPredictionVSUNet.with_rotation_tta(
            vsunet.model, reduction=reduction
        )
        click.echo(
            f"[{position_name}] Test-time augmentation: on ({reduction} over 4 rotations)"
        )
    else:
        predictor = AugmentedPredictionVSUNet(model=vsunet.model)
        click.echo(f"[{position_name}] Test-time augmentation: off")
    predictor = predictor.eval().to(device)

    # Normalization: reuse the configured NormalizeSampled transforms directly.
    normalize = Compose(instances.data.normalizations)

    source_channel = cfg.data.init_args.source_channel
    source_channel = source_channel[0] if isinstance(source_channel, list) else source_channel
    target_channels = cfg.data.init_args.target_channel
    target_channels = (
        [target_channels] if isinstance(target_channels, str) else target_channels
    )
    out_channel = len(target_channels)
    step = cfg.sliding_window_step

    with open_ome_zarr(str(input_position_path), mode="r") as input_dataset:
        T, _, Z, Y, X = input_dataset.data.shape
        channel_index = input_dataset.channel_names.index(source_channel)
        # Precomputed statistics written by `viscy preprocess`.
        norm_meta = read_norm_meta(input_dataset)
        if norm_meta is None or source_channel not in norm_meta:
            raise RuntimeError(
                f"Normalization statistics for channel '{source_channel}' were not "
                f"found in the input store. Run `viscy preprocess` on the input "
                f"dataset before virtual staining."
            )

        # Depth of the Z-window fed to the model. In VisCy's own
        # `predict_sliding_windows`, `model.out_stack_depth` is the input window
        # depth: each window is `x[:, :, start:start + out_stack_depth]`. We
        # mirror that here purely to log the window count (the value matches
        # `len(range(0, Z - out_stack_depth + 1, step))` exactly). "?" when the
        # model exposes no such attribute, and 0 when the volume is too shallow
        # for even one window (predict_sliding_windows then raises).
        z_window_depth = getattr(vsunet.model, "out_stack_depth", None)
        n_windows = max(0, (Z - z_window_depth) // step + 1) if z_window_depth else "?"
        click.echo(
            f"[{position_name}] {T} timepoints, volume (Z,Y,X)=({Z},{Y},{X}), "
            f"'{source_channel}' -> {target_channels}, "
            f"{n_windows} sliding windows/timepoint (step={step})"
        )

        with open_ome_zarr(str(output_position_path), mode="r+") as output_dataset:
            for t in range(T):
                t_start = time.perf_counter()
                # (1, 1, Z, Y, X) source volume for this timepoint
                volume = np.asarray(
                    input_dataset.data[t : t + 1, channel_index : channel_index + 1]
                )
                source = torch.from_numpy(volume).float().to(device)
                # Apply VisCy's configured normalization via a sample dict.
                sample = normalize({source_channel: source, "norm_meta": norm_meta})
                source = sample[source_channel]
                with torch.inference_mode():
                    prediction = predictor.predict_sliding_windows(
                        source, out_channel=out_channel, step=step
                    )
                output_dataset.data[t] = prediction[0].cpu().numpy()
                click.echo(
                    f"[{position_name}] timepoint {t + 1}/{T} done "
                    f"({time.perf_counter() - t_start:.1f}s)"
                )

    click.echo(
        f"[{position_name}] Completed {T} timepoints in "
        f"{time.perf_counter() - position_start:.1f}s -> {output_position_path}"
    )


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    target_channels: list[str],
    output_ome_zarr_version: str | None = None,
) -> tuple[int, int, int, int, int]:
    """Create (or extend) the empty virtual-stain output plate.

    The output mirrors the input geometry but contains only the predicted
    virtual-stain channels. ``create_empty_plate`` is idempotent, so this is
    safe to call from both the orchestrator and per-position runs.

    ``output_ome_zarr_version`` dictates the output store's OME-Zarr version;
    when None the input store's version is preserved.

    Returns the input ``(T, C, Z, Y, X)`` shape.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        input_shape = input_dataset.data.shape
        scale = input_dataset.scale
    T, C, Z, Y, X = input_shape

    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[Path(p).parts[-3:] for p in input_position_dirpaths],
        channel_names=target_channels,
        shape=(T, len(target_channels), Z, Y, X),
        scale=scale,
        version=resolve_ome_zarr_version(input_position_dirpaths[0], output_ome_zarr_version),
    )

    return (T, C, Z, Y, X)


def virtual_stain(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
):
    """Run cytoland virtual staining on a plate, one GPU job per position.

    Opens the input store, creates an output store containing only the
    predicted virtual-stain channels, and dispatches one job per position. Each
    job runs ``virtual_stain_position``, which loops over timepoints on a single
    GPU.

    Parameters
    ----------
    input_position_dirpaths : list[str]
        Input position directory paths.
    config_filepath : Path
        Path to the VisCy-style predict config.
    output_dirpath : str
        Output plate path.
    sbatch_filepath : str, optional
        SBATCH file overriding default SLURM parameters.
    cluster : str, optional
        Execution cluster: 'slurm' submits to a Slurm cluster, 'local' runs jobs
        as subprocesses on this machine, 'debug' runs jobs in-process in the
        foreground.
    monitor : bool, optional
        Monitor submitted jobs.
    init_only : bool, optional
        Only initialize the output store and exit; skip per-position processing.
    """
    # The 'local' cluster launches every position as a concurrent subprocess
    # (submitit's LocalExecutor ignores `slurm_array_parallelism`, so there is no
    # concurrency cap) and biahub does not pin per-job GPUs, so all jobs target
    # cuda:0. With more than one position that oversubscribes a single GPU. Use
    # 'slurm' for multi-GPU parallelism or 'debug' to run positions one at a time.
    if cluster == "local" and len(input_position_dirpaths) > 1:
        raise ValueError(
            f"cluster='local' cannot run {len(input_position_dirpaths)} positions: "
            "local jobs run concurrently without a parallelism cap and all target "
            "the same GPU. Use cluster='slurm' for multi-GPU parallelism, or "
            "cluster='debug' to run positions sequentially on this machine."
        )

    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    # Validate the config against VisCy's schema up front (using the first
    # position for data_path) and read the predicted channel names.
    _, cfg = load_predict_config(config_filepath, input_position_dirpaths[0])
    target_channels = cfg.data.init_args.target_channel
    target_channels = (
        [target_channels] if isinstance(target_channels, str) else target_channels
    )

    input_shape = _init_output_plate(
        input_position_dirpaths,
        output_dirpath,
        target_channels,
        cfg.output_ome_zarr_version,
    )

    # Timepoints are processed sequentially on a single GPU, so CPU and RAM
    # needs are fixed (independent of dataset size); only wall-time scales with
    # the data (see the T*Z budget below).
    num_cpus, mem_gb = 16, 64
    # Wall-clock budget (minutes) for GPU prediction. Each timepoint runs ~Z
    # sliding windows along Z; this is a GPU step so time scales with T*Z, not
    # with CPU count. Measured ~2 windows/s (0.5 s/window) with median TTA on
    # this model from a completed run's tqdm; budget 1.0 s/window (~2x margin
    # for slower GPUs and larger FOVs) with a 60-minute floor. Computed before
    # the init_only return so --init emits it for the Nextflow pipeline.
    T, Z = input_shape[0], input_shape[2]
    seconds_per_window = 1.0
    time_minutes = int(np.ceil(max(60, T * Z * seconds_per_window / 60)))
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        click.echo(f"Initialized {output_dirpath} ({len(input_position_dirpaths)} positions)")
        return

    output_position_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "virtual-stain",
        "slurm_gres": "gpu:1",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 20,  # process up to 20 positions at a time
        "slurm_time": time_minutes,
        "slurm_partition": "gpu",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths, strict=True
        ):
            jobs.append(
                executor.submit(
                    virtual_stain_position,
                    config_filepath,
                    input_position_path,
                    output_position_path,
                )
            )

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    slurm_out_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a
    # DebugJob but execution only happens when .wait()/.done()/.result() is
    # called. Run each one in the foreground and stream progress; monitor's
    # async polling UI is pointless against synchronous in-process jobs.
    if resolved_cluster == "debug":
        for job, path in zip(jobs, input_position_dirpaths, strict=True):
            job.wait()
            click.echo(f"Virtual staining complete: {path}")
        return

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("virtual-stain")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
def virtual_stain_cli(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
):
    """Virtually stain a label-free dataset using a cytoland (VisCy) model.

    Runs cytoland's ``predict_sliding_windows`` per position, looping over
    timepoints on a single GPU. The config is a ``viscy predict`` YAML and is
    validated against VisCy's own model/data classes.

    \b
    SLURM fan-out of positions across a whole plate:
    >>> biahub virtual-stain -i ./input.zarr/*/*/* -c ./virtual_stain_params.yml -o ./output.zarr

    \b
    Initialize the output plate only (e.g. before running per-position Nextflow workers):
    >>> biahub virtual-stain --init -i ./input.zarr/*/*/* -c ./virtual_stain_params.yml -o ./output.zarr

    \b
    In-process run of a single position (e.g. from a Nextflow worker):
    >>> biahub virtual-stain --cluster debug -i ./input.zarr/A/1/0 -c ./virtual_stain_params.yml -o ./output.zarr
    """  # noqa: D301
    virtual_stain(
        input_position_dirpaths=input_position_dirpaths,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
    )


if __name__ == "__main__":
    virtual_stain_cli()
