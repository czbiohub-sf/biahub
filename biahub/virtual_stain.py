import logging
import os
import shutil
import subprocess

from pathlib import Path

import click
import numpy as np
import submitit
import yaml

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate

from biahub.cli.monitor import monitor_jobs
from biahub.cli.option_eat_all import OptionEatAll
from biahub.cli.parsing import (
    config_filepath,
    init_only,
    local,
    monitor,
    num_processes,
    output_dirpath,
    sbatch_filepath_predict,
    sbatch_filepath_preprocess,
    sbatch_to_submitit,
)
from biahub.cli.utils import (
    estimate_resources,
    get_submitit_cluster,
    read_plate_metadata,
)

logger = logging.getLogger(__name__)


def _optional_validate_paths(
    ctx: click.Context, opt: click.Option, value: tuple | None
) -> list[Path] | None:
    if value is None or len(value) == 0:
        return None
    from natsort import natsorted

    return [p for p in map(Path, natsorted(value)) if p.is_dir()]


def run_viscy_preprocess(
    data_path: str,
    num_workers: int = 32,
    config_file: str = None,
    path_viscy_env: Path = None,
    verbose: bool = False,
):
    cmd = (
        "module load anaconda && "
        f"conda activate {path_viscy_env} && "
        f"viscy preprocess "
        f"--data_path {data_path} "
        f"--num_workers {num_workers} "
    )
    if config_file:
        cmd += f" -c {config_file}"
    else:
        cmd += "--channel_names -1  --block_size 32"
    if verbose:
        click.echo(f"Preprocess FOV: {'/'.join(Path(data_path).parts[-3:])}")
        click.echo(f"Command: {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


def run_viscy_predict(
    data_path: str,
    config_file: str,
    output_store: str,
    log_dir: str,
    path_viscy_env: Path = None,
    verbose: bool = False,
):
    os.chdir(log_dir)
    cmd = (
        "module load anaconda && "
        f"conda activate {path_viscy_env} && "
        "viscy predict "
        f'-c "{config_file}" '
        f'--data.init_args.data_path "{data_path}" '
        f"--trainer.callbacks+=viscy.translation.predict_writer.HCSPredictionWriter "
        f'--trainer.callbacks.output_store "{output_store}" '
        f'--trainer.default_root_dir "{log_dir}"'
    )
    if verbose:
        click.echo(f"Predict FOV: {'/'.join(Path(data_path).parts[-3:])}")
        click.echo(f"Command: {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


def combine_fov_zarrs_to_plate(
    fovs: list[Path],
    temp_dir: Path,
    output_dirpath: Path,
    cleanup: bool = True,
):
    for fov in fovs:
        row, col, pos = fov.parts[-3:]
        nested_fov_path = temp_dir / f"{row}_{col}_{pos}.zarr" / row / col / pos

        if not nested_fov_path.exists():
            print(f"Skipping missing: {nested_fov_path}")
            continue

        dest_path = output_dirpath / row / col / pos

        if dest_path.exists():
            shutil.rmtree(dest_path)

        print(f"Moving {nested_fov_path} → {dest_path}")
        shutil.move(str(nested_fov_path), str(dest_path))

    if cleanup:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Could not remove {temp_dir}: {e}")

    print(f"Combined all FOVs into {output_dirpath}")


def _init_output_plate(
    input_zarr: Path,
    output_zarr: Path,
    config_filepath: Path,
) -> tuple[int, int, int, int, int]:
    """Create the empty virtual-stain output plate.

    Reads target channels from the predict config, creates the output store
    with ``<channel>_prediction`` channel names, and copies per-position
    metadata from the input plate.

    Returns the (T, C_pred, Z, Y, X) output shape.
    """
    with open(config_filepath) as f:
        cfg = yaml.safe_load(f)

    target_channels = cfg["data"]["init_args"]["target_channel"]
    prediction_channels = [f"{ch}_prediction" for ch in target_channels]

    position_keys, _, shape, scale = read_plate_metadata(input_zarr)
    T, _, Z, Y, X = shape

    output_shape = (T, len(prediction_channels), Z, Y, X)

    create_empty_plate(
        store_path=output_zarr,
        position_keys=position_keys,
        channel_names=prediction_channels,
        shape=output_shape,
        scale=scale,
        version="0.5",
        dtype=np.float32,
        copy_metadata_from=input_zarr,
    )
    click.echo(
        f"Created {output_zarr} ({len(position_keys)} positions, "
        f"channels={prediction_channels})"
    )
    return output_shape


def _copy_position(
    temp_zarr: Path,
    output_zarr: Path,
    position: str,
) -> None:
    """Copy viscy prediction from a temp FOV zarr into the output plate position.

    The temp zarr is a per-position HCS plate produced by VisCy's
    HCSPredictionWriter.  This reads the data and attributes from the nested
    position, writes them into the output plate, then removes the temp zarr.
    """
    temp_position = temp_zarr / position
    output_position = output_zarr / position

    with open_ome_zarr(str(temp_position), mode="r") as src:
        src_data = np.asarray(src[0][:])
        src_attrs = dict(src.zattrs)

    with open_ome_zarr(str(output_position), mode="r+") as dst:
        dst[0][:] = src_data
        dst.zattrs.update(src_attrs)

    shutil.rmtree(temp_zarr)
    click.echo(f"Virtual stain copied: {position}")


def virtual_stain(
    input_position_dirpaths: list[str],
    output_dirpath: str,
    predict_config_filepath: str,
    path_viscy_env: str,
    preprocess_config_filepath: str = None,
    sbatch_filepath_preprocess: str = None,
    sbatch_filepath_predict: str = None,
    run_mode: str = "all",
    num_processes: int = 32,
    local: bool = False,
    monitor: bool = True,
    verbose: bool = True,
):
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    cluster = get_submitit_cluster(local)

    job_ids_preprocess = []
    slurm_dependency = None
    path_viscy_env = Path(path_viscy_env)

    if run_mode in ["all", "preprocess"]:
        slurm_args_preprocess = {
            "slurm_job_name": "VS_preprocess",
            "slurm_mem_per_cpu": "8G",
            "slurm_cpus_per_task": num_processes,
            "slurm_array_parallelism": 1,
            "slurm_time": 8 * 60,
            "slurm_partition": "cpu",
        }

        if sbatch_filepath_preprocess:
            slurm_args_preprocess.update(sbatch_to_submitit(sbatch_filepath_preprocess))

        executor = submitit.AutoExecutor(folder=slurm_out_path / "preprocess", cluster=cluster)
        executor.update_parameters(**slurm_args_preprocess)
        click.echo(f"Submitting preprocess job with: {slurm_args_preprocess}")

        plate_path = Path(input_position_dirpaths[0]).parents[2]
        with executor.batch():
            job = executor.submit(
                run_viscy_preprocess,
                data_path=str(plate_path),
                num_workers=num_processes,
                config_file=preprocess_config_filepath,
                path_viscy_env=path_viscy_env,
                verbose=verbose,
            )
            job_ids_preprocess.append(job)

        job_ids = [job.job_id for job in job_ids_preprocess]

        log_path = Path(slurm_out_path / "preprocess" / "submitit_jobs_ids.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            log_file.write("\n".join(job_ids))

        slurm_dependency = f"afterok:{job.job_id}"

    job_ids_predict = []
    job_ids_combine = []

    if run_mode in ["all", "predict"]:
        slurm_args_predict = {
            "slurm_job_name": "VS_predict",
            "slurm_mem_per_cpu": "16G",
            "slurm_cpus_per_task": 32,
            "slurm_array_parallelism": 100,
            "slurm_time": 8 * 60,
            "slurm_partition": "gpu",
            "slurm_gres": "gpu:1",
            "slurm_constraint": "a100|a6000|a40",
            "slurm_use_srun": False,
        }

        if slurm_dependency:
            slurm_args_predict["slurm_dependency"] = slurm_dependency

        if sbatch_filepath_predict:
            slurm_args_predict.update(sbatch_to_submitit(sbatch_filepath_predict))

        executor = submitit.AutoExecutor(folder=slurm_out_path / "predict", cluster=cluster)
        executor.update_parameters(**slurm_args_predict)
        click.echo(f"Submitting predict jobs with: {slurm_args_predict}")
        config_file = str(Path(predict_config_filepath).resolve())
        fovs = []

        with executor.batch():
            for input_position_path in input_position_dirpaths:
                fov = Path(*input_position_path.parts[-3:])
                fovs.append(fov)
                log_dir = (output_dirpath.parent / "logs" / "_".join(fov.parts)).resolve()
                os.makedirs(log_dir, exist_ok=True)
                data_path = str(Path(input_position_path).resolve())

                output_fov_path = (
                    output_dirpath.parent / "temp" / f"{'_'.join(fov.parts)}.zarr"
                )
                output_store = str(Path(output_fov_path).resolve())

                job = executor.submit(
                    run_viscy_predict,
                    data_path=data_path,
                    config_file=config_file,
                    output_store=output_store,
                    log_dir=log_dir,
                    path_viscy_env=path_viscy_env,
                    verbose=verbose,
                )
                job_ids_predict.append(job)

        job_ids = [job.job_id for job in job_ids_predict]

        log_path = Path(slurm_out_path / "predict" / "submitit_jobs_ids.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            log_file.write("\n".join(job_ids))

        with open_ome_zarr(input_position_dirpaths[0]) as dataset:
            T, C, Z, Y, X = dataset.data.shape
            channel_names = dataset.channel_names
            scale = dataset.scale

        output_metadata = {
            "shape": (T, len(channel_names), Z, Y, X),
            "chunks": None,
            "scale": scale,
            "channel_names": channel_names,
            "dtype": np.float32,
        }

        create_empty_plate(
            store_path=output_dirpath,
            position_keys=[p.parts[-3:] for p in input_position_dirpaths],
            **output_metadata,
        )

        slurm_args_combine = {
            "slurm_job_name": "VS_combine",
            "slurm_mem_per_cpu": "8G",
            "slurm_cpus_per_task": num_processes,
            "slurm_array_parallelism": 1,
            "slurm_time": 8 * 60,
            "slurm_partition": "cpu",
            "slurm_dependency": f"afterok:{':'.join([str(job.job_id) for job in job_ids_predict])}",
        }

        executor = submitit.AutoExecutor(folder=slurm_out_path / "combine", cluster=cluster)
        executor.update_parameters(**slurm_args_combine)
        click.echo(f"Submitting combine job with: {slurm_args_combine}")

        plate_path = Path(input_position_dirpaths[0]).parents[2]

        with executor.batch():
            job = executor.submit(
                combine_fov_zarrs_to_plate,
                fovs=fovs,
                temp_dir=output_dirpath.parent / "temp",
                output_dirpath=output_dirpath,
                cleanup=True,
            )
            job_ids_combine.append(job)

        job_ids = [job.job_id for job in job_ids_combine]

        log_path = Path(slurm_out_path / "combine" / "submitit_jobs_ids.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            log_file.write("\n".join(job_ids))

    if monitor:
        job_ids = [
            job.job_id for job in job_ids_predict + job_ids_preprocess + job_ids_combine
        ]
        monitor_jobs(job_ids, input_position_dirpaths)


@click.command("virtual-stain")
@click.option(
    "--input-position-dirpaths",
    "-i",
    required=False,
    cls=OptionEatAll,
    type=tuple,
    callback=_optional_validate_paths,
    help="Paths to input positions (required for --init and full runs).",
)
@output_dirpath()
@config_filepath()
@init_only()
@click.option(
    "--copy",
    "copy_mode",
    is_flag=True,
    default=False,
    help="Copy viscy prediction from temp zarr into output plate position.",
)
@click.option(
    "--temp-zarr",
    "-t",
    default=None,
    type=click.Path(),
    help="Path to temp FOV zarr (required for --copy).",
)
@click.option(
    "--position",
    "-p",
    default=None,
    type=str,
    help="Position key like B/3/000000 (required for --copy).",
)
@sbatch_filepath_preprocess()
@sbatch_filepath_predict()
@num_processes()
@local()
@monitor()
@click.option("--verbose", is_flag=True, default=False, help="Verbose output.")
@click.option(
    "--path-viscy-env",
    default=None,
    help="Conda environment with VisCy installed.",
)
@click.option(
    "--preprocess-config-filepath",
    type=str,
    help="Path to the VisCy preprocess config file.",
)
@click.option(
    "--run-mode",
    type=click.Choice(["all", "preprocess", "predict"]),
    default="all",
    help="Which VisCy stage(s) to run.",
)
def virtual_stain_cli(
    input_position_dirpaths: list[Path] | None,
    output_dirpath: Path,
    config_filepath: Path,
    init_only: bool = False,
    copy_mode: bool = False,
    temp_zarr: str | None = None,
    position: str | None = None,
    path_viscy_env: str | None = None,
    preprocess_config_filepath: str = None,
    run_mode: str = "all",
    num_processes: int = 32,
    sbatch_filepath_preprocess: str = None,
    sbatch_filepath_predict: str = None,
    local: bool = False,
    monitor: bool = True,
    verbose: bool = True,
):
    r"""Run VisCy virtual staining on a zarr plate.

    \b
    Initialize the output plate only (Nextflow init step):
    >>> biahub virtual-stain --init -i ./input.zarr/*/*/* -c ./predict.yml -o ./output.zarr

    \b
    Copy a single position from temp zarr (Nextflow per-position copy step):
    >>> biahub virtual-stain --copy -t ./temp/B_3_000000.zarr -o ./output.zarr -p B/3/000000 -c ./predict.yml

    \b
    Full SLURM run (preprocess + predict + combine):
    >>> biahub virtual-stain -i ./input.zarr/*/*/* -o ./output.zarr \
        -c ./predict.yml --path-viscy-env /path/to/viscy/env --run-mode all
    """
    if copy_mode:
        if temp_zarr is None:
            raise click.UsageError("--temp-zarr / -t is required when using --copy.")
        if position is None:
            raise click.UsageError("--position / -p is required when using --copy.")
        _copy_position(Path(temp_zarr), output_dirpath, position)
        return

    if not input_position_dirpaths:
        raise click.UsageError(
            "--input-position-dirpaths / -i is required for --init and full runs."
        )

    if init_only:
        input_plate = Path(input_position_dirpaths[0]).parents[2]
        output_shape = _init_output_plate(input_plate, output_dirpath, config_filepath)

        num_cpus, mem_per_cpu = estimate_resources(
            shape=output_shape, ram_multiplier=16, max_num_cpus=16
        )
        click.echo(f"RESOURCES:{num_cpus} {num_cpus * mem_per_cpu}")
        return

    if path_viscy_env is None:
        raise click.UsageError("--path-viscy-env is required for full virtual stain runs.")

    virtual_stain(
        input_position_dirpaths=input_position_dirpaths,
        output_dirpath=str(output_dirpath),
        predict_config_filepath=str(config_filepath),
        preprocess_config_filepath=preprocess_config_filepath,
        sbatch_filepath_preprocess=sbatch_filepath_preprocess,
        sbatch_filepath_predict=sbatch_filepath_predict,
        path_viscy_env=path_viscy_env,
        run_mode=run_mode,
        num_processes=num_processes,
        local=local,
        monitor=monitor,
        verbose=verbose,
    )


if __name__ == "__main__":
    virtual_stain_cli()
