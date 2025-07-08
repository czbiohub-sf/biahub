import os
import subprocess
from pathlib import Path
from typing import List
import numpy as np

import click
import submitit
from iohub.ngff import open_ome_zarr

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    input_position_dirpaths,
    output_dirpath,
    sbatch_filepath_preprocess,
    sbatch_filepath_predict,
    sbatch_to_submitit,
    num_processes,
    local,
    monitor,
)
from biahub.cli.utils import create_empty_hcs_zarr


def run_viscy_preprocess(data_path: str, num_workers: int = 32, config_file: str = None, path_viscy_env: Path = None):
    cmd = (
    "module load anaconda && "
    f"conda activate {path_viscy_env} && "
    f"viscy preprocess "
    f"--data_path {data_path} "
    f"--num_workers {num_workers} "
    f"--channel_names -1 "
    f"--block_size 32"
    )
    # if config_file:
    #     cmd += f" -c {config_file}"
    # else:
    #     cmd += " --channel_names -1  --block_size 32"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


def run_viscy_predict(data_path: str, config_file: str, output_store: str, log_dir: str, path_viscy_env: Path = None):
    cmd = (
    "module load anaconda && "
    f"conda activate {path_viscy_env} && "
    f"viscy predict -c {config_file} "
    f"--data.init_args.data_path={data_path} "
    f"--trainer.callbacks+=viscy.translation.predict_writer.HCSPredictionWriter "
    f"--trainer.callbacks.output_store={output_store} "
    f"--trainer.default_root_dir={log_dir}"
    )
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


@click.command("virtual-stain")
@input_position_dirpaths()
@output_dirpath()
@sbatch_filepath_preprocess()
@sbatch_filepath_predict()
@num_processes()
@local()
@monitor()
@click.option("--path-viscy-env", default="/hpc/mydata/taylla.theodoro/anaconda/2022.05/x86_64/envs/viscy", show_default=True, help="Conda environment with VisCy installed.")
@click.option("--preprocess-config-filepath", type=str, help="Path to the VisCy preprocess config file.")
@click.option("--predict-config-filepath", type=str, help="Path to the VisCy predict config file.")
@click.option(
    "--run-mode",
    type=click.Choice(["all", "preprocess", "predict"]),
    default="all",
    help="Which VisCy stage(s) to run.",
)
def virtual_stain_cli(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    predict_config_filepath: str,
    preprocess_config_filepath: str = None,
    path_viscy_env: str = "/hpc/mydata/taylla.theodoro/anaconda/2022.05/x86_64/senvs/viscy",
    run_mode: str = "all",
    num_processes: int = 16,
    sbatch_filepath_preprocess: str = None,
    sbatch_filepath_predict: str = None,
    local: bool = False,
    monitor: bool = True,
):
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

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
    
    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=[p.parts[-3:] for p in input_position_dirpaths],
        **output_metadata,
    )

    cluster = "local" if local else "slurm"
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
            )
            job_ids_preprocess.append(job)

        slurm_dependency = f"afterok:{job.job_id}"

    job_ids_predict = []

    if run_mode in ["all", "predict"]:
        slurm_args_predict = {
            "slurm_job_name": "VS_predict",
            "slurm_mem_per_cpu": "8G",
            "slurm_cpus_per_task": num_processes,
            "slurm_array_parallelism": 100,
            "slurm_time": 8 * 60,
            "slurm_partition": "gpu",
            "slurm_gres": "gpu:1",
            "slurm_constraint": "a100|a6000|a40",
        }

        if slurm_dependency:
            slurm_args_predict["slurm_dependency"] = slurm_dependency

        if sbatch_filepath_predict:
            slurm_args_predict.update(sbatch_to_submitit(sbatch_filepath_predict))

        executor = submitit.AutoExecutor(folder=slurm_out_path / "predict", cluster=cluster)
        executor.update_parameters(**slurm_args_predict)
        click.echo(f"Submitting predict jobs with: {slurm_args_predict}")

        for input_position_path in input_position_dirpaths:
            fov = Path(*input_position_path.parts[-3:])
            log_dir = output_dirpath.parent / "logs" / "_".join(fov.parts)
            log_dir.mkdir(parents=True, exist_ok=True)
            output_fov_path = output_dirpath / fov

            job = executor.submit(
                run_viscy_predict,
                data_path=str(input_position_path),
                config_file=predict_config_filepath,
                output_store=str(output_fov_path),
                log_dir=str(log_dir),
                path_viscy_env=path_viscy_env,
            )
            job_ids_predict.append(job)

    if monitor:
        monitor_jobs(job_ids_predict + job_ids_preprocess, input_position_dirpaths)






if __name__ == "__main__":
    virtual_stain_cli()


