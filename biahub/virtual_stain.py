import os
import shutil
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


def run_viscy_preprocess(data_path: str, num_workers: int = 32, config_file: str = None, path_viscy_env: Path = None, verbose: bool = False):
    
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



# CURRENT_DIR=$(pwd)

# INPUT_DATASET=$(pwd)/../0-reconstruct/${DATASET}.zarr
# OUTPUT_FOLDER=$(pwd)/${DATASET}
# CONFIG_FILE=$(pwd)/predict.yml

# positions=($INPUT_DATASET/*/*/*)

# mkdir -p "$OUTPUT_FOLDER"
# ARRAY_JOB_ID=$(sbatch --parsable --array=0-$((${#positions[@]}-1))%36 predict_slurm.sh $INPUT_DATASET $OUTPUT_FOLDER $CONFIG_FILE)

# # Merge after the previous job is done
# sbatch --parsable --dependency=afterok:$ARRAY_JOB_ID ./combine.sh "$OUTPUT_FOLDER" "$INPUT_DATASET" "${OUTPUT_FOLDER}.zarr"

# positions=()
# for p in $1/*/*/*; do
#     positions+=($p)
# done

# log_dir="$(pwd)/logs/$SLURM_ARRAY_TASK_ID"
# mkdir -p $log_dir
# cd $log_dir

# cat $3

# viscy predict \
#     -c $3 \
#     --data.init_args.data_path ${positions[$SLURM_ARRAY_TASK_ID]} \
#     --trainer.callbacks+=viscy.translation.predict_writer.HCSPredictionWriter \
#     --trainer.callbacks.output_store=$2/$SLURM_ARRAY_TASK_ID.zarr \
#     --trainer.default_root_dir=$log_dir


def run_viscy_predict(data_path: str, config_file: str, output_store: str, log_dir: str, path_viscy_env: Path = None, verbose: bool = False):
    # cd to the log_dir
    os.chdir(log_dir)


    # Compose the shell command
    cmd = (
        "module load anaconda && "
        f"conda activate {path_viscy_env} && "
        "viscy predict "
        f"-c \"{config_file}\" "
        f"--data.init_args.data_path \"{data_path}\" "
        f"--trainer.callbacks+=viscy.translation.predict_writer.HCSPredictionWriter "
        f"--trainer.callbacks.output_store \"{output_store}\" "
        f"--trainer.default_root_dir \"{log_dir}\""
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
    """
    Combine VisCy-predicted FOV Zarrs (in temp) into a single HCS plate Zarr by moving.

    Parameters
    ----------
    fovs : list of Path
        Original FOV paths (used to extract B/1/000000).
    temp_dir : Path
        Directory containing the individual .zarr folders (each named like B_1_000000.zarr).
    output_dirpath : Path
        The plate-level HCS Zarr to merge into.
    cleanup : bool
        Whether to delete the moved files afterwards. Default True.
    """
    for fov in fovs:

        row, col, pos = fov.parts[-3:]
        nested_fov_path = temp_dir / f"{row}_{col}_{pos}.zarr" / row / col / pos

        if not nested_fov_path.exists():
            print(f"Skipping missing: {nested_fov_path}")
            continue

        dest_path = output_dirpath / row / col / pos
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Moving {nested_fov_path} → {dest_path}")
        shutil.move(str(nested_fov_path), str(dest_path))

        # Optionally remove the full temp zarr folder if it's now empty
        if cleanup:
            temp_zarr_dir = temp_dir / f"{row}_{col}_{pos}.zarr"
            try:
                shutil.rmtree(temp_zarr_dir)
            except Exception as e:
                print(f"Could not remove {temp_zarr_dir}: {e}")

    print(f"Combined all FOVs into {output_dirpath}")

@click.command("virtual-stain")
@input_position_dirpaths()
@output_dirpath()
@sbatch_filepath_preprocess()
@sbatch_filepath_predict()
@num_processes()
@local()
@monitor()
@click.option("--verbose", is_flag=True, default=False, help="Verbose output.")
@click.option(
    "--path-viscy-env",
    default="/hpc/mydata/taylla.theodoro/anaconda/2022.05/x86_64/envs/viscy",
    show_default=True,
    help="Conda environment with VisCy installed.",
)
@click.option(
    "--preprocess-config-filepath",
    type=str,
    help="Path to the VisCy preprocess config file.",
)
@click.option(
    "--predict-config-filepath",
    type=str,
    help="Path to the VisCy predict config file.",
)
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
    path_viscy_env: str = "/hpc/mydata/taylla.theodoro/anaconda/2022.05/x86_64/envs/viscy",
    run_mode: str = "all",
    num_processes: int = 16,
    sbatch_filepath_preprocess: str = None,
    sbatch_filepath_predict: str = None,
    local: bool = False,
    monitor: bool = True,
    verbose: bool = True,
):
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

   

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
                verbose=verbose,
            )
            job_ids_preprocess.append(job)

        job_ids = [job.job_id for job in job_ids_preprocess]  # Access job IDs after batch submission

        log_path = Path(slurm_out_path / "preprocess" / "submitit_jobs_ids.log")
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
        fovs =[]
        for input_position_path in input_position_dirpaths:
            fov = Path(*input_position_path.parts[-3:])
            fovs.append(fov)
            log_dir = (output_dirpath.parent / "logs" / "_".join(fov.parts)).resolve()
            os.makedirs(log_dir, exist_ok=True)
            data_path = str(Path(input_position_path).resolve())

            output_fov_path = output_dirpath.parent / "temp" / f"{'_'.join(fov.parts)}.zarr"
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

        job_ids = [job.job_id for job in job_ids_predict]  # Access job IDs after batch submission

        log_path = Path(slurm_out_path / "predict" / "submitit_jobs_ids.log")
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
        
        create_empty_hcs_zarr(
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
                temp_dir=output_dirpath.parent/ "temp",
                output_dirpath=output_dirpath,
                cleanup=True,
            )
            job_ids_combine.append(job)

        job_ids = [job.job_id for job in job_ids_combine]  # Access job IDs after batch submission

        log_path = Path(slurm_out_path / "combine" / "submitit_jobs_ids.log")
        with log_path.open("w") as log_file:
            log_file.write("\n".join(job_ids))

    if monitor:
        job_ids = [job.job_id for job in job_ids_predict + job_ids_preprocess + job_ids_combine]
        monitor_jobs(job_ids, input_position_dirpaths)






if __name__ == "__main__":
    virtual_stain_cli()


