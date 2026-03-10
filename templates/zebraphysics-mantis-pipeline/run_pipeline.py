import glob
import os
import subprocess
import time
from pathlib import Path
from check_logs import check_slurm_logs
from biahub.cli.slurm import wait_for_jobs_to_finish

from iohub import open_ome_zarr
import yaml

def convert_zarr(dataset: str, conda_environment: str):
    """Convert the dataset to Zarr format using the CLI."""
    print("Step 0: Converting dataset to Zarr format...")

    files = os.listdir(os.getcwd())
    # check if dataset_symlink
    if f"{dataset}_symlink" in files:
        print(f"Dataset {dataset}_symlink already exists. Skipping conversion.")

    else:
        # run link_datasets.py

        command = (
            f"module load anaconda && conda activate {conda_environment} && "
            f"export DATASET={dataset} && "
            f"python link_datasets.py"
        )
        print(f"Running command: {command}")
        # run the command
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

        # Check for CLI errors
        if result.returncode != 0:
            raise RuntimeError("Conversion failed. Check the logs for more details.")

    # rename well_map.csv
    command_2 = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"export DATASET={dataset} && "
        f"bash rename_wells.sh"
    )
    print(f"Running command: {command_2}")

    result_2 = subprocess.run(command_2, shell=True, stdout=subprocess.PIPE, text=True)
    # Check for CLI errors

    if result_2.returncode != 0:
        raise RuntimeError("Conversion failed. Check the logs for more details.")
    files = os.listdir()
    command_3 = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"export DATASET={dataset} && "
        f"bash convert_DRAQ5_PSF_FLUOR.sh"
    )
    print(f"Running command: {command_3}")

    result_3 = subprocess.run(command_3, shell=True, stdout=subprocess.PIPE, text=True)

    if result_3.returncode != 0:
        print("Check if conversion is missing...")

    print("Conversion completed.")


def lf_reconstruct_phase(dataset: str, conda_environment: str):
    """Run reconstruction using the CLI and wait for all jobs to finish."""
    print("Step 1: Running reconstruction...")

    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping reconstruction.")
        return

    print(f"Current directory: {os.getcwd()}")

    # # Run the CLI command with environment activation
    command = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"biahub reconstruct "
        f"-i ../../../0-convert/{dataset}_symlink/{dataset}_labelfree_1.zarr/*/*/* "
        f"-c phase_config.yaml -o {dataset}.zarr -j 12 ")

    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True)

    # Check for CLI errors
    if result.returncode != 0:
        raise RuntimeError("Reconstruction failed. Check the logs for more details.")
    # Define paths
    job_ids_log_path = Path("slurm_output/submitit_jobs_ids.log")
    # # Ensure the job_ids.log file exists
    if not job_ids_log_path.exists():
        raise FileNotFoundError(f"Job IDs log file not found at {job_ids_log_path}")

    # Read job IDs from the log file
    print(f"Reading job IDs from {job_ids_log_path}...")
    with job_ids_log_path.open("r") as f:
        job_ids = [line.strip() for line in f if line.strip()]

    print(job_ids_log_path)

    # Wait for all jobs to finish
    wait_for_jobs_to_finish(job_ids)
    print("Reconstruction completed.")

def lf_virtual_stain_preprocess(dataset: str):
    """Run preprocessing as an SLURM job."""
    print("Step 2: Running preprocessing...")

    # get files in the path with .out
    files = os.listdir(os.getcwd())
    # filter files with .out
    files = [file for file in files if file.endswith(".out")]
    for file in files:
        # checki if "calculating channel statistics 0/[0] 48/48" is in the file
        with open(file, "r") as f:
            if "calculating channel statistics 0/[0]" and "100%" in f.read():
                print(f"Preprocessing already completed. Skipping preprocessing.")
                return

    # Submit the SLURM job
    command = f"sbatch --export=DATASET={dataset} preprocess.sh"
    print(f"Submitting preprocessing job with command: {command}")
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    # Check for submission errors
    if result.returncode != 0:
        raise RuntimeError("Failed to submit preprocessing job.")

    # Parse the job ID from the sbatch output
    output = result.stdout.strip()
    print(f"sbatch output: {output}")
    job_id = None
    if "Submitted batch job" in output:
        job_id = output.split()[-1]
        print(f"Preprocessing job submitted with ID: {job_id}")
    else:
        raise RuntimeError("Failed to retrieve job ID from sbatch output.")

    # Wait for the job to complete
    wait_for_jobs_to_finish([job_id])
    print("Preprocessing completed.")


def lf_virtual_stain_prediction(dataset: str):
    """Run prediction as a SLURM array job and merge results."""
    print("Step 3: Running prediction and merging results...")

    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping prediction.")
        return

    # Define dataset paths
    current_dir = os.getcwd()
    input_dataset = f"{current_dir}/../0-reconstruct/{dataset}.zarr"
    output_folder = f"{current_dir}/{dataset}"
    config_file = f"{current_dir}/predict.yml"

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Get positions using glob (equivalent to positions=($INPUT_DATASET/*/*/*))
    positions = sorted(glob.glob(f"{input_dataset}/*/*/*"))

    if not positions:
        print("No positions found for prediction. Exiting.")
        return

    num_positions = len(positions)
    # Submit the SLURM array job for prediction
    command_array_job = (
        f"sbatch --parsable --array=0-{num_positions-1}%36 "
        f"predict_slurm.sh {input_dataset} {output_folder} {config_file}"
    )

    print(f"Submitting prediction array job with command: {command_array_job}")
    array_job_result = subprocess.run(
        command_array_job, shell=True, stdout=subprocess.PIPE, text=True
    )

    # Check for submission errors
    if array_job_result.returncode != 0:
        raise RuntimeError("Failed to submit prediction array job.")

    # Parse the array job ID
    array_job_id = array_job_result.stdout.strip()
    print(f"Prediction array job submitted with ID: {array_job_id}")

    # Submit the merging job with a dependency on the array job
    command_merge_job = (
        f"sbatch --parsable --dependency=afterok:{array_job_id} "
        f"./combine.sh {output_folder} {input_dataset} {output_folder}.zarr"
    )

    print(f"Submitting merge job with command: {command_merge_job}")
    merge_result = subprocess.run(
        command_merge_job, shell=True, stdout=subprocess.PIPE, text=True
    )

    # Check for merge job submission errors
    if merge_result.returncode != 0:
        raise RuntimeError("Failed to submit merge job.")

    # Parse the merge job ID
    merge_job_id = merge_result.stdout.strip()
    print(f"Merge job submitted with ID: {merge_job_id}")

    # Wait for the merge job to complete
    wait_for_jobs_to_finish([merge_job_id])
    print("Prediction and merging completed.")

def lf_estimate_stabilization_phase(
    dataset, conda_environment: str,
):
    """Run stabilization estimation locally and wait for it to finish."""
    print("Step 4: Estimating stabilization parameters...")

    if Path("xyz_stabilization_settings").exists():
        print(
            "Stabilization settings file already exists. Skipping stabilization estimation."
        )
        return

    print(f"Current directory: {os.getcwd()}")
    command = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"biahub estimate-stabilization "
        f"-i ../0-reconstruct/{dataset}.zarr/*/*/* "
        f"-o . "
        f"-c estimate-stabilization-xyz.yml")
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError("Stabilization estimation failed.")
    print("Stabilization estimation completed.")


def lf_apply_stabilization(dataset, conda_environment: str, input_dir,):
    """Apply stabilization using the CLI and wait for all jobs to finish."""
    print("Step 5: Applying stabilization...")

    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping stabilization.")
        return

    print(f"Current directory: {os.getcwd()}")

    # Run the CLI command
    command = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"biahub stabilize "
        f"-i ../../{input_dir}/{dataset}.zarr/*/*/* -o {dataset}.zarr "
        f"-c ../xyz_stabilization_settings/*_*_*.yml"
    )
    
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True)

    # Check for CLI errors
    if result.returncode != 0:
        raise RuntimeError("Stabilization failed. Check the logs for more details.")
    # Define paths
    job_ids_log_path = Path("slurm_output/submitit_jobs_ids.log")
    # Ensure the job_ids.log file exists
    if not job_ids_log_path.exists():
        raise FileNotFoundError(f"Job IDs log file not found at {job_ids_log_path}")

    # Read job IDs from the log file
    print(f"Reading job IDs from {job_ids_log_path}...")
    with job_ids_log_path.open("r") as f:
        job_ids = [line.strip() for line in f if line.strip()]

    # Wait for all jobs to finish
    wait_for_jobs_to_finish(job_ids)
    print("Stabilization completed.")

def lf_check_blank_frames(dataset: str, conda_environment: str):

    """Check for blank frames in the dataset using biahub."""
    print("Step 6: Checking for blank frames...")

    # Check if the blank frames file exists
    if Path(f"blank_frames.csv").exists():
        print(f"Blank frames file blank_frames.csv already exists. Skipping.")
        return

    # Run the CLI command
    command = (
        f"module load anaconda/latest && conda activate {conda_environment} && export DATASET={dataset} && "
        f"python check_blank_frames.py"
    )
    print(f"Running command: {command}")

    # Execute the command
    result = subprocess.run(command, shell=True, executable="/bin/bash")

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(
            "Blank frame check  failed. Check the logs for more details.")

    print("Blank frame check completed.")


def ls_raw_deskew(dataset: str, conda_environment: str):
    """Run deskewing for lightsheet data using biahub."""
    print("Step 7: Running deskewing...")

    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping deskew.")
        return

    # Define the SLURM job submission command
    command = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"biahub deskew "
        f"-i ../../../../0-convert/{dataset}_symlink/{dataset}_lightsheet_1.zarr/*/*/* "
        f"-c deskew_settings.yml "
        f"-o {dataset}.zarr"
    )

    print(f"Running command: {command}")

    # Submit the SLURM job
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    # Check for submission errors
    if result.returncode != 0:
        raise RuntimeError("Deskew job failed. Check the logs for more details.")

    job_ids_log_path = Path("slurm_output/submitit_jobs_ids.log")
    # Ensure the job_ids.log file exists
    if not job_ids_log_path.exists():
        raise FileNotFoundError(f"Job IDs log file not found at {job_ids_log_path}")

    # Read job IDs from the log file
    print(f"Reading job IDs from {job_ids_log_path}...")
    with job_ids_log_path.open("r") as f:
        job_ids = [line.strip() for line in f if line.strip()]

    # Wait for all jobs to finish
    wait_for_jobs_to_finish(job_ids)
    print("Deskewing completed.")


def ls_raw_estimate_registration(
    dataset: str,
    conda_environment: str,
    bead_fov: str = "C/1/000000",
    rt:str = "Phase3D",
    rs:list = ["mCherry EX561 EM600-37", "GFP EX488 EM525-45"],
):
    """Run registration estimation using biahub."""
    print("Step 8: Estimating registration...")

    if Path("stabilization_settings.yml").exists():
        print(
            "Registration estimation settings file already exists. Skipping registration estimation."
        )
        return

    # Run the CLI command
    command = (
        f'module load anaconda && conda activate {conda_environment} && '
        'biahub estimate-registration '
        f'-t ../../../label-free/0-reconstruct/{dataset}.zarr/{bead_fov} '
        f'-s ../0-deskew/{dataset}.zarr/{bead_fov} '
        '-o registration_settings.yml '
        '-c estimate-registration-beads.yml ' # change to manual or ants if needed
        f'-rt {rt} '
        f'-rs "{rs[0]}" '
        f'-rs "{rs[1]}"'
    )

    print(f"Running command: {command}")

    # Execute the command
    result = subprocess.run(command, shell=True, executable="/bin/bash")

    # Check for CLI errors
    if result.returncode != 0:
        raise RuntimeError(
            "Registration estimation failed. Check the logs for more details."
        )

    print("Registration estimation completed.")


def ls_raw_apply_registration(dataset: str, conda_environment: str):
    """Run stabilization using biahub."""
    print("Step 9: Running stabilization...")

    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping stabilization.")
        return
    # Run the CLI command
    command = (
        f"module load anaconda/latest && conda activate {conda_environment} && "
        f"biahub stabilize "
        f"-i ../0-deskew/{dataset}.zarr/*/*/* "
        f"-o {dataset}.zarr "
        f"-c registration_settings.yml"
    )
    print(f"Running command: {command}")

    # Execute the command
    result = subprocess.run(command, shell=True, executable="/bin/bash")

    # Check for CLI errors
    if result.returncode != 0:
        raise RuntimeError("Stabilization job failed. Check the logs for more details.")
    job_ids_log_path = Path("slurm_output/submitit_jobs_ids.log")
    # Ensure the job_ids.log file exists
    if not job_ids_log_path.exists():
        raise FileNotFoundError(f"Job IDs log file not found at {job_ids_log_path}")

    # Read job IDs from the log file
    print(f"Reading job IDs from {job_ids_log_path}...")
    with job_ids_log_path.open("r") as f:
        job_ids = [line.strip() for line in f if line.strip()]

    # Wait for all jobs to finish
    wait_for_jobs_to_finish(job_ids)

    print("Stabilization completed.")

def ls_raw_z_estimate_stabilization_fluorescent(
    dataset, conda_environment: str,
):
    """Run stabilization estimation locally and wait for it to finish."""
    print("Step 10: Estimating stabilization parameters...")

    if Path("z_stabilization_settings").exists():
        print(
            "Stabilization settings file already exists. Skipping stabilization estimation."
        )
        return

    print(f"Current directory: {os.getcwd()}")
    command = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"biahub estimate-stabilization "
        f"-i ../1-register/{dataset}.zarr/*/*/* "
        f"-o . "
        f"-c estimate-stabilization-z.yml")
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError("Stabilization estimation failed.")
    print("Stabilization estimation completed.")

def ls_raw_combine_stabilization(dataset: str, conda_environment: str):
    """Combine the stabilization matrices for the raw dataset."""
    print("Step 11: Combining stabilization matrices...")

    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping combining stabilization matrices.")
        return

    # Construct the command to run
    command = (
        f"module load anaconda/latest && conda activate {conda_environment} && "
        f"python combine_stabilization_matrices.py"
    )
    print(f"Running command: {command}")

    # Execute the command
    result = subprocess.run(command, shell=True, executable="/bin/bash")

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(
            "Combining stabilization matrices failed. Check the logs for more details."
        )

def ls_raw_xyz_apply_stabilization(dataset, conda_environment: str):
    """Apply stabilization using the CLI and wait for all jobs to finish."""
    print("Step 12: Applying stabilization...")

    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping stabilization.")
        return

    print(f"Current directory: {os.getcwd()}")

    # Run the CLI command
    command = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"biahub stabilize "
        f"-i ../1-register/{dataset}.zarr/*/*/* "
        f"-o {dataset}.zarr "
        f"-c combined_stabilization_settings/*_*_*.yml"
    )
    
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True)

    # Check for CLI errors
    if result.returncode != 0:
        raise RuntimeError("Stabilization failed. Check the logs for more details.")
    # Define paths
    job_ids_log_path = Path("slurm_output/submitit_jobs_ids.log")
    # Ensure the job_ids.log file exists
    if not job_ids_log_path.exists():
        raise FileNotFoundError(f"Job IDs log file not found at {job_ids_log_path}")

    # Read job IDs from the log file
    print(f"Reading job IDs from {job_ids_log_path}...")
    with job_ids_log_path.open("r") as f:
        job_ids = [line.strip() for line in f if line.strip()]

    # Wait for all jobs to finish
    wait_for_jobs_to_finish(job_ids)
    print("Stabilization completed.")

def ls_raw_rename_channels(dataset: str):

    """Rename dataset channels by prefixing them with 'raw'."""
    print("Step 13: Renaming channels...")

    dataset_path = f"{dataset}.zarr"

    # Check if the dataset exists
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset {dataset_path} does not exist.")

    # Open the dataset and rename channels
    with open_ome_zarr(dataset_path, mode="a") as ds:
        for pos_name, pos in ds.positions():
            print(f"Processing position: {pos_name}")
            for channel_name in ds.channel_names:
                if "raw" in channel_name:
                    print(f"Channel '{channel_name}' already has 'raw' prefix. Skipping.")
                    continue
                # Rename the channel by prefixing it with 'raw'
                new_channel_name = f"raw {channel_name}"
                print(f"Renaming channel '{channel_name}' to '{new_channel_name}'")
                pos.rename_channel(channel_name, new_channel_name)

    print("Channel renaming completed.")

def ls_deconvolved_estimate_psf(conda_environment: str):
    """Estimate the point spread function (PSF) using biahub."""
    print("Step 14: Estimating PSF...")

    if Path("PSF.zarr").exists():
        print("PSF.zarr already exists. Skipping PSF estimation.")
        return

    # Construct the command to run
    command = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"biahub estimate-psf "
        f"-i ../../../../0-convert/PSF.zarr/0/FOV0/0 "
        f"-c estimate_psf.yml "
        f"-o PSF.zarr"
    )
    print(f"Running command: {command}")

    # Execute the command
    result = subprocess.run(command, shell=True, executable="/bin/bash")

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError("PSF estimation failed. Check the logs for more details.")

    print("PSF estimation completed.")


def ls_deconvolved_deconvolve(dataset: str, conda_environment: str):
    """Run deconvolution using biahub."""
    print("Step 15: Running deconvolution...")
    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping deconvolution.")
        return

    # Construct the command to run
    command = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"biahub deconvolve "
        f"-i ../../../../0-convert/{dataset}_symlink/{dataset}_lightsheet_1.zarr/*/*/* "
        f"-c decon.yml "
        f"-o {dataset}.zarr "
        f"--psf-dirpath PSF.zarr"
    )
    print(f"Running command: {command}")

    # Execute the command
    result = subprocess.run(command, shell=True, executable="/bin/bash")

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError("Deconvolution failed. Check the logs for more details.")
    job_ids_log_path = Path("slurm_output/submitit_jobs_ids.log")
    # Ensure the job_ids.log file exists
    if not job_ids_log_path.exists():
        raise FileNotFoundError(f"Job IDs log file not found at {job_ids_log_path}")

    # Read job IDs from the log file
    print(f"Reading job IDs from {job_ids_log_path}...")
    with job_ids_log_path.open("r") as f:
        job_ids = [line.strip() for line in f if line.strip()]

    # Wait for all jobs to finish
    wait_for_jobs_to_finish(job_ids)

    print("Deconvolution completed.")


def ls_deconvolved_deskew(
    dataset: str,
    conda_environment: str,
):
    """Run deskewing on the deconvolved dataset using biahub."""
    print("Step 16: Deskewing deconvolved dataset...")
    if Path(f"{dataset}.zarr").exists():
        print(
            f"Dataset {dataset}.zarr already exists. Skipping deskewing of deconvolved dataset."
        )
        return
    # Construct the command to run
    command = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"biahub deskew "
        f"-i ../0-decon/{dataset}.zarr/*/*/* "
        f"-c deskew_settings.yml "
        f"-o {dataset}.zarr"
    )
    print(f"Running command: {command}")

    # Execute the command
    result = subprocess.run(command, shell=True, executable="/bin/bash")

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(
            "Deskewing of deconvolved dataset failed. Check the logs for more details."
        )
    job_ids_log_path = Path("slurm_output/submitit_jobs_ids.log")
    # Ensure the job_ids.log file exists
    if not job_ids_log_path.exists():
        raise FileNotFoundError(f"Job IDs log file not found at {job_ids_log_path}")

    # Read job IDs from the log file
    print(f"Reading job IDs from {job_ids_log_path}...")
    with job_ids_log_path.open("r") as f:
        job_ids = [line.strip() for line in f if line.strip()]

    # Wait for all jobs to finish
    wait_for_jobs_to_finish(job_ids)

    print("Deskewing of deconvolved dataset completed.")

def ls_devonvoved_register_stabilize(dataset: str, conda_environment: str):
    """Run registration and stabilization for the deconvolved dataset."""
    print("Step 17: Running registration and stabilization...")

    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping registration and stabilization.")
        return

    # Construct the command to run
    command = (
        f"module load anaconda/latest && conda activate {conda_environment} && "
        f"python combine_stabilization_matrices.py"
    )
    print(f"Running command: {command}")

    # Execute the command
    result = subprocess.run(command, shell=True, executable="/bin/bash")

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(
            "Registration and stabilization failed. Check the logs for more details."
        )

def ls_deconvolved_apply_transform(dataset: str, conda_environment: str):
    """Run the final stabilization step using biahub."""
    print("Step 18: Running final stabilization...")
    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping final stabilization.")
        return

    # Construct the command to run
    command = (
        f"module load anaconda/latest && conda activate {conda_environment} && "
        f"biahub stabilize "
        f"-i ../1-deskew/{dataset}.zarr/*/*/* "
        f"-o {dataset}.zarr "
        f"-c combined_stabilization_settings/*_*_*.yml"
    )
    print(f"Running command: {command}")

    # Execute the command
    result = subprocess.run(command, shell=True, executable="/bin/bash")

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(
            "Final stabilization failed. Check the logs for more details."
        )
    job_ids_log_path = Path("slurm_output/submitit_jobs_ids.log")
    # Ensure the job_ids.log file exists
    if not job_ids_log_path.exists():
        raise FileNotFoundError(f"Job IDs log file not found at {job_ids_log_path}")

    # Read job IDs from the log file
    print(f"Reading job IDs from {job_ids_log_path}...")
    with job_ids_log_path.open("r") as f:
        job_ids = [line.strip() for line in f if line.strip()]

    # Wait for all jobs to finish
    wait_for_jobs_to_finish(job_ids)
    print("Stabilization completed.")


def assemble(dataset: str, conda_environment: str):
    """Concatenate the final processed dataset using biahub."""
    print("Step 19: Assemble dataset...")
    if Path(f"{dataset}.zarr").exists():
        print(f"Dataset {dataset}.zarr already exists. Skipping concatenation.")
        return
    
    with open("concatenate.yml", "r") as file:
        config = yaml.safe_load(file)  # Load YAML file

    config["concat_data_paths"] = [f'../1-preprocess/label-free/2-stabilize/phase/{dataset}.zarr/*/*/*',
                                f'../1-preprocess/label-free/2-stabilize/virtual-stain/{dataset}.zarr/*/*/*',
                                f'../1-preprocess/light-sheet/raw/3-stabilize/{dataset}.zarr/*/*/*',
                                f'../1-preprocess/light-sheet/deconvolved/2-stabilize-register/{dataset}.zarr/*/*/*']  # Modify the list

    with open("concatenate.yml", "w") as file:
        yaml.safe_dump(config, file)  # Save YAML back

    # Construct the command to run
    command = (
        f"module load anaconda && conda activate {conda_environment} && "
        f"biahub concatenate "
        f"-c concatenate.yml "
        f"-o {dataset}.zarr"
    )
    print(f"Running command: {command}")

    # Execute the command
    result = subprocess.run(command, shell=True, executable="/bin/bash")

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError("Concatenation failed. Check the logs for more details.")
    job_ids_log_path = Path("slurm_output/submitit_jobs_ids.log")
    # Ensure the job_ids.log file exists
    if not job_ids_log_path.exists():
        raise FileNotFoundError(f"Job IDs log file not found at {job_ids_log_path}")

    # Read job IDs from the log file
    print(f"Reading job IDs from {job_ids_log_path}...")
    with job_ids_log_path.open("r") as f:
        job_ids = [line.strip() for line in f if line.strip()]

    # Wait for all jobs to finish
    wait_for_jobs_to_finish(job_ids)
    print("Concatenation completed.")



def main():
    # Define dataset and configuration paths
    dataset = os.environ.get("DATASET")
    if dataset is None:
        print("DATASET environmental variable is not set")
        exit()

    get_inital_path = os.getcwd()

    # Paths to working directories
    convert_working_dir = f"0-convert/"
    lf_reconstruction_working_dir = f"1-preprocess/label-free/0-reconstruct/"
    lf_stabilization_working_dir = f"1-preprocess/label-free/2-stabilize/"
    lf_virtual_stain_working_dir = f"1-preprocess/label-free/1-virtual-stain/"
    lf_tracking_working_dir = f"1-preprocess/label-free/3-track/"
    ls_raw_deskew_working_dir = f"1-preprocess/light-sheet/raw/0-deskew/"
    ls_raw_registration_working_dir = f"1-preprocess/light-sheet/raw/1-register/"
    ls_raw_stabilize_working_dir = f"1-preprocess/light-sheet/raw/2-stabilize/"
    ls_deconvolved_decon_working_dir = (
        f"1-preprocess/light-sheet/deconvolved/0-decon/"
    )
    ls_deconvolved_deskew_working_dir = (
        f"1-preprocess/light-sheet/deconvolved/1-deskew/"
    )
    ls_deconvolved_registration_working_dir = (
        f"1-preprocess/light-sheet/deconvolved/2-stabilize-register"
    )
    assemble_working_dir = f"2-assemble/"

    print(f"Running pipeline for dataset: {dataset}")

    try:
        #### label free
        # Step 0: Convert 
        os.chdir(convert_working_dir)
        convert_zarr(dataset=dataset, conda_environment="biahub")

        os.chdir(get_inital_path)
        os.chdir(lf_reconstruction_working_dir)

        # Step 1: Reconstruction
        lf_reconstruct_phase(dataset=dataset, conda_environment="biahub")

        # Step 4: Preprocessing
        os.chdir(get_inital_path)
        os.chdir(lf_virtual_stain_working_dir)

        lf_virtual_stain_preprocess(dataset=dataset)

        # Step 5: Prediction

        lf_virtual_stain_prediction(dataset=dataset)

        # Step 6: Estimate Stabilization

        os.chdir(get_inital_path)
        os.chdir(lf_stabilization_working_dir)

        lf_estimate_stabilization_phase(
            dataset=dataset,
            conda_environment="biahub",
        )
        # Step 7: Apply Stabilization Phase


        os.makedirs("phase", exist_ok=True)
        os.chdir("phase")
        lf_apply_stabilization(
            dataset=dataset,
            conda_environment="biahub",
            input_dir="0-reconstruct",
        )
        

        # Step 8: Apply Stabilization Virtual Stain
        os.chdir(get_inital_path)
        os.chdir(lf_stabilization_working_dir)
        os.makedirs("virtual-stain", exist_ok=True)
        os.chdir("virtual-stain")

        lf_apply_stabilization(
            dataset=dataset,
            conda_environment="biahub",
            input_dir="1-virtual-stain",
        )

        # Step 9: Check for Blank Frames
        os.chdir(get_inital_path)
        os.chdir(lf_tracking_working_dir)
        lf_check_blank_frames(
            dataset=dataset,
            conda_environment="biahub",
        )

        
        #### light sheet raw

        # Step 10: Deskew
        os.chdir(get_inital_path)
        os.chdir(ls_raw_deskew_working_dir)
        
        ls_raw_deskew(
            dataset=dataset,
            conda_environment="biahub",
        )

        # Step 11: Estimate Registration
        os.chdir(get_inital_path)
        os.chdir(ls_raw_registration_working_dir)
        ls_raw_estimate_registration(
            dataset=dataset,
            conda_environment="biahub",
        )
        # Step 12: Apply Registration
        ls_raw_apply_registration(
            dataset=dataset,
            conda_environment="biahub",
        )
        # Step 13: Stabilize Z
        os.chdir(get_inital_path)
        os.chdir(ls_raw_stabilize_working_dir)
        ls_raw_z_estimate_stabilization_fluorescent(
            dataset=dataset,
            conda_environment="biahub",
        )
        # Step 14: Combine Stabilization Matrices
        os.chdir(get_inital_path)
        os.chdir(ls_raw_stabilize_working_dir)
        ls_raw_combine_stabilization(
            dataset=dataset,
            conda_environment="biahub",
        )
        # Step 15: Apply Stabilization
        os.chdir(get_inital_path)
        os.chdir(ls_raw_stabilize_working_dir)
        ls_raw_xyz_apply_stabilization(
            dataset=dataset,
            conda_environment="biahub",
        )   
        # Step 16: Rename Channels
        ls_raw_rename_channels(dataset)
    
        #### light sheet deconvolved

        # Step 17: Estimate PSF
        os.chdir(get_inital_path)
        os.chdir(ls_deconvolved_decon_working_dir)
        
        ls_deconvolved_estimate_psf(
            conda_environment="biahub",
        )
        # Step 18: Deconvolve
      
        ls_deconvolved_deconvolve(
            dataset=dataset,
            conda_environment="biahub",
        )
        # Step 19: Deskew Deconvolved Dataset
        os.chdir(get_inital_path)
        os.chdir(ls_deconvolved_deskew_working_dir)
        ls_deconvolved_deskew(
            dataset=dataset,
            conda_environment="biahub",
        )
        # Step 20: Register and Stabilize
        os.chdir(get_inital_path)
        os.chdir(ls_deconvolved_registration_working_dir)
    
        ls_devonvoved_register_stabilize(
            dataset=dataset,
            conda_environment="biahub",
        )
        # Step 21: Apply Final Stabilization
        ls_deconvolved_apply_transform(
            dataset=dataset,
            conda_environment="biahub",
        )
        # Step 22: Concatenate
        # os.chdir(get_inital_path)
        # os.chdir(assemble_working_dir)
        # assemble(
        #     dataset=dataset,
        #     conda_environment="biahub",
        # )

        print("Pipeline completed successfully!")

    except RuntimeError as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
