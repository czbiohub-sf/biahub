from inspect import formatargvalues
from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr

from biahub.cli.parsing import (
    config_filepath,
    local,
    output_filepath,
    sbatch_filepath,
    source_position_dirpaths,
    target_position_dirpaths,
)
from biahub.cli.utils import (
    model_to_yaml,
    yaml_to_model,
    estimate_resources,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.registration.manual import user_assisted_registration
from biahub.registration.utils import evaluate_transforms, plot_translations
from biahub.settings import (
    EstimateRegistrationSettings,
    RegistrationSettings,
    StabilizationSettings,
)
from biahub.cli.parsing import (
    sbatch_to_submitit,
)
from pathlib import Path
from typing import List

import click
import numpy as np
import submitit
from iohub.ngff import open_ome_zarr
from tqdm import tqdm

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import yaml_to_model
from biahub.registration.utils import (
    evaluate_transforms,
    save_transforms,
)
from biahub.settings import (
    EstimateStabilizationSettings,
    StabilizationSettings,
)

# from biahub.registration.ants import estimate as ants_estimate_tczyx\
from biahub.registration.phase_cross_correlation import estimate as pcc
# from biahub.registration.beads import estimate as beads_estimate_independently
# from biahub.registration.beads import estimate_with_propagation as beads_estimate_with_propagation
# from biahub.registration.stackreg import estimate_tczyx as stackreg_estimate_tczyx
# from biahub.registration.match_z_focus import estimate_tzyx as match_z_focus_estimate_tzyx
from biahub.settings import (
    AntsRegistrationSettings,
    AffineTransformSettings,
    PhaseCrossCorrSettings,
    BeadsMatchSettings,
    StackRegSettings,
    FocusFindingSettings,
)

# One SLURM job per (FOV, T) — each T is independent

## preprocess, estimate, postprocess:  ants, pcc, 



def estimate_transforms(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    config_filepath: str,
    sbatch_filepath: str = None,
    local: bool = False,
    cluster: str = "local",
) -> dict[str, list]:
    """
    Orchestrator for registration/stabilization estimation across FOVs and timepoints.

    Submits one SLURM job per FOV for methods that require all timepoints (stackreg, beads),
    or one job per (FOV, T) for independent-timepoint methods (ants, pcc, focus-finding).

    After all jobs finish, collects per-T transforms saved to disk and returns them
    as a dict keyed by FOV path string.
    """
    from datetime import datetime

    settings = yaml_to_model(config_filepath, EstimateRegistrationSettings)
    method = settings.method
    kwargs = settings.kwargs

    click.echo(f"Settings: {settings}")

    output_dirpath = Path(output_dirpath)
    output_dirpath.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_dirpath / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)
    transforms_out_path = output_dirpath / "transforms_per_position"
    transforms_out_path.mkdir(parents=True, exist_ok=True)

    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(1, C, Z, Y, X), ram_multiplier=16, max_num_cpus=16
    )

    slurm_args = {
        "slurm_job_name": "estimate_registration",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,
        "slurm_time": 60,
        "slurm_partition": "preempted",
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)
    
    ref_channel_name = settings.ref_channel_name
    mov_channel_names = settings.mov_channel_names
    time_indices = settings.time_indices # all or int or list of ints

    if time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(time_indices, int):
        time_indices = [time_indices]
    elif isinstance(time_indices, list):
        time_indices = time_indices
    else:
        raise ValueError(f"Invalid time_indices: {time_indices}")


    click.echo(f"Submitting jobs with resources: {slurm_args}")

    PARALLEL_FOV_T_METHODS = {
        # "ants": {
        #     "function": ants_estimate_tczyx,
        #     "config": kwargs.get("ants_settings", None),
        # },
        "pcc": {}
        # "beads": {
        #     "function": beads_estimate_independently,
        #     "config": kwargs.get("beads_match_settings", None),
        # },
        # "match-z-focus": {
        #     "function": match_z_focus_estimate_tzyx,
        #     "config": settings.focus_finding_settings,
        # },
    }

    # One SLURM job per FOV — T processed sequentially, each T depends on the previous
    SEQUENTIAL_T_METHODS = {
        # "beads-with-propagation": {
        #     "function": beads_estimate_with_propagation,
        #     "config": settings.beads_match_settings,
        # },
    }

    # One SLURM job per FOV — T must all be processed together (e.g. stackreg register_stack)
    PARALLEL_FOV_METHODS = {
        # "stackreg": {
        #     "function": stackreg_estimate_tczyx,
        #     "config": settings.stack_reg_settings,
        # },
    }

    _ALL_METHODS = {**PARALLEL_FOV_T_METHODS, **SEQUENTIAL_T_METHODS, **PARALLEL_FOV_METHODS}
    

    # method_config = _ALL_METHODS.get(method)
    # if method_config is None:
    #     raise ValueError(f"Unknown method: {method}. Available: {list(_ALL_METHODS)}")
    # method_function = method_config["function"]
    # method_settings= method_config["config"]



    jobs = []

    ## kwargs per method

    ### read input data
    if method in SEQUENTIAL_T_METHODS:
        for fov_path in input_position_dirpaths:
            with open_ome_zarr(fov_path) as dataset:
                ref_data = dataset.data[:, ref_channel_name].dask_array()
                mov_data = dataset.data[:, mov_channel_names].dask_array()
                # RUN METHOD FUNCTION
    else:
        with submitit.helpers.clean_env(), executor.batch():
            for fov_path in input_position_dirpaths:
                fov_key = str(Path(*Path(fov_path).parts[-3:]))
                with open_ome_zarr(fov_path) as dataset:
                    if method in PARALLEL_FOV_T_METHODS:
                        for fov_path in input_position_dirpaths:
                            output_dirpath_fov = output_dirpath / "transforms"/ fov_key
                            for t in time_indices:
                                mov_data = dataset.data[t, mov_channel_names].dask_array()
                                ref_data = dataset.data[t, ref_channel_name].dask_array()
                            
                    # elif method in PARALLEL_FOV_METHODS:
                    #             ref_data = dataset.data[:, ref_channel_name].dask_array()
                    #             mov_data = dataset.data[:, mov_channel_names].dask_array()
                                
                    
                    job = executor.submit(
                        pcc, 
                        t=t,
                        fov=fov_path.name,
                        mov=mov_data,
                        ref=ref_data,
                        output_dirpath=output_dirpath_fov,
                        preprocessing=True,
                        phase_cross_corr_settings=settings.phase_cross_corr_settings,
                        verbose=settings.verbose,
                        debug=settings.debug,
                    )
                    jobs.append(job)
                    # RUN METHOD FUNCTION
                    # job = executor.submit(
                    #     method_function,
                    #     fov_path,
                    #     ref_data,
                    #     mov_data,
                    #     method_settings=method_settings,
                    # )
                    # jobs.append(job)

        wait_for_jobs_to_finish(jobs)


    # LOAD TRANSFORMS \
    # from biahub.registration.utils import load_transforms
    # for fov_path in fov_paths:
    #     fov_transforms = load_transforms(transforms_out_path, T)
    # return fov_transforms



    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(slurm_out_path / f"job_ids_{timestamp}.log", "w") as log_f:
        for job in jobs:
            log_f.write(f"{job.job_id}\n")

    wait_for_jobs_to_finish(jobs)

    # Collect per-T transforms saved to disk by each job
    fov_transforms = {}
    for fov_path in input_position_dirpaths:
        fov_key = str(Path(*Path(fov_path).parts[-3:]))
        fov_out_path = transforms_out_path / fov_key.replace("/", "_")
        fov_transforms[fov_key] = [np.load(fov_out_path / f"{t}.npy") for t in range(T)]

    return fov_transforms
