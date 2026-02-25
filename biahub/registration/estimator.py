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

from biahub.registration.ants import estimate_tczyx as ants_estimate_tczyx
from biahub.registration.phase_cross_correlation import estimate_tczyx as pcc_estimate_tczyx
from biahub.registration.beads import estimate_independently as beads_estimate_independently
from biahub.registration.beads import estimate_with_propagation as beads_estimate_with_propagation
from biahub.registration.stackreg import estimate_tczyx as stackreg_estimate_tczyx
from biahub.registration.match_z_focus import estimate_tzyx as match_z_focus_estimate_tzyx
from biahub.settings import (
    AntsRegistrationSettings,
    AffineTransformSettings,
    PhaseCrossCorrSettings,
    BeadsMatchSettings,
    StackRegSettings,
    FocusFindingSettings,
)

# One SLURM job per (FOV, T) — each T is independent
PARALLEL_FOV_T_METHODS = {
    "ants": {
        "function": ants_estimate_tczyx,
        "settings_class": AntsRegistrationSettings,
    },
    "pcc": {
        "function": pcc_estimate_tczyx,
        "settings_class": PhaseCrossCorrSettings,
    },
    "beads-independently": {
        "function": beads_estimate_independently,
        "settings_class": BeadsMatchSettings,
    },
    "match-z-focus": {
        "function": match_z_focus_estimate_tzyx,
        "settings_class": FocusFindingSettings,
    },
}

# One SLURM job per FOV — T processed sequentially, each T depends on the previous
SEQUENTIAL_T_METHODS = {
    "beads-with-propagation": {
        "function": beads_estimate_with_propagation,
        "settings_class": BeadsMatchSettings,
    },
}

# One SLURM job per FOV — T must all be processed together (e.g. stackreg register_stack)
PARALLEL_FOV_METHODS = {
    "stackreg": {
        "function": stackreg_estimate_tczyx,
        "settings_class": StackRegSettings,
    },
}

_ALL_METHODS = {**PARALLEL_FOV_T_METHODS, **SEQUENTIAL_T_METHODS, **PARALLEL_FOV_METHODS}



def estimate_transforms(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    config_filepath: str,
    sbatch_filepath: str = None,
    ref_position_dirpaths: list[Path] = None,  # None for stabilization mode
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

    settings = yaml_to_model(config_filepath, RegistrationSettings)
    method = settings.method
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
    click.echo(f"Submitting jobs with resources: {slurm_args}")

    method_config = _ALL_METHODS.get(method)
    if method_config is None:
        raise ValueError(f"Unknown method: {method}. Available: {list(_ALL_METHODS)}")
    function_to_run = method_config["function"]

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for fov_path in input_position_dirpaths:
            fov_key = str(Path(*Path(fov_path).parts[-3:]))
            fov_out_path = transforms_out_path / fov_key.replace("/", "_")
            fov_out_path.mkdir(parents=True, exist_ok=True)

            if method in PARALLEL_FOV_T_METHODS:
                # one job per (FOV, T)
                for t in range(T):
                    job = executor.submit(
                        function_to_run,
                        fov_path,
                        t,
                        fov_out_path,
                    )
                    jobs.append(job)
            else:
                # one job per FOV — method handles T internally (stackreg, beads)
                job = executor.submit(
                    function_to_run,
                    fov_path,
                    fov_out_path,
                )
                jobs.append(job)

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
