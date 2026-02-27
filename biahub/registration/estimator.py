from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr



from biahub.settings import (
    PhaseCrossCorrSettings,
)

from biahub.registration.phase_cross_correlation import estimate as pcc


# One SLURM job per (FOV, T) — each T is independent

## preprocess, estimate, postprocess:  ants, pcc, 



def estimate_transforms(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
) -> dict[str, list]:
    """
    Orchestrator for registration/stabilization estimation across FOVs and timepoints.

    Submits one SLURM job per FOV for methods that require all timepoints (stackreg, beads),
    or one job per (FOV, T) for independent-timepoint methods (ants, pcc, focus-finding).

    After all jobs finish, collects per-T transforms saved to disk and returns them
    as a dict keyed by FOV path string.
    """

    # settings = yaml_to_model(config_filepath, EstimateRegistrationSettings)
    # method = settings.method
    # kwargs = settings.kwargs

    # click.echo(f"Settings: {settings}")

    # output_dirpath = Path(output_dirpath)
    # output_dirpath.mkdir(parents=True, exist_ok=True)
    # slurm_out_path = output_dirpath / "slurm_output"
    # slurm_out_path.mkdir(parents=True, exist_ok=True)
    # transforms_out_path = output_dirpath / "transforms_per_position"
    # transforms_out_path.mkdir(parents=True, exist_ok=True)


    # Channel used to estimate stabilization parameters across timepoints.
    phase_cross_corr_settings = PhaseCrossCorrSettings(
        t_reference="previous",
        function_type="custom",
        center_crop_xy=[800, 800],
    )

    mode = "registration"
    method = "pcc"
    t_reference = "first"
    ref_channel_name = 0
    mov_channel_names = 0
    time_indices = "all" # all or int or list of ints


    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape



    if time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(time_indices, int):
        time_indices = [time_indices]
    elif isinstance(time_indices, list):
        time_indices = time_indices
    else:
        raise ValueError(f"Invalid time_indices: {time_indices}")



    PARALLEL_FOV_T_METHODS = {
        # "ants": {
        #     "function": ants_estimate_tczyx,
        #     "config": kwargs.get("ants_settings", None),
        # },
        "pcc": {"function": pcc, "kwargs": 
                        {"preprocessing": False, ##all jwargs should be in method settings
                         "phase_cross_corr_settings": phase_cross_corr_settings,
                        "verbose": True}}}
        # "beads": {
        #     "function": beads_estimate_independently,
        #     "config": kwargs.get("beads_match_settings", None),
        # },
        # "match-z-focus": {
        #     "function": match_z_focus_estimate_tzyx,
        #     "config": settings.focus_finding_settings,
        # },
    #}

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
    transforms = []
  
    # if method in SEQUENTIAL_T_METHODS:
    #     for fov_path in input_position_dirpaths:
    #         with open_ome_zarr(fov_path) as dataset:
    #             ref_data = dataset.data[:, ref_channel_name].dask_array()
    #             if mode == "stabilization":
    #                 if t_reference == "previous":
    #             mov_data = dataset.data[:, mov_channel_names].dask_array()
    #             # RUN METHOD FUNCTION
    # else:
    # get att for function and kwargs
    method_config = _ALL_METHODS.get(method)
    if method_config is None:
        raise ValueError(f"Unknown method: {method}. Available: {list(_ALL_METHODS)}")
    run_function = method_config["function"]
    run_kwargs = method_config["kwargs"]

    for fov_path in input_position_dirpaths:
        fov_key = "_".join(fov_path.parts[-3:])
        with open_ome_zarr(fov_path) as mov_dataset:
            if method in PARALLEL_FOV_T_METHODS:
                for fov_path in input_position_dirpaths:
                    output_dirpath_fov = output_dirpath / "transforms"/ fov_key
                    for t in time_indices:
                        mov_data = mov_dataset.data.dask_array()[t, mov_channel_names]
                        if mode == "stabilization":
                            if t_reference == "previous":
                                ref_data = mov_dataset.data.dask_array()[t-1, ref_channel_name]
                            elif t_reference == "first":
                                ref_data = mov_dataset.data.dask_array()[0, ref_channel_name]
                            else:
                                raise ValueError(f"Invalid t_reference: {t_reference}")
                        if mode == "registration":
                            print(f"reg ZONE")
                            with open_ome_zarr(fov_path) as ref_dataset:
                                ref_data = ref_dataset.data.dask_array()[t, ref_channel_name]

                        transform = run_function(
                            t=t,
                            fov=fov_key,
                            mov=mov_data,
                            ref=ref_data,
                            output_dirpath=output_dirpath_fov, 
                            **run_kwargs)
                    transforms.append(transform)
                    print(f"Transform {t}: {transform}")
                    
            # elif method in PARALLEL_FOV_METHODS:
            #             ref_data = dataset.data[:, ref_channel_name].dask_array()
            #             mov_data = dataset.data[:, mov_channel_names].dask_array()
                        
            
            
            
            # RUN METHOD FUNCTION
            # job = executor.submit(
            #     method_function,
            #     fov_path,
            #     ref_data,
            #     mov_data,
            #     method_settings=method_settings,
            # )
            # jobs.append(job)



    # LOAD TRANSFORMS \
    # from biahub.registration.utils import load_transforms
    # for fov_path in fov_paths:
    #     fov_transforms = load_transforms(transforms_out_path, T)
    # return fov_transforms



    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # with open(slurm_out_path / f"job_ids_{timestamp}.log", "w") as log_f:
    #     for job in jobs:
    #         log_f.write(f"{job.job_id}\n")

    # wait_for_jobs_to_finish(jobs)

    # Collect per-T transforms saved to disk by each job
    # fov_transforms = {}
    # for fov_path in input_position_dirpaths:
    #     fov_key = str(Path(*Path(fov_path).parts[-3:]))
    #     fov_out_path = transforms_out_path / fov_key.replace("/", "_")
    #     fov_transforms[fov_key] = [np.load(fov_out_path / f"{t}.npy") for t in range(T)]

    # return fov_transforms

    ### think about saving config file for the transforms


def estimate_transforms_prod(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
) -> dict[str, list]:
    """
    Orchestrator for registration/stabilization estimation across FOVs and timepoints.

    Submits one SLURM job per FOV for methods that require all timepoints (stackreg, beads),
    or one job per (FOV, T) for independent-timepoint methods (ants, pcc, focus-finding).

    After all jobs finish, collects per-T transforms saved to disk and returns them
    as a dict keyed by FOV path string.
    """
    from glob import glob

    input_position_dirpaths = [Path(p) for p in glob(str(input_position_dirpaths[0]))]







    # settings = yaml_to_model(config_filepath, EstimateRegistrationSettings)
    # method = settings.method
    # kwargs = settings.kwargs

    # click.echo(f"Settings: {settings}")

    # output_dirpath = Path(output_dirpath)
    # output_dirpath.mkdir(parents=True, exist_ok=True)
    slurm_out_path = output_dirpath / "slurm_output"
    slurm_out_path.mkdir(parents=True, exist_ok=True)
    # transforms_out_path = output_dirpath / "transforms_per_position"
    # transforms_out_path.mkdir(parents=True, exist_ok=True)


    # Channel used to estimate stabilization parameters across timepoints.
    phase_cross_corr_settings = PhaseCrossCorrSettings(
        t_reference="previous",
        function_type="custom",
        center_crop_xy=[800, 800],
    )

    #mode = "stabilization":



    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape
    from biahub.cli.utils import estimate_resources
    import submitit

    num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=(1, C, Z, Y, X), ram_multiplier=16, max_num_cpus=16
    )

    slurm_args = {
        "slurm_job_name": "estimate_registration",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": 1,
        "slurm_array_parallelism": 100,
        "slurm_time": 60,
        "slurm_partition": "preempted",
    }
    # if sbatch_filepath:
    #     slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster="slurm")
    executor.update_parameters(**slurm_args)
    
    ref_channel_name = 0
    mov_channel_names = 0
    time_indices = "all" # all or int or list of ints

    if time_indices == "all":
        time_indices = list(range(T))
    elif isinstance(time_indices, int):
        time_indices = [time_indices]
    elif isinstance(time_indices, list):
        time_indices = time_indices
    else:
        raise ValueError(f"Invalid time_indices: {time_indices}")



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
    mode = "stabilization"
    method = "pcc"
    ### read input data
    transforms = []
    t_reference = "first"
    # if method in SEQUENTIAL_T_METHODS:
    #     for fov_path in input_position_dirpaths:
    #         with open_ome_zarr(fov_path) as dataset:
    #             ref_data = dataset.data[:, ref_channel_name].dask_array()
    #             if mode == "stabilization":
    #                 if t_reference == "previous":
    #             mov_data = dataset.data[:, mov_channel_names].dask_array()
    #             # RUN METHOD FUNCTION
    # else:

    ## chose to parallel in fov, t, or both, or none
    with submitit.helpers.clean_env(), executor.batch():
        for fov_path in input_position_dirpaths:
            fov_key = "_".join(fov_path.parts[-3:])
            with open_ome_zarr(fov_path) as dataset:
                if method in PARALLEL_FOV_T_METHODS:
                    for fov_path in input_position_dirpaths:
                        output_dirpath_fov = output_dirpath / fov_key
                        for t in time_indices:

                            if mode == "stabilization":
                                if t_reference == "previous":
                                    ref_data = dataset.data.dask_array()[t-1, ref_channel_name]
                                elif t_reference == "first":
                                    ref_data = dataset.data.dask_array()[0, ref_channel_name]
                                else:
                                    raise ValueError(f"Invalid t_reference: {t_reference}")
                            mov_data = dataset.data.dask_array()[t, mov_channel_names]

                            job = executor.submit(
                                pcc,
                                t=t,
                                fov= fov_key,
                                mov=mov_data,
                                ref=ref_data,
                                output_dirpath=output_dirpath_fov,
                                preprocessing=False,
                                phase_cross_corr_settings=phase_cross_corr_settings,
                                verbose=True)
                        
                            jobs.append(job)
                            
                        
            # elif method in PARALLEL_FOV_METHODS:
            #             ref_data = dataset.data[:, ref_channel_name].dask_array()
            #             mov_data = dataset.data[:, mov_channel_names].dask_array()
                        
            
            
            
            # RUN METHOD FUNCTION
            # job = executor.submit(
            #     method_function,
            #     fov_path,
            #     ref_data,
            #     mov_data,
            #     method_settings=method_settings,
            # )
            # jobs.append(job)



    # LOAD TRANSFORMS \
    # from biahub.registration.utils import load_transforms
    # for fov_path in fov_paths:
    #     fov_transforms = load_transforms(transforms_out_path, T)
    # return fov_transforms



    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # with open(slurm_out_path / f"job_ids_{timestamp}.log", "w") as log_f:
    #     for job in jobs:
    #         log_f.write(f"{job.job_id}\n")

    # wait_for_jobs_to_finish(jobs)

    # Collect per-T transforms saved to disk by each job
    # fov_transforms = {}
    # for fov_path in input_position_dirpaths:
    #     fov_key = str(Path(*Path(fov_path).parts[-3:]))
    #     fov_out_path = transforms_out_path / fov_key.replace("/", "_")
    #     fov_transforms[fov_key] = [np.load(fov_out_path / f"{t}.npy") for t in range(T)]

    # return fov_transforms



if __name__ == "__main__":

    estimate_transforms(
        input_position_dirpaths=[Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/1-preprocess/label-free/0-reconstruct/2024_11_21_A549_TOMM20_DENV.zarr/B/1/000000")],
        output_dirpath=Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/debug_reg_refactor/reg"),
    )
    # estimate_transforms_prod(
    #     input_position_dirpaths=[Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/1-preprocess/label-free/0-reconstruct/2024_11_21_A549_TOMM20_DENV.zarr/B/1/*")],
    #     output_dirpath=Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_21_A549_TOMM20_DENV/debug_reg_refactor/multifov_saving"),
    # )