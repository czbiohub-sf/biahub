import datetime
import glob
import os
from pathlib import Path

import numpy as np
from biahub.analysis.AnalysisSettings import StabilizationSettings
from biahub.cli.stabilize import apply_stabilization_transform
from biahub.cli.utils import (
    create_empty_hcs_zarr,
    process_single_position_v2,
    yaml_to_model,
)
from iohub import open_ome_zarr
from natsort import natsorted
from slurmkit import SlurmParams, slurm_function, submit_function

# NOTE: this pipeline uses the focus found on well one for all. Perhaps this should be done per FOV(?)

dataset = os.environ.get("DATASET")
if dataset is None:
    print("DATASET environmental variable is not set")
    exit()

# io parameters
input_position_dirpaths = Path(f"../1-deskew/{dataset}.zarr/*/*/*")
target_position_dirpaths = Path(f"../../../label-free/1-stabilize/{dataset}.zarr/*/*/*")
output_dirpath = Path(f"{dataset}.zarr")
config_filepath = Path("combined_transforms.yml")

# batch and resource parameters
partition = "preempted"
cpus_per_task = 16
mem_per_cpu = "12G"
time = 180  # minutes
simultaneous_processes_per_node = 16

# convert to Path
input_position_dirpaths = [
    Path(p) for p in natsorted(glob.glob(str(input_position_dirpaths)))
]
target_position_dirpaths = [
    Path(p) for p in natsorted(glob.glob(str(target_position_dirpaths)))
]

settings = yaml_to_model(config_filepath, StabilizationSettings)
combined_mats = settings.affine_transform_zyx_list
combined_mats = np.array(combined_mats)

with open_ome_zarr(input_position_dirpaths[0]) as source_ds:
    T, C, Z, Y, X = source_ds.data.shape
    channel_names = source_ds.channel_names

with open_ome_zarr(target_position_dirpaths[0]) as target_ds:
    output_shape = (T, C) + target_ds.data.shape[-3:]
    output_scale = target_ds.scale

output_metadata = {
    "shape": output_shape,
    "scale": output_scale,
    "channel_names": channel_names,
    "dtype": np.float32,
}

# Create the output zarr mirroring input_position_dirpaths
create_empty_hcs_zarr(
    store_path=output_dirpath,
    position_keys=[p.parts[-3:] for p in input_position_dirpaths],
    **output_metadata,
)

slurm_out_path = str(
    os.path.join(output_dirpath.parent, "slurm_output/stabilization-%j.out")
)

# prepare slurm parameters
params = SlurmParams(
    partition=partition,
    cpus_per_task=cpus_per_task,
    mem_per_cpu=mem_per_cpu,
    time=datetime.timedelta(minutes=time),
    output=slurm_out_path,
)

# wrap our utils.process_single_position() function with slurmkit

slurm_process_single_position = slurm_function(process_single_position_v2)
stabilization_function = slurm_process_single_position(
    func=apply_stabilization_transform,
    list_of_shifts=combined_mats,
    time_indices=list(range(T)),
    num_processes=simultaneous_processes_per_node,
    output_shape=output_shape[-3:],
)

stabilization_jobs = [
    submit_function(
        stabilization_function,
        slurm_params=params,
        input_data_path=in_path,
        output_path=output_dirpath,
    )
    for in_path in input_position_dirpaths
]
