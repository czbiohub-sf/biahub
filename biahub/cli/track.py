# %%
from iohub import open_ome_zarr
from ultrack import Tracker, MainConfig
from ultrack.imgproc import robust_invert, normalize, inverted_edt, detect_foreground, Cellpose
from ultrack.utils.array import array_apply, create_zarr
import numpy as np
from pathlib import Path
import click
from numpy.typing import ArrayLike
from typing import Tuple, List
import toml
import os
import submitit
from pathlib import Path
from biahub.cli.parsing import (
    config_filepath,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
    input_position_dirpaths
)
from iohub import open_ome_zarr
from pathlib import Path
from biahub.cli.utils import (
    yaml_to_model,
    create_empty_hcs_zarr,
)
import numpy as np
from biahub.analysis.AnalysisSettings import TrackSettings


def detect_empty_frames(arr):
    empty_frames_idx = []
    for f in range(arr.shape[0]):
        if arr[f].sum() == 0.0:
            empty_frames_idx.append(f)
    return empty_frames_idx

def data_preprocessing(im_arr, nuc_arr, mem_arr):

    empty_frames_idx = detect_empty_frames(im_arr)
    prev_idx = 0
    for i in range(len(empty_frames_idx)):
        nuc_arr[empty_frames_idx[i]] = nuc_arr[empty_frames_idx[i]-1]
        mem_arr[empty_frames_idx[i]] = mem_arr[empty_frames_idx[i]-1]
        
        if empty_frames_idx[i] == prev_idx+1 and i < len(empty_frames_idx)-1 and i > 0:
            nuc_arr[empty_frames_idx[i]] = nuc_arr[empty_frames_idx[i]+1]
            mem_arr[empty_frames_idx[i]] = mem_arr[empty_frames_idx[i]+1]
        prev_idx = empty_frames_idx[i]

    # Preprocess the data
    nuc_arr_norm = create_zarr(shape=nuc_arr.shape, dtype=float)
    array_apply(np.array(nuc_arr), out_array=nuc_arr_norm, func=normalize, gamma=0.7)

    mem_arr_norm = create_zarr(shape=mem_arr.shape, dtype=float)
    array_apply(np.array(mem_arr), out_array=mem_arr_norm, func=normalize, gamma=0.7)

    fg_arr = create_zarr(shape=nuc_arr_norm.shape, dtype=bool)
   # array_apply(np.array(nuc_arr_norm), out_array = fg_arr, func = Cellpose(model_type = 'nuclei'))

    array_apply(
        nuc_arr_norm,
        out_array=fg_arr,
        func=detect_foreground,
        sigma=90,
    )
    top_arr = create_zarr(shape=nuc_arr.shape, dtype=np.float32)
    #array_apply(np.array(fg_arr), out_array=top_arr, func=inverted_edt)
    array_apply(np.array(nuc_arr_norm),np.array(mem_arr_norm), out_array=top_arr, func=mem_nuc_contor)
    
    return fg_arr, top_arr

def mem_nuc_contor(nuc_arr, mem_arr):
    contourn = (np.array(mem_arr) + (1- np.array(nuc_arr))) / 2
    return contourn

def ultrack_tracking(tracking_config, fg_arr: ArrayLike, top_arr: ArrayLike, scale: Tuple[float, float], databaset_path):
    print("Tracking...")
    cfg = MainConfig()

    cfg.data_config.working_dir = databaset_path
    cfg.segmentation_config.min_area = tracking_config.min_area
    cfg.segmentation_config.max_area = tracking_config.max_area
    cfg.segmentation_config.n_workers = tracking_config.n_workers_segmentation
    cfg.segmentation_config.min_frontier = tracking_config.min_frontier
    cfg.segmentation_config.max_noise = tracking_config.max_noise

    cfg.linking_config.n_workers = tracking_config.n_workers_linking
    cfg.linking_config.max_distance = tracking_config.max_distance
    cfg.linking_config.distance_weight = tracking_config.distance_weight
    cfg.linking_config.max_neighbors = tracking_config.max_neighbors    

    cfg.tracking_config.n_threads = tracking_config.n_threads
    cfg.tracking_config.disappear_weight = tracking_config.disappear_weight
    cfg.tracking_config.appear_weight = tracking_config.appear_weight
    cfg.tracking_config.division_weight = tracking_config.division_weight
   # cfg.tracking_config.window_size = tracking_config.window_size


    tracker = Tracker(cfg)
    tracker.track(
        detection=fg_arr,
        edges=top_arr,
        scale=scale,
        # overwrite=True,
    )
 
    tracks_df, graph = tracker.to_tracks_layer()
    labels = tracker.to_zarr(
        tracks_df=tracks_df,
    )

    with open(databaset_path/"config.toml", mode="w") as f:
        toml.dump(cfg.dict(by_alias=True), f)

    return (
        labels,
        tracks_df,
        graph,
    )

def tracking_one_position(
        input_lf_dirpath: Path,
        input_vs_path: Path,
        output_dirpath: Path,
        z_slice: tuple,
        vs_projection: str, tracking_config:
        TrackSettings):

    # Using this range to do projection
    z_slices = slice(z_slice[0], z_slice[1])

    input_vs_path = input_vs_path[0]
    
    position_key = input_vs_path.parts[-3:]
    fov = "_".join(position_key)
    click.echo(f"Position key: {position_key}")

    input_im_path = input_lf_dirpath / Path(*position_key)
    im_dataset = open_ome_zarr(input_im_path)
    vs_dataset = open_ome_zarr(input_vs_path)

    T, C, Z, Y, X = vs_dataset.data.shape
    channel_names = vs_dataset.channel_names

    click.echo(f"Channel names: {channel_names}")

    yx_scale = vs_dataset.scale[-2:]
    processing_channels = [f"{channel_names[0]}_labels"]

    output_metadata = {
        "shape": (T, len(processing_channels), 1, Y, X),
        "chunks": None,
        "scale": vs_dataset.scale,
        "channel_names": processing_channels,
        "dtype": np.uint32,
    }

    create_empty_hcs_zarr(
        store_path=output_dirpath, position_keys=[position_key], **output_metadata
    )

    im_arr = im_dataset[0][:, 0, z_slices.start, :, :]
    nuc_c_idx = channel_names.index(channel_names[0])
    mem_c_idx = channel_names.index(channel_names[1])


    if vs_projection == "max":
        nuc_arr = vs_dataset[0][:, nuc_c_idx, z_slices].max(axis=1)    
        mem_arr = vs_dataset[0][:, mem_c_idx, z_slices].max(axis=1)
        
    elif vs_projection == "mean":
        nuc_arr = vs_dataset[0][:, nuc_c_idx, z_slices].mean(axis=1)
        mem_arr = vs_dataset[0][:, mem_c_idx, z_slices].mean(axis=1)
    

    # preprocess to get the the foreground and multi-level contours
    fg_arr, top_arr = data_preprocessing(im_arr,nuc_arr, mem_arr)

    filename = str(output_dirpath).split("/")[-1].split(".")[0]

    databaset_path = output_dirpath.parent /f"{filename}_config_tracking"/ f"{fov}"
    os.makedirs(databaset_path, exist_ok=True)

    #Perform tracking
    labels, tracks_df, _ = ultrack_tracking(tracking_config, fg_arr, top_arr, yx_scale, databaset_path)

    # Save the tracks
    csv_path = output_dirpath / Path(*position_key) / f"tracks_{fov}.csv"
    tracks_df.to_csv(csv_path, index=False)

    click.echo(f"Saved tracks to: {output_dirpath / Path(*position_key)}")

    # Save the labels
    with open_ome_zarr(
        output_dirpath / Path(*position_key), mode="r+"
    ) as output_dataset:
        output_dataset[0][:, 0, 0] = np.array(labels)
        
@click.command()
@input_position_dirpaths()
@output_dirpath()
@config_filepath()
@sbatch_filepath()
@local()
@click.option(
    "-input_lf_dirpaths",
    required=True,
    type=str,
    help="input label free dirpath",
)
def track(
        input_lf_dirpaths:str,
        input_position_dirpaths:List[str],
        output_dirpath:str,
        config_filepath:str,
        sbatch_filepath: str = None,
        local: bool = None):
    
    """
    Track nuclei and membranes in using virtual staining nuclei and membranes data.
    
    This function applied tracking to the virtual staining data for each position in the input.
    To use this function, install the lib as pip install -e .["track"]
    
    Example usage:

    biahub track -i virtual_staining.zarr/*/*/* -input_lf_dirpaths lf_stabilize.zarr -o output.zarr -c config_tracking.yml

    """

    input_dirpath = Path(input_lf_dirpaths)

    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)

    settings = yaml_to_model(config_filepath, TrackSettings)


    with open_ome_zarr(input_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape

    # Estimate resources
    gb_per_element = 4 / 2**30  # bytes_per_float32 / bytes_per_gb
    num_cpus = np.min([T * C, 16])
    input_memory = num_cpus * Z * Y * X * gb_per_element
    gb_ram_request = np.ceil(np.max([1, input_memory])).astype(int)

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "tracking",
        "slurm_mem_per_cpu": f"{gb_ram_request}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 60,
        "slurm_partition": "gpu",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Prepare and submit jobs
    slurm_out_path = output_dirpath.parent / "slurm_output"
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo('Submitting SLURM jobs...')
    jobs = []
    

    with executor.batch():
        for input_vs_position_path in zip(
            input_position_dirpaths):
            job = executor.submit(
                tracking_one_position,
                input_lf_dirpath=input_lf_dirpaths,
                input_vs_path=input_vs_position_path,
                output_dirpath=output_dirpath,
                z_slice=settings.z_slices,
                vs_projection=settings.vs_projection,
                tracking_config=settings.tracking_config,    
            )
            jobs.append(job)


if __name__ == "__main__":
    track()
    
