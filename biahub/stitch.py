from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import scipy.ndimage
import submitit

from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta
from iohub.ngff.nodes import Plate, Position

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.settings import StitchSettings


def list_of_nd_slices_from_array_shape(
    array_shape: tuple[int, int, int], chunk_shape: tuple[int, int, int]
) -> list[tuple[slice, slice, slice]]:
    """
    Return a list of slices dividing an array of shape `array_shape`
    into chunks of shape `chunk_shape`.

    Example:
        list_of_nd_slices_from_array_shape((4, 5, 6), (2, 3, 4))
        # [
        #   (slice(0, 2), slice(0, 3), slice(0, 4)),
        #   (slice(0, 2), slice(0, 3), slice(4, 6)),
        #   (slice(0, 2), slice(3, 5), slice(0, 4)),
        #   (slice(0, 2), slice(3, 5), slice(4, 6)),
        #   (slice(2, 4), slice(0, 3), slice(0, 4)),
        #   (slice(2, 4), slice(0, 3), slice(4, 6)),
        #   (slice(2, 4), slice(3, 5), slice(0, 4)),
        #   (slice(2, 4), slice(3, 5), slice(4, 6)),
        # ]
    """
    chunk_slices: list[tuple[slice, slice, slice]] = []
    for idx in product(*[range(0, s, c) for s, c in zip(array_shape, chunk_shape)]):
        chunk_slices.append(
            tuple(slice(i, min(i + c, s)) for i, c, s in zip(idx, chunk_shape, array_shape))
        )
    return chunk_slices


def check_overlap(
    chunk: Tuple[slice, slice, slice],
    fov_shift: Tuple[float, float, float],
    fov_extent: Tuple[int, int, int],
) -> bool:
    """
    Check if a chunk overlaps with a field of view (FOV).

    Parameters
    ----------
    chunk : Tuple[slice, slice, slice]
        3D chunk defined by slice objects for each dimension.
    fov_shift : Tuple[float, float, float]
        Translation offset of the FOV in (z, y, x) order.
    fov_extent : Tuple[int, int, int]
        Size of the FOV in (z, y, x) order.

    Returns
    -------
    bool
        True if the chunk overlaps with the FOV, False otherwise.
    """
    for dim in range(3):
        if (
            chunk[dim].start >= fov_shift[dim] + fov_extent[dim]
            or chunk[dim].stop <= fov_shift[dim]
        ):
            return False
    return True


def overlap_slices(
    chunk_corner: Tuple[float, float, float],
    chunk_extent: Tuple[float, float, float],
    fov_corner: Tuple[float, float, float],
    fov_extent: Tuple[int, int, int],
) -> Tuple[Optional[Tuple[slice, slice, slice]], Optional[Tuple[slice, slice, slice]]]:
    """
    Calculate slice objects for overlapping regions between a chunk and FOV.

    Parameters
    ----------
    chunk_corner : Tuple[float, float, float]
        Corner position of the chunk in (z, y, x) order.
    chunk_extent : Tuple[float, float, float]
        Size of the chunk in (z, y, x) order.
    fov_corner : Tuple[float, float, float]
        Corner position of the FOV in (z, y, x) order.
    fov_extent : Tuple[int, int, int]
        Size of the FOV in (z, y, x) order.

    Returns
    -------
    Tuple[Optional[Tuple[slice, slice, slice]], Optional[Tuple[slice, slice, slice]]]
        A tuple containing (fixed_slice, moving_slice) for the overlapping region,
        or (None, None) if no overlap exists.
    """
    fixed, moving = [], []
    for d in range(3):
        start = max(chunk_corner[d], fov_corner[d])
        stop = min(chunk_corner[d] + chunk_extent[d], fov_corner[d] + fov_extent[d])
        if stop <= start:
            return None, None
        fixed_slice = slice(int(start - chunk_corner[d]), int(stop - chunk_corner[d]))
        moving_slice = slice(int(start - fov_corner[d]), int(stop - fov_corner[d]))
        # Ensure both slices are the same size
        fixed_len = fixed_slice.stop - fixed_slice.start
        moving_len = moving_slice.stop - moving_slice.start
        max_len = max(fixed_len, moving_len)
        fixed.append(slice(fixed_slice.start, fixed_slice.start + max_len))
        moving.append(slice(moving_slice.start, moving_slice.start + max_len))
    return tuple(fixed), tuple(moving)


def find_contributing_fovs(
    chunk: Tuple[slice, slice, slice],
    fov_shifts: Dict[str, Tuple[float, float, float]],
    fov_extent: Tuple[int, int, int],
) -> List[str]:
    """
    Find all FOVs that contribute data to a given chunk.

    Parameters
    ----------
    chunk : Tuple[slice, slice, slice]
        3D chunk defined by slice objects for each dimension.
    fov_shifts : Dict[str, Tuple[float, float, float]]
        Dictionary mapping FOV names to their translation offsets in (z, y, x) order.
    fov_extent : Tuple[int, int, int]
        Size of each FOV in (z, y, x) order.

    Returns
    -------
    List[str]
        List of FOV names that overlap with the given chunk.
    """
    contributing_fovs = []
    for fov_key, fov_shift in fov_shifts.items():
        if check_overlap(chunk, fov_shift, fov_extent):
            contributing_fovs.append(fov_key)
    return contributing_fovs


def get_output_shape(
    shifts: Dict[str, Tuple[float, float, float]], tile_shape: Tuple[int, ...]
) -> Tuple[int, int, int]:
    """
    Calculate the output shape of the stitched image from FOV shifts.

    Parameters
    ----------
    shifts : Dict[str, Tuple[float, float, float]]
        Dictionary mapping FOV names to their translation offsets in (z, y, x) order.
    tile_shape : Tuple[int, ...]
        Shape of individual tiles/FOVs.

    Returns
    -------
    Tuple[int, int, int]
        Output shape of the stitched image in (z, y, x) order.
    """

    z_shifts = [shift[0] for shift in shifts.values()]
    y_shifts = [shift[1] for shift in shifts.values()]
    x_shifts = [shift[2] for shift in shifts.values()]

    max_z = int(np.max(np.asarray(z_shifts)))
    max_y = int(np.max(np.asarray(y_shifts)))
    max_x = int(np.max(np.asarray(x_shifts)))

    return max_z + tile_shape[-3], max_y + tile_shape[-2], max_x + tile_shape[-1]


def write_output_chunk(
    output_chunk_slices: Tuple[slice, slice, slice],
    fov_shifts: Dict[str, Tuple[float, float, float]],
    channel_idx: int,
    input_plate: Plate,
    input_fov_shape: Tuple[int, int, int, int, int],
    output_position: Position,
    verbose: bool,
    blending_exponent: float = 1.0,
) -> None:
    """
    Write a single output chunk by blending contributing FOVs with distance-based weighting.

    This function processes one chunk of the final stitched image by:
    1. Finding all FOVs that contribute to this chunk
    2. Computing distance-based weight maps for smooth blending
    3. Applying weights to FOV data and summing contributions
    4. Writing the result to the output array

    Parameters
    ----------
    output_chunk_slices : Tuple[slice, slice, slice]
        Slice objects defining the chunk region in the output image.
    fov_shifts : Dict[str, Tuple[float, float, float]]
        Dictionary mapping FOV names to their translation offsets in (z, y, x) order.
    channel_idx : int
        Index of the channel to process.
    input_plate : Plate
        Input plate containing all FOV data.
    input_fov_shape : Tuple[int, int, int, int, int]
        Shape of input FOVs in (T, C, Z, Y, X) order.
    output_position : Position
        Output position where the stitched data will be written.
    verbose : bool
        Whether to print detailed progress information.
    blending_exponent : float, default=1.0
        Exponent for distance-based blending weights. Higher values create sharper transitions.
    """
    # For each output chunk, find the input fovs that contribute to it
    contributing_fov_names = find_contributing_fovs(
        output_chunk_slices, fov_shifts, input_fov_shape[-3:]
    )
    chunk_corner = np.array([output_chunk_slices[dim].start for dim in range(3)])
    chunk_extent = np.array(
        [output_chunk_slices[dim].stop - output_chunk_slices[dim].start for dim in range(3)]
    )

    output_array = output_position["0"]
    array_shape = output_array[(slice(None), channel_idx, *output_chunk_slices)].shape
    output_chunk = np.zeros(array_shape)

    # Compute overlap slices
    fixed_slices = []
    moving_slices = []
    for fov_name in contributing_fov_names:
        fov_corner = np.array([fov_shifts[fov_name][d] for d in range(3)])
        fov_extent = np.array([input_fov_shape[d + 2] for d in range(3)])
        fixed_slice, moving_slice = overlap_slices(
            chunk_corner, chunk_extent, fov_corner, fov_extent
        )
        if fixed_slice is None or moving_slice is None:
            continue
        else:
            fixed_slices.append(fixed_slice)
            moving_slices.append(moving_slice)

    # Precompute a single distance-from-edge map for a complete FOV
    # Note: this computes distance from the XY edges, will need extension for 3D
    fov_temp = np.zeros(fov_extent)
    fov_temp[:, 1:-1, 1:-1] = 1
    mask = fov_temp != 0
    centered_distance_map_2d = scipy.ndimage.distance_transform_edt(mask[0])
    centered_distance_map = np.tile(
        centered_distance_map_2d[None, :, :], (output_chunk.shape[-3], 1, 1)
    )

    # Slice into the precomputed distance map to build the distance maps for
    # each contributing fov
    distance_maps = np.zeros((len(contributing_fov_names),) + output_chunk.shape[-3:])
    for i, (fixed_slice, moving_slice) in enumerate(zip(fixed_slices, moving_slices)):
        if verbose:
            click.echo(f"\t\tComputing distance map for {contributing_fov_names[i]}")
        distance_maps[(i, *fixed_slice)] = centered_distance_map[(*moving_slice,)]

    # Compute weight maps for each contributing fov
    if verbose:
        click.echo("\t\tBuilding weight maps")
    w = np.power(distance_maps, blending_exponent, where=(distance_maps > 0))
    sum_w = np.sum(w, axis=0, keepdims=True)
    weight_maps = w / (sum_w + 1e-8)

    # Apply weights to each contributing fov and sum
    for i, (fov_name, fixed_slice, moving_slice) in enumerate(
        zip(contributing_fov_names, fixed_slices, moving_slices)
    ):
        if verbose:
            click.echo(f"\t\tApplying weight maps to {fov_name}")
        # Get the fov data
        fov_data = input_plate[fov_name].data

        # Apply weights to the fov data
        weighted_output = (
            fov_data[(slice(None), channel_idx, *moving_slice)]
            * weight_maps[(i, *fixed_slice)]
        )

        # Add to the output chunk
        output_chunk[(slice(None), channel_idx, *fixed_slice)] += weighted_output

    # Write chunk to output array
    if verbose:
        click.echo(f"\t\tWriting chunk to output array: {output_chunk_slices}")
    output_array[(slice(None), channel_idx, *output_chunk_slices)] = output_chunk


@click.command("stitch")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    type=bool,
    help="Verbose stitching output. Default is False.",
)
@click.option(
    "--blending-exponent",
    "-b",
    type=float,
    default=1.0,
    help="Exponent for blending weights. 0.0 is average blending, 1.0 is linear blending, and >1.0 is progressively sharper S-curve blending.",
)
def stitch_cli(
    input_position_dirpaths: List[str],
    output_dirpath: str,
    config_filepath: str,
    verbose: bool = False,
    sbatch_filepath: str = None,
    local: bool = False,
    blending_exponent: float = 1.0,
) -> None:
    """
    Stitch FOVs in each well together into a single FOV.
    Uses shift from configuration file generated with `biahub estimate-stitch`.

    >> biahub stitch -i ./input.zarr/*/*/* -c ./config.yaml -o ./output.zarr
    """

    click.echo("Starting stitching...")
    settings = yaml_to_model(config_filepath, StitchSettings)
    input_plate = open_ome_zarr(input_position_dirpaths[0].parents[2], mode="r")
    all_shifts = settings.total_translation

    input_channels = input_plate.channel_names
    if settings.channels is None:
        settings.channels = input_channels
    channel_idx = np.asarray([input_channels.index(ch) for ch in settings.channels])

    if not all(channel in input_channels for channel in settings.channels):
        raise ValueError("Invalid channel(s) provided.")

    # Create output store
    output_plate = open_ome_zarr(
        output_dirpath, layout='hcs', mode="w", channel_names=settings.channels
    )

    # Group shift metadata by well
    shifts_by_well = defaultdict(dict)
    for key, value in all_shifts.items():
        well_name = "/".join(key.split("/")[:2])
        shifts_by_well[well_name][key] = value

    # Prepare jobs
    job_args_list = []
    for well_name, fov_shifts in shifts_by_well.items():
        if verbose:
            click.echo(
                f"Processing well {list(shifts_by_well.keys()).index(well_name)+1}/{len(shifts_by_well)}: {well_name}"
            )
        first_fov_name = list(shifts_by_well[well_name].keys())[0]
        input_fov_shape = input_plate[first_fov_name].data.shape
        output_shape_zyx = get_output_shape(fov_shifts, input_fov_shape)
        output_chunk_size = (
            1,
            1,
            output_shape_zyx[0],
            *input_plate[first_fov_name].data.chunks[-2:],
        )
        output_scale = input_plate[first_fov_name].scale

        output_shape = (
            input_fov_shape[0],
            len(channel_idx),
        ) + output_shape_zyx

        # Create the output array
        # note that output shape is different for each well, so we are not
        # using iohub.ngff.utils.create_empty_plate here
        output_position = output_plate.create_position(
            first_fov_name.split("/")[0],
            first_fov_name.split("/")[1],
            "0",
        )
        _ = output_position.create_zeros(
            "0",
            shape=output_shape,
            chunks=(1, 1, 10, output_chunk_size[-2], output_chunk_size[-1]),
            dtype=np.float16,
            transform=[TransformationMeta(type="scale", scale=output_scale)],
        )

        # Split the output array into chunks
        chunk_list = list_of_nd_slices_from_array_shape(
            output_shape_zyx,
            output_chunk_size[2:],
        )

        # Append job arguments for each chunk
        for chunk in chunk_list:
            if verbose:
                click.echo(
                    f"\tPreparing job for chunk {chunk_list.index(chunk)+1}/{len(chunk_list)}: {chunk}"
                )
            job_args_list.append(
                (
                    chunk,
                    fov_shifts,
                    channel_idx,
                    input_plate,
                    input_fov_shape,
                    output_position,
                    verbose,
                    blending_exponent,
                )
            )

    # Prepare for SLURM submission

    # Estimate resources
    num_cpus, gb_ram = estimate_resources(
        shape=input_fov_shape, ram_multiplier=25, max_num_cpus=16
    )

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "stitch",
        "slurm_mem_per_cpu": f"{gb_ram}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 output chunks at a time
        "slurm_time": 60,
        "slurm_partition": "preempted",
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
    click.echo(f"Preparing jobs: {slurm_args}")
    slurm_out_path = output_dirpath.parent / "slurm_output"
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    with executor.batch():
        for job_args in job_args_list:
            jobs.append(
                executor.submit(
                    write_output_chunk,
                    *job_args,
                )
            )

    job_ids = [job.job_id for job in jobs]  # Access job IDs after batch submission

    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))


if __name__ == '__main__':
    stitch_cli()
