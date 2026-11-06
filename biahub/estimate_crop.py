import itertools
import shutil

from ast import literal_eval
from pathlib import Path

import click
import dask.array as da
import largestinteriorrectangle as lir
import numpy as np
import pandas as pd
import submitit

from iohub import open_ome_zarr
from matplotlib import pyplot as plt

from biahub.cli.parsing import (
    config_filepath,
    local,
    output_filepath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.core.transform import Transform
from biahub.registration.preprocess import build_mask
from biahub.settings import ConcatenateSettings, MaskCircle
from biahub.utils.cluster import (
    apply_sbatch_overrides,
    estimate_resources,
    get_submitit_cluster,
)
from biahub.utils.config import model_to_yaml, yaml_to_model

# Cropping: the largest box inside an overlap. Two searches, both kept: `find_lir` (the
# rectangle at the middle z slice, z clipped to where it still fits; estimate-crop's) and
# `largest_box` (exact over every z-range, on a coarse grid; apply-transform's shared_crop).


def find_lir(registered_zyx: np.ndarray, plot: bool = False) -> tuple:
    registered_zyx = np.asarray(registered_zyx, dtype=bool)

    # Find the lir in YX at Z//2
    registered_yx = registered_zyx[registered_zyx.shape[0] // 2].copy()
    coords_yx = lir.lir(registered_yx)
    coords_yx = list(map(int, coords_yx))

    x, y, width, height = coords_yx
    x_start, x_stop = x, x + width
    y_start, y_stop = y, y + height
    x_slice = slice(x_start, x_stop)
    y_slice = slice(y_start, y_stop)

    # Iterate over ZX and ZY slices to find optimal Z cropping params
    _coords = []
    for _x in (x_start, x_start + (x_stop - x_start) // 2, x_stop - 1):
        registered_zy = registered_zyx[:, y_slice, _x].copy()
        coords_zy = lir.lir(registered_zy)
        _, z, _, depth = coords_zy
        z_start, z_stop = z, z + depth
        _coords.append((z_start, z_stop))
    for _y in (y_start, y_start + (y_stop - y_start) // 2, y_stop - 1):
        registered_zx = registered_zyx[:, _y, x_slice].copy()
        coords_zx = lir.lir(registered_zx)
        _, z, _, depth = coords_zx
        z_start, z_stop = z, z + depth
        _coords.append((z_start, z_stop))

    _coords = np.asarray(_coords)
    z_start = int(_coords.max(axis=0)[0])
    z_stop = int(_coords.min(axis=0)[1])
    z_slice = slice(z_start, z_stop)

    if plot:
        xy_corners = ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
        rectangle_yx = plt.Polygon(
            xy_corners,
            closed=True,
            fill=None,
            edgecolor="r",
        )
        # Add the rectangle to the plot
        _, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(registered_yx)
        ax[0].add_patch(rectangle_yx)

        zx_corners = ((x, z), (x + width, z), (x + width, z + depth), (x, z + depth))
        rectangle_zx = plt.Polygon(
            zx_corners,
            closed=True,
            fill=None,
            edgecolor="r",
        )
        ax[1].imshow(registered_zx)
        ax[1].add_patch(rectangle_zx)
        plt.savefig("./lir.png")

    return (z_slice, y_slice, x_slice)


Slices = tuple[slice, slice, slice]


def _factors(downsample: int) -> tuple[int, int, int]:
    """Return the coarse grid's step per axis: z 1, y and x `downsample`.

    z stays at full resolution: a stack has few planes, and each lost costs the most.
    """
    f = max(1, int(downsample))
    return (1, f, f)


def _coarse(shape_zyx: tuple[int, int, int], factors: tuple) -> tuple[int, int, int]:
    return tuple(max(1, int(np.ceil(s / f))) for s, f in zip(shape_zyx, factors, strict=True))


def overlap_mask(
    moving_shape_zyx: tuple[int, int, int],
    reference_shape_zyx: tuple[int, int, int],
    transform: Transform,
    downsample: int = 4,
) -> np.ndarray:
    """Reference-grid blocks the warped moving volume covers whole, on a coarser grid (`_factors`).

    Each coarse voxel is the block of reference voxels it stands for (`downsample` x
    `downsample` in yx, one plane in z). The covered region is convex (the preimage of
    the moving volume's box), so a block is covered whole when its corner voxels are: each corner
    is tested with `Transform.coverage` (the warp's own edge, without warping) against
    the moving volume's full-resolution bounds. Cheap for a 240-timepoint series; a crop from it
    holds only covered voxels (a partial last block counts as uncovered).
    """
    factors = _factors(downsample)
    coarse_shape = _coarse(reference_shape_zyx, factors)
    scale = Transform(np.diag([*map(float, factors), 1.0]))
    mask = np.ones(coarse_shape, dtype=bool)
    for corner in itertools.product(*[sorted({0, f - 1}) for f in factors]):
        # coarse index k stands for reference voxel factors * k + corner
        to_reference = Transform.from_translation(corner) @ scale
        mask &= (to_reference.invert() @ transform).coverage(moving_shape_zyx, coarse_shape)
    return mask


def largest_box(mask: np.ndarray) -> Slices:
    """Exact largest axis-aligned box of True voxels in a (Z, Y, X) mask.

    For every z-range the slices are ANDed and the exact largest 2D rectangle found
    (`largestinteriorrectangle`); the largest volume wins. `find_lir` (above) is a
    heuristic (rectangle at the middle slice, z clipped to where that rectangle still
    fits) and on a sheared overlap gives up most of z; this is O(Z^2) 2D searches, which
    on shared_crop's coarse grid is a few seconds.
    """
    mask = np.asarray(mask, dtype=bool)
    best, best_volume = None, 0
    for z0 in range(mask.shape[0]):
        combined = mask[z0].copy()
        for z1 in range(z0, mask.shape[0]):
            combined &= mask[z1]
            if not combined.any():
                break
            depth = z1 - z0 + 1
            if depth * combined.sum() <= best_volume:
                continue  # even the full remaining area cannot beat the best box
            x, y, width, height = map(int, lir.lir(combined))
            volume = depth * width * height
            if volume > best_volume:
                best, best_volume = (
                    (slice(z0, z1 + 1), slice(y, y + height), slice(x, x + width)),
                    volume,
                )
    if best is None:
        raise ValueError("empty mask")
    return best


def _mask_box(
    mask: np.ndarray, reference_shape_zyx: tuple[int, int, int], downsample: int
) -> Slices:
    """Largest box inside a coarse mask, back on the full-resolution grid."""
    if not mask.any():
        raise click.UsageError(
            "the transform leaves no overlapping region between moving and reference; "
            "drop --crop-to-overlap or check the transform"
        )
    box = largest_box(mask)
    return tuple(
        slice(min(s.start * f, dim), min(s.stop * f, dim))
        for s, f, dim in zip(box, _factors(downsample), reference_shape_zyx, strict=True)
    )


def overlap_slices(
    moving_shape_zyx: tuple[int, int, int],
    reference_shape_zyx: tuple[int, int, int],
    transform: Transform,
    downsample: int = 4,
) -> Slices:
    """Largest box inside the overlap of the warped moving volume and the reference grid."""
    mask = overlap_mask(moving_shape_zyx, reference_shape_zyx, transform, downsample)
    return _mask_box(mask, reference_shape_zyx, downsample)


def shared_crop(
    moving_shape_zyx: tuple[int, int, int],
    reference_shape_zyx: tuple[int, int, int],
    transforms: list[Transform],
    keep_overhang: bool,
    downsample: int = 4,
) -> Slices:
    """Find the one crop every transform shares: the largest box in their common overlap.

    `transforms` are every (forward) transform the output must hold -- each timepoint and
    each position of the file, since one output store has one shape -- and the crop is
    the ZYX slices of the largest z-y-x box inside all their overlaps (z planes a shift
    or tilt leaves uncovered are cropped too). The masks are intersected first and the
    exact largest box found once (`largest_box`). Finding a box per transform and intersecting the boxes is much
    smaller: on a rotated overlap each timepoint's box trades the axes differently.

    With `keep_overhang` the full reference grid is kept (content the transform pushes
    outside it is lost, content it pulls in from outside is blank).
    """
    if keep_overhang:
        return tuple(slice(0, dim) for dim in reference_shape_zyx)
    # many timepoints share a transform: one mask each
    unique = {t.matrix.round(9).tobytes(): t for t in transforms}
    mask = None
    for transform in unique.values():
        current = overlap_mask(moving_shape_zyx, reference_shape_zyx, transform, downsample)
        mask = current if mask is None else mask & current
    if mask is None or not mask.any():
        raise click.UsageError(
            "the transforms share no overlapping region across timepoints; "
            "drop --crop-to-overlap or check the transforms"
        )
    box = _mask_box(mask, reference_shape_zyx, downsample)
    # the coarse mask counts partly covered blocks as uncovered: win their voxels back
    steps = [f - 1 for f in _factors(downsample)]
    return _grow_covered(
        box, list(unique.values()), moving_shape_zyx, reference_shape_zyx, steps
    )


def _grow_covered(box: Slices, transforms, moving_shape, reference_shape, steps) -> Slices:
    """Grow each face of a covered box outward, voxel by voxel (up to `steps` per axis).

    The region every transform covers is convex, so a box lies inside it when its 8
    corner voxels do (`Transform.covers`): each step checks only those.
    """
    bounds = [[s.start, s.stop] for s in box]

    def covered(trial) -> bool:
        corners = list(itertools.product(*[(start, stop - 1) for start, stop in trial]))
        return all(t.covers(corners, moving_shape).all() for t in transforms)

    for axis in range(3):
        for end, step in ((1, 1), (0, -1)):  # the stop outward, then the start
            for _ in range(steps[axis]):
                trial = [list(b) for b in bounds]
                trial[axis][end] += step
                if not 0 <= trial[axis][0] < trial[axis][1] <= reference_shape[axis]:
                    break
                if not covered(trial):
                    break
                bounds = trial
    return tuple(slice(start, stop) for start, stop in bounds)


def estimate_crop_one_position(
    lf_dir: np.ndarray,
    ls_dir: np.ndarray,
    lf_mask_radius: float = None,
    output_dir: Path = None,
):
    """
    Estimate a crop region where both phase and fluorescence volumes are non-zero.

    Parameters
    ----------
    lf_dir : Path
        Path to the phase channel.
    ls_dir : Path
        Path to the fluorescence channel.
    lf_mask_radius : float
        Radius of the circular mask which will be applied to the phase channel. If None, no masking will be applied
    output_dir : Path
        Path to save the output CSV file.

    Returns
    -------
    tuple
        Tuple of slices for Z, Y, and X dimensions.

    """
    fov = "/".join(lf_dir.parts[-3:])

    click.echo(f"Processing FOV: {fov}")
    with open_ome_zarr(lf_dir) as lf_dataset:
        lf_data = lf_dataset.data.dask_array()[:, :1]  # Pick only first channel
        lf_mask = ((lf_data != 0) & (~da.isnan(lf_data))).compute()

    with open_ome_zarr(ls_dir) as ls_dataset:
        ls_data = ls_dataset.data.dask_array()[:, :1]
        ls_mask = ((ls_data != 0) & (~da.isnan(ls_data))).compute()

    if lf_mask.ndim != 5 or ls_mask.ndim != 5:
        raise ValueError("Both phase_data and fluor_data must be 5D arrays.")

    # Both volumes cut to the shape they share (the same shape, normally)
    lf_shape = lf_mask.shape[-3:]
    ls_shape = ls_mask.shape[-3:]
    if lf_shape != ls_shape:
        click.echo(
            "WARNING: Phase and fluorescence datasets should have the same shape, got"
            f" phase shape: {lf_shape}, fluorescence shape: {ls_shape}"
        )
    Z, Y, X = np.minimum(lf_shape, ls_shape)
    lf_mask = lf_mask[..., :Z, :Y, :X]
    ls_mask = ls_mask[..., :Z, :Y, :X]

    # Concatenate the data arrays along the channel axis
    data = np.concatenate([lf_mask, ls_mask], axis=1)

    # Create a mask to find time points and channels where any data is non-zero
    volume = np.sum(data, axis=(2, 3, 4))  # more robust selection
    median_volume = np.median(volume)
    valid_T, valid_C = np.where(
        (volume > 0.8 * median_volume) & (volume < 1.2 * median_volume)
    )

    if len(valid_T) == 0:
        click.echo("No valid data found for current position, will not crop.")
        return ([0, int(Z)], [0, int(Y)], [0, int(X)])
    valid_data = data[valid_T, valid_C]

    # Compute a mask where all voxels are non-zero along time and channel dimensions
    combined_mask = np.all(valid_data, axis=0)

    # Create a circular boolean mask of radius phase_mask_radius to apply to the phase channel
    if lf_mask_radius is not None:
        click.echo(f"Applying circular mask of radius {lf_mask_radius} to phase channel.")
        if not (0 < lf_mask_radius <= 1):
            raise ValueError(
                "lf_mask_radius is a fraction of the half-width of the shorter yx axis "
                "(0 < lf_mask_radius <= 1)"
            )

        # the preprocess block's circle, on the phase field it belongs to, then cut to
        # the shared shape like the volumes
        circle = MaskCircle(target=["reference"], radius=lf_mask_radius)
        phase_circle = build_mask(lf_shape, [circle], "reference")
        combined_mask = combined_mask * phase_circle[:Z, :Y, :X]

    # Compute overlapping region
    z_slice, y_slice, x_slice = find_lir(combined_mask)

    click.echo(
        f"Estimated crop for FOV {fov}:\n"
        f"Z: {z_slice.start} - {z_slice.stop}\n"
        f"Y: {y_slice.start} - {y_slice.stop}\n"
        f"X: {x_slice.start} - {x_slice.stop}"
    )

    if output_dir:
        df = pd.DataFrame(
            [
                {
                    "fov": fov,
                    "Z": [z_slice.start, z_slice.stop],
                    "Y": [y_slice.start, y_slice.stop],
                    "X": [x_slice.start, x_slice.stop],
                }
            ]
        )
        df.to_csv(output_dir / f"{fov.replace('/', '_')}.csv", index=False)

    return (
        [z_slice.start, z_slice.stop],
        [y_slice.start, y_slice.stop],
        [x_slice.start, x_slice.stop],
    )


def estimate_crop(
    config_filepath: str,
    output_filepath: str,
    lf_mask_radius: float | None = None,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """
    Estimate a crop region where both phase and fluorescence volumes are non-zero.

    Parameters
    ----------
    config_filepath : str
        Path to a yaml ConcatenateSettings file.
        This file will be replicated in the output with modified XYZ slicing parametrs.
    output_filepath : str
        Path to save the output config file.
    lf_mask_radius : float
        Radius of the circular mask given as fraction of image width to apply to the phase channel.
        A good value if 0.95.
    sbatch_filepath : str
        Path to a SLURM submission script.
    local : bool
        If True, run the jobs locally.

    """
    if config_filepath.suffix not in [".yml", ".yaml"]:
        raise ValueError("Config file must be a yaml file")

    config_filepath = Path(config_filepath)
    settings = yaml_to_model(config_filepath, ConcatenateSettings)
    output_dir = Path(output_filepath).parent
    slurm_out_path = output_dir / "slurm_output"
    output_path_csv = output_dir / "crop_estimates"
    output_path_csv.mkdir(exist_ok=True, parents=True)

    # Assume phase dataset is first and fluor dataset is second in input_model.concat_data_paths
    if not settings.concat_data_paths or len(settings.concat_data_paths) < 2:
        raise ValueError(
            "estimate-crop needs concat_data_paths in the config: the phase positions "
            "first, the fluorescence positions second."
        )
    lf_paths = config_filepath.parent.glob(settings.concat_data_paths[0])
    lf_position_dirpaths = [p for p in lf_paths if p.is_dir()]
    click.echo(f"Found {len(lf_position_dirpaths)} phase channels.")
    ls_paths = config_filepath.parent.glob(settings.concat_data_paths[1])
    ls_position_dirpaths = [p for p in ls_paths if p.is_dir()]
    click.echo(f"Found {len(ls_position_dirpaths)} fluorescence channels.")

    if len(lf_position_dirpaths) != len(ls_position_dirpaths):
        raise ValueError("Number of phase and fluorescence channels must be the same.")

    # Estimate resources from a sample
    with open_ome_zarr(lf_position_dirpaths[0]) as dataset:
        T, C, Z, Y, X = dataset.data.shape

    _, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16
    )
    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "estimate_crop",
        "slurm_mem_per_cpu": f"{gb_ram_per_cpu}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": 30,
        "slurm_partition": "preempted",
    }

    # Override defaults if sbatch_filepath is provided
    if sbatch_filepath:
        slurm_args = apply_sbatch_overrides(slurm_args, sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    cluster = get_submitit_cluster(local)

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    click.echo("Submitting SLURM jobs...")
    jobs = []

    with submitit.helpers.clean_env(), executor.batch():
        for ls_dir, lf_dir in zip(ls_position_dirpaths, lf_position_dirpaths, strict=True):
            job = executor.submit(
                estimate_crop_one_position,
                lf_mask_radius=lf_mask_radius,
                lf_dir=lf_dir,
                ls_dir=ls_dir,
                output_dir=output_path_csv,
            )
            jobs.append(job)

    # Wait for jobs to finish
    wait_for_jobs_to_finish(jobs)

    # Read and merge results. The same ROI crop will be applied to all positions.
    # Here we estimate the smallest common crop region across all positions.
    estimate_crop_csvs = list(output_path_csv.glob("*.csv"))
    if not estimate_crop_csvs:
        click.echo("No crop CSV files found. Exiting.")
        return

    df = pd.concat(
        [pd.read_csv(f, dtype={"fov": str}) for f in estimate_crop_csvs], ignore_index=True
    )
    df = df.drop_duplicates(subset=["fov", "Z", "Y", "X"])
    df = df.sort_values("fov")
    for col in ["X", "Y", "Z"]:
        df[col] = df[col].apply(literal_eval)
    df.to_csv(output_dir / "crop_slices.csv", index=False)

    # Compute standardized crop
    all_ranges = []
    for _, row in df.iterrows():
        all_ranges.append([row["Z"], row["Y"], row["X"]])

    all_ranges = np.array(all_ranges)
    standardized_ranges = np.concatenate(
        [
            all_ranges[..., 0].max(axis=0, keepdims=True),
            all_ranges[..., 1].min(axis=0, keepdims=True),
        ]
    )

    click.echo(
        f"Standardized ranges:\nZ: {standardized_ranges[:, 0].tolist()}\n"
        f"Y: {standardized_ranges[:, 1].tolist()}\n"
        f"X: {standardized_ranges[:, 2].tolist()}"
    )

    # Save updated YAML config
    output_model = settings.model_copy()
    output_model.Z_slice = standardized_ranges[:, 0].tolist()
    output_model.Y_slice = standardized_ranges[:, 1].tolist()
    output_model.X_slice = standardized_ranges[:, 2].tolist()
    model_to_yaml(output_model, output_filepath)

    shutil.rmtree(output_path_csv)
    click.echo("Done.")


@click.command("estimate-crop")
@config_filepath()
@output_filepath()
@sbatch_filepath()
@local()
@click.option(
    "--lf-mask-radius",
    type=float,
    help=(
        "Also crop to a circle on the phase channel, of this radius as a fraction of the "
        "half-width of the shorter yx axis (e.g. 0.95, to leave out a bright well edge). "
        "Left out: no circle."
    ),
    required=False,
)
def estimate_crop_cli(
    config_filepath: str,
    output_filepath: str,
    lf_mask_radius: float | None = None,
    sbatch_filepath: str = None,
    local: bool = False,
):
    """Estimate a crop region where both phase and fluorescence volumes are non-zero.

    >>> biahub estimate-crop \
        -c ./concat.yml \
        -o ./cropped_concat.yml \
        --local
    """
    estimate_crop(
        config_filepath=config_filepath,
        output_filepath=output_filepath,
        lf_mask_radius=lf_mask_radius,
        sbatch_filepath=sbatch_filepath,
        local=local,
    )


if __name__ == "__main__":
    estimate_crop_cli()
