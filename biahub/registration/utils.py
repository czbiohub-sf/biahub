import os

from pathlib import Path
from typing import Literal, Tuple, Union

import ants
import click
import largestinteriorrectangle as lir
import numpy as np

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from biahub.cli.utils import (
    model_to_yaml,
)
from biahub.settings import (
    AffineTransformSettings,
    RegistrationSettings,
    StabilizationSettings,
)

# TODO: see if at some point these globals should be hidden or exposed.
NA_DETECTION_SOURCE = 1.35
NA_DETECTION_TARGET = 1.35
WAVELENGTH_EMISSION_SOURCE_CHANNEL = 0.45  # in um
WAVELENGTH_EMISSION_TARGET_CHANNEL = 0.6  # in um
FOCUS_SLICE_ROI_WIDTH = 150  # size of central ROI used to find focal slice


def validate_transforms(
    transforms: list[ArrayLike],
    shape_zyx: tuple[int, int, int],
    window_size: int = 10,
    tolerance: float = 100.0,
    verbose: bool = False,
) -> list[ArrayLike]:
    """
    Validate that a provided list of transforms do not deviate beyond the tolerance threshold
    relative to the average transform within a given window size.

    Parameters
    ----------
    transforms : list[ArrayLike]
        List of affine transformation matrices (4x4), one for each timepoint.
    shape_zyx : tuple[int, int, int]
        Shape of the source (i.e. moving) volume (Z, Y, X).
    window_size : int
        Size of the moving window for smoothing transformations.
    tolerance : float
        Maximum allowed difference between consecutive transformations for validation.
    verbose : bool
        If True, prints detailed logs of the validation process.

    Returns
    -------
    list[ArrayLike]
        List of affine transformation matrices with invalid or inconsistent values replaced by None.
    """
    valid_transforms = []
    reference_transform = None

    for i, transform in enumerate(transforms):
        if transform is not None:
            if len(valid_transforms) < window_size:
                # Bootstrap the buffer without validating yet
                valid_transforms.append(transform)
                reference_transform = np.mean(valid_transforms, axis=0)
                if verbose:
                    click.echo(
                        f"[Bootstrap] Accepting transform at timepoint {i} (no validation)"
                    )
            elif check_transforms_difference(
                transform, reference_transform, shape_zyx, tolerance, verbose
            ):
                valid_transforms.append(transform)
                if len(valid_transforms) > window_size:
                    valid_transforms.pop(0)
                reference_transform = np.mean(valid_transforms, axis=0)
                if verbose:
                    click.echo(f"Transform at timepoint {i} is valid")
            else:
                transforms[i] = None
                if verbose:
                    click.echo(
                        f"Transform at timepoint {i} is invalid and will be interpolated"
                    )
        else:
            transforms[i] = None
            if verbose:
                click.echo(f"Transform at timepoint {i} is None and will be interpolated")

    return transforms


def interpolate_transforms(
    transforms: list[ArrayLike],
    window_size: int = 3,
    interpolation_type: Literal["linear", "cubic"] = "linear",
    verbose: bool = False,
):
    """
    Interpolate missing transforms (None) in a list of affine transformation matrices.

    Parameters
    ----------
    transforms : list[ArrayLike]
        List of affine transformation matrices (4x4), one for each timepoint.
    window_size : int
        Local window radius for interpolation. If 0, global interpolation is used.
    interpolation_type : Literal["linear", "cubic"]
        Interpolation type.
    verbose : bool
        If True, prints detailed logs of the interpolation process.

    Returns
    -------
    list[ArrayLike]
        List of affine transformation matrices with missing values filled via linear interpolation.
    """
    n = len(transforms)
    valid_transform_indices = [i for i, t in enumerate(transforms) if t is not None]
    valid_transforms = [np.array(transforms[i]) for i in valid_transform_indices]

    if not valid_transform_indices or len(valid_transform_indices) < 2:
        raise ValueError("At least two valid transforms are required for interpolation.")

    missing_indices = [i for i in range(n) if transforms[i] is None]

    if not missing_indices:
        return transforms  # nothing to do
    if verbose:
        click.echo(f"Interpolating missing transforms at timepoints: {missing_indices}")

    if window_size > 0:
        for idx in missing_indices:
            # Define local window
            start = max(0, idx - window_size)
            end = min(n, idx + window_size + 1)

            local_x = []
            local_y = []

            for j in range(start, end):
                if j in valid_transform_indices:
                    local_x.append(j)
                    local_y.append(np.array(transforms[j]))

            if len(local_x) < 2:
                # Not enough neighbors for interpolation. Assign to closes valid transform
                closest_valid_idx = valid_transform_indices[
                    np.argmin(np.abs(np.asarray(valid_transform_indices) - idx))
                ]
                transforms[idx] = transforms[closest_valid_idx]
                if verbose:
                    click.echo(
                        f"Not enough interpolation neighbors were found for timepoint {idx} using closest valid transform at timepoint {closest_valid_idx}"
                    )
                continue

            f = interp1d(
                local_x, local_y, axis=0, kind=interpolation_type, fill_value='extrapolate'
            )
            transforms[idx] = f(idx).tolist()
            if verbose:
                click.echo(f"Interpolated timepoint {idx} using neighbors: {local_x}")

    else:
        # Global interpolation using all valid transforms
        f = interp1d(
            valid_transform_indices,
            valid_transforms,
            axis=0,
            kind='linear',
            fill_value='extrapolate',
        )
        transforms = [
            f(i).tolist() if transforms[i] is None else transforms[i] for i in range(n)
        ]

    return transforms


def check_transforms_difference(
    tform1: ArrayLike,
    tform2: ArrayLike,
    shape_zyx: tuple[int, int, int],
    threshold: float = 5.0,
    verbose: bool = False,
):
    """
    Evaluate the difference between two affine transforms by calculating the
    Mean Squared Error (MSE) of a grid of points transformed by each matrix.

    Parameters
    ----------
    tform1 : ArrayLike
        First affine transform (4x4 matrix).
    tform2 : ArrayLike
        Second affine transform (4x4 matrix).
    shape_zyx : tuple[int, int, int]
        Shape of the source (i.e. moving) volume (Z, Y, X).
    threshold : float
        The maximum allowed MSE difference.
    verbose : bool
        Flag to print the MSE difference.

    Returns
    -------
    bool
        True if the MSE difference is within the threshold, False otherwise.
    """
    tform1 = np.array(tform1)
    tform2 = np.array(tform2)
    (Z, Y, X) = shape_zyx

    zz, yy, xx = np.meshgrid(
        np.linspace(0, Z - 1, 10), np.linspace(0, Y - 1, 10), np.linspace(0, X - 1, 10)
    )

    grid_points = np.vstack([zz.ravel(), yy.ravel(), xx.ravel(), np.ones(zz.size)]).T

    points_tform1 = np.dot(tform1, grid_points.T).T
    points_tform2 = np.dot(tform2, grid_points.T).T

    differences = np.linalg.norm(points_tform1[:, :3] - points_tform2[:, :3], axis=1)
    mse = np.mean(differences)

    if verbose:
        click.echo(f'MSE of transformed points: {mse:.2f}; threshold: {threshold:.2f}')
    return mse <= threshold


def evaluate_transforms(
    transforms: ArrayLike,
    shape_zyx: tuple[int, int, int],
    validation_window_size: int = 10,
    validation_tolerance: float = 100.0,
    interpolation_window_size: int = 3,
    interpolation_type: Literal["linear", "cubic"] = "linear",
    verbose: bool = False,
) -> ArrayLike:
    """
    Evaluate a list of affine transformation matrices.
    Transform matrices are checked for deviation from the average within a given window size.
    If a transform is found to lead to shift larger than the given tolerance,
    that transform will be replaced by interpolation of valid transforms within a given window size.

    Parameters
    ----------
    transforms : ArrayLike
        List of affine transformation matrices (4x4), one for each timepoint.
    shape_zyx : tuple[int, int, int]
        Shape of the source (i.e. moving) volume (Z, Y, X).
    validation_window_size : int
        Size of the moving window for smoothing transformations.
    validation_tolerance : float
        Maximum allowed difference between consecutive transformations for validation.
    interpolation_window_size : int
        Size of the local window for interpolation.
    interpolation_type : Literal["linear", "cubic"]
        Interpolation type.
    verbose : bool
        If True, prints detailed logs of the evaluation and validation process.

    Returns
    -------
    list[ArrayLike]
        List of affine transformation matrices with missing values filled via linear interpolation.
    """

    if not isinstance(transforms, list):
        transforms = transforms.tolist()
    if len(transforms) < validation_window_size:
        raise Warning(
            f"Not enough transforms for validation and interpolation. "
            f"Required: {validation_window_size}, "
            f"Provided: {len(transforms)}"
        )
    else:
        transforms = validate_transforms(
            transforms=transforms,
            window_size=validation_window_size,
            tolerance=validation_tolerance,
            shape_zyx=shape_zyx,
            verbose=verbose,
        )

    if len(transforms) < interpolation_window_size:
        raise Warning(
            f"Not enough transforms for interpolation. "
            f"Required: {interpolation_window_size}, "
            f"Provided: {len(transforms)}"
        )
    else:
        transforms = interpolate_transforms(
            transforms=transforms,
            window_size=interpolation_window_size,
            interpolation_type=interpolation_type,
            verbose=verbose,
        )
    return transforms


def save_transforms(
    model: Union[AffineTransformSettings, StabilizationSettings, RegistrationSettings],
    transforms: list[ArrayLike],
    output_filepath_settings: Path,
    output_filepath_plot: Path = None,
    verbose: bool = False,
):
    """
    Save the transforms to a yaml file and plot the translations.

    Parameters
    ----------
    model : Union[AffineTransformSettings, StabilizationSettings, RegistrationSettings]
        Model to save the transforms to.
    transforms : list[ArrayLike]
        List of affine transformation matrices (4x4), one for each timepoint.
    output_filepath_settings : Path
        Path to the output settings file.
    output_filepath_plot : Path
        Path to the output plot file.
    verbose : bool
        If True, prints detailed logs of the saving process.

    Returns
    -------
    None

    Notes
    -----
    The transforms are saved to a yaml file and a plot of the translations is saved to a png file.
    The plot is saved in the same directory as the settings file and is named "translations.png".

    """
    if transforms is None or len(transforms) == 0:
        raise ValueError("Transforms are empty")

    if not isinstance(transforms, list):
        transforms = transforms.tolist()

    model.affine_transform_zyx_list = transforms

    if output_filepath_settings.suffix not in [".yml", ".yaml"]:
        output_filepath_settings = output_filepath_settings.with_suffix(".yml")

    output_filepath_settings.parent.mkdir(parents=True, exist_ok=True)
    model_to_yaml(model, output_filepath_settings)

    if verbose and output_filepath_plot is not None:
        if output_filepath_plot.suffix not in [".png"]:
            output_filepath_plot = output_filepath_plot.with_suffix(".png")
        output_filepath_plot.parent.mkdir(parents=True, exist_ok=True)

        plot_translations(np.asarray(transforms), output_filepath_plot)


def plot_translations(
    transforms_zyx: ArrayLike,
    output_filepath: Path,
):
    """
    Plot the translations of a list of affine transformation matrices.

    Parameters
    ----------
    transforms_zyx : ArrayLike
        List of affine transformation matrices (4x4), one for each timepoint.
    output_filepath : Path
        Path to the output plot file.
    Returns
    -------
    None

    Notes
    -----
    The plot is saved as a png file.
    The plot is saved in the same directory as the output file.
    The plot is saved as a png file.
    """
    transforms_zyx = np.asarray(transforms_zyx)
    os.makedirs(output_filepath.parent, exist_ok=True)

    z_transforms = transforms_zyx[:, 0, 3]
    y_transforms = transforms_zyx[:, 1, 3]
    x_transforms = transforms_zyx[:, 2, 3]
    _, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(z_transforms)
    axs[0].set_title("Z-Translation")
    axs[1].plot(x_transforms)
    axs[1].set_title("X-Translation")
    axs[2].plot(y_transforms)
    axs[2].set_title("Y-Translation")
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()


def convert_transform_to_ants(T_numpy: np.ndarray):
    """Homogeneous 3D transformation matrix from numpy to ants

    Parameters
    ----------
    numpy_transform :4x4 homogenous matrix

    Returns
    -------
    Ants transformation matrix object
    """
    assert T_numpy.shape == (4, 4)

    T_ants_style = T_numpy[:, :-1].ravel()
    T_ants_style[-3:] = T_numpy[:3, -1]
    T_ants = ants.new_ants_transform(
        transform_type="AffineTransform",
    )
    T_ants.set_parameters(T_ants_style)

    return T_ants


def convert_transform_to_numpy(T_ants):
    """
    Convert the ants transformation matrix to numpy 3D homogenous transform

    Modified from Jordao's dexp code

    Parameters
    ----------
    T_ants : Ants transfromation matrix object

    Returns
    -------
    np.array
        Converted Ants to numpy array

    """

    T_numpy = T_ants.parameters.reshape((3, 4), order="F")
    T_numpy[:, :3] = T_numpy[:, :3].transpose()
    T_numpy = np.vstack((T_numpy, np.array([0, 0, 0, 1])))

    # Reference:
    # https://sourceforge.net/p/advants/discussion/840261/thread/9fbbaab7/
    # https://github.com/netstim/leaddbs/blob/a2bb3e663cf7fceb2067ac887866124be54aca7d/helpers/ea_antsmat2mat.m
    # T = original translation offset from A
    # T = T + (I - A) @ centering

    T_numpy[:3, -1] += (np.eye(3) - T_numpy[:3, :3]) @ T_ants.fixed_parameters

    return T_numpy


def find_lir(registered_zyx: np.ndarray, plot: bool = False) -> Tuple:
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


def find_overlapping_volume(
    input_zyx_shape: Tuple,
    target_zyx_shape: Tuple,
    transformation_matrix: np.ndarray,
    method: str = "LIR",
    plot: bool = False,
) -> Tuple:
    """
    Find the overlapping rectangular volume after registration of two 3D datasets

    Parameters
    ----------
    input_zyx_shape : Tuple
        shape of input array
    target_zyx_shape : Tuple
        shape of target array
    transformation_matrix : np.ndarray
        affine transformation matrix
    method : str, optional
        method of finding the overlapping volume, by default 'LIR'

    Returns
    -------
    Tuple
        ZYX slices of the overlapping volume after registration

    """

    # Make dummy volumes
    moving_volume = np.ones(tuple(input_zyx_shape), dtype=np.float32)
    fixed_volume = np.ones(tuple(target_zyx_shape), dtype=np.float32)

    # Convert to ants objects
    fixed_volume_ants = ants.from_numpy(fixed_volume)
    moving_volume_ants = ants.from_numpy(moving_volume)

    tform_ants = convert_transform_to_ants(transformation_matrix)

    # Now apply the transform using this grid
    registered_volume = tform_ants.apply_to_image(
        moving_volume_ants, reference=fixed_volume_ants
    ).numpy()
    if method == "LIR":
        click.echo("Starting Largest interior rectangle (LIR) search")
        mask = (registered_volume > 0) & (fixed_volume > 0)
        z_slice, y_slice, x_slice = find_lir(mask, plot=plot)

    else:
        raise ValueError(f"Unknown method {method}")

    return (z_slice, y_slice, x_slice)


def rescale_voxel_size(affine_matrix, input_scale):
    return np.linalg.norm(affine_matrix, axis=1) * input_scale


def load_transforms(transforms_path: Path, T: int, verbose: bool = False) -> list[ArrayLike]:
    # Load the transforms
    transforms = []
    for t in range(T):
        file_path = transforms_path / f"{t}.npy"
        if not os.path.exists(file_path):
            transforms.append(None)
            if verbose:
                click.echo(f"Transform for timepoint {t} not found.")

        else:
            matrix = np.load(file_path)
            transforms.append(matrix.tolist())

            if verbose:
                click.echo(f"Transform for timepoint {t}: {matrix}")

    return transforms


def get_3D_rescaling_matrix(start_shape_zyx, scaling_factor_zyx=(1, 1, 1), end_shape_zyx=None):
    center_Y_start, center_X_start = np.array(start_shape_zyx)[-2:] / 2
    if end_shape_zyx is None:
        center_Y_end, center_X_end = (center_Y_start, center_X_start)
    else:
        center_Y_end, center_X_end = np.array(end_shape_zyx)[-2:] / 2

    scaling_matrix = np.array(
        [
            [scaling_factor_zyx[-3], 0, 0, 0],
            [
                0,
                scaling_factor_zyx[-2],
                0,
                -center_Y_start * scaling_factor_zyx[-2] + center_Y_end,
            ],
            [
                0,
                0,
                scaling_factor_zyx[-1],
                -center_X_start * scaling_factor_zyx[-1] + center_X_end,
            ],
            [0, 0, 0, 1],
        ]
    )
    return scaling_matrix


def get_3D_rotation_matrix(
    start_shape_zyx: Tuple, angle: float = 0.0, end_shape_zyx: Tuple = None
) -> np.ndarray:
    """
    Rotate Transformation Matrix

    Parameters
    ----------
    start_shape_zyx : Tuple
        Shape of the input
    angle : float, optional
        Angles of rotation in degrees
    end_shape_zyx : Tuple, optional
       Shape of output space

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    # TODO: make this 3D?
    center_Y_start, center_X_start = np.array(start_shape_zyx)[-2:] / 2
    if end_shape_zyx is None:
        center_Y_end, center_X_end = (center_Y_start, center_X_start)
    else:
        center_Y_end, center_X_end = np.array(end_shape_zyx)[-2:] / 2

    theta = np.radians(angle)

    rotation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [
                0,
                np.cos(theta),
                -np.sin(theta),
                -center_Y_start * np.cos(theta)
                + np.sin(theta) * center_X_start
                + center_Y_end,
            ],
            [
                0,
                np.sin(theta),
                np.cos(theta),
                -center_Y_start * np.sin(theta)
                - center_X_start * np.cos(theta)
                + center_X_end,
            ],
            [0, 0, 0, 1],
        ]
    )
    return rotation_matrix


def get_3D_fliplr_matrix(start_shape_zyx: tuple, end_shape_zyx: tuple = None) -> np.ndarray:
    """
    Get 3D left-right flip transformation matrix.

    Parameters
    ----------
    start_shape_zyx : tuple
        Shape of the source volume (Z, Y, X).
    end_shape_zyx : tuple, optional
        Shape of the target volume (Z, Y, X). If None, uses start_shape_zyx.

    Returns
    -------
    np.ndarray
        4x4 transformation matrix for left-right flip.
    """
    center_X_start = start_shape_zyx[-1] / 2
    if end_shape_zyx is None:
        center_X_end = center_X_start
    else:
        center_X_end = end_shape_zyx[-1] / 2

    # Flip matrix: reflects across X axis and translates to maintain center
    flip_matrix = np.array(
        [
            [1, 0, 0, 0],  # Z unchanged
            [0, 1, 0, 0],  # Y unchanged
            [0, 0, -1, 2 * center_X_end],  # X flipped and translated
            [0, 0, 0, 1],  # Homogeneous coordinate
        ]
    )
    return flip_matrix



def apply_affine_transform(
    zyx_data: np.ndarray,
    matrix: np.ndarray,
    output_shape_zyx: Tuple,
    method="ants",
    interpolation: str = "linear",
    crop_output_slicing: bool = None,
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    zyx_data : np.ndarray
        3D input array to be transformed
    matrix : np.ndarray
        3D Homogenous transformation matrix
    output_shape_zyx : Tuple
        output target zyx shape
    method : str, optional
        method to use for transformation, by default 'ants'
    interpolation: str, optional
        interpolation mode for ants, by default "linear"
    crop_output : bool, optional
        crop the output to the largest interior rectangle, by default False

    Returns
    -------
    np.ndarray
        registered zyx data
    """

    Z, Y, X = output_shape_zyx
    if crop_output_slicing is not None:
        Z_slice, Y_slice, X_slice = crop_output_slicing
        Z = Z_slice.stop - Z_slice.start
        Y = Y_slice.stop - Y_slice.start
        X = X_slice.stop - X_slice.start

    # TODO: based on the signature of this function, it should not be called on 4D array
    if zyx_data.ndim == 4:
        registered_czyx = np.zeros((zyx_data.shape[0], Z, Y, X), dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            registered_czyx[c] = apply_affine_transform(
                zyx_data[c],
                matrix,
                output_shape_zyx,
                method=method,
                interpolation=interpolation,
                crop_output_slicing=crop_output_slicing,
            )
        return registered_czyx
    else:
        # Convert nans to 0
        zyx_data = np.nan_to_num(zyx_data, nan=0)

        # NOTE: default set to ANTS apply_affine method until we decide we get a benefit from using cupy
        # The ants method on CPU is 10x faster than scipy on CPU. Cupy method has not been bencharked vs ANTs

        if method == "ants":
            # The output has to be a ANTImage Object
            empty_target_array = np.zeros((output_shape_zyx), dtype=np.float32)
            target_zyx_ants = ants.from_numpy(empty_target_array)

            T_ants = convert_transform_to_ants(matrix)

            zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
            registered_zyx = T_ants.apply_to_image(
                zyx_data_ants, reference=target_zyx_ants, interpolation=interpolation
            ).numpy()

        elif method == "scipy":
            registered_zyx = scipy.ndimage.affine_transform(zyx_data, matrix, output_shape_zyx)

        else:
            raise ValueError(f"Unknown method {method}")

        # Crop the output to the largest interior rectangle
        if crop_output_slicing is not None:
            registered_zyx = registered_zyx[Z_slice, Y_slice, X_slice]

    return registered_zyx

