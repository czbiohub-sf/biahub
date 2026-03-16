import contextlib
import inspect
import io
import itertools
import multiprocessing as mp

from functools import partial
from pathlib import Path

import click
import numpy as np
import yaml

from iohub.ngff import Position, open_ome_zarr
from iohub.ngff.models import TransformationMeta
from numpy.typing import DTypeLike
from tqdm import tqdm


def update_model(model_instance, update_dict: dict) -> type:
    """
    Properly update a Pydantic model with only the provided values while keeping the defaults.

    This ensures that nested models retain missing values instead of getting overwritten.

    Parameters
    ----------
    model_instance
        The Pydantic model instance to update.
    update_dict : dict
        Dictionary of field names and values to update.

    Returns
    -------
    type
        A new model instance with updated fields.
    """
    updated_fields = {}
    for key, value in update_dict.items():
        if isinstance(value, dict) and hasattr(model_instance, key):
            # If it's a nested dict, update the nested Pydantic model properly
            nested_model = getattr(model_instance, key)
            updated_fields[key] = nested_model.copy(update=value)
        else:
            # Otherwise, just update the value directly
            updated_fields[key] = value

    # Create a new instance with updated fields
    return model_instance.copy(update=updated_fields)


# TODO: convert all code to use this function from now on
def create_empty_hcs_zarr(
    store_path: Path,
    position_keys: list[tuple[str, str, str]],
    channel_names: list[str],
    shape: tuple[int, int, int, int, int],
    chunks: tuple[int, int, int, int, int] | None = None,
    scale: tuple[float, float, float, float, float] = (1, 1, 1, 1, 1),
    dtype: DTypeLike = np.float32,
    max_chunk_size_bytes: float = 500e6,
) -> None:
    """
    Create an empty HCS plate Zarr store or append positions and channels if it exists.

    If the plate does not exist, create an empty zarr plate. If the plate exists,
    append positions and channels if they are not already in the plate.

    Parameters
    ----------
    store_path : Path
        HCS plate path.
    position_keys : list[tuple[str, str, str]]
        Position keys, will append if not present in the plate.
        e.g. [("A", "1", "0"), ("A", "1", "1")].
    channel_names : list[str]
        Channel names, will append if not present in metadata.
    shape : tuple[int, int, int, int, int]
        Shape of the data array in (T, C, Z, Y, X) format.
    chunks : tuple[int, int, int, int, int] | None, optional
        Chunk size in (T, C, Z, Y, X) format. If None, chunks are automatically
        calculated to fit within max_chunk_size_bytes, by default None.
    scale : tuple[float, float, float, float, float], optional
        Physical scale of the data in (T, C, Z, Y, X) format, by default (1, 1, 1, 1, 1).
    dtype : DTypeLike, optional
        Data type of the array, by default np.float32.
    max_chunk_size_bytes : float, optional
        Maximum chunk size in bytes for automatic chunk calculation, by default 500e6.

    Returns
    -------
    None
        The Zarr store is created or modified in-place.

    Notes
    -----
    Modifying from recOrder
    https://github.com/mehta-lab/recOrder/blob/d31ad910abf84c65ba927e34561f916651cbb3e8/recOrder/cli/utils.py#L12
    """
    MAX_CHUNK_SIZE = max_chunk_size_bytes  # in bytes
    bytes_per_pixel = np.dtype(dtype).itemsize

    # Limiting the chunking to 500MB
    if chunks is None:
        chunk_zyx_shape = list(shape[-3:])
        # chunk_zyx_shape[-3] > 1 ensures while loop will not stall if single
        # XY image is larger than MAX_CHUNK_SIZE
        while (
            chunk_zyx_shape[-3] > 1
            and np.prod(chunk_zyx_shape) * bytes_per_pixel > MAX_CHUNK_SIZE
        ):
            chunk_zyx_shape[-3] = np.ceil(chunk_zyx_shape[-3] / 2).astype(int)
        chunk_zyx_shape = tuple(chunk_zyx_shape)

        chunks = 2 * (1,) + chunk_zyx_shape

    # Create plate
    output_plate = open_ome_zarr(
        str(store_path), layout="hcs", mode="a", channel_names=channel_names
    )
    transform = [TransformationMeta(type="scale", scale=scale)]

    # Create positions
    for position_key in position_keys:
        position_key_string = "/".join(position_key)
        # Check if position is already in the store, if not create it
        if position_key_string not in output_plate.zgroup:
            position = output_plate.create_position(*position_key)
            _ = position.create_zeros(
                name="0",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                transform=transform,
            )
        else:
            position = output_plate[position_key_string]

    # Check if channel_names are already in the store, if not append them
    for channel_name in channel_names:
        # Read channel names directly from metadata to avoid race conditions
        metadata_channel_names = [
            channel.label for channel in position.metadata.omero.channels
        ]
        if channel_name not in metadata_channel_names:
            position.append_channel(channel_name, resize_arrays=True)


def get_output_paths(
    input_paths: list[Path],
    output_zarr_path: Path,
    ensure_unique_positions: bool | None = None,
) -> list[Path]:
    """
    Generates a mirrored output path list given an input list of positions

    Parameters
    ----------
    input_paths : list[Path]
        List of input position paths
    output_zarr_path : Path
        Base output zarr path
    ensure_unique_positions : bool | None, optional
        If True, ensures unique output position paths by appending a suffix to the column part
        when duplicate position names are detected.
        For example, if "A/1/0" is duplicated, it becomes "A/1d0/0", "A/1d1/0", etc., by default None.

    Returns
    -------
    list[Path]
        List of output position paths
    """
    list_output_path = []

    # Track position names to ensure uniqueness if required
    position_name_counts = {}

    for path in input_paths:
        # Select the Row/Column/FOV parts of input path
        path_strings = Path(path).parts[-3:]
        position_name = "/".join(path_strings)

        # If we need to ensure uniqueness and this position name has been seen before
        if ensure_unique_positions and position_name in position_name_counts:
            # Increment the count for this position name
            position_name_counts[position_name] += 1

            # Create a new position name by appending a suffix to the column part
            # For example, "A/1/0" becomes "A/1d0/0", "A/1d1/0", etc.
            modified_path_strings = list(path_strings)

            # Append the suffix to the column part
            modified_path_strings[1] = (
                f"{modified_path_strings[1]}d{position_name_counts[position_name]}"
            )

            # Append the modified position path
            list_output_path.append(Path(output_zarr_path, *modified_path_strings))
        else:
            # First time seeing this position name or uniqueness not required
            if ensure_unique_positions:
                position_name_counts[position_name] = 0

            # Append the original position path
            list_output_path.append(Path(output_zarr_path, *path_strings))

    return list_output_path


def apply_function_to_zyx_and_save(
    func, position: Position, output_path: Path, t_idx: int, c_idx: int, **kwargs
) -> None:
    """
    Load a ZYX array from a Position object, apply a transformation and save the result to file.

    Parameters
    ----------
    func
        Function to apply to the ZYX array.
    position : Position
        Position object containing the input data.
    output_path : Path
        Path to the output Zarr store where the result will be saved.
    t_idx : int
        Time index to process.
    c_idx : int
        Channel index to process.
    **kwargs
        Additional keyword arguments to pass to the function.

    Returns
    -------
    None
        The processed data is written to the output store.
    """
    click.echo(f"Processing c={c_idx}, t={t_idx}")

    zyx_data = position[0][t_idx, c_idx]
    if _check_nan_n_zeros(zyx_data):
        click.echo(f"Skipping c={c_idx}, t={t_idx} due to all zeros or nans")
    else:
        # Apply function
        processed_zyx = func(zyx_data, **kwargs)

        # Write to file
        with open_ome_zarr(output_path, mode="r+") as output_dataset:
            output_dataset[0][t_idx, c_idx] = processed_zyx

        click.echo(f"Finished Writing.. c={c_idx}, t={t_idx}")


def apply_transform_to_zyx_and_save_v2(
    func,
    position: Position,
    output_path: Path,
    input_channel_indices: list[int],
    output_channel_indices: list[int],
    t_idx: int,
    t_idx_out: int,
    c_idx: int | None = None,
    **kwargs,
) -> None:
    """
    Load a ZYX or CZYX array from a Position object, apply a transformation and save the result to file.

    Parameters
    ----------
    func
        Function to apply to the data array.
    position : Position
        Position object containing the input data.
    output_path : Path
        Path to the output Zarr store where the result will be saved.
    input_channel_indices : list[int]
        List of input channel indices to process.
    output_channel_indices : list[int]
        List of output channel indices where results will be written.
    t_idx : int
        Input time index to process.
    t_idx_out : int
        Output time index where results will be written.
    c_idx : int | None, optional
        Channel index (used when input_channel_indices is empty), by default None.
    **kwargs
        Additional keyword arguments to pass to the function.

    Returns
    -------
    None
        The processed data is written to the output store.
    """

    # TODO: temporary fix to slumkit issue
    if _is_nested(input_channel_indices):
        input_channel_indices = [int(x) for x in input_channel_indices if x.isdigit()]
    if _is_nested(output_channel_indices):
        output_channel_indices = [int(x) for x in output_channel_indices if x.isdigit()]

    # Check if t_idx should be added to the func kwargs
    # This is needed when a different processing is needed for each time point, for example during stabilization
    all_func_params = inspect.signature(func).parameters.keys()
    if "t_idx" in all_func_params:
        kwargs["t_idx"] = t_idx

    # Process CZYX vs ZYX
    if input_channel_indices is not None and len(input_channel_indices) > 0:
        click.echo(f"Processing t={t_idx}")

        czyx_data = position.data.oindex[t_idx, input_channel_indices]
        if not _check_nan_n_zeros(czyx_data):
            transformed_czyx = func(czyx_data, **kwargs)
            # Write to file
            with open_ome_zarr(output_path, mode="r+") as output_dataset:
                output_dataset[0].oindex[t_idx_out, output_channel_indices] = transformed_czyx
            click.echo(f"Finished Writing.. t={t_idx}")
        else:
            click.echo(f"Skipping t={t_idx} due to all zeros or nans")
    else:
        click.echo(f"Processing c={c_idx}, t={t_idx}")

        czyx_data = position.data.oindex[t_idx, c_idx : c_idx + 1]
        # Checking if nans or zeros and skip processing
        if not _check_nan_n_zeros(czyx_data):
            # Apply transformation
            transformed_czyx = func(czyx_data, **kwargs)

            # Write to file
            with open_ome_zarr(output_path, mode="r+") as output_dataset:
                output_dataset[0][t_idx_out, c_idx : c_idx + 1] = transformed_czyx

            click.echo(f"Finished Writing.. c={c_idx}, t={t_idx}")
        else:
            click.echo(f"Skipping c={c_idx}, t={t_idx} due to all zeros or nans")


def process_single_position(
    func,
    input_data_path: Path,
    output_path: Path,
    num_processes: int = mp.cpu_count(),
    **kwargs,
) -> None:
    """
    Process a single position with multiprocessing parallelization over T and C axes.

    Parameters
    ----------
    func
        Function to apply to each ZYX array.
    input_data_path : Path
        Path to the input OME-Zarr dataset.
    output_path : Path
        Path to the output OME-Zarr dataset.
    num_processes : int, optional
        Number of processes to use for parallelization, by default mp.cpu_count().
    **kwargs
        Additional keyword arguments to pass to the function.

    Returns
    -------
    None
        Processed data is written to the output dataset.
    """
    # Function to be applied
    click.echo(f"Function to be applied: \t{func}")

    # Get the reader and writer
    click.echo(f"Input data path:\t{input_data_path}")
    click.echo(f"Output data path:\t{str(output_path)}")
    input_dataset = open_ome_zarr(str(input_data_path))
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        input_dataset.print_tree()
    click.echo(f" Input data tree: {stdout_buffer.getvalue()}")

    T, C, _, _, _ = input_dataset.data.shape

    # Check the arguments for the function
    all_func_params = inspect.signature(func).parameters.keys()
    # Extract the relevant kwargs for the function 'func'
    func_args = {}
    non_func_args = {}

    for k, v in kwargs.items():
        if k in all_func_params:
            func_args[k] = v
        else:
            non_func_args[k] = v

    # Write the settings into the metadata if existing
    # TODO: alternatively we can throw all extra arguments as metadata.
    if 'extra_metadata' in non_func_args:
        # For each dictionary in the nest
        with open_ome_zarr(output_path, mode='r+') as output_dataset:
            for params_metadata_keys in kwargs['extra_metadata'].keys():
                output_dataset.zattrs['extra_metadata'] = non_func_args['extra_metadata']

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"\nStarting multiprocess pool with {num_processes} processes")
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial(
                apply_function_to_zyx_and_save,
                func,
                input_dataset,
                str(output_path),
                **func_args,
            ),
            itertools.product(range(T), range(C)),
        )


# TODO: modifiy how we get the time and channesl like recOrder (isinstance(input, list) or instance(input,int) or all)
def process_single_position_v2(
    func,
    input_data_path: Path,
    output_path: Path,
    time_indices: list[int] | str = [0],
    time_indices_out: list[int] | None = None,
    input_channel_idx: list[int] | None = None,
    output_channel_idx: list[int] | None = None,
    num_processes: int = mp.cpu_count(),
    **kwargs,
) -> None:
    """
    Process a single position with multiprocessing parallelization over T and C axes.

    This is an improved version that supports flexible time and channel indexing.

    Parameters
    ----------
    func
        Function to apply to each data array.
    input_data_path : Path
        Path to the input OME-Zarr dataset.
    output_path : Path
        Path to the output OME-Zarr dataset.
    time_indices : list[int] | str, optional
        List of time indices to process, or "all" to process all timepoints, by default [0].
    time_indices_out : list[int] | None, optional
        List of output time indices. If None, matches time_indices, by default None.
    input_channel_idx : list[int] | None, optional
        List of input channel indices. If None or empty, processes all channels individually, by default None.
    output_channel_idx : list[int] | None, optional
        List of output channel indices. If None or empty, matches input_channel_idx, by default None.
    num_processes : int, optional
        Number of processes to use for parallelization, by default mp.cpu_count().
    **kwargs
        Additional keyword arguments to pass to the function.

    Returns
    -------
    None
        Processed data is written to the output dataset.
    """
    # Function to be applied
    click.echo(f"Function to be applied: \t{func}")

    # Get the reader and writer
    click.echo(f"Input data path:\t{input_data_path}")
    click.echo(f"Output data path:\t{str(output_path)}")
    input_dataset = open_ome_zarr(str(input_data_path))
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        input_dataset.print_tree()
    click.echo(f" Input data tree: {stdout_buffer.getvalue()}")

    # Find time indices
    if time_indices == "all":
        time_indices = range(input_dataset.data.shape[0])
        time_indices_out = time_indices
    elif isinstance(time_indices, list):
        time_indices_out = range(len(time_indices))

    # Check for invalid times
    time_ubound = input_dataset.data.shape[0] - 1
    if np.max(time_indices) > time_ubound:
        raise ValueError(
            f"time_indices = {time_indices} includes a time index beyond the maximum index of the dataset = {time_ubound}"
        )

    # Check the arguments for the function
    all_func_params = inspect.signature(func).parameters.keys()
    # Extract the relevant kwargs for the function 'func'
    func_args = {}
    non_func_args = {}

    for k, v in kwargs.items():
        if k in all_func_params:
            func_args[k] = v
        else:
            non_func_args[k] = v

    # Write the settings into the metadata if existing
    if 'extra_metadata' in non_func_args:
        # For each dictionary in the nest
        with open_ome_zarr(output_path, mode='r+') as output_dataset:
            for params_metadata_keys in kwargs['extra_metadata'].keys():
                output_dataset.zattrs['extra_metadata'] = non_func_args['extra_metadata']

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"\nStarting multiprocess pool with {num_processes} processes")

    if input_channel_idx is None or len(input_channel_idx) == 0:
        # If C is not empty, use itertools.product with both ranges
        _, C, _, _, _ = input_dataset.data.shape
        iterable = [
            (time_idx, time_idx_out, c)
            for (time_idx, time_idx_out), c in itertools.product(
                zip(time_indices, time_indices_out), range(C)
            )
        ]
        partial_apply_transform_to_zyx_and_save = partial(
            apply_transform_to_zyx_and_save_v2,
            func,
            input_dataset,
            output_path / Path(*input_data_path.parts[-3:]),
            input_channel_idx,
            output_channel_idx,
            **func_args,
        )
    else:
        # If C is empty, use only the range for time_indices
        iterable = list(zip(time_indices, time_indices_out))
        partial_apply_transform_to_zyx_and_save = partial(
            apply_transform_to_zyx_and_save_v2,
            func,
            input_dataset,
            output_path / Path(*input_data_path.parts[-3:]),
            input_channel_idx,
            output_channel_idx,
            c_idx=0,
            **func_args,
        )

    click.echo(f"\nStarting multiprocess pool with {num_processes} processes")
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial_apply_transform_to_zyx_and_save,
            iterable,
        )


def copy_n_paste(zyx_data: np.ndarray, zyx_slicing_params: list[slice]) -> np.ndarray:
    """
    Crop a ZYX array given a list of ZYX slices.

    Parameters
    ----------
    zyx_data : np.ndarray
        Data array to crop, with shape (Z, Y, X).
    zyx_slicing_params : list[slice]
        List of slicing parameters for z, y, x dimensions.
        Each element is a slice object: [z_slice, y_slice, x_slice].

    Returns
    -------
    np.ndarray
        Cropped array from the input zyx_data given the slicing parameters.
    """
    # Replace NaN values with zeros
    zyx_data = np.nan_to_num(zyx_data, nan=0)
    zyx_data_sliced = zyx_data[
        zyx_slicing_params[0],
        zyx_slicing_params[1],
        zyx_slicing_params[2],
    ]
    return zyx_data_sliced


def copy_n_paste_czyx(czyx_data: np.ndarray, czyx_slicing_params: list[slice]) -> np.ndarray:
    """
    Crop a CZYX array given a list of ZYX slices.

    Parameters
    ----------
    czyx_data : np.ndarray
        Data array to crop, with shape (C, Z, Y, X).
    czyx_slicing_params : list[slice]
        List of slicing parameters for z, y, x dimensions.
        Each element is a slice object: [z_slice, y_slice, x_slice].

    Returns
    -------
    np.ndarray
        Cropped array from the input czyx_data given the slicing parameters.
    """
    czyx_data_sliced = czyx_data[
        :,
        czyx_slicing_params[0],
        czyx_slicing_params[1],
        czyx_slicing_params[2],
    ]
    return czyx_data_sliced


def append_channels(input_data_path: Path, target_data_path: Path) -> None:
    """
    Append channels from an input Zarr store to a target Zarr store.

    Parameters
    ----------
    input_data_path : Path
        Path to the input Zarr store containing channels to append.
    target_data_path : Path
        Path to the target Zarr store where channels will be appended.

    Returns
    -------
    None
        Channels are appended to the target store in-place.
    """
    appending_dataset = open_ome_zarr(input_data_path, mode="r")
    appending_channel_names = appending_dataset.channel_names
    with open_ome_zarr(target_data_path, mode="r+") as dataset:
        target_data_channel_names = dataset.channel_names
        num_channels = len(target_data_channel_names) - 1
        print(f"channels in target {target_data_channel_names}")
        print(f"adding channels {appending_channel_names}")
        for name, position in tqdm(dataset.positions(), desc="Positions"):
            for i, appending_channel_idx in enumerate(
                tqdm(appending_channel_names, desc="Channel", leave=False)
            ):
                position.append_channel(appending_channel_idx)
                position["0"][:, num_channels + i + 1] = appending_dataset[str(name)][0][:, i]
        dataset.print_tree()
    appending_dataset.close()


def model_to_yaml(model, yaml_path: Path) -> None:
    """
    Save a model's dictionary representation to a YAML file.

    Borrowing from recOrder==0.4.0

    Parameters
    ----------
    model : object
        The model object to convert to YAML.
    yaml_path : Path
        The path to the output YAML file.

    Raises
    ------
    TypeError
        If the `model` object does not have a `dict()` method.

    Notes
    -----
    This function converts a model object into a dictionary representation
    using the `dict()` method. It removes any fields with None values before
    writing the dictionary to a YAML file.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = MyModel()
    >>> model_to_yaml(model, 'model.yaml')

    """
    yaml_path = Path(yaml_path)

    if not hasattr(model, "dict"):
        raise TypeError("The 'model' object does not have a 'dict()' method.")

    model_dict = model.model_dump()

    # Remove None-valued fields
    clean_model_dict = {key: value for key, value in model_dict.items() if value is not None}

    with open(yaml_path, "w+") as f:
        yaml.dump(clean_model_dict, f, default_flow_style=False, sort_keys=False)


def yaml_to_model(yaml_path: Path, model):
    """
    Load model settings from a YAML file and create a model instance.

    Borrowing from recOrder==0.4.0

    Parameters
    ----------
    yaml_path : Path
        The path to the YAML file containing the model settings.
    model : class
        The model class used to create an instance with the loaded settings.

    Returns
    -------
    object
        An instance of the model class with the loaded settings.

    Raises
    ------
    TypeError
        If the provided model is not a class or does not have a callable constructor.
    FileNotFoundError
        If the YAML file specified by `yaml_path` does not exist.

    Notes
    -----
    This function loads model settings from a YAML file using `yaml.safe_load()`.
    It then creates an instance of the provided `model` class using the loaded settings.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = yaml_to_model('model.yaml', MyModel)

    """
    yaml_path = Path(yaml_path)

    if not callable(getattr(model, "__init__", None)):
        raise TypeError("The provided model must be a class with a callable constructor.")

    try:
        with open(yaml_path, "r") as file:
            raw_settings = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The YAML file '{yaml_path}' does not exist.")

    return model(**raw_settings)


def _is_nested(lst: list) -> bool:
    """
    Check if a list contains nested lists or strings.

    Parameters
    ----------
    lst : list
        List to check for nested structures.

    Returns
    -------
    bool
        True if the list contains nested lists or strings, False otherwise.
    """
    return any(isinstance(i, list) for i in lst) or any(isinstance(i, str) for i in lst)


def _check_nan_n_zeros(input_array: np.ndarray) -> bool:
    """
    Checks if data are all zeros or nan.

    Parameters
    ----------
    input_array : np.ndarray
        Input array (N-dimensional).

    Returns
    -------
    bool
        True if the array is entirely zeros or NaNs, False otherwise.
    """
    return np.all(np.isnan(input_array)) or np.all(input_array == 0)


def get_empty_frame_indices(input_array: np.ndarray) -> list[int]:
    """
    Get the indices of the empty frames in a 3D array.

    Parameters
    ----------
    input_array : np.ndarray
        Input array with shape (Z, Y, X).

    Returns
    -------
    list[int]
        List of Z indices that are entirely zeros or NaNs.

    Raises
    ------
    ValueError
        If the input array is not 3D.
    """
    indices = []

    if len(input_array.shape) == 3:  # 3D array (e.g., Z, Y, X)
        for z in range(input_array.shape[0]):
            if _check_nan_n_zeros(input_array[z, :, :]):
                indices.append(z)  # Add Z index if it's empty
        return indices

    else:
        raise ValueError("Input array must be 3D.")


def estimate_resources(
    shape: tuple[int, int, int, int, int],
    dtype: DTypeLike = np.float32,
    ram_multiplier: float = 1.0,
    max_num_cpus: int = 64,
    min_ram_per_cpu: int = 4,
) -> tuple[int, int]:
    """
    Estimate the number of CPUs and the amount of RAM required for processing a given data volume.

    Parameters
    ----------
    shape : tuple[int, int, int, int, int]
        The shape of the data as a tuple (T, C, Z, Y, X).
    dtype : DTypeLike, optional
        The data type of the elements, by default np.float32.
    ram_multiplier : float, optional
        Multiplier to scale the required memory for processing a given ZYX volume. For example,
        if a processing pipeline makes two copies of the input data, then the ram_multiplier
        should be at least 3, by default 1.0.
    max_num_cpus : int, optional
        Maximum number of available CPUs, by default 64.
    min_ram_per_cpu : int, optional
        Minimum amount of RAM per CPU in GB, by default 4.

    Returns
    -------
    tuple[int, int]
        A tuple containing the estimated number of CPUs and the required amount of RAM per CPU in GB.
        These values can be passed to the `--cpus_per_task` and `--mem_per_cpu` parameters of sbatch.

    Raises
    ------
    ValueError
        If the shape is not a 5-tuple.
    """
    if len(shape) != 5:
        raise ValueError("The shape must be a 5-tuple (T, C, Z, Y, X).")

    T, C, Z, Y, X = shape
    gb_per_element = np.dtype(dtype).itemsize / 2**30  # bytes_per_element / bytes_per_gb
    num_cpus = min(T * C, max_num_cpus)
    gb_ram_per_volume = Z * Y * X * gb_per_element
    gb_ram_per_cpu = np.ceil(max(min_ram_per_cpu, gb_ram_per_volume * ram_multiplier))

    return int(num_cpus), int(gb_ram_per_cpu)
