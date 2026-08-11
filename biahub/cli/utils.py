import hashlib
import json
import logging
import os

from pathlib import Path
from typing import Literal

import click
import numpy as np
import yaml

from iohub.ngff import open_ome_zarr
from numpy.typing import DTypeLike
from tqdm import tqdm

logger = logging.getLogger(__name__)

#: ``fnmatch`` patterns for the per-position zattrs keys that steps carry
#: forward into their output store, passed as ``create_empty_plate``'s
#: ``metadata_keys``. These are the provenance records each step stamps on its
#: output — keeping them means a derived plate still describes how it was made.
#:
#: This is deliberately an allowlist rather than a denylist. Everything else an
#: upstream store happens to carry is dropped, which matters most for the
#: acquisition writer's ``ome_writers`` blob: it holds one record per raw
#: Z-frame (~18 MB per position on a mantis plate), and its frame indices stop
#: describing the data as soon as a step changes the shape. Copying it forward
#: made every ``--init`` step read and rewrite tens of MB per position, and
#: taxed every subsequent read of the position for no benefit.
#:
#: ``normalization`` (written by the reconstruction step) is intentionally
#: absent — it describes that step's inputs, not its output.
PROVENANCE_METADATA_KEYS = ("biahub-*", "waveorder", "cytoland")


def echo_resources(num_cpus: int, mem_gb: int, time_minutes: int) -> None:
    """Emit the per-position resource request consumed by the Nextflow pipeline.

    Every step CLI calls this from its ``--init`` path so there is a single
    source of truth for per-position CPU, memory, and wall-clock time. The
    Nextflow ``init_*`` process captures this line on stdout and
    ``parse_resources`` (``nextflow/modules/common.nf``) reads the JSON payload
    to set the per-position task's ``cpus``/``memory``/``time`` directives. The
    same values also feed the CLI's own ``slurm_*`` submission args, so the
    SLURM fan-out and the Nextflow fan-out request identical resources.

    A single JSON payload keeps the contract order-independent and extensible
    (new fields can be added without breaking the positional parsing).

    Parameters
    ----------
    num_cpus : int
        CPUs per position.
    mem_gb : int
        TOTAL memory per position in GB (not per-CPU).
    time_minutes : int
        Wall-clock budget per position in minutes.
    """
    # Coerce to plain int: estimators may return numpy integers, which json
    # cannot serialize.
    payload = {"cpus": int(num_cpus), "mem_gb": int(mem_gb), "time_minutes": int(time_minutes)}
    click.echo("RESOURCES:" + json.dumps(payload))


def settings_fingerprint(settings) -> str:
    """Stable short hash of a settings model.

    Passed to ``process_single_position(resume_token=...)`` so that the
    per-unit completion records a resumed run relies on belong to the settings
    that produced them. Re-running a step with a changed config against an
    existing output store then recomputes instead of skipping units whose data
    would now be different.
    """
    payload = json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_submitit_cluster(
    local: bool = False,
    cluster: str | None = None,
) -> str:
    """Return the submitit cluster type.

    'debug' is forced in CI. Otherwise the explicit `cluster` string wins;
    if no cluster is given, falls back to the legacy `local` boolean.
    """
    if os.environ.get("CI") == "true":
        return "debug"
    if cluster is not None:
        return cluster
    return "local" if local else "slurm"


def resolve_ome_zarr_version(
    reference_store_path: Path,
    override: Literal["0.4", "0.5"] | None,
) -> Literal["0.4", "0.5"]:
    """Pick the OME-Zarr version to use for an output store.

    When ``override`` is set it wins; otherwise the version is read from
    ``reference_store_path`` so the output preserves the input's version.
    """
    if override is not None:
        return override
    with open_ome_zarr(str(reference_store_path), mode="r") as dataset:
        return dataset.version


def update_model(model_instance, update_dict):
    """
    Properly updates a Pydantic model with only the provided values while keeping the defaults.

    This ensures that nested models retain missing values instead of getting overwritten.
    """
    updated_fields = {}
    for key, value in update_dict.items():
        if isinstance(value, dict) and hasattr(model_instance, key):
            # If it's a nested dict, update the nested Pydantic model properly
            nested_model = getattr(model_instance, key)
            updated_fields[key] = nested_model.model_copy(update=value)
        else:
            # Otherwise, just update the value directly
            updated_fields[key] = value

    # Create a new instance with updated fields
    return model_instance.model_copy(update=updated_fields)


def get_output_paths(
    input_paths: list[Path], output_zarr_path: Path, ensure_unique_positions: bool = None
) -> list[Path]:
    """
    Generate a mirrored output path list given an input list of positions.

    Parameters
    ----------
    input_paths : list[Path]
        List of input position paths
    output_zarr_path : Path
        Base output zarr path
    ensure_unique_positions : bool, optional
        If True, ensures unique output position paths by appending a suffix to the column part
        when duplicate position names are detected.
        For example, if "A/1/0" is duplicated, it becomes "A/1d0/0", "A/1d1/0", etc.

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


def copy_n_paste(zyx_data: np.ndarray, zyx_slicing_params: list) -> np.ndarray:
    """
    Load a zyx array and crop given a list of ZYX slices().

    Parameters
    ----------
    zyx_data : np.ndarray
        data to copy
    zyx_slicing_params : list
        list of slicing parameters for z,y,x
        Each element is a single slice object [z_slice, y_slice, x_slice]

    Returns
    -------
    np.ndarray
        crop of the input zyx_data given the slicing parameters
    """
    # Replace NaN values with zeros
    zyx_data = np.nan_to_num(zyx_data, nan=0)
    zyx_data_sliced = zyx_data[
        zyx_slicing_params[0],
        zyx_slicing_params[1],
        zyx_slicing_params[2],
    ]
    return zyx_data_sliced


def copy_n_paste_czyx(czyx_data: np.ndarray, czyx_slicing_params: list) -> np.ndarray:
    """
    Load a zyx array and crop given a list of ZYX slices().

    Parameters
    ----------
    czyx_data : np.ndarray
        data to copy
    czyx_slicing_params : list
        list of slicing parameters for z,y,x
        Each element is a single slice object [z_slice, y_slice, x_slice]

    Returns
    -------
    np.ndarray
        crop of the input czyx_data given the slicing parameters
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
    Append channels to a target zarr store.

    Parameters
    ----------
    input_data_path : Path
        input zarr path = /input.zarr
    target_data_path : Path
        target zarr path  = /target.zarr
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
    >>> model_to_yaml(model, "model.yaml")

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
    >>> model = yaml_to_model("model.yaml", MyModel)

    """
    yaml_path = Path(yaml_path)

    if not callable(getattr(model, "__init__", None)):
        raise TypeError("The provided model must be a class with a callable constructor.")

    try:
        with open(yaml_path) as file:
            raw_settings = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The YAML file '{yaml_path}' does not exist.") from None

    return model(**raw_settings)


def _check_nan_n_zeros(input_array: np.ndarray) -> bool:
    """
    Check if data are all zeros or nan.

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
        Input array (3D).

    Returns
    -------
    List[int]
        List of Z indices that are entirely zeros or NaNs.
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
    time_multiplier: float = 1.0,
    max_num_cpus: int = 64,
    min_ram_per_cpu: int = 4,
    min_time_minutes: int = 30,
):
    """Estimate wall-time, CPUs, and RAM required to process a data volume.

    Both RAM and wall-time key on the ZYX volume, the natural unit of work here:
    RAM scales with a single volume (the per-CPU working set), and wall-time
    scales with the NUMBER of volumes processed (T * C).

    Counting volumes -- rather than voxels -- is deliberate. Per-voxel
    throughput is not a stable quantity: it depends on the CPU/GPU model, the
    filesystem write speed, and the chunking, so a voxel-rate calibrated on one
    run does not transfer to the next. Volume count is a property of the
    dataset alone. The spread in per-volume cost between, say, an A549 volume
    and a neuromast volume is absorbed by ``time_multiplier``, which is a fudge
    factor, not a physical constant -- over-requesting 2x on one dataset and
    1.5x on another is fine and expected.

    ``time_multiplier`` mirrors ``ram_multiplier``: it is the per-step scaling
    knob, in minutes of wall-time per ZYX volume, calibrated from observed
    COMPLETED runs (see each call site). Callers that only need CPUs and RAM can
    ignore the time estimate:

        _, num_cpus, gb_ram_per_cpu = estimate_resources(shape, ram_multiplier=8)

    Parameters
    ----------
    shape : Tuple[int, int, int, int, int]
        The shape of the data as a tuple (T, C, Z, Y, X).
    dtype : DTypeLike, optional
        The data type of the elements. Default is np.float32.
    ram_multiplier : float, optional
        Multiplier to scale the required memory for processing a given ZYX volume.
        For example, if a pipeline makes two copies of the input data, the
        ram_multiplier should be at least 3. Default is 1.0.
    time_multiplier : float, optional
        Wall-time in minutes per ZYX volume processed (T*C volumes total). The
        per-step calibration knob, analogous to ram_multiplier. Default is 1.0.
    max_num_cpus : int, optional
        Maximum number of available CPUs. Default is 64.
    min_ram_per_cpu : int, optional
        Minimum amount of RAM per CPU in GB. Default is 4.
    min_time_minutes : int, optional
        Minimum wall-time so tiny inputs still get a sane request. Default 30.

    Returns
    -------
    Tuple[int, int, int]
        (time_minutes, num_cpus, gb_ram_per_cpu). time_minutes is rounded up to
        the nearest 10 minutes; num_cpus and gb_ram_per_cpu map to sbatch's
        --time, --cpus_per_task, and --mem_per_cpu.
    """
    if len(shape) != 5:
        raise ValueError("The shape must be a 5-tuple (T, C, Z, Y, X).")
    if ram_multiplier <= 0 or time_multiplier <= 0:
        raise ValueError("ram_multiplier and time_multiplier must be > 0.")

    T, C, Z, Y, X = shape
    gb_per_element = np.dtype(dtype).itemsize / 2**30  # bytes_per_element / bytes_per_gb
    # In CI/tests, run serially: the test data is tiny, so spawning a worker
    # pool costs far more (per-process re-imports) than the work itself.
    num_cpus = 1 if os.environ.get("CI") == "true" else min(T * C, max_num_cpus)
    gb_ram_per_volume = Z * Y * X * gb_per_element
    gb_ram_per_cpu = np.ceil(max(min_ram_per_cpu, gb_ram_per_volume * ram_multiplier))

    # Wall-time from the number of ZYX volumes processed, scaled by the per-step
    # time_multiplier, then rounded up to the nearest 10 minutes for tidy SLURM
    # requests.
    num_volumes = T * C
    minutes = max(min_time_minutes, num_volumes * time_multiplier)
    time_minutes = int(np.ceil(minutes / 10.0) * 10)

    return time_minutes, int(num_cpus), int(gb_ram_per_cpu)
