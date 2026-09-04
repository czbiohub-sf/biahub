"""OME-Zarr store helpers: output paths, version, provenance, channels."""

from pathlib import Path
from typing import Literal

from iohub.ngff import open_ome_zarr
from tqdm import tqdm

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
