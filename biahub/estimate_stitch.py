from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from iohub import open_ome_zarr
from iohub.ngff.nodes import Plate

from biahub.cli.parsing import InputPositionDirpaths, OutputFilepath, local, monitor
from biahub.settings import StitchSettings
from biahub.utils.config import model_to_yaml
from biahub.vendor.stitch.tile import optimal_positions, pairwise_shifts


def extract_stage_position(
    plate_dataset: Plate, position_name: str
) -> tuple[float, float, float]:
    """
    Extract stage position coordinates from plate metadata.

    Parameters
    ----------
    plate_dataset : Plate
        Plate dataset containing stage position metadata.
    position_name : str
        Name of the position to extract coordinates for.

    Returns
    -------
    Tuple[float, float, float]
        Stage position coordinates in (z, y, x) order in micrometers.
    """
    stage_positions = plate_dataset.zattrs["Summary"]["StagePositions"]
    for stage_position in stage_positions:  # TODO: fail if this loop reaches the end
        if stage_position["Label"] == position_name:
            # Initialize default values
            xpos, ypos, zpos = 0, 0, 0

            if "DevicePositions" in stage_position.keys():
                # Handle DevicePositions case
                xy_stage_name = stage_position.get("DefaultXYStage", "")
                non_z_devices = {xy_stage_name}

                for device in stage_position["DevicePositions"]:
                    if device["Device"] == xy_stage_name and xy_stage_name:
                        xpos, ypos = device["Position_um"]
                    elif device["Device"] not in non_z_devices:
                        zpos += device["Position_um"][0]
            else:
                # Handle direct stage keys case - separate try blocks for independent failure
                try:
                    xy_stage_name = stage_position["DefaultXYStage"]
                    xpos, ypos = stage_position[xy_stage_name]
                except KeyError:
                    pass

                try:
                    z_stage_name = stage_position["DefaultZStage"]
                    zpos = stage_position[z_stage_name]
                except KeyError:
                    pass

    return zpos, ypos, xpos


def estimate_stitch_cli(
    input_position_dirpaths: InputPositionDirpaths,
    output_filepath: OutputFilepath,
    fliplr: Annotated[
        bool,
        typer.Option("--fliplr", help="Flip images left-right before stitching"),
    ] = False,
    flipud: Annotated[
        bool,
        typer.Option("--flipud", help="Flip images up-down before stitching"),
    ] = False,
    flipxy: Annotated[
        bool,
        typer.Option("--flipxy", help="Flip images along the diagonal before stitching"),
    ] = False,
    pcc_channel_name: Annotated[
        str | None,
        typer.Option(
            "--pcc-channel-name",
            help=(
                "Channel name to use for phase cross-correlation optimization "
                "(default: None, disables optimization)"
            ),
        ),
    ] = None,
    pcc_z_index: Annotated[
        int,
        typer.Option(
            "--pcc-z-index",
            help="Z slice index to use for phase cross-correlation optimization (default: 0)",
        ),
    ] = 0,
    add_offset: Annotated[
        bool,
        typer.Option(
            "--add_offset",
            help="add the offset to estimated shifts, needed for OPS experiments",
        ),
    ] = False,
    local: local = False,
    monitor: monitor = False,
):
    """Estimate stitching parameters for positions in wells of a zarr store.

    This routine uses micro-manager stage position metadata and iohub scale
    metadata to generate translation parameters for stitching. Translations are
    saved in pixel units.

    This function estimates translations using metadata alone. More precise
    translations require phase cross-correlation using `--pcc-channel`.

    >>> biahub estimate-stitch -i ./input.zarr/*/*/* -o ./stitch_params.yml
    """
    input_plate_path = Path(*input_position_dirpaths[0].parts[:-3])
    output_filepath = Path(output_filepath)

    # Collect raw stage positions
    print("Reading stage positions...")
    translation_dict = {}
    for input_position_dirpath in input_position_dirpaths:
        fov_name = "/".join(input_position_dirpath.parts[-3:])

        # Find position name from position-level omero metadata
        with open_ome_zarr(input_position_dirpath) as input_position_dataset:
            position_name = input_position_dataset.zattrs["omero"]["name"]

        # Use position name to index into micromanager plate-level metadata
        with open_ome_zarr(input_plate_path) as input_plate_dataset:
            zyx_position = extract_stage_position(input_plate_dataset, position_name)

        print(f"Found metadata: {fov_name}: {zyx_position}")
        translation_dict[fov_name] = zyx_position

    # Group by well
    grouped_wells = defaultdict(dict)
    for key, value in translation_dict.items():
        well_name = "/".join(key.split("/")[:2])
        grouped_wells[well_name][key] = value

    # Prepare stage positions in pixel coordinates for each well
    final_translation_dict = {}
    for key, value in grouped_wells.items():
        zyx_array = []
        for my_value in value.values():
            zyx_array.append(my_value)
        zyx_well_array = np.array(zyx_array)

        # Shift so that (0, 0, 0) is the lowermost corner
        zyx_well_array -= np.min(zyx_well_array, axis=0)

        # Scale to pixel coordinates
        zyx_well_array /= open_ome_zarr(input_position_dirpaths[0]).scale[2:]

        # Optimization using phase cross-correlation if pcc_channel is provided
        if pcc_channel_name is not None:
            well_positions = grouped_wells[key]
            tile_lut = {t.split("/")[-1]: i for i, t in enumerate(well_positions)}
            initial_guess = {
                key: {
                    "i": zyx_well_array[:, 1],
                    "j": zyx_well_array[:, 2],
                }
            }
            channel_index = open_ome_zarr(input_plate_path).get_channel_index(pcc_channel_name)

            edge_list, confidence_dict = pairwise_shifts(
                well_positions,
                input_plate_path,
                key,
                flipud=flipud,
                fliplr=fliplr,
                rot90=False,
                overlap=300,  # good default for pcc
                channel_index=channel_index,
                z_index=pcc_z_index,
            )
            print("Confidence scores:")
            for v in confidence_dict.values():
                print(f"{v[0]}: {v[-1]:.2f}")

            # Get actual tile size from the first position's data shape
            first_position_path = list(well_positions.keys())[0]
            with open_ome_zarr(input_plate_path / first_position_path) as first_position:
                tile_size = first_position.data.shape[-2:]  # Get (Y, X) dimensions

            opt_shift_dict = optimal_positions(
                edge_list, tile_lut, key, tile_size=tile_size, initial_guess=initial_guess
            )
            zyx_well_array[:, 1] = [a[0] for a in opt_shift_dict.values()]
            zyx_well_array[:, 2] = [a[1] for a in opt_shift_dict.values()]

        # Flip coordinates
        if fliplr:
            zyx_well_array[:, 2] *= -1
        if flipud:
            zyx_well_array[:, 1] *= -1
        if flipxy:
            zyx_well_array[:, [1, 2]] = zyx_well_array[:, [2, 1]]

        # Shift all columns so that the minimum value in each column is zero
        zyx_well_array -= np.minimum(zyx_well_array.min(axis=0), 0)

        # Write back into flat dictionary
        for i, fov_name in enumerate(grouped_wells[key].keys()):
            final_translation_dict[fov_name] = list(np.round(zyx_well_array[i], 2))

    # Validate and save
    settings = StitchSettings(
        channels=None,
        total_translation=final_translation_dict,
    )
    model_to_yaml(settings, output_filepath)
