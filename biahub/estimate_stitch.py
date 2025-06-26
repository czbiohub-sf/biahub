from collections import defaultdict
from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr
from stitch.stitch.tile import optimal_positions, pairwise_shifts

from biahub.cli.parsing import input_position_dirpaths, output_filepath
from biahub.cli.utils import model_to_yaml
from biahub.settings import StitchSettings


def extract_stage_position(plate_dataset, position_name):
    stage_positions = plate_dataset.zattrs["Summary"]["StagePositions"]
    for stage_position in stage_positions:  # TODO: fail if this loop reaches the end
        if stage_position["Label"] == position_name:

            # Read XY positions
            try:
                xy_stage_name = stage_position["DefaultXYStage"]
                if "DevicePositions" in stage_position.keys():
                    for device in stage_position["DevicePositions"]:
                        if device["Device"] == xy_stage_name:
                            xpos, ypos = device["Position_um"]
                            break
                else:
                    xpos, ypos = stage_position[xy_stage_name]
            except KeyError:
                xpos, ypos = 0, 0
                pass

            # Read Z positions
            
            try:
                xy_stage_name = stage_position["DefaultXYStage"]
                z_stage_name = stage_position['DefaultZStage']
                non_z_devices = {xy_stage_name}
                if "DevicePositions" in stage_position.keys():
                    zpos = 0
                    for device in stage_position["DevicePositions"]:
                        if device["Device"] not in non_z_devices:
                            zpos+= device["Position_um"][0]     
                else:
                    zpos = stage_position[z_stage_name]
            except KeyError:
                zpos = 0
                pass
    return zpos, ypos, xpos


@click.command("estimate-stitch")
@input_position_dirpaths()
@output_filepath()
@click.option("--fliplr", is_flag=True, help="Flip images left-right before stitching")
@click.option("--flipud", is_flag=True, help="Flip images up-down before stitching")
@click.option("--flipxy", is_flag=True, help="Flip images along the diagonal before stitching")
@click.option(
    "--pcc-channel-name",
    default=None,
    type=str,
    help="Channel name to use for phase cross-correlation optimization (default: None, disables optimization)",
)
@click.option(
    "--pcc-z-index",
    default=0,
    type=int,
    help="Z slice index to use for phase cross-correlation optimization (default: 0)",
)
def estimate_stitch_cli(
    input_position_dirpaths: list[Path],
    output_filepath: str,
    fliplr: bool,
    flipud: bool,
    flipxy: bool,
    pcc_channel_name: str,
    pcc_z_index: int,
):
    """
    Estimate stitching parameters for positions in wells of a zarr store.

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
            position_name = input_position_dataset.zattrs['omero']['name']

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
    for i, (key, value) in enumerate(grouped_wells.items()):
        zyx_array = []
        for my_value in grouped_wells[key].values():
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
            for k, v in confidence_dict.items():
                print(f"{v[0]}: {v[-1]:.2f}")

            opt_shift_dict = optimal_positions(
                edge_list, tile_lut, key, tile_size=(2048, 2048), initial_guess=initial_guess
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


if __name__ == "__main__":
    estimate_stitch_cli()
