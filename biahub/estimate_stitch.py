from collections import defaultdict
from pathlib import Path

import click
import numpy as np

from iohub import open_ome_zarr

from biahub.cli.parsing import input_position_dirpaths, output_filepath
from biahub.cli.utils import model_to_yaml
from biahub.settings import ProcessingSettings, StitchSettings


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
                z_stage_name = stage_position["DefaultZStage"]
                if "DevicePositions" in stage_position.keys():
                    for device in stage_position["DevicePositions"]:
                        if device["Device"] == z_stage_name:
                            zpos = device["Position_um"][0]
                            break
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
@click.option(
    "--rot90",
    default=0,
    type=int,
    help="rotate the images 90 counterclockwise n times before stitching",
)
def estimate_stitch_cli(
    input_position_dirpaths: list[Path],
    output_filepath: str,
    fliplr: bool,
    flipud: bool,
    rot90: int,
):
    """
    Estimate stitching parameters for positions in wells of a zarr store.

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

        # Write back into flat dictionary
        for i, fov_name in enumerate(grouped_wells[key].keys()):
            final_translation_dict[fov_name] = list(np.round(zyx_well_array[i], 2))

    # Validate and save
    settings = StitchSettings(
        channels=None,
        preprocessing=ProcessingSettings(fliplr=fliplr, flipud=flipud, rot90=rot90),
        postprocessing=ProcessingSettings(),
        total_translation=final_translation_dict,
    )
    model_to_yaml(settings, output_filepath)


if __name__ == "__main__":
    estimate_stitch_cli()
