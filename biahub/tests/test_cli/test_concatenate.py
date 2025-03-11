import numpy as np

from iohub import open_ome_zarr

from biahub.analysis.AnalysisSettings import ConcatenateSettings
from biahub.cli.concatenate import concatenate


def test_concatenate_channels(example_plate_2, tmp_path):
    """
    Test concatenating channels across zarr stores with the same layout
    """
    # Load example plate with three positions - A/1/0, B/1/0, B/2/0, and two channels - GFP, RFP
    plate_2_path, plate_2 = example_plate_2
    plate_2_channels = plate_2.channel_names
    data_shape = plate_2["A/1/0"].data.shape

    # Create another plate with the same structure and different set of channels
    plate_1_path = tmp_path / "plate_1.zarr"
    position_list = [pos_name for pos_name, _ in plate_2.positions()]
    plate_1_channels = ["DAPI", "Cy5"]

    # Generate input dataset
    plate_1 = open_ome_zarr(
        plate_1_path,
        layout="hcs",
        mode="w",
        channel_names=plate_1_channels,
    )

    for pos_name in position_list:
        position = plate_1.create_position(*pos_name.split("/"))
        position["0"] = np.random.randint(
            100, np.iinfo(np.uint16).max, size=data_shape, dtype=np.uint16
        )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=['all', 'all'],
        time_indices='all',
    )

    output_path = tmp_path / "output.zarr"
    concatenate(
        settings=settings,
        output_dirpath=output_path,
        local=True,
    )

    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has all the channels from the input plates
    # channel ordering might be different
    output_channels = output_plate.channel_names
    assert set(output_channels) == set(plate_1_channels + plate_2_channels)

    # Check that the output plate has the right number of positions
    output_positions = [pos_name for pos_name, _ in output_plate.positions()]
    assert set(output_positions) == set(position_list)
