from iohub import open_ome_zarr

from biahub.analysis.AnalysisSettings import ConcatenateSettings
from biahub.concatenate import concatenate


def test_concatenate_channels(create_custom_plate, tmp_path):
    """
    Test concatenating channels across zarr stores with the same layout
    """
    # Create example plates with same layout and different channels
    position_list = ["A/1/0", "B/1/0", "B/2/0"]
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / 'zarr1',
        position_list=[p.split("/") for p in position_list],
        channel_names=['DAPI', 'Cy3'],
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / 'zarr2',
        position_list=[p.split("/") for p in position_list],
        channel_names=["GFP", "RFP", "Phase3D"],
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
    assert set(output_channels) == set(plate_1.channel_names + plate_2.channel_names)

    # Check that the output plate has the right number of positions
    output_positions = [pos_name for pos_name, _ in output_plate.positions()]
    assert set(output_positions) == set(position_list)


def test_concatenate_specific_channels(create_custom_plate, tmp_path):
    """
    Test concatenating specific channels from zarr stores
    """

    # Create test plates
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / 'zarr1', channel_names=["DAPI", "Cy5"]
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / 'zarr2', channel_names=["GFP", "RFP"]
    )

    # Select only specific channels from each plate
    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=[
            ["DAPI"],
            ["GFP"],
        ],  # Only select DAPI from plate_1 and GFP from plate_2
        time_indices='all',
    )

    output_path = tmp_path / "output.zarr"
    concatenate(
        settings=settings,
        output_dirpath=output_path,
        local=True,
    )

    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has only the selected channels
    output_channels = output_plate.channel_names
    assert set(output_channels) == {"DAPI", "GFP"}


def test_concatenate_with_time_indices(create_custom_plate, tmp_path):
    """
    Test concatenating with specific time indices
    """

    # Create test plates
    plate_1_path, plate_1 = create_custom_plate(tmp_path / 'zarr1', time_points=10)
    plate_2_path, plate_2 = create_custom_plate(tmp_path / 'zarr2', time_points=5)

    # Select only specific time indices
    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=['all', 'all'],
        time_indices=[2, 3],  # Select specific time points
    )

    output_path = tmp_path / "output.zarr"
    concatenate(
        settings=settings,
        output_dirpath=output_path,
        local=True,
    )

    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has the two time points
    assert output_plate["A/1/0"].data.shape[0] == 2


def test_concatenate_with_single_slice_to_all(create_custom_plate, tmp_path):
    """
    Test concatenating with a single slice applied to all datasets
    """
    # Create test plates with same shape
    (T, Z, Y, X) = (3, 4, 6, 8)
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / 'zarr1',
        time_points=T,
        z_size=Z,
        y_size=Y,
        x_size=X,
        channel_names=["GFP", "RFP", "Phase3D"],
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / 'zarr2',
        time_points=T,
        z_size=Z,
        y_size=Y,
        x_size=X,
        channel_names=["DAPI"],
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=['all', 'all'],
        time_indices='all',
        Z_slice=[0, 2],
        Y_slice=[0, 3],
        X_slice=[0, 4],
    )

    output_path = tmp_path / "output.zarr"
    concatenate(
        settings=settings,
        output_dirpath=output_path,
        local=True,
    )

    output_plate = open_ome_zarr(output_path)

    assert output_plate["A/1/0"].data.shape == (T, 4, 2, 3, 4)


def test_concatenate_with_cropping(create_custom_plate, tmp_path):
    """
    Test concatenating with cropping
    """
    Z, Y, X = 4, 6, 8
    # Create example plates
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / 'zarr1', channel_names=["DAPI", "Cy5"], z_size=Z, y_size=Y, x_size=X
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / 'zarr2', channel_names=["GFP", "RFP"], z_size=Z, y_size=Y, x_size=X
    )

    # Define crop parameters
    z_start, z_end = 0, Z // 2
    y_start, y_end = 0, Y // 2
    x_start, x_end = 0, X // 2

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=['all', 'all'],
        time_indices='all',
        Z_slice=[[z_start, z_end], [z_start, z_end]],
        Y_slice=[[y_start, y_end], [y_start, y_end]],
        X_slice=[[x_start, x_end], [x_start, x_end]],
    )

    output_path = tmp_path / "output.zarr"
    concatenate(
        settings=settings,
        output_dirpath=output_path,
        local=True,
    )

    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has the expected cropped dimensions
    _, _, output_Z, output_Y, output_X = output_plate['A/1/0'].data.shape
    assert output_Z == z_end - z_start
    assert output_Y == y_end - y_start
    assert output_X == x_end - x_start


def test_concatenate_with_custom_chunks(create_custom_plate, tmp_path):
    """
    Test concatenating with custom chunk sizes
    """
    # Create example plates
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / 'zarr1',
        channel_names=["DAPI", "Cy5"],
        time_points=3,
        z_size=4,
        y_size=8,
        x_size=6,
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / 'zarr2',
        channel_names=["GFP", "RFP"],
        time_points=3,
        z_size=4,
        y_size=8,
        x_size=6,
    )

    # Define custom chunk sizes
    custom_chunks = [1, 2, 4, 3]  # [C, Z, Y, X]

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=['all', 'all'],
        time_indices='all',
        chunks_czyx=custom_chunks,
    )

    output_path = tmp_path / "output.zarr"
    concatenate(
        settings=settings,
        output_dirpath=output_path,
        local=True,
    )

    # We can't easily check the chunks directly, but we can verify the operation completed successfully
    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has all the channels from the input plates
    output_channels = output_plate.channel_names
    assert set(output_channels) == set(plate_1.channel_names + plate_2.channel_names)


def test_concatenate_multiple_plates(create_custom_plate, tmp_path):
    """
    Test concatenating multiple plates
    """
    common_params = {"time_points": 3, "z_size": 4, "y_size": 5, "x_size": 6}

    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / 'zarr1',
        channel_names=["GFP", "RFP", "DAPI", "Cy5", "Phase3D"],
        **common_params,
    )

    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / 'zarr2', channel_names=["GFP", "RFP"], **common_params
    )

    plate_3_path, plate_3 = create_custom_plate(
        tmp_path / 'zarr3', channel_names=["Phase3D"], **common_params
    )

    settings = ConcatenateSettings(
        concat_data_paths=[
            str(plate_1_path) + "/A/1/0",
            str(plate_2_path) + "/A/1/0",
            str(plate_3_path) + "/B/1/0",
            str(plate_1_path) + "/A/1/0",
        ],
        channel_names=['all', ['GFP', 'RFP'], ['Phase3D'], 'all'],
        time_indices='all',
    )

    output_path = tmp_path / "output.zarr"
    concatenate(
        settings=settings,
        output_dirpath=output_path,
        local=True,
    )

    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has the right number of positions
    output_positions = [pos_name for pos_name, _ in output_plate.positions()]
    assert len(output_positions) == 2  # merges 'A/1/0'

    # Check that the output plate has the right channels
    output_channels = output_plate.channel_names
    assert set(output_channels) == {"GFP", "RFP", "DAPI", "Cy5", "Phase3D"}

    # Check that the output plate has the right shape
    assert output_plate["A/1/0"].data.shape[0] == 3  # time points
    assert output_plate["A/1/0"].data.shape[1] == 5  # channels


def test_concatenate_mismatched_with_cropping(create_custom_plate, tmp_path):
    """
    Test concatenating zarr stores of mismatched shapes with cropping to the
    same output shape
    """
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / 'zarr1', time_points=3, z_size=2, y_size=3, x_size=3
    )

    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / 'zarr2', time_points=3, z_size=4, y_size=6, x_size=6
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=['all', 'all'],
        time_indices='all',
        Z_slice=['all', [0, 2]],
        Y_slice=['all', [0, 3]],
        X_slice=['all', [0, 3]],
    )

    output_path = tmp_path / "output.zarr"
    concatenate(
        settings=settings,
        output_dirpath=output_path,
        local=True,
    )

    output_plate = open_ome_zarr(output_path)

    assert output_plate["A/1/0"].data.shape == (3, 3, 2, 3, 3)


def test_concatenate_with_mixed_slice_formats(create_custom_plate, tmp_path):
    """
    Test concatenating with mixed slice formats like [[0,1], 'all']
    """
    # Create a plate with larger dimensions to test mixed slice formats
    plate_path_1, plate_1 = create_custom_plate(
        tmp_path / 'large_plate_1',
        time_points=2,
        z_size=10,  # Larger Z dimension for testing
        y_size=20,  # Larger Y dimension for testing
        x_size=20,  # Larger X dimension for testing
    )
    plate_path_2, plate_2 = create_custom_plate(
        tmp_path / 'large_plate_2',
        time_points=2,
        z_size=5,  # Larger Z dimension for testing
        y_size=4,  # Larger Y dimension for testing
        x_size=8,  # Larger X dimension for testing
    )

    # Define mixed slice formats
    z_slices = [
        [0, 5],
        "all",
    ]  # First 5 slices, all slices, and last 5 slices
    y_slices = [[2, 6], 'all']  # First 10 slices and last 10 slices
    x_slices = [[4, 12], 'all']  # All slices in X dimension

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_path_1) + "/*/*/*", str(plate_path_2) + "/*/*/*"],
        channel_names=['all', 'all'],
        time_indices='all',
        Z_slice=z_slices,
        Y_slice=y_slices,
        X_slice=x_slices,
    )

    output_path = tmp_path / "output_mixed_slice.zarr"
    concatenate(
        settings=settings,
        output_dirpath=output_path,
        local=True,
    )

    output_plate = open_ome_zarr(output_path)

    # Expect the shape to be the same
    assert output_plate["A/1/0"].data.shape[-3:] == (
        z_slices[0][1] - z_slices[0][0],
        y_slices[0][1] - y_slices[0][0],
        x_slices[0][1] - x_slices[0][0],
    )


def test_concatenate_with_unique_positions(create_custom_plate, tmp_path):
    """
    Similar to test_concatenate_channels, but with ensure_unique_positions=True
    to prevent overwriting when multiple inputs have the same position names
    """
    # Create example plates with same layout and different channels
    position_list = ["A/1/0", "B/1/0"]
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / 'zarr1',
        position_list=[p.split("/") for p in position_list],
        channel_names=['DAPI', 'Cy5'],
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / 'zarr2',
        position_list=[p.split("/") for p in position_list],
        channel_names=["GFP", "RFP"],
    )

    # Now test with ensure_unique_positions=True
    settings_unique = ConcatenateSettings(
        concat_data_paths=[
            str(plate_1_path) + "/A/1/0",
            str(plate_2_path) + "/A/1/0",  # Same position name
        ],
        channel_names=['all', 'all'],
        time_indices='all',
        ensure_unique_positions=True,  # Enable unique positions
    )

    output_path_unique = tmp_path / "output_unique.zarr"
    concatenate(
        settings=settings_unique,
        output_dirpath=output_path_unique,
        local=True,
    )

    output_plate_unique = open_ome_zarr(output_path_unique)

    # Check that there are two positions (both inputs were preserved with unique names)
    output_positions_unique = [pos_name for pos_name, _ in output_plate_unique.positions()]
    assert len(output_positions_unique) == 2

    # The first position should keep its original name, the second should have a suffix
    assert "A/1/0" in output_positions_unique
    assert "A/1d1/0" in output_positions_unique  # Second position has a suffix

    # Check that both positions have the expected channels
    for pos_name, pos in output_plate_unique.positions():
        # Both positions should have all channels
        assert set(pos.channel_names) == {"DAPI", "Cy5", "GFP", "RFP"}
