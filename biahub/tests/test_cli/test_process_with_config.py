import numpy as np
import pytest

from biahub.process_data import binning_czyx, process_with_config


@pytest.fixture(scope="function")
def example_process_plate(tmp_path):
    """
    Create a test plate for process_with_config testing
    """
    plate_path = tmp_path / "process_plate.zarr"

    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
    )

    # Create plate with test channels
    from iohub.ngff import open_ome_zarr

    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["BF", "GFP"],
    )

    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        # Create test data with known values for verification
        # Shape: (T, C, Z, Y, X) = (3, 2, 4, 32, 32)
        data = np.zeros((3, 2, 4, 32, 32), dtype=np.float32)

        # Fill with test pattern for easy verification
        for t in range(3):
            for c in range(2):
                for z in range(4):
                    # Create a simple pattern: increasing values from top-left
                    for y in range(32):
                        for x in range(32):
                            data[t, c, z, y, x] = (y + x) / 100.0 + 0.1

        position["0"] = data

    yield plate_path, plate_dataset


def test_process_with_config_binning_2x2(tmp_path, example_process_plate, monkeypatch):
    """
    Test process_with_config functionality with 2x2 binning
    """
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_process_plate
    output_path = tmp_path / "processed_output_binning.zarr"

    # Create test configuration for 2x2 binning
    config_content = """processing_functions:
  - function: biahub.process_data.binning_czyx
    input_channels: ["BF"]
    kwargs:
      binning_factor_zyx: [1, 2, 2]
      mode: "sum"
"""

    config_path = tmp_path / "test_process_config_binning.yml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    # Get input position paths
    input_position_paths = [
        plate_path / "A" / "1" / "0",
        plate_path / "B" / "1" / "0",
    ]

    # Call process_with_config function directly
    process_with_config(
        input_position_dirpaths=input_position_paths,
        config_filepath=config_path,
        output_dirpath=output_path,
        local=True,
    )

    # Verify output exists
    assert output_path.exists()

    # Verify output structure
    for position in ["A/1/0", "B/1/0"]:
        position_path = output_path / position
        assert position_path.exists()

    # Verify the processed data by loading a sample
    from iohub import open_ome_zarr

    with open_ome_zarr(output_path / "A" / "1" / "0") as output_dataset:
        # Check that the output has the expected shape after binning
        T, C, Z, Y, X = output_dataset.data.shape

        # Original shape was (3, 2, 4, 32, 32)
        # After 2x2 binning in Y and X: (3, 2, 4, 16, 16)
        expected_shape = (3, 2, 4, 16, 16)
        assert (T, C, Z, Y, X) == expected_shape

        # Check that channel names are preserved
        assert output_dataset.channel_names == ["BF", "GFP"]

        print(f"Binning test - Output shape: {output_dataset.data.shape}")
        print(f"Binning test - Channel names: {output_dataset.channel_names}")


def test_process_with_config_squaring(tmp_path, example_process_plate, monkeypatch):
    """
    Test process_with_config functionality with squaring pixel values
    """
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_process_plate
    output_path = tmp_path / "processed_output_squaring.zarr"

    # Create test configuration for squaring
    config_content = """processing_functions:
  - function: np.square
    input_channels: ["GFP"]
    kwargs: {}
"""

    config_path = tmp_path / "test_process_config_squaring.yml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    # Get input position paths
    input_position_paths = [
        plate_path / "A" / "1" / "0",
        plate_path / "B" / "1" / "0",
    ]

    # Call process_with_config function directly
    process_with_config(
        input_position_dirpaths=input_position_paths,
        config_filepath=config_path,
        output_dirpath=output_path,
        local=True,
    )

    # Verify output exists
    assert output_path.exists()

    # Verify the processed data by loading a sample
    from iohub import open_ome_zarr

    with open_ome_zarr(output_path / "A" / "1" / "0") as output_dataset:
        # Check that the output has the same shape (squaring doesn't change dimensions)
        T, C, Z, Y, X = output_dataset.data.shape

        # Original shape should be preserved: (3, 2, 4, 32, 32)
        expected_shape = (3, 2, 4, 32, 32)
        assert (T, C, Z, Y, X) == expected_shape

        # Check that channel names are preserved
        assert output_dataset.channel_names == ["BF", "GFP"]

        # Verify that values have been squared
        # Load original data for comparison
        with open_ome_zarr(plate_path / "A" / "1" / "0") as original_dataset:
            original_data = original_dataset.data[0, 1, 0, 0, 0]  # First GFP pixel
            processed_data = output_dataset.data[0, 1, 0, 0, 0]  # First GFP pixel

            # Check that the processed value is the square of the original
            expected_squared = original_data**2
            assert np.allclose(processed_data, expected_squared, atol=1e-6)

        print(f"Squaring test - Output shape: {output_dataset.data.shape}")
        print(f"Squaring test - Channel names: {output_dataset.channel_names}")


def test_process_with_config_binning_and_squaring(
    tmp_path, example_process_plate, monkeypatch
):
    """
    Test process_with_config functionality with both binning (2x2) and squaring pixel values
    """
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_process_plate
    output_path = tmp_path / "processed_output_combined.zarr"

    # Create test configuration for binning and squaring
    config_content = """processing_functions:
  - function: biahub.process_data.binning_czyx
    input_channels: ["BF"]
    kwargs:
      binning_factor_zyx: [1, 2, 2]
      mode: "sum"
  - function: np.square
    input_channels: ["GFP"]
    kwargs: {}
"""

    config_path = tmp_path / "test_process_config_combined.yml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    # Get input position paths
    input_position_paths = [
        plate_path / "A" / "1" / "0",
        plate_path / "B" / "1" / "0",
    ]

    # Call process_with_config function directly
    process_with_config(
        input_position_dirpaths=input_position_paths,
        config_filepath=config_path,
        output_dirpath=output_path,
        local=True,
    )

    # Verify output exists
    assert output_path.exists()

    # Verify output structure
    for position in ["A/1/0", "B/1/0"]:
        position_path = output_path / position
        assert position_path.exists()

    # Verify the processed data by loading a sample
    from iohub import open_ome_zarr

    with open_ome_zarr(output_path / "A" / "1" / "0") as output_dataset:
        # Check that the output has the expected shape after binning
        T, C, Z, Y, X = output_dataset.data.shape

        # Original shape was (3, 2, 4, 32, 32)
        # After 2x2 binning in Y and X: (3, 2, 4, 16, 16)
        expected_shape = (3, 2, 4, 16, 16)
        assert (T, C, Z, Y, X) == expected_shape

        # Check that channel names are preserved
        assert output_dataset.channel_names == ["BF", "GFP"]

        print(f"Combined test - Output shape: {output_dataset.data.shape}")
        print(f"Combined test - Channel names: {output_dataset.channel_names}")


def test_binning_function():
    """
    Test the binning function directly with simple test data
    """
    # Create a larger test array with a linear ramp
    # Shape before: (C=1, Z=1, Y=8, X=8)
    # After 2x2 binning: (1, 1, 4, 4)
    data = np.arange(1, 1 + 8 * 8, dtype=np.float32).reshape((1, 1, 8, 8))

    # Apply binning: no binning in Z, 2x2 in Y and X
    binned = binning_czyx(data, binning_factor_zyx=[1, 2, 2], mode="sum")

    # Check shape
    assert binned.shape == (1, 1, 4, 4)

    # Recalculate what the unnormalized sums should be:
    expected_sums = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):  # Y bins
        for j in range(4):  # X bins
            block = data[0, 0, i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2]
            expected_sums[i, j] = block.sum()

    # Now apply normalization (same as in binning_czyx)
    min_val = expected_sums.min()
    max_val = expected_sums.max()
    expected_normalized = (expected_sums - min_val) * 65535 / (max_val - min_val)

    # Compare all elements
    assert np.allclose(binned[0, 0], expected_normalized, atol=1e-3)


def test_binning_function_mean_mode():
    """
    Test the binning function with mean mode
    """
    # Create test data: 2x2x2 array
    test_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)  # Z=0  # Z=1

    # Add channel dimension: (1, 2, 2, 2)
    test_data = test_data[np.newaxis, :, :, :]

    # Apply 2x2 binning with mean mode
    binned = binning_czyx(test_data, binning_factor_zyx=[1, 2, 2], mode="mean")

    # Expected result: (1, 2, 1, 1) with averaged values
    expected_shape = (1, 2, 1, 1)
    assert binned.shape == expected_shape

    # Check that values are averaged correctly
    # Z=0: mean of [1,2,3,4] = 2.5
    # Z=1: mean of [5,6,7,8] = 6.5
    assert np.allclose(binned[0, 0, 0, 0], 2.5)
    assert np.allclose(binned[0, 1, 0, 0], 6.5)

    print("Mean mode binning function test passed!")


def test_squaring_function_direct():
    """
    Test the squaring function directly
    """
    test_data = np.array([2.0, 3.0, 4.0])
    squared = np.square(test_data)
    expected = np.array([4.0, 9.0, 16.0])
    assert np.allclose(squared, expected)
    print("Direct squaring function test passed!")


def test_process_with_config_invalid_function(tmp_path, example_process_plate, monkeypatch):
    """
    Test that process_with_config fails with invalid function
    """
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_process_plate
    output_path = tmp_path / "processed_output_invalid.zarr"

    # Create configuration with invalid function
    config_content = """processing_functions:
  - function: nonexistent.module.invalid_function
    input_channels: ["BF"]
    kwargs: {}
"""

    config_path = tmp_path / "test_process_config_invalid.yml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    input_position_paths = [plate_path / "A" / "1" / "0"]

    # Test that it raises an exception with invalid function
    with pytest.raises(Exception):
        process_with_config(
            input_position_dirpaths=input_position_paths,
            config_filepath=config_path,
            output_dirpath=output_path,
            local=True,
        )


def test_process_with_config_invalid_channel(tmp_path, example_process_plate, monkeypatch):
    """
    Test that process_with_config fails with invalid channel name
    """
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_process_plate
    output_path = tmp_path / "processed_output_invalid_channel.zarr"

    # Create configuration with invalid channel
    config_content = """processing_functions:
  - function: np.square
    input_channels: ["INVALID_CHANNEL"]
    kwargs: {}
"""

    config_path = tmp_path / "test_process_config_invalid_channel.yml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    input_position_paths = [plate_path / "A" / "1" / "0"]

    # Test that it raises an exception with invalid channel
    with pytest.raises(Exception):
        process_with_config(
            input_position_dirpaths=input_position_paths,
            config_filepath=config_path,
            output_dirpath=output_path,
            local=True,
        )


def test_process_with_config_empty_functions(tmp_path, example_process_plate, monkeypatch):
    """
    Test that process_with_config fails with empty processing functions
    """
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_process_plate
    output_path = tmp_path / "processed_output_empty.zarr"

    # Create configuration with empty processing functions
    config_content = """processing_functions: []
"""

    config_path = tmp_path / "test_process_config_empty.yml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    input_position_paths = [plate_path / "A" / "1" / "0"]

    # Test that it raises an exception with empty functions
    with pytest.raises(Exception):
        process_with_config(
            input_position_dirpaths=input_position_paths,
            config_filepath=config_path,
            output_dirpath=output_path,
            local=True,
        )


def test_process_with_config_multiple_channels(tmp_path, example_process_plate, monkeypatch):
    """
    Test process_with_config with processing multiple channels
    """
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_process_plate
    output_path = tmp_path / "processed_output_multiple.zarr"

    # Create test configuration for multiple channels
    config_content = """processing_functions:
  - function: biahub.process_data.binning_czyx
    input_channels: ["BF"]
    kwargs:
      binning_factor_zyx: [1, 2, 2]
      mode: "sum"
  - function: np.square
    input_channels: ["GFP"]
    kwargs: {}
  - function: np.sqrt
    input_channels: ["BF"]
    kwargs: {}
"""

    config_path = tmp_path / "test_process_config_multiple.yml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    # Get input position paths
    input_position_paths = [
        plate_path / "A" / "1" / "0",
    ]

    # Call process_with_config function directly
    process_with_config(
        input_position_dirpaths=input_position_paths,
        config_filepath=config_path,
        output_dirpath=output_path,
        local=True,
    )

    # Verify output exists
    assert output_path.exists()

    # Verify the processed data by loading a sample
    from iohub import open_ome_zarr

    with open_ome_zarr(output_path / "A" / "1" / "0") as output_dataset:
        # Check that the output has the expected shape after binning
        T, C, Z, Y, X = output_dataset.data.shape

        # Original shape was (3, 2, 4, 32, 32)
        # After 2x2 binning in Y and X: (3, 2, 4, 16, 16)
        expected_shape = (3, 2, 4, 16, 16)
        assert (T, C, Z, Y, X) == expected_shape

        # Check that channel names are preserved
        assert output_dataset.channel_names == ["BF", "GFP"]

        print(f"Multiple channels test - Output shape: {output_dataset.data.shape}")
        print(f"Multiple channels test - Channel names: {output_dataset.channel_names}")
