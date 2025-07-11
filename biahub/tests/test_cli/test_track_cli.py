import numpy as np
import pandas as pd
import pytest
import os
from click.testing import CliRunner

from biahub.cli.main import cli


@pytest.fixture(scope="function")
def example_tracking_plate(tmp_path):
    """
    Create a test plate with nuclei and membrane prediction channels for tracking
    """
    plate_path = tmp_path / "tracking_plate.zarr"
    
    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
        ("B", "2", "0"),
    )
    
    # Create plate with nuclei and membrane prediction channels
    from iohub.ngff import open_ome_zarr
    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["nuclei_prediction", "membrane_prediction"],
    )
    
    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        # Create realistic nuclei and membrane data
        # Shape: (T, C, Z, Y, X) = (5, 2, 3, 64, 64)
        data = np.random.uniform(0.1, 0.3, size=(5, 2, 3, 64, 64)).astype(np.float32)
        
        # Add some bright nuclei spots to channel 0
        for t in range(5):
            for z in range(3):
                # Add 3-5 nuclei per frame
                for _ in range(np.random.randint(3, 6)):
                    y, x = np.random.randint(10, 54, 2)
                    data[t, 0, z, y-3:y+4, x-3:x+4] = np.random.uniform(0.7, 1.0)
        
        # Add some membrane boundaries to channel 1
        for t in range(5):
            for z in range(3):
                # Add some membrane structures
                for _ in range(np.random.randint(2, 4)):
                    y, x = np.random.randint(15, 49, 2)
                    # Create circular membrane structures
                    yy, xx = np.ogrid[:64, :64]
                    mask = (yy - y)**2 + (xx - x)**2 <= 25
                    data[t, 1, z][mask] = np.random.uniform(0.6, 0.9)
        
        position["0"] = data
    
    yield plate_path, plate_dataset


@pytest.fixture(scope="function")
def example_blank_frames_csv(tmp_path):
    """
    Create a CSV file with blank frame information for testing
    """
    csv_path = tmp_path / "blank_frames.csv"
    
    # Create sample blank frame data
    data = {
        'FOV': ['A_1_0', 'B_1_0', 'B_2_0'],
        't': ['[0]', '[2]', '[]']  # A_1_0 has blank frame 0, B_1_0 has blank frame 2, B_2_0 has no blank frames
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    yield csv_path


def test_track_cli_local(tmp_path, example_tracking_plate, example_track_settings, sbatch_file, monkeypatch):
    # Set environment variable globally for the entire test process
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")
    os.environ["ULTRACK_ARRAY_MODULE"] = "numpy"
    
    # Create a custom sbatch file that forces local execution
    custom_sbatch_file = tmp_path / "custom_sbatch.txt"
    with open(custom_sbatch_file, "w") as f:
        f.write("#SBATCH --cpus-per-task=1\n")
        f.write("#SBATCH --array-parallelism=1\n")  # Force only 1 job at a time
        f.write("#LOCAL --cpus-per-task=1\n")
        f.write("#LOCAL --timeout-min=5\n")
        f.write("#LOCAL --array-parallelism=1\n")  # Force local to use only 1 process
    
    plate_path, _ = example_tracking_plate
    config_path, _ = example_track_settings
    output_path = tmp_path / "tracking_output.zarr"
    # Create a modified config for testing
    test_config_path = tmp_path / "test_track_config.yml"
    with open(config_path) as f:
        config_content = f.read()
    config_content = config_content.replace("/path/to/virtual_staining.zarr", str(plate_path))
    with open(test_config_path, 'w') as f:
        f.write(config_content)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "track",
            "-o",
            str(output_path),
            "-c",
            str(test_config_path),
            "--local",
            "--sbatch-filepath",
            str(custom_sbatch_file),
        ],
        catch_exceptions=False,
    )
    print(f"CLI Output: {result.output}")
    print(f"Exit Code: {result.exit_code}")
    
    # Test that the CLI runs without crashing
    assert result.exit_code == 0
    
    # Test that the output directory is created
    assert output_path.exists()
    
    # Test that the basic structure is created (we don't expect the full pipeline to work in tests)
    for position in ["A/1/0", "B/1/0", "B/2/0"]:
        position_path = output_path / position
        # The position directories should exist even if tracking fails
        assert position_path.exists()


def test_track_cli_with_blank_frames(tmp_path, example_tracking_plate, example_track_settings, example_blank_frames_csv, sbatch_file, monkeypatch):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")
    """
    Test the track CLI command with blank frames CSV
    """
    plate_path, _ = example_tracking_plate
    config_path, _ = example_track_settings
    output_path = tmp_path / "tracking_output_blank_frames.zarr"
    
    # Create a modified config for testing with blank frames
    test_config_path = tmp_path / "test_track_config_blank_frames.yml"
    with open(config_path) as f:
        config_content = f.read()
    
    # Update the path in the config to point to our test data
    config_content = config_content.replace(
        "/path/to/virtual_staining.zarr", 
        str(plate_path)
    )
    # Update blank frames path
    config_content = config_content.replace(
        "blank_frames.csv", 
        str(example_blank_frames_csv)
    )
    
    with open(test_config_path, 'w') as f:
        f.write(config_content)
    
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "track",
            "-o",
            str(output_path),
            "-c",
            str(test_config_path),
            "--local",
            "--sbatch-filepath",
            str(sbatch_file),
        ],
        catch_exceptions=False,
    )
    
    # Check that the command executed successfully
    assert result.exit_code == 0
    assert output_path.exists()


def test_track_cli_invalid_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")
    """
    Test that the track command fails with invalid config
    """
    output_path = tmp_path / "output.zarr"
    invalid_config_path = tmp_path / "invalid_config.yml"
    
    # Create an invalid config file
    with open(invalid_config_path, 'w') as f:
        f.write("invalid: yaml: content")
    
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "track",
            "-o",
            str(output_path),
            "-c",
            str(invalid_config_path),
            "--local",
        ],
        catch_exceptions=True,
    )
    
    # Should fail due to invalid config
    assert result.exit_code != 0


def test_track_cli_missing_input_path(tmp_path, example_track_settings, monkeypatch):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")
    """
    Test that the track command fails when input path doesn't exist
    """
    config_path, _ = example_track_settings
    output_path = tmp_path / "output.zarr"
    
    # Create a config with non-existent input path
    test_config_path = tmp_path / "test_track_config_missing.yml"
    with open(config_path) as f:
        config_content = f.read()
    
    # Update the path to a non-existent location
    config_content = config_content.replace(
        "/path/to/virtual_staining.zarr", 
        "/non/existent/path"
    )
    
    with open(test_config_path, 'w') as f:
        f.write(config_content)
    
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "track",
            "-o",
            str(output_path),
            "-c",
            str(test_config_path),
            "--local",
        ],
        catch_exceptions=True,
    )
    
    # Should fail due to missing input path
    assert result.exit_code != 0
