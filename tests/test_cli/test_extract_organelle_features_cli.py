import numpy as np
import pandas as pd
import yaml

from iohub import open_ome_zarr

from biahub.extract_organelle_features import extract_organelle_features_cli


def test_extract_organelle_features_cli_basic(create_custom_plate, tmp_path, sbatch_file):
    """Test feature extraction CLI with synthetic labels"""

    # Create a test plate with labels and intensity channels
    input_path, input_plate = create_custom_plate(
        tmp_path / "input",
        channel_names=["mito", "mito_labels"],
        time_points=2,
        z_size=1,
        y_size=64,
        x_size=64,
    )

    # Add labeled objects and intensity data
    with open_ome_zarr(input_path, mode="r+") as plate:
        for pos_name, pos in plate.positions():
            # Create labels (channel 1)
            labels = np.zeros((2, 1, 64, 64), dtype=np.uint16)
            labels[0, 0, 10:20, 10:20] = 1  # Object 1 at t=0
            labels[0, 0, 30:40, 30:40] = 2  # Object 2 at t=0
            labels[1, 0, 15:25, 15:25] = 1  # Object 1 at t=1

            # Create intensity (channel 0)
            intensity = np.zeros((2, 1, 64, 64), dtype=np.uint16)
            intensity[0, 0, 10:20, 10:20] = 30000  # Bright
            intensity[0, 0, 30:40, 30:40] = 10000  # Dim
            intensity[1, 0, 15:25, 15:25] = 25000

            pos.data[:, 0] = intensity
            pos.data[:, 1] = labels

    # Create config file
    config_path = tmp_path / "feature_extraction_config.yml"
    output_csv_path = tmp_path / "features.csv"

    config = {
        "labels_channel": "mito_labels",
        "intensity_channel": "mito",
        "frangi_channel": None,
        "tracking_csv_path": None,
        "spacing": [1.0, 1.0, 1.0],
        "properties": ["label", "area", "mean_intensity"],
        "extra_properties": ["aspect_ratio", "circularity"],
        "output_csv_path": str(output_csv_path),
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run the CLI
    extract_organelle_features_cli.callback(
        input_position_dirpaths=[str(input_path / "A/1/0")],
        config_filepath=config_path,
        sbatch_filepath=str(sbatch_file),
        local=True,
        monitor=False,
    )

    # Verify output CSV exists
    assert output_csv_path.exists()

    # Load and verify CSV contents
    features_df = pd.read_csv(output_csv_path)

    # Check that features were extracted
    assert len(features_df) > 0

    # Check expected columns
    assert "label" in features_df.columns
    assert "area" in features_df.columns
    assert "mean_intensity" in features_df.columns
    assert "aspect_ratio" in features_df.columns
    assert "circularity" in features_df.columns
    assert "fov_name" in features_df.columns
    assert "t" in features_df.columns

    # Check metadata
    assert "A/1/0" in features_df["fov_name"].values


def test_extract_organelle_features_cli_no_extra_properties(
    create_custom_plate, tmp_path, sbatch_file
):
    """Test feature extraction with only base properties"""

    # Create a simple test plate
    input_path, input_plate = create_custom_plate(
        tmp_path / "input",
        channel_names=["intensity", "labels"],
        time_points=1,
        z_size=1,
        y_size=32,
        x_size=32,
    )

    # Add simple labels
    with open_ome_zarr(input_path, mode="r+") as plate:
        for pos_name, pos in plate.positions():
            labels = np.zeros((1, 1, 32, 32), dtype=np.uint16)
            labels[0, 0, 10:20, 10:20] = 1

            intensity = np.ones((1, 1, 32, 32), dtype=np.uint16) * 1000
            intensity[0, 0, 10:20, 10:20] = 20000

            pos.data[:, 0] = intensity
            pos.data[:, 1] = labels

    # Create config without extra properties
    config_path = tmp_path / "config.yml"
    output_csv_path = tmp_path / "features.csv"

    config = {
        "labels_channel": "labels",
        "intensity_channel": "intensity",
        "spacing": [1.0, 1.0, 1.0],
        "properties": ["label", "area"],
        "extra_properties": [],
        "output_csv_path": str(output_csv_path),
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run CLI
    extract_organelle_features_cli.callback(
        input_position_dirpaths=[str(input_path / "A/1/0")],
        config_filepath=config_path,
        sbatch_filepath=str(sbatch_file),
        local=True,
        monitor=False,
    )

    # Verify output
    assert output_csv_path.exists()
    features_df = pd.read_csv(output_csv_path)
    assert len(features_df) > 0
    assert "label" in features_df.columns
    assert "area" in features_df.columns


def test_extract_organelle_features_cli_multiple_positions(
    create_custom_plate, tmp_path, sbatch_file
):
    """Test feature extraction across multiple positions"""

    # Create plate with multiple positions
    input_path, input_plate = create_custom_plate(
        tmp_path / "input",
        position_list=[("A", "1", "0"), ("B", "1", "0")],
        channel_names=["intensity", "labels"],
        time_points=1,
        z_size=1,
        y_size=32,
        x_size=32,
    )

    # Add labels to both positions
    with open_ome_zarr(input_path, mode="r+") as plate:
        for pos_name, pos in plate.positions():
            labels = np.zeros((1, 1, 32, 32), dtype=np.uint16)
            labels[0, 0, 10:15, 10:15] = 1

            intensity = np.ones((1, 1, 32, 32), dtype=np.uint16) * 1000

            pos.data[:, 0] = intensity
            pos.data[:, 1] = labels

    # Create config
    config_path = tmp_path / "config.yml"
    output_csv_path = tmp_path / "features.csv"

    config = {
        "labels_channel": "labels",
        "intensity_channel": "intensity",
        "spacing": [1.0, 1.0, 1.0],
        "properties": ["label", "area"],
        "extra_properties": [],
        "output_csv_path": str(output_csv_path),
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run CLI with multiple positions
    extract_organelle_features_cli.callback(
        input_position_dirpaths=[str(input_path / "A/1/0"), str(input_path / "B/1/0")],
        config_filepath=config_path,
        sbatch_filepath=str(sbatch_file),
        local=True,
        monitor=False,
    )

    # Verify output has features from both positions
    features_df = pd.read_csv(output_csv_path)
    assert len(features_df) >= 2  # At least one object from each position
    assert "A/1/0" in features_df["fov_name"].values
    assert "B/1/0" in features_df["fov_name"].values
