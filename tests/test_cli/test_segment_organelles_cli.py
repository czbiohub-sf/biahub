import numpy as np
import yaml

from iohub import open_ome_zarr

from biahub.segment_organelles import segment_organelles_cli


def test_segment_organelles_cli_basic(create_custom_plate, tmp_path, sbatch_file):
    """Test organelle segmentation CLI with synthetic data"""

    # Create a test plate with organelle channels
    input_path, input_plate = create_custom_plate(
        tmp_path / "input",
        channel_names=["mito", "er"],
        time_points=2,
        z_size=1,
        y_size=64,
        x_size=64,
    )

    # Add some structure to the data to segment
    with open_ome_zarr(input_path, mode="r+") as plate:
        for pos_name, pos in plate.positions():
            # Add bright spots for mitochondria (channel 0)
            pos.data[0, 0, 0, 20:30, 20:30] = 30000
            pos.data[1, 0, 0, 40:50, 40:50] = 30000

            # Add bright spots for ER (channel 1)
            pos.data[0, 1, 0, 10:15, 10:15] = 25000
            pos.data[1, 1, 0, 50:55, 50:55] = 25000

    # Create a temporary config file
    config_path = tmp_path / "organelle_segment_config.yml"
    config = {
        "spacing": [1.0, 1.0, 1.0],
        "channels": {
            "mito": {
                "segment_kwargs": {
                    "sigma_range": [1.0, 2.0],
                    "sigma_steps": 2,
                    "threshold_method": "otsu",
                    "min_object_size": 5,
                    "apply_morphology": False,
                }
            },
            "er": {
                "segment_kwargs": {
                    "sigma_range": [1.0, 2.0],
                    "sigma_steps": 2,
                    "threshold_method": "triangle",
                    "min_object_size": 3,
                    "apply_morphology": False,
                }
            },
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run the CLI
    output_path = tmp_path / "output.zarr"

    segment_organelles_cli.callback(
        input_position_dirpaths=[str(input_path / "A/1/0")],
        config_filepath=config_path,
        output_dirpath=str(output_path),
        sbatch_filepath=str(sbatch_file),
        local=True,
        monitor=False,
    )

    # Verify output
    output_plate = open_ome_zarr(output_path, mode="r")

    # Check channel names
    assert "mito_labels" in output_plate.channel_names
    assert "er_labels" in output_plate.channel_names

    # Check output shape
    T, C, Z, Y, X = output_plate["A/1/0"].data.shape
    assert T == 2  # Same number of timepoints
    assert C == 2  # Two segmentation channels
    assert Z == 1
    assert Y == 64
    assert X == 64

    # Check that labels were created (at least background label 0)
    labels_data = output_plate["A/1/0"].data[:]
    assert labels_data.max() >= 0

    # Check dtype
    assert labels_data.dtype == np.uint16


def test_segment_organelles_cli_single_channel(create_custom_plate, tmp_path, sbatch_file):
    """Test organelle segmentation CLI with single channel"""

    # Create a test plate with single organelle channel
    input_path, input_plate = create_custom_plate(
        tmp_path / "input",
        channel_names=["mito"],
        time_points=1,
        z_size=1,
        y_size=32,
        x_size=32,
    )

    # Add some structure
    with open_ome_zarr(input_path, mode="r+") as plate:
        for pos_name, pos in plate.positions():
            pos.data[0, 0, 0, 10:20, 10:20] = 30000

    # Create config
    config_path = tmp_path / "config.yml"
    config = {
        "spacing": [1.0, 1.0, 1.0],
        "channels": {
            "mito": {
                "segment_kwargs": {
                    "sigma_range": [1.0, 2.0],
                    "threshold_method": "otsu",
                    "min_object_size": 1,
                }
            }
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run CLI
    output_path = tmp_path / "output.zarr"

    segment_organelles_cli.callback(
        input_position_dirpaths=[str(input_path / "A/1/0")],
        config_filepath=config_path,
        output_dirpath=str(output_path),
        sbatch_filepath=str(sbatch_file),
        local=True,
        monitor=False,
    )

    # Verify output
    output_plate = open_ome_zarr(output_path, mode="r")
    assert len(output_plate.channel_names) == 1
    assert "mito_labels" in output_plate.channel_names
