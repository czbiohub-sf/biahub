import numpy as np
import pytest
import yaml

from iohub.ngff import open_ome_zarr

# These fixtures return paired
# - paths for testing CLIs
# - objects for testing underlying functions


@pytest.fixture(scope="function")
def example_deskew_settings():
    settings_path = "./settings/example_deskew_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_register_settings():
    settings_path = "./settings/example_register_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_stabilize_timelapse_settings():
    settings_path = "./settings/example_stabilize_timelapse_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_concatenate_settings():
    settings_path = "./settings/example_concatenate_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_estimate_registration_settings():
    settings_path = "./settings/example_estimate_registration_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_track_settings():
    settings_path = "./settings/example_track_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_process_with_config_settings():
    settings_path = "./settings/example_process_with_config_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture()
def sbatch_file(tmp_path):
    filepath = tmp_path / "sbatch.txt"
    with open(filepath, "w") as f:
        f.write("#SBATCH --cpus-per-task=1\n")
        f.write("#SBATCH --array-parallelism=2\n")
        f.write("#LOCAL --cpus-per-task=1\n")
        f.write("#LOCAL --timeout-min=1\n")
    yield filepath


@pytest.fixture(scope="function")
def example_plate(tmp_path):
    """
    Example HCS plate with 3 positions and 6 channels and float32 data
    """
    plate_path = tmp_path / "plate.zarr"

    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
        ("B", "2", "0"),
    )

    # Generate input dataset
    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["GFP", "RFP", "Phase3D", "Orientation", "Retardance", "Birefringence"],
    )

    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        position["0"] = np.random.uniform(0.0, 255.0, size=(3, 6, 4, 5, 6)).astype(np.float32)

    yield plate_path, plate_dataset


@pytest.fixture(scope="function")
def example_plate_2(tmp_path):
    """
    Example HCS plate with 3 positions and 2 channels and uint16 data
    """
    plate_path = tmp_path / "plate_2.zarr"

    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
        ("B", "2", "0"),
    )

    # Generate input dataset
    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["GFP", "RFP"],
    )

    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        position["0"] = np.random.randint(
            100, np.iinfo(np.uint16).max, size=(3, 2, 4, 5, 6), dtype=np.uint16
        )
    yield plate_path, plate_dataset


@pytest.fixture(scope="function")
def create_custom_plate():
    """
    Factory fixture that creates an HCS plate with customizable channel names

    Returns a function that creates a plate with the specified channel names
    """

    def _create_plate(
        tmp_path,
        position_list=[("A", "1", "0"), ("B", "1", "0"), ("B", "2", "0")],
        channel_names=["GFP", "RFP", "Phase3D"],
        time_points=3,
        z_size=4,
        y_size=5,
        x_size=6,
    ):
        """
        Create a plate with custom channel names

        Args:
            tmp_path: Temporary path for the plate
            channel_names: List of channel names
            time_points: Number of time points (default: 3)
            z_size: Size of Z dimension (default: 4)
            y_size: Size of Y dimension (default: 5)
            x_size: Size of X dimension (default: 6)

        Returns:
            Tuple of (plate_path, plate_dataset)
        """
        plate_path = tmp_path / f"plate_custom_{'-'.join(channel_names)}.zarr"

        # Generate input dataset
        plate_dataset = open_ome_zarr(
            plate_path,
            layout="hcs",
            mode="w",
            channel_names=channel_names,
        )

        for row, col, fov in position_list:
            position = plate_dataset.create_position(row, col, fov)
            position["0"] = np.random.randint(
                100,
                np.iinfo(np.uint16).max,
                size=(time_points, len(channel_names), z_size, y_size, x_size),
                dtype=np.uint16,
            )

        return plate_path, plate_dataset

    return _create_plate
