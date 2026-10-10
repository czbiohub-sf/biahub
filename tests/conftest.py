import os
import sys
import types

import numpy as np
import pytest
import yaml

from iohub.ngff import TransformationMeta, open_ome_zarr

# Use submitit debug executor (in-process, no forking) for fast tests
os.environ["CI"] = "true"

# iohub's default zarr implementation is a *process-global*. biahub relies on
# the ``zarrs-python`` (Rust) pipeline for its sharded zarr-v3 writes
# (flat-field, concatenate, ...). Importing ``cytoland``/``viscy_data`` runs
# ``viscy_data._zarr_codec`` at import time, which flips that global to
# ``zarr-python`` via ``set_default_implementation``. zarr-python's pure-Python
# pipeline mishandles orthogonally-indexed sharded rank-5 writes, so once any
# test imports cytoland (the virtual-stain config test, which also imports it
# at module scope during collection) every later sharded write in the same
# process fails with a broadcast ``shape mismatch``. In production each CLI
# command runs in its own process so the two never collide, but pytest imports
# them all together. Snapshot iohub's pristine default here -- conftest is
# imported before any test module, i.e. before cytoland is imported -- and
# restore it around every test below.
from iohub.core import registry as _iohub_registry  # noqa: E402

_IOHUB_DEFAULT_IMPLEMENTATION = _iohub_registry._default


@pytest.fixture(autouse=True)
def _restore_iohub_default_implementation():
    """Pin iohub's default zarr implementation for the duration of each test.

    See the module-level note above: keeps cytoland's import-time override from
    leaking into biahub's sharded zarr writes.
    """
    _iohub_registry.set_default_implementation(_IOHUB_DEFAULT_IMPLEMENTATION)
    try:
        yield
    finally:
        _iohub_registry.set_default_implementation(_IOHUB_DEFAULT_IMPLEMENTATION)


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
    settings_path = "./settings/example_registration_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_estimate_stabilization_settings():
    settings_path = "./settings/example_estimate_stabilization_settings.yml"
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
def example_stitch_settings():
    settings_path = "./settings/example_stitch_settings.yml"
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

    # Lateral pixel size matches example_deskew_settings.yml (pixel_size_um:
    # 0.116) so deskew doesn't warn about a config/metadata scale mismatch.
    scale = (1, 1, 1.0, 0.116, 0.116)
    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        position.create_image(
            "0",
            np.random.uniform(0.0, 255.0, size=(3, 6, 4, 5, 6)).astype(np.float32),
            transform=[TransformationMeta(type="scale", scale=scale)],
        )

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
        position_list=(("A", "1", "0"), ("B", "1", "0"), ("B", "2", "0")),
        channel_names=("GFP", "RFP", "Phase3D"),
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


# --- cellpose stand-in shared by segment/track tests (the test extra does not install cellpose)


class RecordingModel:
    """Stand-in for cellpose.models.CellposeModel that records how it is used."""

    built = []
    evaluated = []

    def __init__(self, gpu=False, pretrained_model="cpsam_v2", model_type=None, device=None):
        # Like cellpose 4: model_type is accepted and ignored.
        self.pretrained_model, self.device = pretrained_model, device
        RecordingModel.built.append(pretrained_model)

    def eval(
        self,
        x,
        channels=None,
        channel_axis=None,
        z_axis=None,
        diameter=None,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        do_3D=False,
        anisotropy=None,
        min_size=15,
        stitch_threshold=0.0,
    ):
        # Cellpose 4 refuses a z axis for 2D processing (cellpose/transforms.py:491).
        if z_axis is not None and not do_3D and not stitch_threshold:
            raise ValueError("2D image processing selected, but z_axis is not None.")
        RecordingModel.evaluated.append(np.array(x))
        # Like cellpose: masks drop the channel axis and come back squeezed.
        shape = x.shape[1:] if channel_axis == 0 else x.shape
        return np.ones(shape, dtype=np.uint16).squeeze(), None, None


@pytest.fixture()
def fake_cellpose(monkeypatch, tmp_path):
    models = types.ModuleType("cellpose.models")
    models.CellposeModel = RecordingModel
    models.MODEL_NAMES = ["cpsam_v2", "cpsam", "cpdino", "cpdino-vitb"]
    models.get_user_models = lambda: []
    package = types.ModuleType("cellpose")
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(tmp_path / "no-weights"))
    RecordingModel.built, RecordingModel.evaluated = [], []
    from biahub import segment

    segment._MODEL_CACHE.clear()  # models are cached per process
    return RecordingModel
