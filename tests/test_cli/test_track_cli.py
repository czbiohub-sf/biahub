import os
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_labels

from biahub.cli.main import cli
from biahub.settings import TrackingSettings
from biahub.track import _init_labels, track, write_tracking_labels
from biahub.utils.ngff import LABEL_SHARDS, label_chunks, label_layout

# The label image `biahub segment` would write (its model name) and the one track writes.
INPUT_LABEL = "nuclei"
LABEL = "tracks_nuclei"


@pytest.fixture(scope="function")
def example_tracking_plate(tmp_path):
    """
    Create a test plate with nuclei and membrane prediction channels for tracking, and
    the instance labels of the nuclei spots in ``labels/nuclei`` as segment would write.
    """
    from skimage.measure import label as connected_components

    plate_path = tmp_path / "tracking_plate.zarr"

    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
        ("B", "2", "0"),
    )

    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["nuclei_prediction", "membrane_prediction"],
    )

    # Deterministic spots: on random data ultrack occasionally finds no solution, and
    # the failing in-process job drops submitit into pdb (which pytest reports as a
    # read from stdin).
    np.random.seed(0)
    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        # A projected (Z=1) plate, what 2D tracking reads: (T, C, Z, Y, X) = (5, 2, 1, 64, 64)
        data = np.random.uniform(0.1, 0.3, size=(5, 2, 1, 64, 64)).astype(np.float32)

        # Add some bright nuclei spots to channel 0 (11 x 11 px, above the configs'
        # min_area of 100; a single plane no longer merges spots across Z)
        for t in range(5):
            for z in range(1):
                for _ in range(np.random.randint(3, 6)):
                    y, x = np.random.randint(10, 54, 2)
                    data[t, 0, z, y - 5 : y + 6, x - 5 : x + 6] = np.random.uniform(0.7, 1.0)

        # Add some membrane boundaries to channel 1
        for t in range(5):
            for z in range(1):
                for _ in range(np.random.randint(2, 4)):
                    y, x = np.random.randint(15, 49, 2)
                    yy, xx = np.ogrid[:64, :64]
                    mask = (yy - y) ** 2 + (xx - x) ** 2 <= 25
                    data[t, 1, z][mask] = np.random.uniform(0.6, 0.9)

        position["0"] = data

    plate_dataset.close()
    positions = [plate_path / Path(*key) for key in position_list]
    create_empty_labels(
        plate_path, list(position_list), [INPUT_LABEL], label_metadata={"biahub-segment": {}}
    )
    for position_path in positions:
        with open_ome_zarr(str(position_path), mode="r+") as position:
            nuclei = np.asarray(position["0"][:, 0])  # (T, Z, Y, X)
            labels = np.stack([connected_components(frame > 0.5) for frame in nuclei])
            position.get_label(INPUT_LABEL)["0"][:] = labels[:, None].astype(
                np.uint32
            )  # (T, 1, Z, Y, X)
    plate_dataset = open_ome_zarr(plate_path, mode="r")

    yield plate_path, plate_dataset


@pytest.fixture(scope="function")
def example_blank_frames_csv(tmp_path):
    """
    Create a CSV file with blank frame information for testing
    """
    csv_path = tmp_path / "blank_frames.csv"

    data = {
        "FOV": ["A_1_0", "B_1_0", "B_2_0"],
        "t": [
            "[0]",
            "[2]",
            "[]",
        ],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    yield csv_path


def _assert_tracking_outputs(position_path):
    """Labels in the plate's one plane, a valid GEFF linked to them, and the CSV."""
    import geff

    with open_ome_zarr(str(position_path), mode="r") as pos:
        labels = pos.get_label(LABEL)["0"][:]
    assert labels.dtype == np.uint32
    assert labels.shape[1:3] == (1, 1)  # one channel; the projected (Z=1) plate's Z
    assert labels.any()

    geff_path = position_path / "tracks.geff"
    geff.validate_structure(str(geff_path))
    _, metadata = geff.read(str(geff_path), backend="networkx")
    (related,) = metadata.related_objects
    assert related.path == f"../labels/{LABEL}"
    assert related.node_prop == "seg_id"

    (csv_path,) = position_path.glob("tracks_*.csv")
    tracks = pd.read_csv(csv_path)
    assert set(tracks["track_id"]) <= set(np.unique(labels)) - {0}

    # The full Ultrack config is provenance on the label image, not a file beside it.
    label_attrs = yaml.safe_load((position_path / "labels" / LABEL / "zarr.json").read_text())
    ultrack_config = label_attrs["attributes"]["biahub-track"]["ultrack_config"]
    assert "solution_gap" in ultrack_config["tracking"]
    assert "working_dir" not in ultrack_config["data"]


_ULTRACK_CONFIG = {
    "segmentation_config": {
        "min_area": 100,
        "max_area": 80000,
        "n_workers": 1,
        "min_frontier": 0.4,
        "max_noise": 0.05,
    },
    "linking_config": {
        "n_workers": 1,
        "max_distance": 15,
        "distance_weight": -0.0001,
        "max_neighbors": 3,
    },
    "tracking_config": {
        "n_threads": 1,
        "disappear_weight": -0.0001,
        "appear_weight": -0.001,
        "division_weight": -0.0001,
    },
}


def _make_tracking_config(plate_path, tmp_path):
    """A minimal config tracking the plate's `labels/nuclei` image (the default path)."""
    config_path = tmp_path / "track_config.yml"
    config = {
        "output_mode": "2D",
        "fov": "*/*/*",
        "labels": INPUT_LABEL,
        "labels_sigma": 1.0,
        "tracking_config": _ULTRACK_CONFIG,
    }
    config_path.write_text(yaml.dump(config))
    return config_path


def _make_foreground_contour_config(plate_path, tmp_path):
    """A config on the legacy foreground+contour path, which processes image channels."""
    config_path = tmp_path / "track_config.yml"
    config = {
        "output_mode": "2D",
        "fov": "*/*/*",
        "target_channel": LABEL,
        "segmentation_method": "foreground_contour",
        "input_images": [
            {
                # Primary data source left null -> resolved from the -i input plate.
                # The plate is projected (Z=1), so the channels need no processing.
                "path": None,
                "channels": {"nuclei_prediction": [], "membrane_prediction": []},
            },
            {
                "path": None,
                "channels": {
                    "foreground": [
                        {
                            "function": "ultrack.imgproc.detect_foreground",
                            "input_channels": ["nuclei_prediction"],
                            "kwargs": {"sigma": 90},
                        },
                    ],
                    "contour": [
                        {
                            "function": "biahub.track.mem_nuc_contour",
                            "input_channels": [
                                "nuclei_prediction",
                                "membrane_prediction",
                            ],
                            "kwargs": {},
                        },
                    ],
                },
            },
        ],
        "tracking_config": _ULTRACK_CONFIG,
    }
    config_path.write_text(yaml.dump(config))
    return config_path


def test_track_cli_local(
    tmp_path, example_tracking_plate, example_track_settings, sbatch_file, monkeypatch
):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")
    os.environ["ULTRACK_ARRAY_MODULE"] = "numpy"

    custom_sbatch_file = tmp_path / "custom_sbatch.txt"
    with open(custom_sbatch_file, "w") as f:
        f.write("#SBATCH --cpus-per-task=1\n")
        f.write("#SBATCH --array-parallelism=1\n")
        f.write("#LOCAL --cpus-per-task=1\n")
        f.write("#LOCAL --timeout-min=5\n")
        f.write("#LOCAL --array-parallelism=1\n")

    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    output_path = tmp_path / "tracking_output"

    track(
        input_position_dirpaths=[
            str(plate_path / "A" / "1" / "0"),
            str(plate_path / "B" / "1" / "0"),
            str(plate_path / "B" / "2" / "0"),
        ],
        output_dirpath=str(output_path),
        config_filepath=str(config_path),
        sbatch_filepath=str(custom_sbatch_file),
        cluster="local",
    )

    for position in ["A/1/0", "B/1/0", "B/2/0"]:
        _assert_tracking_outputs(plate_path / position)
        assert (output_path / position.replace("/", "_")).is_dir()  # Ultrack database
        with open_ome_zarr(str(plate_path / position), mode="r") as pos:
            # The segmentation labels are left as they were, beside the tracked ones.
            assert pos.label_names() == [INPUT_LABEL, LABEL]
            assert pos.get_label(INPUT_LABEL)["0"][:].any()


def test_track_cli_foreground_contour_path(tmp_path, example_tracking_plate, monkeypatch):
    """The legacy path processes image channels into a mask and contour map."""
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")
    plate_path, _ = example_tracking_plate
    config_path = _make_foreground_contour_config(plate_path, tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out"),
            "-c",
            str(config_path),
            "--cluster",
            "debug",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Tracking with foreground + contour" in result.output
    _assert_tracking_outputs(plate_path / "A" / "1" / "0")


def test_track_cli_with_blank_frames(
    tmp_path,
    example_tracking_plate,
    example_track_settings,
    example_blank_frames_csv,
    sbatch_file,
    monkeypatch,
):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    output_path = tmp_path / "tracking_output_blank_frames"

    # Add blank_frames_path to config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["blank_frames_path"] = str(example_blank_frames_csv)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    track(
        input_position_dirpaths=[
            str(plate_path / "A" / "1" / "0"),
            str(plate_path / "B" / "1" / "0"),
            str(plate_path / "B" / "2" / "0"),
        ],
        output_dirpath=str(output_path),
        config_filepath=str(config_path),
        sbatch_filepath=str(sbatch_file),
        cluster="local",
    )

    _assert_tracking_outputs(plate_path / "A" / "1" / "0")


def test_track_cli_invalid_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    output_path = tmp_path / "output.zarr"
    invalid_config_path = tmp_path / "invalid_config.yml"

    with open(invalid_config_path, "w") as f:
        f.write("invalid: yaml: content")

    with pytest.raises(Exception):  # noqa: B017
        track(
            input_position_dirpaths=[str(tmp_path / "nonexistent" / "A" / "1" / "0")],
            output_dirpath=str(output_path),
            config_filepath=str(invalid_config_path),
            cluster="local",
        )


def test_track_cli_missing_input_path(tmp_path, example_track_settings, monkeypatch):
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    config_path, _ = example_track_settings
    output_path = tmp_path / "output.zarr"

    # -i points at a nonexistent plate, so plate init fails before any tracking.
    with pytest.raises((FileNotFoundError, ValueError, KeyError)):
        track(
            input_position_dirpaths=[str(tmp_path / "nonexistent" / "A" / "1" / "0")],
            output_dirpath=str(output_path),
            config_filepath=str(config_path),
            cluster="local",
        )


def test_track_cli_init_only(tmp_path, example_tracking_plate):
    """--init creates empty label images (the plate's Z) in the -i positions and emits RESOURCES."""
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "track",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            str(plate_path / "B" / "1" / "0"),
            str(plate_path / "B" / "2" / "0"),
            "-c",
            str(config_path),
            "--init",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "RESOURCES:" in result.output

    for position in ["A/1/0", "B/1/0", "B/2/0"]:
        with open_ome_zarr(str(plate_path / position), mode="r") as pos:
            assert pos.label_names() == [INPUT_LABEL, LABEL]
            assert pos.channel_names == ["nuclei_prediction", "membrane_prediction"]
            label = pos.get_label(LABEL)
            assert label["0"].shape == (5, 1, 1, 64, 64)  # one channel, the plate's Z
            assert label["0"].dtype == np.uint32
            assert not label["0"][:].any()
            assert [ax.name.lower() for ax in label.axes] == ["t", "c", "z", "y", "x"]
        label_attrs = yaml.safe_load(
            (plate_path / position / "labels" / LABEL / "zarr.json").read_text()
        )["attributes"]
        assert label_attrs["biahub-track"]["labels"] == INPUT_LABEL
        # The labels listing lives on the labels group (OME-NGFF 0.5).
        labels_attrs = yaml.safe_load(
            (plate_path / position / "labels" / "zarr.json").read_text()
        )["attributes"]
        assert labels_attrs["ome"]["labels"] == [INPUT_LABEL, LABEL]


def test_track_init_requires_the_input_label_image(tmp_path, create_custom_plate):
    """A plate segment has not run on fails at --init, naming `biahub segment`."""
    plate_path, _ = create_custom_plate(tmp_path, z_size=1)
    config_path = _make_tracking_config(plate_path, tmp_path)

    result = CliRunner().invoke(
        cli, ["track", "--init", "-i", str(plate_path / "A/1/0"), "-c", str(config_path)]
    )

    assert result.exit_code != 0
    assert "biahub segment" in str(result.exception)
    assert not (plate_path / "A/1/0" / "labels").exists()


def test_track_cli_debug_single_position(tmp_path, example_tracking_plate, monkeypatch):
    """Test that --cluster debug processes a single position in-process."""
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    # No -o: the Ultrack database goes to $TMPDIR and the submitit folder to ./
    temp_dir = tmp_path / "tmpdir"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    monkeypatch.chdir(tmp_path)

    def run():
        return CliRunner().invoke(
            cli,
            [
                "track",
                "-i",
                str(plate_path / "A" / "1" / "0"),
                "-c",
                str(config_path),
                "--cluster",
                "debug",
            ],
        )

    result = run()
    assert result.exit_code == 0, result.output
    assert "Tracking complete:" in result.output
    _assert_tracking_outputs(plate_path / "A" / "1" / "0")
    # The temporary database was deleted. (ultrack's own temporary arrays go when
    # they are garbage collected, at the latest when the worker process exits.)
    assert not list(temp_dir.glob("biahub-track-*"))
    assert (tmp_path / "slurm_output").is_dir()

    # A retry overwrites the same outputs.
    result = run()
    assert result.exit_code == 0, result.output
    _assert_tracking_outputs(plate_path / "A" / "1" / "0")


# ---------------------------------------------------------------------------
# Labels and dimensionality: track neither projects nor selects Z (that is
# `biahub project`); the label image takes the plate's Z
# ---------------------------------------------------------------------------


def _minimal_settings(**overrides):
    """Legacy-path settings: they need no input label image, only the plate."""
    base = {
        "target_channel": LABEL,
        "segmentation_method": "foreground_contour",
        "input_images": [{"path": None, "channels": {"nuclei_prediction": []}}],
    }
    base.update(overrides)
    return TrackingSettings(**base)


def test_labels_is_the_default_and_names_the_output():
    with pytest.raises(ValueError, match="`labels` is not set"):
        TrackingSettings()
    settings = TrackingSettings(labels="nuclei")
    assert settings.segmentation_method == "labels"
    assert settings.tracked_label == "tracks_nuclei"
    assert settings.input_images == []
    with pytest.raises(ValueError, match="target_channel is for"):
        TrackingSettings(labels="nuclei", target_channel="x")


def test_foreground_contour_needs_a_target_channel_and_input_images():
    with pytest.raises(ValueError, match="needs target_channel"):
        TrackingSettings(segmentation_method="foreground_contour")
    with pytest.raises(ValueError, match="needs input_images"):
        TrackingSettings(segmentation_method="foreground_contour", target_channel="x")
    assert _minimal_settings().tracked_label == LABEL


def test_output_label_names_the_tracked_image_and_the_tracks_files():
    default = TrackingSettings(labels="nuclei")
    assert (default.tracked_label, default.tracks_stem) == ("tracks_nuclei", "tracks")
    custom = TrackingSettings(labels="nuclei", output_label="nuclei_sigma3")
    assert (custom.tracked_label, custom.tracks_stem) == (
        "nuclei_sigma3",
        "tracks_nuclei_sigma3",
    )
    with pytest.raises(ValueError, match="is the label image being tracked"):
        TrackingSettings(labels="nuclei", output_label="nuclei")
    with pytest.raises(ValueError, match="output_label is for"):
        _minimal_settings(output_label="x")


def test_track_debug_run_with_an_output_label(tmp_path, example_tracking_plate, monkeypatch):
    """Two tracking runs of one segmentation keep their labels, GEFFs and CSVs apart."""
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["output_label"] = "nuclei_run2"
    config_path.write_text(yaml.safe_dump(config))
    position_path = plate_path / "A" / "1" / "0"

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "debug",
            "-i",
            str(position_path),
            "-o",
            str(tmp_path / "o"),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    with open_ome_zarr(str(position_path), mode="r") as pos:
        assert pos.label_names() == [INPUT_LABEL, "nuclei_run2"]
        assert pos.get_label("nuclei_run2")["0"][:].any()
    assert (position_path / "tracks_nuclei_run2.geff").is_dir()
    assert (position_path / "tracks_nuclei_run2_A_1_0.csv").is_file()
    assert not (position_path / "tracks.geff").exists()
    import geff

    _, metadata = geff.read(str(position_path / "tracks_nuclei_run2.geff"), backend="networkx")
    assert metadata.related_objects[0].path == "../labels/nuclei_run2"


def test_cellpose_config_is_rejected_with_a_pointer_to_segment():
    with pytest.raises(ValueError, match="biahub segment.*labels"):
        TrackingSettings(labels="nuclei", cellpose_config={"diameter": 80})
    with pytest.raises(ValueError, match="biahub segment"):
        TrackingSettings(segmentation_method="cellpose")


def test_init_labels_take_the_input_z(tmp_path, example_tracking_plate, create_custom_plate):
    """The label image has the plate's Z: one plane for a projected plate, the stack in 3D."""
    flat_path, _ = example_tracking_plate  # Z=1
    position_path = flat_path / "A" / "1" / "0"
    shape = _init_labels([position_path], _minimal_settings())
    assert shape[2] == 1
    with open_ome_zarr(str(position_path), mode="r") as pos:
        assert pos.get_label(LABEL)["0"].shape == (5, 1, 1, 64, 64)

    stack_path, _ = create_custom_plate(tmp_path, z_size=3)
    position_path = stack_path / "A" / "1" / "0"
    shape = _init_labels([position_path], _minimal_settings(output_mode="3D"))
    assert shape[2] == 3
    with open_ome_zarr(str(position_path), mode="r") as pos:
        assert pos.get_label(LABEL)["0"].shape == (3, 1, 3, 5, 6)


@pytest.mark.parametrize(
    "output_mode, z_size, message",
    [("2D", 3, "input is 3D.*biahub project"), ("3D", 1, "one Z-plane")],
)
def test_init_labels_refuses_a_dimensionality_mismatch(
    tmp_path, create_custom_plate, output_mode, z_size, message
):
    plate_path, _ = create_custom_plate(tmp_path, z_size=z_size)
    position_path = plate_path / "A" / "1" / "0"
    with pytest.raises(ValueError, match=message):
        _init_labels([position_path], _minimal_settings(output_mode=output_mode))
    assert not (position_path / "labels").exists()


def test_init_labels_leaves_existing_labels(tmp_path, example_tracking_plate):
    plate_path, _ = example_tracking_plate
    position_path = plate_path / "A" / "1" / "0"
    settings = _minimal_settings()

    _init_labels([position_path], settings)
    with open_ome_zarr(str(position_path), mode="r+") as pos:
        pos.get_label(LABEL)["0"][0, 0, 0, 0, 0] = 7
        image = pos["0"][:]
    _init_labels([position_path], settings)
    with open_ome_zarr(str(position_path), mode="r") as pos:
        assert pos.get_label(LABEL)["0"][0, 0, 0, 0, 0] == 7
        assert np.array_equal(pos["0"][:], image)


def test_label_layout(tmp_path, example_tracking_plate):
    """1 MB uint32 chunks (DCA SHOULD) and a 200 MB shard target, as iohub's sizer takes it."""
    assert label_chunks((67, 1, 86, 1664, 1193)) == (16, 1, 1, 128, 128)
    assert label_chunks((5, 1, 1, 64, 64)) == (5, 1, 1, 64, 64)  # clamped to small arrays

    plate_path, _ = example_tracking_plate
    shape, layout = label_layout(plate_path / "A" / "1" / "0")
    assert shape == (5, 2, 1, 64, 64)
    assert layout == {
        "label_dtype": np.uint32,
        "label_chunks": (5, 1, 1, 64, 64),
        "label_shards": LABEL_SHARDS,
    }
    # What the sizer makes of it on the A549 projected plate: the 16-timepoint chunk by
    # the full frame, rounded up to whole 128 px chunks (136 MB); a 32-timepoint shard
    # would exceed 200 MB.
    from iohub.core.sharding import resolve_shards

    assert resolve_shards(
        LABEL_SHARDS,
        shape=(67, 1, 1, 1664, 1193),
        chunks=label_chunks((67, 1, 1, 1664, 1193)),
        dtype=np.uint32,
        dimension_names=["t", "c", "z", "y", "x"],
    ) == (16, 1, 1, 1664, 1280)


def test_label_layout_copies_the_source_label(tmp_path, create_custom_plate):
    """A derived label (track's from segment's) takes the source's dtype, chunks and shards."""
    plate_path, _ = create_custom_plate(tmp_path, position_list=(("A", "1", "0"),), z_size=1)
    position = plate_path / "A" / "1" / "0"
    with open_ome_zarr(str(position), mode="r+") as dataset:
        dataset.create_label_zeros(
            "cells",
            shape=(3, 1, 1, 5, 6),
            dtype=np.uint16,
            chunks=(1, 1, 1, 5, 3),
            shards=(3, 1, 1, 5, 6),
        )
    _, layout = label_layout(position, source_label="cells")
    assert layout == {
        "label_dtype": np.uint16,
        "label_chunks": (1, 1, 1, 5, 3),
        "label_shards": (3, 1, 1, 5, 6),
    }


def test_write_tracking_labels_keeps_the_label_dtype_and_refuses_overflow():
    label_array = np.zeros((2, 1, 1, 4, 4), dtype=np.uint16)
    write_tracking_labels(label_array, np.full((2, 4, 4), 65535, dtype=np.int64), "2D")
    assert label_array.dtype == np.uint16 and label_array.max() == 65535
    with pytest.raises(ValueError, match="more than the label image's uint16"):
        write_tracking_labels(label_array, np.full((2, 4, 4), 65536, dtype=np.int64), "2D")


def test_write_tracking_labels_2d_fills_the_single_plane():
    label_array = np.zeros((2, 1, 1, 4, 4), dtype=np.uint32)

    write_tracking_labels(label_array, np.ones((2, 4, 4), dtype=np.int64), "2D")
    assert label_array.all()
    # A single-plane series that kept its Z axis.
    write_tracking_labels(label_array, 2 * np.ones((2, 1, 4, 4)), "2D")
    assert (label_array == 2).all()
    with pytest.raises(ValueError, match="2D"):
        write_tracking_labels(label_array, np.ones((2, 3, 4, 4)), "2D")


def test_write_tracking_labels_3d_fills_the_stack():
    label_array = np.zeros((2, 1, 6, 4, 4), dtype=np.uint32)

    write_tracking_labels(label_array, np.ones((2, 6, 4, 4), dtype=np.int64), "3D")
    assert label_array.all()
    with pytest.raises(ValueError, match="3D"):
        write_tracking_labels(label_array, np.ones((2, 5, 4, 4)), "3D")


def test_track_cli_refuses_a_3d_input_for_2d_tracking(tmp_path, create_custom_plate):
    plate_path, _ = create_custom_plate(tmp_path, z_size=3)
    config_path = _make_tracking_config(plate_path, tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "-i",
            str(plate_path / "A/1/0"),
            "-o",
            str(tmp_path / "o.zarr"),
            "-c",
            str(config_path),
            "--init",
        ],
    )

    assert result.exit_code != 0
    assert "biahub project" in str(result.exception)


def test_z_slicing_is_rejected_with_a_pointer_to_project():
    with pytest.raises(ValueError, match="z_slicing was removed.*biahub project"):
        _minimal_settings(z_slicing={"method": "focus"})


# ---------------------------------------------------------------------------
# Input-path resolution
# ---------------------------------------------------------------------------


def test_input_images_path_override(tmp_path, example_tracking_plate, monkeypatch):
    """A null primary path is filled from --input-images-path when provided."""
    monkeypatch.setenv("ULTRACK_ARRAY_MODULE", "numpy")

    plate_path, _ = example_tracking_plate
    output_path = tmp_path / "override_output"
    config_path = _make_foreground_contour_config(plate_path, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "track",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(output_path),
            "-c",
            str(config_path),
            "--cluster",
            "debug",
            "--input-images-path",
            str(plate_path),
        ],
    )

    assert result.exit_code == 0, result.output
    _assert_tracking_outputs(plate_path / "A" / "1" / "0")


def test_load_labels_fills_blank_frames(tmp_path, example_tracking_plate):
    from biahub.track import load_labels

    plate_path, _ = example_tracking_plate
    position_path = plate_path / "A" / "1" / "0"
    with open_ome_zarr(str(position_path), mode="r+") as pos:
        pos.get_label(INPUT_LABEL)["0"][2] = 0  # a dropped timepoint: nothing segmented
    csv_path = tmp_path / "blank.csv"
    pd.DataFrame({"FOV": ["A/1/0"], "t": ["[2]"]}).to_csv(csv_path, index=False)

    labels = load_labels(position_path, INPUT_LABEL, "2D", csv_path)

    assert labels.shape == (5, 64, 64)
    assert np.array_equal(labels[2], labels[1])  # filled from the previous frame
    assert not load_labels(position_path, INPUT_LABEL, "2D")[2].any()
    with pytest.raises(ValueError, match="biahub segment"):
        load_labels(position_path, "missing", "2D")


def test_filter_short_siblings_merges_a_one_frame_daughter():
    """A daughter that lives 1 frame (e.g. a VS hallucination linked as a division) is
    removed and its sibling continues the parent track, in the tracks and the labels."""
    from biahub.track import filter_short_siblings

    labels = np.zeros((6, 8, 8), dtype=np.int32)
    rows = []
    for t in range(3):  # parent, track 1
        labels[t, 1:3, 1:3] = 1
        rows.append((1, t, 2.0, 2.0, -1))
    for t in range(3, 6):  # long daughter, track 2
        labels[t, 1:3, 1:3] = 2
        rows.append((2, t, 2.0, 2.0, 1))
    labels[3, 5:7, 5:7] = 3  # one-frame daughter, track 3
    rows.append((3, 3, 6.0, 6.0, 1))
    tracks = pd.DataFrame(rows, columns=["track_id", "t", "y", "x", "parent_track_id"])

    new_tracks, new_labels = filter_short_siblings(tracks, labels, min_length=2)

    assert set(new_tracks.track_id) == {1}
    assert (new_tracks.parent_track_id == -1).all()
    assert not (new_labels == 3).any()
    assert (new_labels[:, 1:3, 1:3] == 1).all()  # the parent continues through t=5


def test_min_sibling_length_is_off_by_default(example_track_settings):
    _, settings = example_track_settings
    settings.pop("min_sibling_length", None)
    assert TrackingSettings(**settings).min_sibling_length is None


def test_track_passes_min_sibling_length_to_each_position(
    tmp_path, example_tracking_plate, monkeypatch
):
    import contextlib
    import types

    submitted = {}

    class Executor:
        def __init__(self, folder, cluster):
            pass

        def update_parameters(self, **kwargs):
            pass

        def batch(self):
            return contextlib.nullcontext()

        def submit(self, fn, **kwargs):
            submitted.update(kwargs)
            return types.SimpleNamespace(job_id="0", wait=lambda: None)

    monkeypatch.setattr("biahub.track.submitit.AutoExecutor", Executor)
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["min_sibling_length"] = 3
    config_path.write_text(yaml.safe_dump(config))

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "debug",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert submitted["min_sibling_length"] == 3


def _lineage(short_len, long_len=3, parent_len=3):
    """Parent track 1, daughters 2 (long_len frames) and 3 (short_len frames)."""
    T = parent_len + max(long_len, short_len)
    labels = np.zeros((T, 8, 8), dtype=np.int32)
    rows = []
    for t in range(parent_len):
        labels[t, 1:3, 1:3] = 1
        rows.append((1, t, 2.0, 2.0, -1))
    for t in range(parent_len, parent_len + long_len):
        labels[t, 1:3, 1:3] = 2
        rows.append((2, t, 2.0, 2.0, 1))
    for t in range(parent_len, parent_len + short_len):
        labels[t, 5:7, 5:7] = 3
        rows.append((3, t, 6.0, 6.0, 1))
    return pd.DataFrame(rows, columns=["track_id", "t", "y", "x", "parent_track_id"]), labels


def test_filter_removes_daughters_of_min_length_or_fewer():
    """Same meaning as ultrack: a daughter of exactly ``min_length`` frames is removed."""
    from biahub.track import filter_short_siblings

    tracks, labels = _lineage(short_len=3, long_len=5)
    new_tracks, new_labels = filter_short_siblings(tracks, labels, min_length=3)
    assert set(new_tracks.track_id) == {1}
    assert not (new_labels == 3).any()

    tracks, labels = _lineage(short_len=4, long_len=5)
    new_tracks, _ = filter_short_siblings(tracks, labels, min_length=3)
    assert set(new_tracks.track_id) == {1, 2, 3}


def test_filter_keeps_a_division_when_both_daughters_are_short():
    from biahub.track import filter_short_siblings

    tracks, labels = _lineage(short_len=2, long_len=2)
    new_tracks, _ = filter_short_siblings(tracks, labels, min_length=3)
    assert set(new_tracks.track_id) == {1, 2, 3}


def test_filter_leaves_no_temporary_files_and_sorts_rows(tmp_path, monkeypatch):
    import tempfile

    from biahub.track import filter_short_siblings

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)  # re-read TMPDIR
    tracks, labels = _lineage(short_len=1, long_len=4)

    new_tracks, _ = filter_short_siblings(tracks, labels, min_length=2)

    assert list(tmp_path.iterdir()) == []
    assert new_tracks[["track_id", "t"]].equals(
        new_tracks[["track_id", "t"]].sort_values(["track_id", "t"]).reset_index(drop=True)
    )


def test_track_debug_run_applies_the_sibling_filter(tmp_path, example_tracking_plate):
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["min_sibling_length"] = 3
    config_path.write_text(yaml.safe_dump(config))
    position = str(plate_path / "A" / "1" / "0")
    out = str(tmp_path / "out.zarr")

    assert (
        CliRunner()
        .invoke(cli, ["track", "--init", "-i", position, "-o", out, "-c", str(config_path)])
        .exit_code
        == 0
    )
    result = CliRunner().invoke(
        cli, ["track", "--cluster", "debug", "-i", position, "-o", out, "-c", str(config_path)]
    )

    assert result.exit_code == 0, result.output
    assert "Sibling filter (min_length=3)" in result.output


def test_track_jobs_ask_for_no_gpu(tmp_path, example_tracking_plate, monkeypatch):
    """Ultrack is CPU work (segmentation is `biahub segment`'s): the cpu partition."""
    import contextlib

    recorded = {}

    class Executor:
        def __init__(self, folder, cluster):
            recorded["cluster"] = cluster

        def update_parameters(self, **kwargs):
            recorded.update(kwargs)

        def batch(self):
            return contextlib.nullcontext()

        def submit(self, *args, **kwargs):
            return types.SimpleNamespace(job_id="0", wait=lambda: None)

    import types

    monkeypatch.setattr("biahub.track.submitit.AutoExecutor", Executor)
    monkeypatch.delenv("CI", raising=False)
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "local",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded["cluster"] == "local"
    assert "gpus_per_node" not in recorded
    assert recorded["slurm_partition"] == "preempted"
    assert (
        recorded.get("timeout_min") == recorded["slurm_time"]
    )  # local default kills at 2 min


def test_track_local_sbatch_file_overrides_computed_parameters(
    tmp_path, example_tracking_plate, monkeypatch
):
    """'#LOCAL --timeout-min' must win over the computed local timeout, not collide
    (TypeError: multiple values for keyword argument 'timeout_min')."""
    import contextlib
    import types

    recorded = {}

    class Executor:
        def __init__(self, folder, cluster):
            recorded["cluster"] = cluster

        def update_parameters(self, **kwargs):
            recorded.update(kwargs)

        def batch(self):
            return contextlib.nullcontext()

        def submit(self, *args, **kwargs):
            return types.SimpleNamespace(job_id="0", wait=lambda: None)

    monkeypatch.setattr("biahub.track.submitit.AutoExecutor", Executor)
    monkeypatch.delenv("CI", raising=False)  # conftest forces CI (debug) otherwise
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    sbatch = tmp_path / "local.sbatch"
    sbatch.write_text("#LOCAL --timeout-min=5\n")

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "local",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
            "-sb",
            str(sbatch),
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded["timeout_min"] == 5


def test_track_local_timeout_ignores_a_slurm_time_string(
    tmp_path, example_tracking_plate, monkeypatch
):
    """'#SBATCH --time=3:00:00' sets slurm_time to a string; the local timeout_min must
    stay the computed number of minutes."""
    import contextlib
    import types

    recorded = {}

    class Executor:
        def __init__(self, folder, cluster):
            recorded["cluster"] = cluster

        def update_parameters(self, **kwargs):
            recorded.update(kwargs)

        def batch(self):
            return contextlib.nullcontext()

        def submit(self, *args, **kwargs):
            return types.SimpleNamespace(job_id="0", wait=lambda: None)

    monkeypatch.setattr("biahub.track.submitit.AutoExecutor", Executor)
    monkeypatch.delenv("CI", raising=False)  # conftest forces CI (debug) otherwise
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    sbatch = tmp_path / "slurm.sbatch"
    sbatch.write_text("#SBATCH --time=3:00:00\n")

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "local",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
            "-sb",
            str(sbatch),
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded["slurm_time"] == "3:00:00"
    assert recorded["timeout_min"] == 60  # track's computed time, in minutes


def test_track_mem_per_cpu_override_drops_the_computed_total(
    tmp_path, example_tracking_plate, monkeypatch
):
    """'#SBATCH --mem-per-cpu=16G' left --mem set too, and SLURM rejects both."""
    import contextlib
    import types

    recorded = {}

    class Executor:
        def __init__(self, folder, cluster):
            recorded["cluster"] = cluster

        def update_parameters(self, **kwargs):
            recorded.update(kwargs)

        def batch(self):
            return contextlib.nullcontext()

        def submit(self, *args, **kwargs):
            return types.SimpleNamespace(job_id="0", wait=lambda: None)

    monkeypatch.setattr("biahub.track.submitit.AutoExecutor", Executor)
    monkeypatch.delenv("CI", raising=False)  # conftest forces CI (debug) otherwise
    plate_path, _ = example_tracking_plate
    config_path = _make_tracking_config(plate_path, tmp_path)
    sbatch = tmp_path / "mem.sbatch"
    sbatch.write_text("#SBATCH --mem-per-cpu=16G\n")

    result = CliRunner().invoke(
        cli,
        [
            "track",
            "--cluster",
            "slurm",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-o",
            str(tmp_path / "out.zarr"),
            "-c",
            str(config_path),
            "-sb",
            str(sbatch),
        ],
    )

    assert result.exit_code == 0, result.output
    assert recorded["slurm_mem_per_cpu"] == "16G"
    assert "slurm_mem" not in recorded
