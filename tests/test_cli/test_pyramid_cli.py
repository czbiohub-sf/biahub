"""biahub pyramid: multiscale levels on a plate's images and label images, in place."""

import json

import numpy as np
import pytest

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli
from biahub.pyramid import PROVENANCE_KEY, pyramid_dims

POSITIONS = (("A", "1", "0"), ("B", "1", "0"))


@pytest.fixture
def plate(create_custom_plate, tmp_path):
    """(T=3, C=3, Z=4, Y=16, X=12) uint16 with a uint32 label image in each position."""
    plate_path, _ = create_custom_plate(
        tmp_path, position_list=POSITIONS, z_size=4, y_size=16, x_size=12
    )
    for key in POSITIONS:
        with open_ome_zarr(plate_path.joinpath(*key), mode="r+") as position:
            labels = np.zeros((3, 1, 4, 16, 12), dtype=np.uint32)  # (T, 1, Z, Y, X)
            labels[..., 2:10, 1:7] = 5
            labels[..., 12:16, 8:12] = 9
            position.create_label("nuclei", labels)
    return plate_path


def _run(*args):
    return CliRunner().invoke(cli, ["pyramid", *map(str, args)])


def _levels(position_path):
    with open_ome_zarr(position_path, mode="r") as position:
        label = position.get_label("nuclei")
        return (
            {level: position[level].shape for level in position.array_keys()},
            {level: label[level].shape for level in label.array_keys()},
            dict(position.zattrs).get(PROVENANCE_KEY),
        )


def test_pyramid_dims_skips_singleton_axes():
    assert pyramid_dims((4, 16, 12)) == {"z", "y", "x"}
    assert pyramid_dims((1, 16, 12)) == {"y", "x"}


def test_pyramid_writes_image_and_label_levels(plate):
    positions = [plate.joinpath(*key) for key in POSITIONS]

    result = _run("--cluster", "debug", "-i", *positions, "--levels", "3")

    assert result.exit_code == 0, result.output
    for position_path in positions:
        image_levels, label_levels, record = _levels(position_path)
        assert image_levels == {
            "0": (3, 3, 4, 16, 12),
            "1": (3, 3, 2, 8, 6),
            "2": (3, 3, 1, 4, 3),
        }
        assert label_levels == {"0": (3, 1, 4, 16, 12), "1": (3, 1, 2, 8, 6), "2": (3, 1, 1, 4, 3)}
        assert record == {
            "levels": 3,
            "method": "mean",
            "label_method": "max",
            "dims": ["x", "y", "z"],
        }
        with open_ome_zarr(position_path, mode="r") as position:
            label = position.get_label("nuclei")
            for level in ("1", "2"):
                # Max pooling keeps only ids that exist at level 0, and keeps both blobs.
                assert set(np.unique(label[level][:])) == {0, 5, 9}
                assert position.get_effective_scale(level)[2:] == [2.0 ** int(level)] * 3
                assert label.get_effective_scale(level)[2:] == [2.0 ** int(level)] * 3
            assert position[1][:].mean() > 0


def test_pyramid_z1_plate_is_downsampled_in_yx_only(create_custom_plate, tmp_path):
    plate_path, _ = create_custom_plate(
        tmp_path, position_list=POSITIONS[:1], z_size=1, y_size=16, x_size=12
    )
    position_path = plate_path.joinpath(*POSITIONS[0])
    with open_ome_zarr(position_path, mode="r+") as position:
        position.create_label("nuclei", np.ones((3, 1, 1, 16, 12), dtype=np.uint32))

    result = _run("--cluster", "debug", "-i", position_path, "--label-method", "stride")

    assert result.exit_code == 0, result.output
    image_levels, label_levels, record = _levels(position_path)
    assert image_levels["2"] == (3, 3, 1, 4, 3)
    assert label_levels["2"] == (3, 1, 1, 4, 3)
    assert record["dims"] == ["x", "y"]
    assert record["label_method"] == "stride"
    with open_ome_zarr(position_path, mode="r") as position:
        # Z keeps its size and its scale at every level.
        assert position.get_effective_scale("2")[2] == 1.0
        assert position.get_label("nuclei").get_effective_scale("2")[2] == 1.0


def test_pyramid_rerun_recomputes_and_different_depth_is_refused(plate):
    position_path = plate.joinpath(*POSITIONS[0])
    assert _run("--cluster", "debug", "-i", position_path).exit_code == 0

    rerun = _run("--cluster", "debug", "-i", position_path, "--method", "max")
    assert rerun.exit_code == 0, rerun.output
    assert _levels(position_path)[2]["method"] == "max"

    deeper = _run("--cluster", "debug", "-i", position_path, "--levels", "4")
    assert deeper.exit_code != 0
    assert "different depth" in str(deeper.exception)
    assert "'image': 3" in str(deeper.exception)


def test_pyramid_init_validates_and_reports_resources(plate):
    positions = [plate.joinpath(*key) for key in POSITIONS]

    result = _run("--init", "-i", *positions)

    assert result.exit_code == 0, result.output
    line = next(ln for ln in result.output.splitlines() if ln.startswith("RESOURCES:"))
    assert set(json.loads(line.removeprefix("RESOURCES:"))) == {
        "cpus",
        "mem_gb",
        "time_minutes",
    }
    assert "Validated 3-level pyramids for 2 positions" in result.output
    # Nothing was written.
    image_levels, label_levels, record = _levels(positions[0])
    assert list(image_levels) == ["0"] and list(label_levels) == ["0"] and record is None


def test_pyramid_refuses_mean_for_labels_and_one_level(plate):
    position_path = plate.joinpath(*POSITIONS[0])
    assert _run("--init", "-i", position_path, "--label-method", "mean").exit_code == 2
    one_level = _run("--init", "-i", position_path, "--levels", "1")
    assert one_level.exit_code != 0
    assert "levels" in str(one_level.exception)
