import numpy as np
import yaml

from click.testing import CliRunner
from iohub import open_ome_zarr

from biahub.cli.main import cli


def _write_config(tmp_path, settings: dict):
    config_path = tmp_path / "edit.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(settings, f)
    return config_path


def _run(args):
    return CliRunner().invoke(cli, args)


def test_edit_zarr_crop(tmp_path, example_plate):
    """Crop in T and ZYX; channels untouched."""
    plate_path, _ = example_plate
    config_path = _write_config(
        tmp_path,
        {"time_indices": [0, 1], "Z_slice": [1, 3], "Y_slice": [0, 4], "X_slice": [2, 6]},
    )
    output_path = tmp_path / "output.zarr"

    result = _run(
        [
            "edit-zarr",
            "-i",
            str(plate_path) + "/A/1/0",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "--cluster",
            "debug",
        ]
    )
    assert result.exit_code == 0, result.output

    with (
        open_ome_zarr(str(plate_path) + "/A/1/0", mode="r") as src,
        open_ome_zarr(str(output_path) + "/A/1/0", mode="r") as out,
    ):
        assert out.data.shape == (2, 6, 2, 4, 4)
        assert out.channel_names == src.channel_names
        np.testing.assert_array_equal(out.data[:], src.data[0:2, :, 1:3, 0:4, 2:6])


def test_edit_zarr_drop_and_rename(tmp_path, example_plate):
    """Keep a subset of channels and rename one of them."""
    plate_path, _ = example_plate
    config_path = _write_config(
        tmp_path,
        {"channels": [{"input": "GFP", "output": "membrane"}, {"input": "Phase3D"}]},
    )
    output_path = tmp_path / "output.zarr"

    result = _run(
        [
            "edit-zarr",
            "-i",
            str(plate_path) + "/A/1/0",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "--cluster",
            "debug",
        ]
    )
    assert result.exit_code == 0, result.output

    with (
        open_ome_zarr(str(plate_path) + "/A/1/0", mode="r") as src,
        open_ome_zarr(str(output_path) + "/A/1/0", mode="r") as out,
    ):
        assert out.channel_names == ["membrane", "Phase3D"]
        gfp = src.channel_names.index("GFP")
        phase = src.channel_names.index("Phase3D")
        np.testing.assert_array_equal(out.data[:, 0], src.data[:, gfp])
        np.testing.assert_array_equal(out.data[:, 1], src.data[:, phase])


def test_edit_zarr_init_only(tmp_path, example_plate):
    plate_path, _ = example_plate
    config_path = _write_config(tmp_path, {"channels": "all"})
    output_path = tmp_path / "output.zarr"

    result = _run(
        [
            "edit-zarr",
            "-i",
            str(plate_path) + "/A/1/0",
            str(plate_path) + "/B/1/0",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "--init",
        ]
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "RESOURCES:" in result.output


def test_edit_zarr_divide_by_channels(tmp_path, example_plate):
    plate_path, _ = example_plate
    config_path = _write_config(
        tmp_path,
        {
            "divide": {
                "by": "channels",
                "groups": [
                    {"name": "fluor", "channels": ["GFP", "RFP"]},
                    {"name": "phase", "channels": ["Phase3D"]},
                ],
            }
        },
    )
    output_path = tmp_path / "divided"

    result = _run(
        [
            "edit-zarr",
            "-i",
            str(plate_path) + "/A/1/0",
            str(plate_path) + "/B/1/0",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "--cluster",
            "debug",
        ]
    )
    assert result.exit_code == 0, result.output

    with open_ome_zarr(str(output_path / "fluor.zarr") + "/A/1/0", mode="r") as out:
        assert out.channel_names == ["GFP", "RFP"]
    with open_ome_zarr(str(output_path / "phase.zarr") + "/B/1/0", mode="r") as out:
        assert out.channel_names == ["Phase3D"]


def test_edit_zarr_divide_by_positions(tmp_path, example_plate):
    plate_path, _ = example_plate
    config_path = _write_config(
        tmp_path,
        {
            "divide": {
                "by": "positions",
                "groups": [
                    {"name": "groupA", "positions": ["A/1/0"]},
                    {"name": "groupB", "positions": ["B/1/0", "B/2/0"]},
                ],
            }
        },
    )
    output_path = tmp_path / "divided"

    result = _run(
        [
            "edit-zarr",
            "-i",
            str(plate_path) + "/A/1/0",
            str(plate_path) + "/B/1/0",
            str(plate_path) + "/B/2/0",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "--cluster",
            "debug",
        ]
    )
    assert result.exit_code == 0, result.output

    with open_ome_zarr(str(output_path / "groupA.zarr"), mode="r") as out:
        assert [name for name, _ in out.positions()] == ["A/1/0"]
    with open_ome_zarr(str(output_path / "groupB.zarr"), mode="r") as out:
        assert sorted(name for name, _ in out.positions()) == ["B/1/0", "B/2/0"]
