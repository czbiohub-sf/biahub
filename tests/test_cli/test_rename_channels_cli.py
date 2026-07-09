from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli


def test_rename_channels_prefix(tmp_path, example_plate):
    plate_path, _ = example_plate
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "rename-channels",
            "-i",
            str(plate_path),
            "-p",
            "A/1/0",
            "--prefix",
            "ls_",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Renamed channels" in result.output

    with open_ome_zarr(str(plate_path / "A" / "1" / "0"), mode="r") as ds:
        for name in ds.channel_names:
            assert name.startswith("ls_")


def test_rename_channels_suffix(tmp_path, example_plate):
    plate_path, _ = example_plate
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "rename-channels",
            "-i",
            str(plate_path),
            "-p",
            "B/1/0",
            "--suffix",
            "_raw",
        ],
    )

    assert result.exit_code == 0, result.output

    with open_ome_zarr(str(plate_path / "B" / "1" / "0"), mode="r") as ds:
        for name in ds.channel_names:
            assert name.endswith("_raw")


def test_rename_channels_no_prefix_or_suffix(tmp_path, example_plate):
    plate_path, _ = example_plate
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "rename-channels",
            "-i",
            str(plate_path),
            "-p",
            "A/1/0",
        ],
    )

    assert result.exit_code != 0
    assert "Provide at least --prefix or --suffix" in result.output
