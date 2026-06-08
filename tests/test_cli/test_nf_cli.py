from click.testing import CliRunner

from biahub.cli.main import cli


def test_list_positions(example_plate):
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["nf", "list-positions", "-i", str(plate_path)])
    assert result.exit_code == 0, result.output

    lines = [line for line in result.output.strip().split("\n") if line]
    assert len(lines) == 3
    assert "A/1/0" in lines
    assert "B/1/0" in lines
    assert "B/2/0" in lines


def test_init_resources(example_plate):
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["nf", "init-resources", "-i", str(plate_path), "-r", "2.0"])
    assert result.exit_code == 0, result.output
    assert result.output.strip().startswith("RESOURCES:")

    parts = result.output.strip().replace("RESOURCES:", "").split()
    assert len(parts) == 2
    cpus = int(parts[0])
    mem = int(parts[1])
    assert cpus >= 1
    assert mem >= 1


def test_clean_temp(tmp_path):
    temp_dir = tmp_path / "temp_to_clean"
    temp_dir.mkdir()
    (temp_dir / "somefile.txt").write_text("data")

    runner = CliRunner()
    result = runner.invoke(cli, ["nf", "clean-temp", str(temp_dir)])
    assert result.exit_code == 0
    assert not temp_dir.exists()


def test_clean_temp_nonexistent(tmp_path):
    temp_dir = tmp_path / "does_not_exist"

    runner = CliRunner()
    result = runner.invoke(cli, ["nf", "clean-temp", str(temp_dir)])
    assert result.exit_code == 0


def test_clean_intermediates(tmp_path, example_plate):
    plate_path, plate_dataset = example_plate
    plate_dataset.close()

    output_dir = tmp_path / "output"
    dataset_name = "test_dataset"

    for dirname in ["0-flatfield", "1-deskew", "2-reconstruct", "3-virtual-stain"]:
        zarr_path = output_dir / dirname / f"{dataset_name}.zarr"
        zarr_path.mkdir(parents=True)
        (zarr_path / ".zattrs").write_text("{}")

    dirs = ["0-flatfield", "1-deskew", "2-reconstruct", "3-virtual-stain"]
    dir_flags = [flag for d in dirs for flag in ["-i", d]]

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["nf", "clean-intermediates", "-o", str(output_dir), "-d", dataset_name] + dir_flags,
    )
    assert result.exit_code == 0

    for dirname in dirs:
        assert not (output_dir / dirname / f"{dataset_name}.zarr").exists()


def test_clean_intermediates_missing(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["nf", "clean-intermediates", "-o", str(output_dir), "-d", "missing",
         "-i", "0-flatfield", "-i", "1-deskew"],
    )
    assert result.exit_code == 0
