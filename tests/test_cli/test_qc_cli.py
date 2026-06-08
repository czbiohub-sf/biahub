import yaml

from click.testing import CliRunner

from biahub.cli.main import cli


def test_generate_report_spec_basic(tmp_path):
    """generate-report-spec creates a valid YAML with tabs derived from zarr paths."""
    zarr1 = tmp_path / "0-flatfield" / "plate.zarr"
    zarr2 = tmp_path / "2-reconstruct" / "plate.zarr"
    zarr1.mkdir(parents=True)
    zarr2.mkdir(parents=True)
    out = tmp_path / "spec.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate-report-spec",
            "-o",
            str(out),
            str(zarr1),
            str(zarr2),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()

    spec = yaml.safe_load(out.read_text())
    assert spec["title"] == "QC Report"
    assert len(spec["tabs"]) == 2

    tab1 = spec["tabs"][0]
    assert tab1["label"] == "0-flatfield"
    assert tab1["zarr_path"] == str(zarr1)
    assert tab1["qc_dir"] == str(tmp_path / "0-flatfield" / "plate_qc")
    assert "config" not in tab1

    tab2 = spec["tabs"][1]
    assert tab2["label"] == "2-reconstruct"
    assert tab2["zarr_path"] == str(zarr2)


def test_generate_report_spec_with_config_dir(tmp_path):
    """--config-dir adds a config key to each tab."""
    zarr = tmp_path / "step" / "plate.zarr"
    zarr.mkdir(parents=True)
    config_dir = tmp_path / "qc_configs"
    config_dir.mkdir()
    out = tmp_path / "spec.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate-report-spec",
            "-o",
            str(out),
            "--config-dir",
            str(config_dir),
            str(zarr),
        ],
    )

    assert result.exit_code == 0, result.output
    spec = yaml.safe_load(out.read_text())
    assert spec["tabs"][0]["config"] == str(config_dir)


def test_generate_report_spec_custom_title(tmp_path):
    """--title overrides the default report title."""
    zarr = tmp_path / "data" / "plate.zarr"
    zarr.mkdir(parents=True)
    out = tmp_path / "spec.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate-report-spec",
            "-o",
            str(out),
            "--title",
            "My Custom Report",
            str(zarr),
        ],
    )

    assert result.exit_code == 0, result.output
    spec = yaml.safe_load(out.read_text())
    assert spec["title"] == "My Custom Report"


def test_generate_report_spec_ome_suffix(tmp_path):
    """Zarr names with .ome.zarr have the .ome stripped for qc_dir derivation."""
    zarr = tmp_path / "step" / "plate.ome.zarr"
    zarr.mkdir(parents=True)
    out = tmp_path / "spec.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate-report-spec",
            "-o",
            str(out),
            str(zarr),
        ],
    )

    assert result.exit_code == 0, result.output
    spec = yaml.safe_load(out.read_text())
    assert spec["tabs"][0]["qc_dir"] == str(tmp_path / "step" / "plate_qc")


def test_generate_report_spec_no_zarr_paths(tmp_path):
    """No zarr paths should fail."""
    out = tmp_path / "spec.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate-report-spec",
            "-o",
            str(out),
        ],
    )

    assert result.exit_code != 0


def test_generate_report_spec_creates_parent_dirs(tmp_path):
    """Output parent directories are created if they don't exist."""
    zarr = tmp_path / "data" / "plate.zarr"
    zarr.mkdir(parents=True)
    out = tmp_path / "nested" / "deep" / "spec.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate-report-spec",
            "-o",
            str(out),
            str(zarr),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
