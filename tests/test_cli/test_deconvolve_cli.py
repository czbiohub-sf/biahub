import numpy as np
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from biahub.cli.main import cli


@pytest.fixture()
def psf_store(tmp_path):
    """Create a small PSF zarr store for testing."""
    psf_path = tmp_path / "psf.zarr"
    scale = (1.0, 1.0, 1.0, 1.0, 1.0)

    with open_ome_zarr(psf_path, layout="hcs", mode="w", channel_names=["PSF"]) as ds:
        pos = ds.create_position("0", "0", "0")
        psf_data = np.zeros((1, 1, 4, 5, 6), dtype=np.float32)
        psf_data[0, 0, 2, 2, 3] = 1.0  # delta function
        pos.create_image(
            "0",
            psf_data,
            chunks=(1, 1, 4, 5, 6),
            transform=[TransformationMeta(type="scale", scale=scale)],
        )

    return psf_path


@pytest.fixture()
def deconvolve_config(tmp_path):
    config_path = tmp_path / "deconvolve.yml"
    config = {
        "regularization_strength": 0.001,
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_deconvolve_init_only(tmp_path, example_plate, psf_store, deconvolve_config):
    plate_path, _ = example_plate
    output_path = tmp_path / "deconv.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "deconvolve",
            "--init",
            "-i",
            str(plate_path) + "/A/1/0",
            str(plate_path) + "/B/1/0",
            "-p",
            str(psf_store),
            "-c",
            str(deconvolve_config),
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "RESOURCES:" in result.output
    assert "Initialized" in result.output
    assert output_path.exists()

    # Transfer function should have been computed
    tf_path = output_path.parent / "transfer_function.zarr"
    assert tf_path.exists()

    with open_ome_zarr(str(output_path / "A/1/0"), mode="r") as ds:
        assert ds.data.shape == (3, 6, 4, 5, 6)

    with open_ome_zarr(str(output_path / "B/1/0"), mode="r") as ds:
        assert ds.data.shape == (3, 6, 4, 5, 6)


def test_deconvolve_init_requires_psf(tmp_path, example_plate, deconvolve_config):
    plate_path, _ = example_plate
    output_path = tmp_path / "deconv.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "deconvolve",
            "--init",
            "-i",
            str(plate_path) + "/A/1/0",
            "-c",
            str(deconvolve_config),
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code != 0


def test_deconvolve_cluster_debug(tmp_path, example_plate, psf_store, deconvolve_config):
    plate_path, _ = example_plate
    output_path = tmp_path / "deconv.zarr"
    positions = ["A/1/0", "B/1/0"]

    runner = CliRunner()
    # Step 1: init with all positions
    result = runner.invoke(
        cli,
        [
            "deconvolve",
            "--init",
            "-i",
            *[str(plate_path / p) for p in positions],
            "-p",
            str(psf_store),
            "-c",
            str(deconvolve_config),
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output

    # Step 2: run single position with --cluster debug
    result = runner.invoke(
        cli,
        [
            "deconvolve",
            "--cluster",
            "debug",
            "-i",
            str(plate_path / "A/1/0"),
            "-c",
            str(deconvolve_config),
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Deconvolution complete" in result.output

    with open_ome_zarr(str(output_path / "A/1/0"), mode="r") as ds:
        assert ds.data.shape == (3, 6, 4, 5, 6)
        assert not np.all(ds.data[:] == 0)
