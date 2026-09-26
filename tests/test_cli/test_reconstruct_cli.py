import numpy as np
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import TransformationMeta, open_ome_zarr

from biahub.cli.main import cli


@pytest.fixture()
def reconstruct_config(tmp_path):
    cfg = {
        "input_channel_names": ["Phase3D"],
        "time_indices": "all",
        "reconstruction_dimension": 3,
        "phase": {
            "transfer_function": {
                "wavelength_illumination": 0.450,
                "yx_pixel_size": 0.1,
                "z_pixel_size": 0.25,
                "z_padding": 0,
                "index_of_refraction_media": 1.3,
                "numerical_aperture_detection": 1.2,
                "numerical_aperture_illumination": 0.5,
                "invert_phase_contrast": False,
            },
            "apply_inverse": {
                "reconstruction_algorithm": "Tikhonov",
                "regularization_strength": 1e-3,
            },
        },
    }
    config_path = tmp_path / "reconstruct.yml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_path


@pytest.fixture()
def reconstruct_plate(tmp_path):
    """Small plate with a Phase3D channel for reconstruction tests."""
    plate_path = tmp_path / "input.zarr"

    position_list = (("A", "1", "0"), ("B", "1", "0"))

    plate = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["Phase3D"],
    )

    for row, col, fov in position_list:
        position = plate.create_position(row, col, fov)
        # Match the pixel sizes declared in reconstruct_config so waveorder
        # doesn't emit a PixelSizeMismatchWarning during reconstruction.
        position.create_image(
            "0",
            np.random.uniform(1.0, 100.0, size=(1, 1, 5, 8, 8)).astype(np.float32),
            transform=[TransformationMeta(type="scale", scale=(1, 1, 0.25, 0.1, 0.1))],
        )

    plate.close()
    return plate_path


def test_apply_inv_tf_cli_init_only(tmp_path, reconstruct_plate, reconstruct_config):
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "apply-inv-tf",
            "--init",
            "-i",
            str(reconstruct_plate) + "/A/1/0",
            str(reconstruct_plate) + "/B/1/0",
            "-c",
            str(reconstruct_config),
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "RESOURCES:" in result.output


def test_apply_inv_tf_cli_debug_single_position(
    tmp_path, reconstruct_plate, reconstruct_config
):
    """Exercise biahub's --cluster debug submitit fan-out path.

    compute-tf (a thin waveorder pass-through) and the reconstruction math are
    covered upstream in waveorder; here we only assert that biahub's debug
    orchestration runs to completion against a real position.
    """
    output_path = tmp_path / "output.zarr"
    tf_path = tmp_path / "tf.zarr"

    runner = CliRunner()

    # Setup: init the output plate, then compute a TF to feed the debug run.
    for cmd in (
        [
            "apply-inv-tf",
            "--init",
            "-i",
            str(reconstruct_plate) + "/A/1/0",
            "-c",
            str(reconstruct_config),
            "-o",
            str(output_path),
        ],
        [
            "compute-tf",
            "-i",
            str(reconstruct_plate) + "/A/1/0",
            "-c",
            str(reconstruct_config),
            "-o",
            str(tf_path),
        ],
    ):
        result = runner.invoke(cli, cmd)
        assert result.exit_code == 0, result.output

    # Apply inv TF in debug mode (the biahub-specific in-process fan-out).
    result = runner.invoke(
        cli,
        [
            "apply-inv-tf",
            "--cluster",
            "debug",
            "-i",
            str(reconstruct_plate) + "/A/1/0",
            "-t",
            str(tf_path),
            "-c",
            str(reconstruct_config),
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Apply-inv-tf complete:" in result.output


@pytest.mark.parametrize(
    "flag, expected", [([], False), (["--resume"], True), (["--no-resume"], False)]
)
def test_apply_inv_tf_cli_passes_resume(
    tmp_path, reconstruct_plate, reconstruct_config, monkeypatch, flag, expected
):
    """--resume reaches waveorder's per-position call, which does the skipping."""
    calls = []
    monkeypatch.setattr(
        "biahub.apply_inverse_transfer_function.apply_inverse_transfer_function_single_position",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    result = CliRunner().invoke(
        cli,
        [
            "apply-inv-tf",
            "--cluster",
            "debug",
            "-i",
            str(reconstruct_plate) + "/A/1/0",
            "-t",
            str(tmp_path / "tf.zarr"),
            "-c",
            str(reconstruct_config),
            "-o",
            str(tmp_path / "output.zarr"),
            *flag,
        ],
    )

    assert result.exit_code == 0, result.output
    assert [call["resume"] for call in calls] == [expected]


def test_apply_inv_tf_cli_resume_skips_finished_timepoints(tmp_path, reconstruct_config):
    """End to end through biahub's init and --cluster debug path: a --resume rerun
    keeps the timepoints an earlier run finished instead of recomputing them."""
    plate_path = tmp_path / "input.zarr"
    with open_ome_zarr(plate_path, layout="hcs", mode="w", channel_names=["Phase3D"]) as plate:
        plate.create_position("A", "1", "0").create_image(
            "0",
            np.random.default_rng(0)
            .uniform(1.0, 100.0, size=(3, 1, 5, 8, 8))
            .astype(np.float32),
            transform=[TransformationMeta(type="scale", scale=(1, 1, 0.25, 0.1, 0.1))],
        )
    position = str(plate_path) + "/A/1/0"
    output_path = tmp_path / "output.zarr"
    tf_path = tmp_path / "tf.zarr"
    runner = CliRunner()

    def run(*args):
        result = runner.invoke(cli, list(args))
        assert result.exit_code == 0, result.output
        return result

    run(
        "apply-inv-tf",
        "--init",
        "-i",
        position,
        "-c",
        str(reconstruct_config),
        "-o",
        str(output_path),
    )
    run("compute-tf", "-i", position, "-c", str(reconstruct_config), "-o", str(tf_path))
    apply = (
        "apply-inv-tf", "--cluster", "debug", "-i", position, "-t", str(tf_path),
        "-c", str(reconstruct_config), "-o", str(output_path),
    )  # fmt: skip
    run(*apply)
    with open_ome_zarr(output_path / "A" / "1" / "0", mode="r+") as result:
        first = result["0"][:]
        result["0"][0] = 123.0  # stands in for "already finished; must not be recomputed"

    run(*apply, "--resume")
    with open_ome_zarr(output_path / "A" / "1" / "0") as result:
        np.testing.assert_array_equal(result["0"][0], 123.0)
        np.testing.assert_array_equal(result["0"][1:], first[1:])

    run(*apply, "--no-resume")
    with open_ome_zarr(output_path / "A" / "1" / "0") as result:
        np.testing.assert_array_equal(result["0"][:], first)


def test_apply_inv_tf_init_keeps_one_provenance_key_per_reconstruction(
    tmp_path, reconstruct_plate, reconstruct_config
):
    """Two configs initialized into one plate each keep their settings, under
    their own top-level key, and a downstream plate carries both forward."""
    from iohub.ngff.utils import create_empty_plate
    from waveorder.cli.settings import FluorescenceSettings, ReconstructionSettings

    from biahub.utils.config import model_to_yaml, yaml_to_model
    from biahub.utils.ngff import PROVENANCE_METADATA_KEYS

    phase_settings = yaml_to_model(reconstruct_config, ReconstructionSettings)
    fluor_settings = ReconstructionSettings(
        input_channel_names=["Phase3D"],
        reconstruction_dimension=3,
        fluorescence=FluorescenceSettings(),
    )
    fluor_config = tmp_path / "fluor.yml"
    model_to_yaml(fluor_settings, fluor_config)
    output_path = tmp_path / "output.zarr"
    position = str(reconstruct_plate) + "/A/1/0"

    for config in (reconstruct_config, fluor_config):
        result = CliRunner().invoke(
            cli,
            [
                "apply-inv-tf",
                "--init",
                "-i",
                position,
                "-c",
                str(config),
                "-o",
                str(output_path),
            ],
        )
        assert result.exit_code == 0, result.output

    with open_ome_zarr(output_path / "A" / "1" / "0") as reconstructed:
        assert reconstructed.channel_names == ["Phase3D", "Phase3D_Density3D"]
        assert reconstructed.zattrs["waveorder-Phase3D"] == phase_settings.model_dump()
        assert (
            reconstructed.zattrs["waveorder-Phase3D_Density3D"] == fluor_settings.model_dump()
        )
        assert "waveorder" not in reconstructed.zattrs

    downstream_path = tmp_path / "downstream.zarr"
    create_empty_plate(
        store_path=downstream_path,
        position_keys=[("A", "1", "0")],
        channel_names=["Phase3D"],
        shape=(1, 1, 5, 8, 8),
        metadata_sources=output_path,
        metadata_keys=PROVENANCE_METADATA_KEYS,
    )
    with open_ome_zarr(downstream_path / "A" / "1" / "0") as downstream:
        assert {"waveorder-Phase3D", "waveorder-Phase3D_Density3D"} <= set(downstream.zattrs)


def test_legacy_waveorder_provenance_still_carries_forward(tmp_path):
    """Stores written before the switch keep the nested ``waveorder`` key."""
    from iohub.ngff.utils import create_empty_plate

    from biahub.utils.ngff import PROVENANCE_METADATA_KEYS

    legacy_path = tmp_path / "legacy.zarr"
    legacy = {"Phase3D": {"input_channel_names": ["BF"]}}
    create_empty_plate(
        store_path=legacy_path,
        position_keys=[("A", "1", "0")],
        channel_names=["Phase3D"],
        shape=(1, 1, 2, 4, 4),
        extra_metadata={"waveorder": legacy},
    )
    downstream_path = tmp_path / "downstream.zarr"
    create_empty_plate(
        store_path=downstream_path,
        position_keys=[("A", "1", "0")],
        channel_names=["Phase3D"],
        shape=(1, 1, 2, 4, 4),
        metadata_sources=legacy_path,
        metadata_keys=PROVENANCE_METADATA_KEYS,
    )
    with open_ome_zarr(downstream_path / "A" / "1" / "0") as downstream:
        assert downstream.zattrs["waveorder"] == legacy
