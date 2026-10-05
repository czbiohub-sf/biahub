import numpy as np
import pytest
import yaml

from click.testing import CliRunner
from scipy.ndimage import shift as ndi_shift

from biahub.cli.main import cli
from biahub.settings import load_transform_settings
from tests.test_estimate_transform import (
    APPLIED_SHIFT_ZYX,
    SHAPE,
    _synthetic_bead_volume,
    _write_plate,
)

PEAKS = {"threshold_abs": 100, "nms_distance": 4, "min_distance": 0, "block_size": [8, 8, 8]}


@pytest.fixture(autouse=True)
def _in_process_jobs(monkeypatch):
    monkeypatch.setenv("CI", "true")  # get_submitit_cluster -> "debug": jobs run in-process


@pytest.fixture
def beads_plate(tmp_path):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    return _write_plate(tmp_path / "beads.zarr", [(ref, mov), (ref, mov)])


def _legacy_registration_config(path):
    path.write_text(
        yaml.safe_dump(
            {
                "source_channel_name": "GFP",
                "target_channel_name": "Phase3D",
                "estimation_method": "beads",
                "beads_match_settings": {
                    "source_peaks_settings": PEAKS,
                    "target_peaks_settings": PEAKS,
                },
                "affine_transform_settings": {
                    "transform_type": "euclidean",
                    "use_prev_t_transform": False,
                },
            }
        )
    )
    return path


def test_aliases_are_hidden_from_help():
    output = CliRunner().invoke(cli, ["--help"]).output
    hidden = ("estimate-registration", "estimate-stabilization", "register ", "stabilize ")
    for name in (*hidden, "convert-settings"):
        assert name not in output
    assert "estimate-transform" in output and "apply-transform" in output


def test_estimate_registration_alias_runs_estimate_transform_and_warns(beads_plate, tmp_path):
    config = _legacy_registration_config(tmp_path / "estimate-registration.yml")
    output = tmp_path / "out" / "registration_settings.yml"

    result = CliRunner().invoke(
        cli,
        [
            "estimate-registration",
            "-s",
            str(beads_plate),
            "-t",
            str(beads_plate),
            "-c",
            str(config),
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DeprecationWarning" in result.output and "estimate-transform" in result.output
    model = load_transform_settings(output)  # the new transforms file, at the old path
    inverse = [model._as(e.matrix, "inverse")[:3, 3] for e in model.transforms]
    for row in inverse:
        np.testing.assert_allclose(row, APPLIED_SHIFT_ZYX, atol=0.5)


def test_register_alias_applies_a_legacy_registration_config(beads_plate, tmp_path):
    config = tmp_path / "register.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "source_channel_names": ["GFP"],
                "target_channel_name": "Phase3D",
                "affine_transform_zyx": np.eye(4).tolist(),
                "keep_overhang": True,
            }
        )
    )
    output = tmp_path / "registered.zarr"

    result = CliRunner().invoke(
        cli,
        [
            "register",
            "-s",
            str(beads_plate),
            "-t",
            str(beads_plate),
            "-c",
            str(config),
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DeprecationWarning" in result.output and "apply-transform" in result.output
    assert (output / "A" / "1" / "0").exists()


def test_optimize_registration_points_to_the_replacement():
    result = CliRunner().invoke(cli, ["optimize-registration", "-c", "x.yml"])
    assert result.exit_code != 0 and "estimate-transform" in result.output


@pytest.fixture
def drifting_plate(tmp_path):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    frames = [
        ndi_shift(ref, shift=tuple(t * np.array(APPLIED_SHIFT_ZYX)), order=1, mode="constant")
        for t in range(3)
    ]
    return _write_plate(tmp_path / "drift.zarr", [(f, f) for f in frames])


def test_estimate_stabilization_alias_converts_a_legacy_pcc_config(drifting_plate, tmp_path):
    config = tmp_path / "estimate-stabilization.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "stabilization_estimation_channel": "GFP",
                "stabilization_channels": ["GFP"],
                "stabilization_type": "xyz",
                "stabilization_method": "phase-cross-corr",
                "phase_cross_corr_settings": {
                    "t_reference": "first",
                    "center_crop_xy": [SHAPE[1], SHAPE[2]],
                },
            }
        )
    )
    result = CliRunner().invoke(
        cli,
        [
            "estimate-stabilization",
            "-i",
            str(drifting_plate),
            "-c",
            str(config),
            "-o",
            str(tmp_path / "stab"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DeprecationWarning" in result.output
    model = load_transform_settings(tmp_path / "stab" / "transforms.yml")
    for t, entry in enumerate(model.transforms):
        np.testing.assert_allclose(
            model._as(entry.matrix, "inverse")[:3, 3],
            t * np.array(APPLIED_SHIFT_ZYX),
            atol=0.5,
        )


def test_stabilize_alias_applies_a_legacy_stabilization_config(drifting_plate, tmp_path):
    config = tmp_path / "stabilize.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "stabilization_estimation_channel": "GFP",
                "stabilization_type": "xyz",
                "stabilization_method": "phase-cross-corr",
                "stabilization_channels": ["GFP"],
                "affine_transform_zyx_list": [np.eye(4).tolist()] * 3,
                "output_voxel_size": [1, 1, 1, 1, 1],
            }
        )
    )
    output = tmp_path / "stabilized.zarr"
    result = CliRunner().invoke(
        cli, ["stabilize", "-i", str(drifting_plate), "-c", str(config), "-o", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert "DeprecationWarning" in result.output and (output / "A" / "1" / "0").exists()
