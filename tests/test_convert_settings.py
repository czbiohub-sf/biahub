import numpy as np
import pytest

from click.testing import CliRunner

from biahub.convert_settings import (
    EstimateRegistrationSettings,
    EstimateStabilizationSettings,
    RegistrationSettings,
    StabilizationSettings,
    convert_settings,
    convert_settings_cli,
    load_legacy_settings,
)
from biahub.settings import (
    EstimateTransformSettings,
    TransformSettings,
    load_estimate_transform_settings,
    load_transform_settings,
)

PULL = np.eye(4)
PULL[:3, 3] = [2.0, -3.0, 4.0]


def test_estimate_registration_converts_to_a_cross_reference_estimate():
    legacy = EstimateRegistrationSettings(
        source_channel_name="GFP",
        target_channel_name="Phase3D",
        estimation_method="beads",
        affine_transform_settings={
            "approx_transform": PULL.tolist(),
            "transform_type": "affine",
        },
        time_indices=[0, 5],
    )
    unified, notes = convert_settings(legacy)
    assert isinstance(unified, EstimateTransformSettings)
    assert unified.reference.frame == "cross" and unified.reference.channel == "Phase3D"
    assert unified.moving.channel == "GFP" and unified.method == "beads"
    assert (
        unified.transform.seed == PULL.tolist() and unified.transform.seed_direction == "pull"
    )
    assert unified.transform.type == "affine" and unified.time_indices == [0, 5]
    assert unified.beads is not None
    assert any("use_prev_t_transform" in n for n in notes)


def test_same_channel_estimate_registration_is_self_stabilization():
    legacy = EstimateRegistrationSettings(
        source_channel_name="GFP",
        target_channel_name="GFP",
        estimation_method="phase-cross-corr",
        affine_transform_settings={"t_reference": "previous", "use_prev_t_transform": False},
        eval_transform_settings={"validation_window_size": 5},
    )
    unified, notes = convert_settings(legacy)
    assert unified.reference.channel is None and unified.reference.frame == "previous"
    assert unified.method == "phase-cross-corr"
    assert [n for n in notes] == [n for n in notes if "eval_transform_settings" in n]


@pytest.mark.parametrize("stabilization_type", ["z", "xy", "xyz"])
def test_focus_finding_stabilization_converts_type_to_axes(stabilization_type):
    legacy = EstimateStabilizationSettings(
        stabilization_estimation_channel="Phase3D",
        stabilization_channels=["Phase3D"],
        stabilization_type=stabilization_type,
        stabilization_method="focus-finding",
        stack_reg_settings={"center_crop_xy": [600, 500], "t_reference": "previous"},
        focus_finding_settings={"center_crop_xy": [600, 500]},
        affine_transform_settings={"use_prev_t_transform": False},
    )
    unified, notes = convert_settings(legacy)
    assert unified.method == "focus-finding" and unified.reference.frame == "previous"
    assert unified.focus_finding.axes == stabilization_type
    assert unified.focus_finding.center_crop_xy == [600, 500]
    assert notes == []


def test_register_config_converts_to_a_single_pull_matrix():
    legacy = RegistrationSettings(
        source_channel_names=["GFP", "mCherry"],
        target_channel_name="Phase3D",
        affine_transform_zyx=PULL.tolist(),
        keep_overhang=True,
    )
    unified, notes = convert_settings(legacy)
    assert isinstance(unified, TransformSettings)
    assert unified.direction == "pull" and unified.series_wide
    assert unified.transforms[0].matrix == PULL.tolist()
    assert (
        unified.moving_channels == ["GFP", "mCherry"]
        and unified.reference_channel == "Phase3D"
    )
    assert any("--keep-overhang" in n for n in notes)
    np.testing.assert_allclose(unified.matrix_for(0, "forward")[:3, 3], [-2, 3, -4])


def test_stabilize_config_converts_to_a_self_grid_series():
    legacy = StabilizationSettings(
        stabilization_estimation_channel="Phase3D",
        stabilization_type="xyz",
        stabilization_method="phase-cross-corr",
        stabilization_channels=["GFP"],
        affine_transform_zyx_list=[PULL.tolist()] * 3,
        output_voxel_size=[1, 1, 2, 0.5, 0.5],
    )
    unified, _ = convert_settings(legacy)
    assert unified.direction == "pull" and unified.timepoints() == [0, 1, 2]
    assert unified.moving_channels == ["GFP", "Phase3D"] and unified.reference_channel is None
    assert unified.method == "phase-cross-corr" and unified.voxel_size == [1, 1, 2, 0.5, 0.5]


def test_convert_settings_cli_writes_a_config_the_strict_loaders_accept(tmp_path):
    (tmp_path / "estimate.yml").write_text(
        "source_channel_name: GFP\ntarget_channel_name: Phase3D\nestimation_method: ants\n"
    )
    (tmp_path / "register.yml").write_text(
        "source_channel_names: [GFP]\ntarget_channel_name: Phase3D\n"
        f"affine_transform_zyx: {PULL.tolist()}\n"
    )
    runner = CliRunner()
    for name in ("estimate", "register"):
        result = runner.invoke(
            convert_settings_cli,
            ["-c", str(tmp_path / f"{name}.yml"), "-o", str(tmp_path / f"{name}.unified.yml")],
        )
        assert result.exit_code == 0, result.output
    estimate = load_estimate_transform_settings(tmp_path / "estimate.unified.yml")
    assert estimate.method == "ants" and estimate.ants is not None
    transform = load_transform_settings(tmp_path / "register.unified.yml")
    assert transform.direction == "pull"

    (tmp_path / "junk.yml").write_text("nonsense: 1\n")
    with pytest.raises(ValueError, match="not a legacy"):
        load_legacy_settings(tmp_path / "junk.yml")
