import numpy as np
import pytest

from click.testing import CliRunner

from biahub.registration.legacy.convert_settings import (
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

INVERSE = np.eye(4)
INVERSE[:3, 3] = [2.0, -3.0, 4.0]


def test_estimate_registration_converts_to_a_cross_reference_estimate():
    legacy = EstimateRegistrationSettings(
        source_channel_name="GFP",
        target_channel_name="Phase3D",
        estimation_method="beads",
        affine_transform_settings={
            "approx_transform": INVERSE.tolist(),
            "transform_type": "affine",
        },
        time_indices=[0, 5],
    )
    unified, notes = convert_settings(legacy)
    assert isinstance(unified, EstimateTransformSettings)
    assert unified.reference.frame == "cross" and unified.reference.channel == "Phase3D"
    assert unified.moving.channel == "GFP" and unified.method == "beads"
    assert (
        unified.transform.seed == INVERSE.tolist()
        and unified.transform.seed_direction == "inverse"
    )
    assert unified.transform.type == "affine" and unified.time_indices == [0, 5]
    assert unified.beads is not None
    # legacy use_prev_t_transform (default true) is propagation, no longer dropped
    assert unified.transform.seed_from == "previous_timepoint"
    assert not any("use_prev_t_transform" in n for n in notes)


def test_same_channel_estimate_registration_stays_a_cross_registration():
    legacy = EstimateRegistrationSettings(
        source_channel_name="GFP",
        target_channel_name="GFP",
        estimation_method="phase-cross-corr",
        affine_transform_settings={"t_reference": "previous", "use_prev_t_transform": False},
        eval_transform_settings={"validation_window_size": 5},
    )
    unified, notes = convert_settings(legacy)
    # Legacy estimate-registration read GFP from the source store and GFP from the target
    # store: a cross-store registration, not stabilization against itself.
    assert unified.reference.frame == "cross" and unified.reference.channel == "GFP"
    assert unified.method == "phase-cross-corr"
    assert any("kept as a cross-store registration" in n for n in notes)


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


def test_register_config_converts_to_a_single_inverse_matrix():
    legacy = RegistrationSettings(
        source_channel_names=["GFP", "mCherry"],
        target_channel_name="Phase3D",
        affine_transform_zyx=INVERSE.tolist(),
        keep_overhang=False,
    )
    unified, notes = convert_settings(legacy)
    assert isinstance(unified, TransformSettings)
    assert unified.direction == "inverse" and unified.series_wide
    assert unified.transforms[0].matrix == INVERSE.tolist()
    assert (
        unified.moving_channels == ["GFP", "mCherry"]
        and unified.reference_channel == "Phase3D"
    )
    assert any("--crop-to-overlap" in n for n in notes)
    np.testing.assert_allclose(unified.matrix_for(0, "forward")[:3, 3], [-2, 3, -4])


def test_stabilize_config_converts_to_a_self_grid_series():
    legacy = StabilizationSettings(
        stabilization_estimation_channel="Phase3D",
        stabilization_type="xyz",
        stabilization_method="phase-cross-corr",
        stabilization_channels=["GFP"],
        affine_transform_zyx_list=[INVERSE.tolist()] * 3,
        output_voxel_size=[1, 1, 2, 0.5, 0.5],
    )
    unified, _ = convert_settings(legacy)
    assert unified.direction == "inverse" and unified.timepoints() == [0, 1, 2]
    assert unified.moving_channels == ["GFP", "Phase3D"] and unified.reference_channel is None
    assert unified.method == "phase-cross-corr" and unified.voxel_size == [1, 1, 2, 0.5, 0.5]


def test_convert_settings_cli_writes_a_config_the_strict_loaders_accept(tmp_path):
    (tmp_path / "estimate.yml").write_text(
        "source_channel_name: GFP\ntarget_channel_name: Phase3D\nestimation_method: ants\n"
    )
    (tmp_path / "register.yml").write_text(
        "source_channel_names: [GFP]\ntarget_channel_name: Phase3D\n"
        f"affine_transform_zyx: {INVERSE.tolist()}\n"
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
    assert transform.direction == "inverse"

    (tmp_path / "junk.yml").write_text("nonsense: 1\n")
    with pytest.raises(ValueError, match="not a legacy"):
        load_legacy_settings(tmp_path / "junk.yml")


def test_convert_settings_folds_per_position_stabilize_configs(tmp_path):
    from biahub.utils.config import model_to_yaml

    folder = tmp_path / "xyz_stabilization_settings"
    folder.mkdir()
    for fov, dx in (("000", 1.0), ("001", 5.0)):
        inverse = INVERSE.copy()
        inverse[2, 3] = dx
        model_to_yaml(
            StabilizationSettings(
                stabilization_estimation_channel="GFP",
                stabilization_type="xyz",
                stabilization_method="phase-cross-corr",
                stabilization_channels=["GFP"],
                affine_transform_zyx_list=[inverse.tolist()] * 2,
                output_voxel_size=[1, 1, 2, 0.5, 0.5],
            ),
            folder / f"0_8_{fov}.yml",
        )
    output = tmp_path / "transforms.yml"

    result = CliRunner().invoke(
        convert_settings_cli, ["-c", str(folder / "*.yml"), "-o", str(output)]
    )

    assert result.exit_code == 0, result.output
    transforms = load_transform_settings(output)
    assert sorted(transforms.positions) == ["0/8/000", "0/8/001"]
    assert transforms.matrix_for(1, "inverse", "0/8/001")[2, 3] == 5.0
    assert transforms.matrix_for(1, "inverse", "0/8/000")[2, 3] == 1.0


def test_convert_settings_refuses_to_fold_files_that_are_not_per_position_stabilize(tmp_path):
    for name in ("0_8_000", "0_8_001"):
        (tmp_path / f"{name}.yml").write_text(
            "source_channel_name: GFP\ntarget_channel_name: Phase3D\nestimation_method: ants\n"
        )
    result = CliRunner().invoke(
        convert_settings_cli, ["-c", str(tmp_path / "*.yml"), "-o", str(tmp_path / "out.yml")]
    )
    assert result.exit_code != 0 and "per-position stabilize" in result.output


def test_estimate_registration_without_propagation_converts_to_independent_estimates():
    legacy = EstimateRegistrationSettings(
        source_channel_name="GFP",
        target_channel_name="Phase3D",
        estimation_method="beads",
        affine_transform_settings={"use_prev_t_transform": False},
    )
    unified, _ = convert_settings(legacy)
    assert unified.transform.seed_from == "input"


@pytest.mark.parametrize(
    "legacy, expected",
    [
        # beads honoured use_prev_t_transform (default true): propagation
        (
            EstimateRegistrationSettings(
                source_channel_name="GFP",
                target_channel_name="Phase3D",
                estimation_method="beads",
            ),
            "previous_timepoint",
        ),
        # ants ignored it: independent, whatever the flag says
        (
            EstimateRegistrationSettings(
                source_channel_name="GFP",
                target_channel_name="Phase3D",
                estimation_method="ants",
                affine_transform_settings={"use_prev_t_transform": True},
            ),
            "input",
        ),
        # stabilization by phase cross-correlation ignored it too
        (
            EstimateStabilizationSettings(
                stabilization_estimation_channel="GFP",
                stabilization_channels=["GFP"],
                stabilization_type="xyz",
                stabilization_method="phase-cross-corr",
                phase_cross_corr_settings={},
            ),
            "input",
        ),
    ],
)
def test_use_prev_t_transform_maps_to_propagation_only_for_beads(legacy, expected):
    unified, _ = convert_settings(legacy)
    assert unified.transform.seed_from == expected


def test_legacy_manual_time_index_becomes_time_indices():
    legacy = EstimateRegistrationSettings(
        source_channel_name="GFP",
        target_channel_name="Phase3D",
        estimation_method="manual",
        manual_registration_settings={"time_index": 82, "affine_fliplr": True},
    )
    unified, _ = convert_settings(legacy)
    assert unified.time_indices == 82
    assert unified.manual.affine_fliplr is True


def test_legacy_pcc_reference_and_beads_fov_convert_to_the_unified_fields():
    legacy = EstimateStabilizationSettings(
        stabilization_estimation_channel="GFP",
        stabilization_channels=["GFP"],
        stabilization_type="xyz",
        stabilization_method="phase-cross-corr",
        phase_cross_corr_settings={"t_reference": "previous", "skip_beads_fov": "0/2/000000"},
    )
    unified, notes = convert_settings(legacy)
    assert unified.reference.frame == "previous"
    assert unified.phase_cross_corr.t_reference == "first"
    assert unified.phase_cross_corr.skip_beads_fov == "0"
    assert any("skip_beads_fov" in note for note in notes)
