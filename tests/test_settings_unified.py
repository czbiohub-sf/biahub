from pathlib import Path

import numpy as np
import pytest

from biahub.settings import (
    ChannelSettings,
    EstimateRegistrationSettings,
    EstimateStabilizationSettings,
    EstimateTransformSettings,
    RegistrationSettings,
    StabilizationSettings,
    TransformSettings,
    load_estimate_transform_settings,
    load_transform_settings,
)
from biahub.utils.config import model_to_yaml, yaml_to_model

SETTINGS_DIR = Path("settings")


def test_cross_reference_needs_a_target_and_self_reference_forbids_one():
    with pytest.raises(ValueError, match="needs a target"):
        EstimateTransformSettings(source=ChannelSettings(channel="GFP"), method="beads")
    with pytest.raises(ValueError, match="drop target"):
        EstimateTransformSettings(
            source=ChannelSettings(channel="GFP"),
            target=ChannelSettings(channel="GFP"),
            reference="first",
            method="beads",
        )
    stabilization = EstimateTransformSettings(
        source=ChannelSettings(channel="GFP"), reference="previous", method="phase-cross-corr"
    )
    assert stabilization.target_channel == "GFP"
    assert stabilization.phase_cross_corr is not None, "the chosen method's block is filled in"
    assert stabilization.beads is None
    assert stabilization.effective_score_metric == "correlation"


@pytest.mark.parametrize(
    "example",
    sorted(p.name for p in SETTINGS_DIR.glob("example_estimate_registration_settings*.yml")),
)
def test_every_legacy_estimate_registration_example_converts(example):
    legacy = yaml_to_model(SETTINGS_DIR / example, EstimateRegistrationSettings)
    unified = EstimateTransformSettings.from_legacy(legacy)
    assert unified.reference == "cross"
    assert unified.source.channel == legacy.source_channel_name
    assert unified.target.channel == legacy.target_channel_name
    assert unified.method == legacy.estimation_method
    assert unified.transform.seed == legacy.affine_transform_settings.approx_transform
    assert unified.transform.seed_direction == "pull"


@pytest.mark.parametrize(
    "example",
    sorted(p.name for p in SETTINGS_DIR.glob("example_estimate_stabilization_settings*.yml")),
)
def test_every_legacy_estimate_stabilization_example_converts_or_says_why_not(example):
    legacy = yaml_to_model(SETTINGS_DIR / example, EstimateStabilizationSettings)
    if legacy.stabilization_method == "focus-finding":
        with pytest.raises(ValueError, match="focus-finding"):
            EstimateTransformSettings.from_legacy(legacy)
        return
    unified = EstimateTransformSettings.from_legacy(legacy)
    assert unified.target is None and unified.reference in ("first", "previous")
    assert unified.source.channel == legacy.stabilization_estimation_channel
    assert unified.method == legacy.stabilization_method


def test_loader_accepts_unified_and_legacy_estimate_configs(tmp_path):
    unified = EstimateTransformSettings(
        source=ChannelSettings(channel="GFP"),
        target=ChannelSettings(channel="Phase3D"),
        method="beads",
        score_metric="residual",
    )
    model_to_yaml(unified, tmp_path / "unified.yml")
    assert load_estimate_transform_settings(tmp_path / "unified.yml") == unified

    legacy = SETTINGS_DIR / "example_estimate_registration_settings_beads.yml"
    converted = load_estimate_transform_settings(legacy)
    assert converted.method == "beads" and converted.reference == "cross"

    (tmp_path / "junk.yml").write_text("nonsense: 1\n")
    with pytest.raises(ValueError, match="matches none of"):
        load_estimate_transform_settings(tmp_path / "junk.yml")


def test_transform_settings_round_trip_to_legacy_inverts_the_direction():
    forward = np.eye(4)
    forward[:3, 3] = [-2.0, 3.0, -4.0]
    series = TransformSettings(
        direction="forward",
        matrices=[forward.tolist()] * 3,
        source_channels=["GFP"],
        target_channel="Phase3D",
        voxel_size=[1, 1, 1, 1, 1],
    )

    stabilization = series.to_stabilization_settings()
    assert isinstance(stabilization, StabilizationSettings)
    np.testing.assert_allclose(
        np.asarray(stabilization.affine_transform_zyx_list[0])[:3, 3], [2.0, -3.0, 4.0]
    )
    assert stabilization.stabilization_channels == ["GFP", "Phase3D"]

    single = TransformSettings(
        direction="forward",
        matrices=[forward.tolist()],
        source_channels=["GFP"],
        target_channel="Phase3D",
    )
    registration = single.to_registration_settings()
    assert isinstance(registration, RegistrationSettings)
    np.testing.assert_allclose(
        np.asarray(registration.affine_transform_zyx)[:3, 3], [2.0, -3.0, 4.0]
    )
    with pytest.raises(ValueError):
        series.to_registration_settings()  # three matrices are not one transform

    back = TransformSettings.from_legacy(stabilization)
    assert back.direction == "pull"
    np.testing.assert_allclose(back.as_direction("forward"), series.as_direction("forward"))


def test_loader_reads_legacy_register_and_stabilize_configs():
    registration = load_transform_settings(SETTINGS_DIR / "example_registration_settings.yml")
    assert registration.direction == "pull" and len(registration.matrices) == 1
    stabilization = load_transform_settings(
        SETTINGS_DIR / "example_stabilize_timelapse_settings.yml"
    )
    assert stabilization.direction == "pull" and len(stabilization.matrices) >= 1
