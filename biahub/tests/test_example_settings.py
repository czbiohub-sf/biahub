from pathlib import Path

import numpy as np
import pytest
import yaml

from pydantic import ValidationError

from biahub.settings import (
    CharacterizeSettings,
    ConcatenateSettings,
    DeskewSettings,
    EstimateRegistrationSettings,
    EstimateStabilizationSettings,
    ProcessingImportFuncSettings,
    RegistrationSettings,
    SegmentationSettings,
    StabilizationSettings,
    StitchSettings,
    TrackingSettings,
)

settings_files_dir = (Path(__file__) / "../../../settings").resolve()

example_settings_params = [
    ("example_characterize_settings.yml", CharacterizeSettings),
    ("example_concatenate_multi_position.yml", ConcatenateSettings),
    ("example_concatenate_settings_organelle_dynamics.yml", ConcatenateSettings),
    ("example_concatenate_settings.yml", ConcatenateSettings),
    ("example_deskew_settings.yml", DeskewSettings),
    ("example_estimate_registration_settings_beads.yml", EstimateRegistrationSettings),
    ("example_estimate_registration_settings_manual.yml", EstimateRegistrationSettings),
    (
        "example_estimate_stabilization_settings_xy_focus-finding.yml",
        EstimateStabilizationSettings,
    ),
    ("example_estimate_stabilization_settings_xyz_beads.yml", EstimateStabilizationSettings),
    (
        "example_estimate_stabilization_settings_xyz_focus-finding.yml",
        EstimateStabilizationSettings,
    ),
    ("example_estimate_stabilization_settings_xyz_pcc.yml", EstimateStabilizationSettings),
    (
        "example_estimate_stabilization_settings_z_focus-finding.yml",
        EstimateStabilizationSettings,
    ),
    ("example_process_with_config_settings.yml", ProcessingImportFuncSettings),
    ("example_registration_settings.yml", RegistrationSettings),
    ("example_segmentation_settings.yml", SegmentationSettings),
    ("example_stabilize_timelapse_settings.yml", StabilizationSettings),
    ("example_stitch_settings.yml", StitchSettings),
    ("example_track_settings.yml", TrackingSettings),
]

try:
    import cellpose  # noqa: F401

    cellpose_available = True
except ImportError:
    cellpose_available = False


def test_all_example_settings_tested():
    num_settings_files = len(
        list(settings_files_dir.glob("*.yml")) + list(settings_files_dir.glob("*.yaml"))
    )
    assert num_settings_files == len(example_settings_params), (
        "Not all example settings files are tested. "
        f"Found {num_settings_files} files, but {len(example_settings_params)} are tested in test_example_settings."
    )


@pytest.mark.parametrize("path,settings_cls", example_settings_params)
def test_example_settings(path, settings_cls):
    # Skip test if cellpose isn't installed and we're testing SegmentationSettings
    if not cellpose_available and settings_cls == SegmentationSettings:
        pytest.skip("Cellpose not installed; skipping SegmentationSettings validation.")

    with open(settings_files_dir / path) as file:
        yaml_settings = yaml.safe_load(file)

    settings_cls(**yaml_settings)


def test_deskew_settings():
    # Test extra parameter
    with pytest.raises(ValidationError):
        DeskewSettings(
            pixel_size_um=0.116, ls_angle_deg=36, scan_step_um=0.313, typo_param="test"
        )

    # Test negative value
    with pytest.raises(ValidationError):
        DeskewSettings(pixel_size_um=-3, ls_angle_deg=36, scan_step_um=0.313)

    # Test light sheet angle range
    with pytest.raises(ValueError):
        DeskewSettings(pixel_size_um=0.116, ls_angle_deg=90, scan_step_um=0.313)

    # Test px_to_scan_ratio logic
    with pytest.raises(ValueError):
        DeskewSettings(pixel_size_um=0.116, ls_angle_deg=36, scan_step_um=None)


def test_register_settings():
    # Test extra parameter
    with pytest.raises(ValidationError):
        RegistrationSettings(
            source_channel_index=0,
            target_channel_index=0,
            affine_transform_zyx=np.identity(4).tolist(),
            typo_param="test",
        )

    # Test wrong output shape size
    with pytest.raises(ValidationError):
        RegistrationSettings(
            source_channel_index=0,
            target_channel_index=0,
            affine_transform_zyx=np.identity(4).tolist(),
            typo_param="test",
        )

    # Test wrong matrix shape
    with pytest.raises(ValidationError):
        RegistrationSettings(
            source_channel_index=0,
            target_channel_index=0,
            affine_transform_zyx=np.identity(5).tolist(),
            typo_param="test",
        )
