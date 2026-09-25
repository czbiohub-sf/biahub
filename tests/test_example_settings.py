import inspect
import re

from pathlib import Path

import pytest
import yaml

from pydantic import ValidationError

from biahub.settings import (
    CharacterizeSettings,
    ConcatenateSettings,
    DeskewSettings,
    EstimateTransformSettings,
    FlatFieldCorrectionSettings,
    ProcessingImportFuncSettings,
    SegmentationSettings,
    StitchSettings,
    TrackingSettings,
    TransformSettings,
)

settings_files_dir = (Path(__file__) / "../../settings").resolve()

example_settings_params = [
    ("example_characterize_settings.yml", CharacterizeSettings),
    ("example_concatenate_multi_position.yml", ConcatenateSettings),
    ("example_concatenate_settings_organelle_dynamics.yml", ConcatenateSettings),
    ("example_concatenate_settings.yml", ConcatenateSettings),
    ("example_deskew_settings.yml", DeskewSettings),
    ("example_estimate_transform_settings.yml", EstimateTransformSettings),
    ("example_estimate_transform_settings_ants.yml", EstimateTransformSettings),
    ("example_estimate_transform_settings_manual.yml", EstimateTransformSettings),
    ("example_estimate_transform_settings_stabilization_pcc.yml", EstimateTransformSettings),
    (
        "example_estimate_transform_settings_stabilization_focus_finding.yml",
        EstimateTransformSettings,
    ),
    ("example_transform_settings.yml", TransformSettings),
    ("example_transform_settings_stabilization.yml", TransformSettings),
    ("example_process_with_config_settings.yml", ProcessingImportFuncSettings),
    ("example_segmentation_settings.yml", SegmentationSettings),
    ("example_stitch_settings.yml", StitchSettings),
    ("example_track_settings.yml", TrackingSettings),
    ("example_flat_field_settings.yml", FlatFieldCorrectionSettings),
]

# Example configs validated against VisCy's own classes via jsonargparse rather
# than a biahub pydantic Settings model (virtual staining keeps no model schema).
jsonargparse_settings_files = [
    "example_virtual_stain_settings.yml",
]

try:
    import cellpose  # noqa: F401

    cellpose_available = True
except ImportError:
    cellpose_available = False

try:
    import cytoland  # noqa: F401

    cytoland_available = True
except ImportError:
    cytoland_available = False


def test_all_example_settings_tested():
    num_settings_files = len(
        list(settings_files_dir.glob("*.yml")) + list(settings_files_dir.glob("*.yaml"))
    )
    num_tested = len(example_settings_params) + len(jsonargparse_settings_files)
    assert num_settings_files == num_tested, (
        "Not all example settings files are tested. "
        f"Found {num_settings_files} files, but {num_tested} are tested in test_example_settings."
    )


@pytest.mark.parametrize("path", jsonargparse_settings_files)
def test_example_jsonargparse_settings(path):
    # These configs are validated against VisCy's VSUNet/HCSDataModule classes,
    # which require the optional `stain` extra (cytoland).
    if not cytoland_available:
        pytest.skip("cytoland not installed; skipping VisCy-config validation.")

    from biahub.virtual_stain import load_predict_config

    # data_path is injected per position at runtime; any placeholder validates.
    load_predict_config(settings_files_dir / path, Path("/placeholder.zarr/A/1/0"))


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


def test_ants_settings_are_all_consumed_by_the_engine():
    """Every AntsRegistrationSettings field must be read by AntsEstimator.from_settings
    (which is what the config-driven ANTs path runs), so a config knob can't be silently
    ignored."""
    from biahub.registration.engine import build_estimator
    from biahub.registration.methods.ants import AntsEstimator
    from biahub.settings import AntsRegistrationSettings

    source = inspect.getsource(AntsEstimator.from_settings) + inspect.getsource(
        build_estimator
    )
    read = set(re.findall(r"ants_(?:registration_)?settings\.(\w+)", source))
    missing = set(AntsRegistrationSettings.model_fields) - read
    assert not missing, f"AntsRegistrationSettings fields nothing reads: {missing}"


def test_ants_settings_defaults_match_preprocess_zyx():
    """Preprocessing defaults must agree with ``preprocess_zyx``, which consumes them."""
    from biahub.registration.methods.ants import preprocess_zyx
    from biahub.settings import AntsRegistrationSettings

    settings = AntsRegistrationSettings()
    params = inspect.signature(preprocess_zyx).parameters
    for field in AntsRegistrationSettings.model_fields:
        assert field in params, f"{field} is not a preprocess_zyx parameter"
        assert getattr(settings, field) == params[field].default, (
            f"default mismatch for {field}: settings={getattr(settings, field)} "
            f"preprocess_zyx={params[field].default}"
        )


@pytest.mark.parametrize("radius", [0, -0.2, 1.5])
def test_ants_ref_mask_radius_rejects_out_of_range(radius):
    from biahub.settings import AntsRegistrationSettings

    with pytest.raises(ValidationError):
        AntsRegistrationSettings(ref_mask_radius=radius)


def test_example_stitch_settings(example_stitch_settings):
    _, settings = example_stitch_settings
    validated_settings = StitchSettings(**settings)

    # Check that leading z = 0 is added to total_translation
    for value in validated_settings.total_translation.values():
        assert len(value) == 3
        assert value[0] == 0.0
