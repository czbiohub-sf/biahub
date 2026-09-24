import json

import click
import numpy as np
import pytest

from iohub import open_ome_zarr
from scipy.ndimage import shift as ndi_shift

from biahub.estimate_transform import estimate_transform
from biahub.settings import (
    AffineTransformSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    EstimateRegistrationSettings,
    StabilizationSettings,
)
from biahub.utils.config import model_to_yaml, yaml_to_model

APPLIED_SHIFT_ZYX = (2.0, -3.0, 4.0)


def _synthetic_bead_volume(rng, shape, n_beads=15, sigma=2.0, amplitude=500.0, noise_std=5.0):
    margin = 8
    centers = rng.uniform([margin] * 3, np.asarray(shape) - margin, size=(n_beads, 3))
    grid = np.indices(shape, dtype=float)
    volume = np.zeros(shape, dtype=np.float32)
    for c in centers:
        d2 = sum((g - ci) ** 2 for g, ci in zip(grid, c, strict=True))
        volume += (amplitude * np.exp(-d2 / (2 * sigma**2))).astype(np.float32)
    volume += rng.normal(0, noise_std, size=shape).astype(np.float32)
    return volume


@pytest.fixture
def beads_plate(tmp_path):
    """Two timepoints; the GFP channel is Phase3D shifted by APPLIED_SHIFT_ZYX."""
    rng = np.random.default_rng(11)
    shape = (40, 60, 60)
    ref = _synthetic_bead_volume(rng, shape)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    data = np.stack([np.stack([ref, mov])] * 2).astype(np.float32)  # (T=2, C=2, Z, Y, X)

    plate_path = tmp_path / "beads.zarr"
    with open_ome_zarr(
        plate_path, layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]
    ) as plate:
        plate.create_position("A", "1", "0")["0"] = data
    return plate_path / "A" / "1" / "0"


def _write_config(tmp_path, **overrides):
    peaks = DetectPeaksSettings(
        threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    fields = {
        "target_channel_name": "Phase3D",
        "source_channel_name": "GFP",
        "estimation_method": "beads",
        "beads_match_settings": BeadsMatchSettings(
            source_peaks_settings=peaks, target_peaks_settings=peaks
        ),
        "affine_transform_settings": AffineTransformSettings(transform_type="euclidean"),
        **overrides,
    }
    settings = EstimateRegistrationSettings(**fields)
    path = tmp_path / "estimate.yml"
    model_to_yaml(settings, path)
    return path


def test_estimate_transform_writes_a_register_compatible_series(beads_plate, tmp_path):
    output = tmp_path / "out" / "registration_settings.yml"

    estimate_transform([beads_plate], [beads_plate], _write_config(tmp_path), output)

    model = yaml_to_model(output, StabilizationSettings)
    assert len(model.affine_transform_zyx_list) == 2
    for matrix in model.affine_transform_zyx_list:
        # The stored matrix is the legacy pull direction: it points from the reference
        # grid back to where the content sits in the moving image, i.e. +APPLIED_SHIFT.
        np.testing.assert_allclose(np.asarray(matrix)[:3, 3], APPLIED_SHIFT_ZYX, atol=0.5)

    report = json.loads((output.parent / "estimate_transform_report.json").read_text())
    assert set(report["scores"]) == {"0", "1"}
    assert all(score > 0.5 for score in report["scores"].values())
    assert report["errors"] == {} and report["filled_from_neighbour"] == []
    assert (output.parent / "run_journal.json").exists()


def test_estimate_transform_single_timepoint_writes_registration_settings(
    beads_plate, tmp_path
):
    from biahub.settings import RegistrationSettings

    output = tmp_path / "out" / "registration_settings.yml"
    estimate_transform(
        [beads_plate], [beads_plate], _write_config(tmp_path, time_indices=1), output
    )

    model = yaml_to_model(output, RegistrationSettings)
    np.testing.assert_allclose(
        np.asarray(model.affine_transform_zyx)[:3, 3], APPLIED_SHIFT_ZYX, atol=0.5
    )


def test_estimate_transform_rejects_non_beads_methods(beads_plate, tmp_path):
    config = _write_config(tmp_path, estimation_method="manual")
    with pytest.raises(click.UsageError, match="'beads'"):
        estimate_transform([beads_plate], [beads_plate], config, tmp_path / "out.yml")
