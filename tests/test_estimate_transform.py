import json

import numpy as np
import pytest

from iohub import open_ome_zarr
from scipy.ndimage import shift as ndi_shift

from biahub.estimate_transform import estimate_transform
from biahub.settings import (
    AffineTransformSettings,
    AntsRegistrationSettings,
    BeadsMatchSettings,
    DetectPeaksSettings,
    EstimateRegistrationSettings,
    ManualRegistrationSettings,
    PhaseCrossCorrSettings,
    RegistrationSettings,
    StabilizationSettings,
)
from biahub.utils.config import model_to_yaml, yaml_to_model

APPLIED_SHIFT_ZYX = (2.0, -3.0, 4.0)
SHAPE = (40, 60, 60)


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


def _write_plate(path, frames):
    """frames: list of (ref, mov) per timepoint -> channels ["Phase3D", "GFP"]."""
    data = np.stack([np.stack(pair) for pair in frames]).astype(np.float32)
    with open_ome_zarr(
        path, layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]
    ) as plate:
        plate.create_position("A", "1", "0")["0"] = data
    return path / "A" / "1" / "0"


@pytest.fixture
def beads_plate(tmp_path):
    """Two timepoints; GFP is Phase3D shifted by APPLIED_SHIFT_ZYX."""
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    return _write_plate(tmp_path / "beads.zarr", [(ref, mov), (ref, mov)])


@pytest.fixture
def beads_plate_with_a_blank_timepoint(tmp_path):
    """Three timepoints; the last GFP frame has no beads, only noise."""
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    blank = rng.normal(0, 5.0, size=SHAPE).astype(np.float32)
    return _write_plate(tmp_path / "beads_blank.zarr", [(ref, mov), (ref, mov), (ref, blank)])


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
    path = tmp_path / "estimate.yml"
    model_to_yaml(EstimateRegistrationSettings(**fields), path)
    return path


def _run(plate, config, output, **kwargs):
    estimate_transform([plate], [plate], config, output, cluster="debug", **kwargs)


def test_estimate_transform_writes_a_register_compatible_series(beads_plate, tmp_path):
    output = tmp_path / "out" / "registration_settings.yml"

    _run(beads_plate, _write_config(tmp_path), output)

    model = yaml_to_model(output, StabilizationSettings)
    assert len(model.affine_transform_zyx_list) == 2
    for matrix in model.affine_transform_zyx_list:
        # Stored in the legacy pull direction: from the reference grid back to where the
        # content sits in the moving image, i.e. +APPLIED_SHIFT.
        np.testing.assert_allclose(np.asarray(matrix)[:3, 3], APPLIED_SHIFT_ZYX, atol=0.5)

    report = json.loads((output.parent / "estimate_transform_report.json").read_text())
    assert set(report["scores"]) == {"0", "1"}
    assert all(score > 0.5 for score in report["scores"].values())
    assert report["errors"] == {} and report["filled_from_neighbour"] == []
    assert (output.parent / "run_journal.json").exists()
    assert sorted(p.name for p in (output.parent / "timepoints").iterdir()) == [
        "0.json",
        "1.json",
    ]


def test_estimate_transform_single_timepoint_writes_registration_settings(
    beads_plate, tmp_path
):
    output = tmp_path / "out" / "registration_settings.yml"

    _run(beads_plate, _write_config(tmp_path, time_indices=1), output)

    model = yaml_to_model(output, RegistrationSettings)
    np.testing.assert_allclose(
        np.asarray(model.affine_transform_zyx)[:3, 3], APPLIED_SHIFT_ZYX, atol=0.5
    )


def test_estimate_transform_resume_keeps_existing_records(beads_plate, tmp_path):
    output = tmp_path / "out" / "registration_settings.yml"
    planted = np.eye(4)
    planted[:3, 3] = [7.0, 7.0, 7.0]
    timepoints = output.parent / "timepoints"
    timepoints.mkdir(parents=True)
    (timepoints / "0.json").write_text(
        json.dumps({"t": 0, "matrix": planted.tolist(), "score": 0.9, "error": None})
    )

    _run(beads_plate, _write_config(tmp_path), output, resume=True)

    model = yaml_to_model(output, StabilizationSettings)
    # t=0 came from the planted record (forward +7 -> pull -7), t=1 was estimated.
    np.testing.assert_allclose(
        np.asarray(model.affine_transform_zyx_list[0])[:3, 3], [-7.0, -7.0, -7.0]
    )
    np.testing.assert_allclose(
        np.asarray(model.affine_transform_zyx_list[1])[:3, 3], APPLIED_SHIFT_ZYX, atol=0.5
    )


def test_estimate_transform_flags_and_tries_to_repair_a_failed_timepoint(
    beads_plate_with_a_blank_timepoint, tmp_path
):
    output = tmp_path / "out" / "registration_settings.yml"

    _run(beads_plate_with_a_blank_timepoint, _write_config(tmp_path), output)

    report = json.loads((output.parent / "estimate_transform_report.json").read_text())
    assert "2" in report["errors"] and "EstimationError" in report["errors"]["2"]
    assert report["flagged"] == [2]
    repair = report["repairs"]["2"]
    assert repair["accepted"] is False
    # No beads in the frame: every candidate fails the same way, and each failure is named.
    assert set(repair["candidate_failures"]) == {"t-1", "consensus_full", "config_seed"}
    failures = repair["candidate_failures"]
    assert (
        "EstimationError" in failures["t-1"] and "EstimationError" in failures["config_seed"]
    )
    assert failures["consensus_full"].startswith(
        "ValueError: Consensus seed: only 2 timepoints"
    )
    assert report["filled_from_neighbour"] == [2]
    assert (output.parent / "repairs" / "2.json").exists()

    journal = json.loads((output.parent / "run_journal.json").read_text())
    (attempt,) = journal["attempts"]
    assert attempt["t"] == 2 and attempt["accepted"] is False and attempt["failures"]

    model = yaml_to_model(output, StabilizationSettings)
    assert len(model.affine_transform_zyx_list) == 3
    np.testing.assert_allclose(  # filled from t=1
        model.affine_transform_zyx_list[2], model.affine_transform_zyx_list[1]
    )


def test_estimate_transform_ants_method_recovers_the_shift(beads_plate, tmp_path):
    output = tmp_path / "out" / "registration_settings.yml"
    config = _write_config(
        tmp_path,
        estimation_method="ants",
        ants_registration_settings=AntsRegistrationSettings(),
        affine_transform_settings=AffineTransformSettings(transform_type="similarity"),
        time_indices=0,
    )

    _run(beads_plate, config, output)

    model = yaml_to_model(output, RegistrationSettings)
    np.testing.assert_allclose(
        np.asarray(model.affine_transform_zyx)[:3, 3], APPLIED_SHIFT_ZYX, atol=0.5
    )
    report = json.loads((output.parent / "estimate_transform_report.json").read_text())
    assert report["scores"]["0"] > 0.9  # correlation score


@pytest.fixture
def drifting_plate(tmp_path):
    """Three timepoints of one channel drifting by APPLIED_SHIFT_ZYX per timepoint."""
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    frames = [
        ndi_shift(
            ref,
            shift=tuple(t * s for s in APPLIED_SHIFT_ZYX),
            order=1,
            mode="constant",
            cval=0.0,
        )
        for t in range(3)
    ]
    return _write_plate(tmp_path / "drift.zarr", [(f, f) for f in frames])


@pytest.mark.parametrize(
    ("t_reference", "expected_pull_factor"),
    [("first", [0, 1, 2]), ("previous", [0, 1, 1])],
)
def test_estimate_transform_stabilizes_a_channel_against_itself(
    drifting_plate, tmp_path, t_reference, expected_pull_factor
):
    output = tmp_path / "out" / "stabilization_settings.yml"
    config = _write_config(
        tmp_path,
        source_channel_name="GFP",
        target_channel_name="GFP",
        affine_transform_settings=AffineTransformSettings(
            transform_type="euclidean", t_reference=t_reference
        ),
    )

    _run(drifting_plate, config, output)

    model = yaml_to_model(output, StabilizationSettings)
    assert model.stabilization_channels == ["GFP"]
    for matrix, factor in zip(
        model.affine_transform_zyx_list, expected_pull_factor, strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(matrix)[:3, 3], [factor * s for s in APPLIED_SHIFT_ZYX], atol=0.5
        )


def test_estimate_registration_beads_path_runs_through_the_engine(beads_plate, tmp_path):
    from biahub.estimate_registration import estimate_registration

    output = tmp_path / "out" / "registration_settings.yml"
    estimate_registration(
        source_position_dirpaths=[beads_plate],
        target_position_dirpaths=[beads_plate],
        output_filepath=output,
        config_filepath=_write_config(tmp_path),
        registration_target_channel=None,
        registration_source_channel=[],
        local=True,
    )

    model = yaml_to_model(output, StabilizationSettings)
    assert len(model.affine_transform_zyx_list) == 2
    np.testing.assert_allclose(
        np.asarray(model.affine_transform_zyx_list[1])[:3, 3], APPLIED_SHIFT_ZYX, atol=0.5
    )
    assert (output.parent / "estimate_transform_report.json").exists()


def test_estimate_transform_phase_cross_corr_stabilizes_against_the_first_frame(
    drifting_plate, tmp_path
):
    output = tmp_path / "out" / "stabilization_settings.yml"
    config = _write_config(
        tmp_path,
        source_channel_name="GFP",
        target_channel_name="GFP",
        estimation_method="phase-cross-corr",
        phase_cross_corr_settings=PhaseCrossCorrSettings(
            t_reference="first", center_crop_xy=[40, 40]
        ),
        affine_transform_settings=AffineTransformSettings(transform_type="euclidean"),
    )

    _run(drifting_plate, config, output)

    model = yaml_to_model(output, StabilizationSettings)
    assert model.stabilization_method == "phase-cross-corr"
    for matrix, factor in zip(model.affine_transform_zyx_list, [0, 1, 2], strict=True):
        np.testing.assert_allclose(
            np.asarray(matrix)[:3, 3], [factor * s for s in APPLIED_SHIFT_ZYX], atol=0.5
        )


def test_estimate_transform_manual_runs_in_process_on_one_timepoint(
    beads_plate, tmp_path, monkeypatch
):
    pull = np.eye(4)
    pull[:3, 3] = APPLIED_SHIFT_ZYX  # the annotation tool returns the pull matrix
    calls = []

    def fake_user_assisted_registration(**kwargs):
        calls.append(kwargs)
        return (pull,)

    monkeypatch.setattr(
        "biahub.registration.estimators.user_assisted_registration",
        fake_user_assisted_registration,
    )
    output = tmp_path / "out" / "registration_settings.yml"
    config = _write_config(
        tmp_path,
        estimation_method="manual",
        manual_registration_settings=ManualRegistrationSettings(
            time_index=1, affine_90degree_rotation=1
        ),
    )

    estimate_transform([beads_plate], [beads_plate], config, output, cluster="slurm")

    assert len(calls) == 1 and calls[0]["pre_affine_90degree_rotation"] == 1
    model = yaml_to_model(output, RegistrationSettings)
    np.testing.assert_allclose(
        np.asarray(model.affine_transform_zyx)[:3, 3], APPLIED_SHIFT_ZYX
    )
