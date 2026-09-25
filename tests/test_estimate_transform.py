import json

import numpy as np
import pytest

from iohub import open_ome_zarr
from scipy.ndimage import shift as ndi_shift

from biahub.estimate_transform import estimate_transform
from biahub.settings import (
    AntsRegistrationSettings,
    BeadsMatchSettings,
    ChannelSettings,
    DetectPeaksSettings,
    EstimateTransformSettings,
    FocusSettings,
    ManualRegistrationSettings,
    PhaseCrossCorrSettings,
    TransformSettings,
    load_transform_settings,
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


def _write_config(
    tmp_path,
    *,
    source="GFP",
    target="Phase3D",
    reference="cross",
    method="beads",
    transform_type="euclidean",
    **blocks,
):
    peaks = DetectPeaksSettings(
        threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    if method == "beads":
        blocks.setdefault(
            "beads",
            BeadsMatchSettings(source_peaks_settings=peaks, target_peaks_settings=peaks),
        )
    settings = EstimateTransformSettings(
        source=ChannelSettings(channel=source),
        target=ChannelSettings(channel=target) if reference == "cross" else None,
        reference=reference,
        method=method,
        transform={"type": transform_type},
        **blocks,
    )
    path = tmp_path / "estimate.yml"
    model_to_yaml(settings, path)
    return path


def _pull_translations(output):
    """Per-timepoint translation in the legacy pull direction (+APPLIED_SHIFT for a hit)."""
    return np.asarray(load_transform_settings(output).as_direction("pull"))[:, :3, 3]


def _run(plate, config, output, **kwargs):
    estimate_transform([plate], [plate], config, output, cluster="debug", **kwargs)


def test_estimate_transform_writes_a_register_compatible_series(beads_plate, tmp_path):
    output = tmp_path / "out" / "registration_settings.yml"

    _run(beads_plate, _write_config(tmp_path), output)

    pull = _pull_translations(output)
    assert len(pull) == 2
    for row in pull:
        # In the pull direction: from the reference grid back to where the content sits
        # in the moving image, i.e. +APPLIED_SHIFT.
        np.testing.assert_allclose(row, APPLIED_SHIFT_ZYX, atol=0.5)

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

    (row,) = _pull_translations(output)
    np.testing.assert_allclose(row, APPLIED_SHIFT_ZYX, atol=0.5)
    # A single estimated matrix is the series' transform: apply-transform must write
    # every timepoint with it, not just the one it was estimated from.
    assert load_transform_settings(output).time_indices == "all"


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

    pull = _pull_translations(output)
    # t=0 came from the planted record (forward +7 -> pull -7), t=1 was estimated.
    np.testing.assert_allclose(pull[0], [-7.0, -7.0, -7.0])
    np.testing.assert_allclose(pull[1], APPLIED_SHIFT_ZYX, atol=0.5)


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

    matrices = load_transform_settings(output).matrices
    assert len(matrices) == 3
    np.testing.assert_allclose(matrices[2], matrices[1])  # filled from t=1


def test_estimate_transform_ants_method_recovers_the_shift(beads_plate, tmp_path):
    output = tmp_path / "out" / "registration_settings.yml"
    config = _write_config(
        tmp_path,
        method="ants",
        ants=AntsRegistrationSettings(),
        transform_type="similarity",
        time_indices=0,
    )

    _run(beads_plate, config, output)

    (row,) = _pull_translations(output)
    np.testing.assert_allclose(row, APPLIED_SHIFT_ZYX, atol=0.5)
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
    # Both references must yield the transform onto the first frame's grid: the
    # per-pair 'previous' estimates are chained.
    [("first", [0, 1, 2]), ("previous", [0, 1, 2])],
)
def test_estimate_transform_stabilizes_a_channel_against_itself(
    drifting_plate, tmp_path, t_reference, expected_pull_factor
):
    output = tmp_path / "out" / "stabilization_settings.yml"
    config = _write_config(tmp_path, source="GFP", reference=t_reference)

    _run(drifting_plate, config, output)

    model = load_transform_settings(output)
    assert model.source_channels == ["GFP"] and model.target_channel is None
    for row, factor in zip(_pull_translations(output), expected_pull_factor, strict=True):
        np.testing.assert_allclose(row, [factor * s for s in APPLIED_SHIFT_ZYX], atol=0.5)


def test_estimate_transform_phase_cross_corr_stabilizes_against_the_first_frame(
    drifting_plate, tmp_path
):
    output = tmp_path / "out" / "stabilization_settings.yml"
    config = _write_config(
        tmp_path,
        source="GFP",
        reference="first",
        method="phase-cross-corr",
        phase_cross_corr=PhaseCrossCorrSettings(t_reference="first", center_crop_xy=[40, 40]),
    )

    _run(drifting_plate, config, output)

    assert load_transform_settings(output).method == "phase-cross-corr"
    for row, factor in zip(_pull_translations(output), [0, 1, 2], strict=True):
        np.testing.assert_allclose(row, [factor * s for s in APPLIED_SHIFT_ZYX], atol=0.5)


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
        "biahub.registration.methods.manual.user_assisted_registration",
        fake_user_assisted_registration,
    )
    output = tmp_path / "out" / "registration_settings.yml"
    config = _write_config(
        tmp_path,
        method="manual",
        manual=ManualRegistrationSettings(time_index=1, affine_90degree_rotation=1),
    )

    estimate_transform([beads_plate], [beads_plate], config, output, cluster="slurm")

    assert len(calls) == 1 and calls[0]["pre_affine_90degree_rotation"] == 1
    (row,) = _pull_translations(output)
    np.testing.assert_allclose(row, APPLIED_SHIFT_ZYX)


def test_estimate_transform_accepts_the_unified_config_and_writes_forward_matrices(
    beads_plate, tmp_path
):
    peaks = DetectPeaksSettings(
        threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    unified = EstimateTransformSettings(
        source=ChannelSettings(channel="GFP"),
        target=ChannelSettings(channel="Phase3D"),
        method="beads",
        beads=BeadsMatchSettings(source_peaks_settings=peaks, target_peaks_settings=peaks),
        score_metric="residual",
    )
    config = tmp_path / "unified.yml"
    model_to_yaml(unified, config)
    output = tmp_path / "out" / "transforms.yml"

    _run(beads_plate, config, output)

    written = yaml_to_model(output, TransformSettings)
    assert written.direction == "forward" and written.method == "beads"
    assert written.source_channels == ["GFP"] and written.target_channel == "Phase3D"
    for matrix in written.matrices:  # forward: content moves by -APPLIED_SHIFT
        np.testing.assert_allclose(
            np.asarray(matrix)[:3, 3], [-s for s in APPLIED_SHIFT_ZYX], atol=0.5
        )
    engine_settings = yaml_to_model(
        output.parent / "estimate_transform_settings.yml", EstimateTransformSettings
    )
    assert engine_settings.score_metric == "residual"


def test_estimate_transform_fallback_settings_reach_flagging_and_repair(
    beads_plate_with_a_blank_timepoint, tmp_path
):
    peaks = DetectPeaksSettings(
        threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    unified = EstimateTransformSettings(
        source=ChannelSettings(channel="GFP"),
        target=ChannelSettings(channel="Phase3D"),
        method="beads",
        beads=BeadsMatchSettings(source_peaks_settings=peaks, target_peaks_settings=peaks),
        fallback={
            "flag": {"k_mad": 2.0, "floor": 0.8, "hard_fail": 0.4},
            "repair": {"candidates": ["seed"]},
        },
    )
    config = tmp_path / "unified.yml"
    model_to_yaml(unified, config)
    output = tmp_path / "out" / "transforms.yml"

    _run(beads_plate_with_a_blank_timepoint, config, output)

    report = json.loads((output.parent / "estimate_transform_report.json").read_text())
    assert report["flagged"] == [2]
    # Only the seed candidate was configured, so only it was tried.
    assert set(report["repairs"]["2"]["candidate_failures"]) | set(
        report["repairs"]["2"]["candidate_scores"]
    ) == {"config_seed"}


@pytest.fixture
def defocusing_plate(tmp_path):
    """One channel whose in-focus plane drifts by (1 slice, 2 rows, -3 columns) per timepoint."""
    from iohub.ngff.models import TransformationMeta
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(5)
    z, y, x = 12, 64, 64
    yy, xx = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")
    plane = np.zeros((y, x), dtype=np.float32)
    for cy, cx in rng.uniform(10, 54, size=(10, 2)):
        plane += 500 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0))
    frames = []
    for t in range(3):
        shifted = ndi_shift(plane, shift=(2 * t, -3 * t), order=1, mode="wrap")
        frames.append(
            np.stack(
                [
                    gaussian_filter(shifted, sigma=0.4 * abs(k - (4 + t)) + 1e-3)
                    for k in range(z)
                ]
            )
        )
    data = np.stack([np.stack([f, f]) for f in frames]).astype(np.float32)
    pixel = 6.5 / 40
    with open_ome_zarr(
        tmp_path / "defocus.zarr", layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]
    ) as plate:
        plate.create_position("A", "1", "0").create_image(
            "0",
            data,
            transform=[TransformationMeta(type="scale", scale=[1, 1, 1, pixel, pixel])],
        )
    return tmp_path / "defocus.zarr" / "A" / "1" / "0"


def test_estimate_transform_focus_finding_stabilizes_z_and_yx_against_the_first_frame(
    defocusing_plate, tmp_path
):
    unified = EstimateTransformSettings(
        source=ChannelSettings(channel="Phase3D"),
        reference="first",
        method="focus-finding",
        focus_finding=FocusSettings(axes="xyz", center_crop_xy=[48, 48]),
    )
    config = tmp_path / "unified.yml"
    model_to_yaml(unified, config)
    output = tmp_path / "out" / "transforms.yml"

    _run(defocusing_plate, config, output)

    written = yaml_to_model(output, TransformSettings)
    assert written.method == "focus-finding" and written.target_channel is None
    assert written.source_channels == ["Phase3D"]
    for t, matrix in enumerate(written.matrices):  # forward: undo the drift
        np.testing.assert_allclose(
            np.asarray(matrix)[:3, 3], [-1 * t, -2 * t, 3 * t], atol=0.5
        )


def test_previous_reference_needs_contiguous_timepoints(drifting_plate, tmp_path):
    config = _write_config(tmp_path, source="GFP", reference="previous", time_indices=[0, 2])
    with pytest.raises(Exception, match="contiguous"):
        _run(drifting_plate, config, tmp_path / "out" / "transforms.yml")
