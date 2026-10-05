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
    ReferenceSettings,
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
    seed_from="input",
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
        moving=ChannelSettings(channel=source),
        reference=ReferenceSettings(
            frame=reference, channel=target if reference == "cross" else None
        ),
        method=method,
        transform={"type": transform_type, "seed_from": seed_from},
        **blocks,
    )
    path = tmp_path / "estimate.yml"
    model_to_yaml(settings, path)
    return path


def _pull_translations(output):
    """Per-entry translation in the legacy pull direction (+APPLIED_SHIFT for a hit)."""
    model = load_transform_settings(output)
    return np.asarray([model._as(e.matrix, "pull") for e in model.transforms])[:, :3, 3]


def _run(plate, config, output, **kwargs):
    estimate_transform(
        [plate], config, output, reference_position_dirpaths=[plate], cluster="debug", **kwargs
    )


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
    assert report["errors"] == {} and report["stand_ins"] == {}
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
    # A single estimated matrix is the series' transform (an entry without t), so
    # apply-transform applies it to every timepoint, not just the one it came from.
    model = load_transform_settings(output)
    assert model.series_wide
    assert model.transforms[0].estimated_at == 1  # where the series' transform came from


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


def test_a_fresh_run_clears_an_earlier_runs_records(beads_plate, tmp_path):
    output = tmp_path / "out" / "registration_settings.yml"
    for name in ("timepoints", "repairs", "sweeps"):
        (output.parent / name).mkdir(parents=True)
        (output.parent / name / "99.json").write_text("{}")

    _run(beads_plate, _write_config(tmp_path), output)

    for name in ("timepoints", "repairs", "sweeps"):
        assert not (output.parent / name / "99.json").exists()
    assert (output.parent / "run_manifest.json").exists()


def test_a_failed_job_is_never_filled_from_a_record_on_disk(tmp_path):
    from biahub.registration.engine import _load_series

    stale = {"t": 1, "matrix": np.eye(4).tolist(), "score": 0.9, "error": None}
    (tmp_path / "1.json").write_text(json.dumps(stale))
    fresh = {0: {"t": 0, "matrix": np.eye(4).tolist(), "score": 0.8, "error": None}}

    failed = _load_series(tmp_path, [0, 1], "affine", records=fresh)
    assert 1 not in failed.transforms and "did not finish" in failed.errors[1]

    resumed = _load_series(tmp_path, [0, 1], "affine", records=fresh, resumed=[1])
    assert resumed.scores[1] == 0.9 and 1 in resumed.transforms


def test_resume_refuses_changed_settings(beads_plate, tmp_path):
    output = tmp_path / "out" / "registration_settings.yml"
    _run(beads_plate, _write_config(tmp_path), output)

    with pytest.raises(Exception, match="settings_sha256 changed"):
        _run(
            beads_plate,
            _write_config(tmp_path, transform_type="similarity"),
            output,
            resume=True,
        )


def test_an_sbatch_time_limit_is_kept_by_every_phase(
    beads_plate_with_a_blank_timepoint, tmp_path, monkeypatch
):
    import submitit

    from biahub.registration import engine

    times = []

    class RecordingExecutor(submitit.AutoExecutor):
        def update_parameters(self, **kwargs):
            if "slurm_time" in kwargs:
                times.append(kwargs["slurm_time"])
            return super().update_parameters(**kwargs)

    monkeypatch.setattr(engine.submitit, "AutoExecutor", RecordingExecutor)
    sbatch = tmp_path / "time.sbatch"
    sbatch.write_text("#!/bin/bash\n#SBATCH --time=7\n")
    output = tmp_path / "out" / "transforms.yml"

    _run(
        beads_plate_with_a_blank_timepoint,
        _write_config(tmp_path),
        output,
        sbatch_filepath=str(sbatch),
    )

    report = json.loads((output.parent / "estimate_transform_report.json").read_text())
    assert report["flagged"] == [2]  # the repair phase ran
    assert times and set(times) == {7}


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
    assert report["stand_ins"] == {"2": "seed"}
    assert (output.parent / "repairs" / "2.json").exists()

    journal = json.loads((output.parent / "run_journal.json").read_text())
    (attempt,) = journal["attempts"]
    assert attempt["t"] == 2 and attempt["accepted"] is False and attempt["failures"]

    entries = load_transform_settings(output).transforms
    assert [e.t for e in entries] == [0, 1, 2]
    np.testing.assert_allclose(entries[2].matrix, np.eye(4))  # the input seed (identity here)
    assert entries[2].score is None and entries[0].score > 0.5
    # ... and the file says so, with the failure, instead of passing as an estimate
    assert entries[2].status == "unreliable" and entries[2].filled_from == "seed"
    assert "EstimationError" in entries[2].note
    assert [e.status for e in entries[:2]] == ["accepted", "accepted"]


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
    assert model.moving_channels == ["GFP"] and model.reference_channel is None
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


def test_estimate_transform_several_positions_each_get_their_own_transforms(tmp_path):
    # Two FOVs drifting in opposite directions: one shared list cannot stabilize both.
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    drift = {"0": np.array(APPLIED_SHIFT_ZYX), "1": -np.array(APPLIED_SHIFT_ZYX)}
    path = tmp_path / "two.zarr"
    with open_ome_zarr(
        path, layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]
    ) as plate:
        for fov, d in drift.items():
            frames = [
                ndi_shift(ref, shift=tuple(t * d), order=1, mode="constant") for t in range(3)
            ]
            plate.create_position("A", "1", fov)["0"] = np.stack(
                [np.stack([f, f]) for f in frames]
            ).astype(np.float32)
    config = _write_config(
        tmp_path,
        source="GFP",
        reference="first",
        method="phase-cross-corr",
        # full extent: a centre crop lets beads drift out of the window and biases PCC
        phase_cross_corr=PhaseCrossCorrSettings(center_crop_xy=[SHAPE[1], SHAPE[2]]),
    )
    output = tmp_path / "out" / "stabilization_settings.yml"

    estimate_transform(
        [path / "A" / "1" / "0", path / "A" / "1" / "1"], config, output, cluster="debug"
    )

    model = load_transform_settings(output)
    assert model.per_position and sorted(model.positions) == ["A/1/0", "A/1/1"]
    for fov, d in drift.items():
        pulls = [model.matrix_for(t, "pull", f"A/1/{fov}")[:3, 3] for t in range(3)]
        for t, row in enumerate(pulls):
            np.testing.assert_allclose(row, t * d, atol=0.5)
    # each position kept its own records, so --resume works per position
    assert (output.parent / "positions" / "A" / "1" / "1" / "timepoints" / "2.json").exists()


def test_estimate_transform_previous_timepoint_runs_one_sequential_job(tmp_path):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    # the moving channel drifts two whole voxels further on each axis every timepoint
    step = np.array([2.0, -2.0, 2.0])
    frames = [
        (ref, ndi_shift(ref, shift=tuple((t + 1) * step), order=1, mode="constant"))
        for t in range(3)
    ]
    plate = _write_plate(tmp_path / "drift_reg.zarr", frames)
    peaks = DetectPeaksSettings(
        threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    # A 1-voxel scoring radius, so a pass's unrefined input does not tie with its refinement
    # (the default 6-voxel radius cannot tell a 2-voxel misalignment from none).
    beads = BeadsMatchSettings(
        source_peaks_settings=peaks,
        target_peaks_settings=peaks,
        qc_settings={"iterations": 2, "score_threshold": 0.4, "score_centroid_mask_radius": 1},
    )
    config = _write_config(tmp_path, seed_from="previous_timepoint", beads=beads)
    output = tmp_path / "out" / "transforms.yml"

    _run(plate, config, output)

    job_ids = (output.parent / "slurm_output" / "estimate_job_ids.log").read_text().split()
    assert len(job_ids) == 1  # one sequential job, not one per timepoint
    for t, row in enumerate(_pull_translations(output)):
        np.testing.assert_allclose(row, (t + 1) * step, atol=0.5)
    record = json.loads((output.parent / "timepoints" / "2.json").read_text())
    assert {"stand_in", "stand_in_from"} <= set(record)


def test_repair_is_skipped_for_a_method_that_ignores_seeds(tmp_path, capsys):
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    blank = rng.normal(0, 5.0, size=SHAPE).astype(np.float32)
    frames = [ref, ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant"), blank]
    plate = _write_plate(tmp_path / "pcc_blank.zarr", [(f, f) for f in frames])
    config = _write_config(
        tmp_path,
        source="GFP",
        reference="first",
        method="phase-cross-corr",
        phase_cross_corr=PhaseCrossCorrSettings(center_crop_xy=[SHAPE[1], SHAPE[2]]),
    )
    output = tmp_path / "out" / "transforms.yml"

    _run(plate, config, output)

    report = json.loads((output.parent / "estimate_transform_report.json").read_text())
    assert report["flagged"]  # the noise frame is still flagged ...
    assert report["repairs"] == {}  # ... but PCC is not re-run from other seeds
    assert "repair skipped" in capsys.readouterr().out


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

    estimate_transform(
        [beads_plate],
        config,
        output,
        reference_position_dirpaths=[beads_plate],
        cluster="slurm",
    )

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
        moving=ChannelSettings(channel="GFP"),
        reference=ReferenceSettings(frame="cross", channel="Phase3D"),
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
    assert written.moving_channels == ["GFP"] and written.reference_channel == "Phase3D"
    assert [e.t for e in written.transforms] == [0, 1]
    for entry in written.transforms:  # forward: content moves by -APPLIED_SHIFT
        np.testing.assert_allclose(
            np.asarray(entry.matrix)[:3, 3], [-s for s in APPLIED_SHIFT_ZYX], atol=0.5
        )
        assert entry.score is not None and entry.repaired_from is None
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
        moving=ChannelSettings(channel="GFP"),
        reference=ReferenceSettings(frame="cross", channel="Phase3D"),
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
        moving=ChannelSettings(channel="Phase3D"),
        reference=ReferenceSettings(frame="first"),
        method="focus-finding",
        focus_finding=FocusSettings(axes="xyz", center_crop_xy=[48, 48]),
    )
    config = tmp_path / "unified.yml"
    model_to_yaml(unified, config)
    output = tmp_path / "out" / "transforms.yml"

    _run(defocusing_plate, config, output)

    written = yaml_to_model(output, TransformSettings)
    assert written.method == "focus-finding" and written.reference_channel is None
    assert written.moving_channels == ["Phase3D"]
    for t, entry in enumerate(written.transforms):  # forward: undo the drift
        assert entry.t == t
        np.testing.assert_allclose(
            np.asarray(entry.matrix)[:3, 3], [-1 * t, -2 * t, 3 * t], atol=0.5
        )


def test_previous_reference_needs_contiguous_timepoints(tmp_path):
    # Rejected when the settings are read, before any job is submitted.
    with pytest.raises(Exception, match="contiguous"):
        _write_config(tmp_path, source="GFP", reference="previous", time_indices=[0, 2])


def test_estimate_transform_sweep_runs_on_flagged_timepoints_and_resumes(
    beads_plate_with_a_blank_timepoint, tmp_path, monkeypatch
):
    peaks = DetectPeaksSettings(
        threshold_abs=100, nms_distance=4, min_distance=0, block_size=[8, 8, 8]
    )
    unified = EstimateTransformSettings(
        moving=ChannelSettings(channel="GFP"),
        reference=ReferenceSettings(frame="cross", channel="Phase3D"),
        method="beads",
        beads=BeadsMatchSettings(source_peaks_settings=peaks, target_peaks_settings=peaks),
        fallback={
            "repair": None,
            "sweep": {
                "grid": [{"beads.hungarian_match_settings.cost_threshold": [0.05, 0.2]}]
            },
        },
    )
    config = tmp_path / "unified.yml"
    model_to_yaml(unified, config)
    output = tmp_path / "out" / "transforms.yml"

    _run(beads_plate_with_a_blank_timepoint, config, output)

    report = json.loads((output.parent / "estimate_transform_report.json").read_text())
    assert report["flagged"] == [2] and report["repairs"] == {}
    swept = report["sweeps"]["2"]
    # A blank frame: both trials fail, each by name, and nothing is kept.
    assert swept["accepted"] is False
    assert set(swept["candidate_failures"]) == {
        "beads.hungarian_match_settings.cost_threshold=0.05",
        "beads.hungarian_match_settings.cost_threshold=0.2",
    }
    assert report["provenance"] == {}
    assert (output.parent / "sweeps" / "2.json").exists()
    journal = json.loads((output.parent / "run_journal.json").read_text())
    assert [(a["t"], a["pass_name"]) for a in journal["attempts"]] == [(2, "sweep")]

    import biahub.registration.engine as engine

    def _must_not_run(*args, **kwargs):
        raise AssertionError("resume must reuse the sweep record")

    monkeypatch.setattr(engine, "_sweep_timepoint_job", _must_not_run)
    _run(beads_plate_with_a_blank_timepoint, config, output, resume=True)
    resumed = json.loads((output.parent / "estimate_transform_report.json").read_text())
    assert resumed["sweeps"] == report["sweeps"]
