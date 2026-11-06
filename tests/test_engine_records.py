"""The per-timepoint and per-pass records: one type each, written by jobs and read back by
resume, finalize and the benchmark."""

import numpy as np

from biahub.core.transform import Transform
from biahub.registration.estimators import EstimatorBase


def test_a_timepoint_record_round_trips_and_ignores_keys_it_does_not_know():
    from biahub.registration.engine.series import TimepointRecord

    record = TimepointRecord(
        t=3, matrix=np.eye(4).tolist(), score=0.9, blocks=[{"block": "1:ants", "matrix": None}]
    )
    data = record.to_dict()
    assert TimepointRecord.from_dict(data) == record
    # older or newer records: missing keys take their defaults, unknown keys are ignored
    assert TimepointRecord.from_dict({"t": 1, "matrix": None, "unknown": 1}).score is None


def test_a_resumed_record_comes_back_with_the_runs_transform_type():
    # propagation resume folded records without the transform type (they came back affine)
    from biahub.registration.engine.series import SeriesResult, TimepointRecord

    record = TimepointRecord(
        t=0, matrix=Transform.from_translation([1.0, 2.0, 3.0]).to_list(), score=0.8
    )
    result = SeriesResult()
    carried = record.fold_into(result, transform_type="euclidean")
    assert carried is result.transforms[0]
    assert result.transforms[0].transform_type == "euclidean"
    assert result.scores[0] == 0.8
    stand_in = TimepointRecord(
        t=1, stand_in=np.eye(4).tolist(), stand_in_from="t=0", error="no beads"
    )
    carried = stand_in.fold_into(result, transform_type="euclidean")
    assert result.stand_ins[1][1] == "t=0" and np.isnan(result.scores[1])
    assert carried.transform_type == "euclidean" and result.errors[1] == "no beads"


def test_propagation_resume_keeps_the_transform_type():
    from biahub.registration.engine.policies import CrossChannel
    from biahub.registration.engine.series import TimepointRecord, estimate_propagated

    class _Echo(EstimatorBase):
        def estimate(self, mov, ref, initial_transform=None):
            return initial_transform

    done = {0: TimepointRecord(t=0, matrix=np.eye(4).tolist(), score=1.0).to_dict()}
    frames = np.ones((2, 2, 2, 2))
    result = estimate_propagated(
        frames,
        CrossChannel(frames),
        _Echo(),
        Transform.identity(3),
        lambda transform, mov, ref: 1.0,
        [0, 1],
        done=done,
        transform_type="similarity",
    )
    assert result.transforms[0].transform_type == "similarity"


def test_a_pass_record_round_trips_and_falls_back_when_nothing_was_accepted():
    from biahub.registration.engine.fallback import PassRecord

    kept = Transform.from_translation([1.0, 0.0, 0.0])
    record = PassRecord(
        t=2, matrix=None, score=None, accepted=False, source="unchanged",
        candidate_scores={"t-1": None}, candidate_failures={"initial": "EstimationError: x"},
    )  # fmt: skip
    assert PassRecord.from_dict(record.to_dict()) == record
    outcome = record.to_result("euclidean", fallback=kept)
    assert outcome.transform is kept and outcome.score == -np.inf
    assert np.isnan(outcome.scores["t-1"]) and outcome.failures == {
        "initial": "EstimationError: x"
    }


def test_the_run_folder_knows_its_layout(tmp_path):
    # one position: the work files sit in the run folder; several: under positions/<key>/
    from biahub.registration.engine.disk import RunFolder

    one = RunFolder(tmp_path / "one")
    one.start(["A/1/0"], {"settings_sha256": "x"})
    assert one.work_dir("A/1/0") == one.path and one.positions() == ["A/1/0"]
    many = RunFolder(tmp_path / "many")
    many.start(["A/1/0", "A/1/1"], {"settings_sha256": "x"})
    assert many.work_dir("A/1/1") == many.path / "positions" / "A/1/1"
    assert [key for key, _ in many.work_dirs()] == ["A/1/0", "A/1/1"]
    assert many.config() == {"settings_sha256": "x"}
    assert RunFolder(tmp_path / "none").work_dirs() == [(None, tmp_path / "none")]


def test_the_config_fingerprint_is_the_resume_manifests_too(tmp_path):
    # one fingerprint: the run's config check (run.json) and the resume manifest agree
    from biahub.registration.engine.disk import _run_fingerprint, config_fingerprint
    from biahub.settings import EstimateTransformSettings

    initial_transform = tmp_path / "plate.yml"
    initial_transform.write_text("transforms: []\n")
    settings = EstimateTransformSettings(
        moving={"channel": "GFP"},
        reference={"frame": "first"},
        method="phase_cross_corr",
        transform={"initial": {"from": "file", "file": str(initial_transform)}},
    )
    config = config_fingerprint(settings)
    assert set(config) == {"settings_sha256", "initial_file_sha256"}
    manifest = _run_fingerprint(settings, tmp_path / "mov", tmp_path / "ref")
    assert {k: manifest[k] for k in config} == config
    initial_transform.write_text("transforms: [changed]\n")
    assert config_fingerprint(settings)["initial_file_sha256"] != config["initial_file_sha256"]


def test_bead_landmarks_find_the_estimates_settings_in_a_multi_position_run(tmp_path):
    # evaluate-transform read run_dir/estimate_transform_settings.yml only; with several
    # positions the settings are under positions/<key>/, so --landmarks beads failed
    from biahub.evaluate_transform import _bead_detection
    from biahub.registration.engine.disk import ENGINE_SETTINGS_FILENAME, RunFolder
    from biahub.settings import EstimateTransformSettings
    from biahub.utils.config import model_to_yaml

    run = RunFolder(tmp_path / "transforms")
    run.start(["A/1/0", "A/1/1"], {"settings_sha256": "x"})
    peaks = {
        "threshold_abs": 321,
        "nms_distance": 4,
        "min_distance": 0,
        "block_size": [8, 8, 8],
    }
    settings = EstimateTransformSettings(
        moving={"channel": "GFP"},
        reference={"frame": "cross", "channel": "Phase3D"},
        method="bead_matching",
        bead_matching={"source_peaks_settings": peaks, "target_peaks_settings": peaks},
    )
    for _, work_dir in run.work_dirs():
        work_dir.mkdir(parents=True)
        model_to_yaml(settings, work_dir / ENGINE_SETTINGS_FILENAME)
    detection = _bead_detection(run.path, None)
    assert detection["moving"]["threshold_abs"] == 321


class _Traced(EstimatorBase):
    """Shifts its initial transform by `d` in x (scored by x); its trace names the call."""

    def __init__(self, d=1.0, fail=False):
        self.d, self.fail, self.calls = d, fail, 0

    def estimate(self, mov, ref, initial_transform=None):
        from biahub.registration.estimators import EstimationError

        self.calls += 1
        if self.fail:
            raise EstimationError("no beads")
        start = initial_transform if initial_transform is not None else Transform.identity(3)
        return Transform.from_translation([0.0, 0.0, self.d]) @ start

    def trace(self):
        return {"best_candidate": f"call{self.calls}", "blocks": [{"block": "1:x"}]}


def _x_score(transform, mov, ref):
    return float(transform.translation[2])


def test_a_series_keeps_each_timepoints_trace_and_drops_it_where_nothing_was_estimated():
    from biahub.registration.engine.policies import (
        CrossChannel,
        FileInitialTransform,
        FixedInitialTransform,
    )
    from biahub.registration.engine.series import estimate_series

    frames = np.ones((3, 2, 4, 4), np.float32)
    frames[1] = 0  # an empty frame: not estimated, no trace
    estimator = _Traced()
    result = estimate_series(
        frames,
        CrossChannel(frames),
        estimator,
        FixedInitialTransform(Transform.identity(3)),
        _x_score,
        [0, 1, 2],
    )
    assert result.best_candidate == {0: "call1", 2: "call2"}
    assert set(result.blocks) == {0, 2}

    # a failed estimate keeps its trace (a stack's blocks say which failed)
    failed = estimate_series(
        frames, CrossChannel(frames), _Traced(fail=True),
        FixedInitialTransform(Transform.identity(3)), _x_score, [0],
    )  # fmt: skip
    assert failed.best_candidate == {0: "call1"} and 0 in failed.errors

    # an initial transform kept as it is (from a file: the policy says so) was not the
    # estimator's: no trace
    kept = estimate_series(
        frames, CrossChannel(frames), _Traced(d=-1.0),
        FileInitialTransform({0: Transform.identity(3)}), _x_score, [0],
    )  # fmt: skip
    assert kept.initialized_from == {0: "file"} and kept.best_candidate == {}


def test_propagation_keeps_the_trace_of_the_start_that_won():
    # run from the previous result (x=1 -> 2) and from the run's initial transform (x=0
    # -> 1): the previous one wins, so t=1's trace is that first call's, not the later
    # one's
    from biahub.registration.engine.policies import CrossChannel
    from biahub.registration.engine.series import estimate_propagated

    frames = np.ones((2, 2, 4, 4), np.float32)
    result = estimate_propagated(
        frames, CrossChannel(frames), _Traced(), Transform.identity(3), _x_score, [0, 1],
    )  # fmt: skip
    assert result.best_candidate == {0: "call1", 1: "call2"}


def test_a_job_reads_nan_as_no_data_so_a_few_nan_voxels_do_not_break_the_estimate():
    # four NaN voxels (an upstream warp's border) made phase correlation return the
    # identity with a NaN score, silently
    from scipy.ndimage import shift as ndi_shift

    from biahub.registration.engine.cluster import _JobFrames
    from biahub.registration.engine.policies import CrossChannel, FixedInitialTransform
    from biahub.registration.engine.series import estimate_series
    from biahub.registration.methods.intensity.phase_cross_corr import PCCEstimator
    from biahub.registration.metrics import correlation_score

    rng = np.random.default_rng(0)
    ref = rng.random((1, 10, 64, 64)).astype(np.float32)
    mov = ndi_shift(ref, (0, 0, 3, -2), order=1).astype(np.float32)
    mov[0, :, :2, :2] = np.nan
    mov_frames, ref_frames = _JobFrames(mov), _JobFrames(ref)
    assert not np.isnan(mov_frames[0]).any()  # passed through
    mov_frames.keep_only([0])
    assert not np.isnan(mov_frames[0]).any()  # kept
    result = estimate_series(
        mov_frames, CrossChannel(ref_frames), PCCEstimator(),
        FixedInitialTransform(Transform.identity(3)), correlation_score, [0],
    )  # fmt: skip
    np.testing.assert_allclose(result.transforms[0].translation, [0, -3, 2], atol=0.5)
    assert np.isfinite(result.scores[0])
