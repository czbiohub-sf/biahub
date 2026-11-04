"""The per-timepoint and per-pass records: one type each, written by jobs and read back by
resume, finalize and the benchmark."""

import numpy as np

from biahub.core.transform import Transform


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

    class _Echo:
        def estimate(self, mov, ref, seed=None):
            return seed

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
