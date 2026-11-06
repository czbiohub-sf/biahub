import numpy as np
import pytest

from biahub.registration.engine import flag_timepoints, select_flagged


def test_flag_timepoints_flags_nothing_on_a_uniformly_good_run():
    scores = np.full(20, 0.95)
    flags = flag_timepoints(scores)
    assert not flags["flagged"].any()


def test_flag_timepoints_flags_outliers_below_the_adaptive_line():
    scores = np.full(20, 0.90)
    scores[5] = 0.10
    scores[12] = 0.05
    flags = flag_timepoints(scores)
    flagged = set(flags.loc[flags["flagged"], "t"])
    assert flagged == {5, 12}


def test_flag_timepoints_uses_the_absolute_floor_not_just_the_adaptive_line():
    # A run whose median is itself low (0.75) should not flag half the run just for
    # sitting below its own median -- the absolute floor (0.80) gates that.
    scores = np.full(20, 0.75)
    flags = flag_timepoints(scores)
    assert not flags["flagged"].any()


def test_flag_timepoints_flags_missing_scores():
    scores = np.array([0.9, 0.9, np.nan, 0.9])
    flags = flag_timepoints(scores)
    row = flags.loc[flags["t"] == 2].iloc[0]
    assert row["flagged"]
    assert "no_score" in row["reasons"]


def test_flag_timepoints_raises_when_no_finite_scores():
    with pytest.raises(ValueError):
        flag_timepoints(np.array([np.nan, np.nan]))


def test_select_flagged_returns_stats_used_for_the_decision():
    scores = np.full(20, 0.90)
    scores[3] = 0.05
    flagged, stats = select_flagged(scores, label="test pass")
    assert flagged == [3]
    assert stats["median"] == pytest.approx(0.90)


def test_a_pass_cap_keeps_the_worst_n_and_reports_what_it_dropped():
    from biahub.registration.engine.fallback import cap_worst

    scores = np.full(20, 0.90)
    scores[[1, 5, 10]] = [0.05, 0.20, 0.01]
    flagged, _ = select_flagged(scores, label="test pass")
    capped = cap_worst(flagged, dict(enumerate(scores)), max_timepoints=2)
    # Worst two by score: t=10 (0.01), t=1 (0.05); t=5 (0.20) is dropped.
    assert sorted(capped) == [1, 10]


def test_select_flagged_returns_empty_when_nothing_flagged():
    scores = np.full(10, 0.9)
    flagged, stats = select_flagged(scores, label="test pass")
    assert flagged == []
    assert stats["median"] == pytest.approx(0.9)


def _run_settings(metric, **fallback):
    from biahub.settings import EstimateTransformSettings

    return EstimateTransformSettings(
        moving={"channel": "GFP"},
        reference={"frame": "cross", "channel": "Phase3D"},
        method="phase_cross_corr" if metric != "overlap" else "bead_matching",
        score={"metric": metric},
        fallback=fallback,
    )


def test_the_absolute_thresholds_are_the_bead_overlaps_and_off_for_other_metrics():
    # floor 0.80 / hard_fail 0.40 / consensus 0.75 are bead-overlap values: on a
    # correlation (a correct cross-modal pair scores ~0.1) they flagged whole runs
    overlap = _run_settings("overlap").fallback_thresholds()
    assert (overlap.floor, overlap.hard_fail, overlap.consensus) == (0.80, 0.40, 0.75)
    correlation = _run_settings("correlation").fallback_thresholds()
    assert (correlation.floor, correlation.hard_fail, correlation.consensus) == (
        None,
        None,
        None,
    )
    given = _run_settings(
        "correlation",
        flag={"floor": 0.2, "hard_fail": 0.05},
        repair={"consensus_threshold": 0.3},
    ).fallback_thresholds()
    assert (given.floor, given.hard_fail, given.consensus) == (0.2, 0.05, 0.3)


def test_without_absolute_thresholds_only_the_runs_own_line_flags():
    scores = np.full(20, 0.10)
    scores[[4, 11]] = [0.02, 0.098]  # 0.02 far below the run, 0.098 within its spread
    scores += np.linspace(0, 0.004, 20)
    flags = flag_timepoints(scores, floor=None, hard_fail=None)
    assert list(flags.loc[flags["flagged"], "t"]) == [4]


def test_without_a_threshold_the_consensus_is_built_from_the_unflagged_timepoints():
    from biahub.core.transform import Transform
    from biahub.registration.engine.policies import ConsensusInitialTransform

    history = {t: Transform.from_translation([float(t), 0.0, 0.0]) for t in range(8)}
    scores = {t: 0.1 for t in history}
    consensus = ConsensusInitialTransform(
        history, scores, score_threshold=None, min_good=3, excluded={5, 6, 7}
    )
    # the median of t = 1, 2, 3, 4 (t = 0 is the one asked for; 5-7 are flagged)
    np.testing.assert_allclose(consensus.initial_for(0).translation, [2.5, 0.0, 0.0])


def test_without_hard_fail_a_flagged_timepoint_is_unreliable_while_below_the_runs_line():
    from biahub.core.transform import Transform
    from biahub.estimate_transform import transform_entries
    from biahub.registration.engine import SeriesResult

    result = SeriesResult(scores={0: 0.10, 1: 0.02, 2: 0.09}, flagged=[1, 2], flag_line=0.08)
    entries = transform_entries(result, [0, 1, 2], [Transform.identity(3)] * 3, hard_fail=None)
    # t=1 still below the line after the fallbacks; t=2 was lifted above it
    assert [e.status for e in entries] == ["accepted", "unreliable", "accepted"]


def test_a_correlation_run_scoring_about_0_1_is_not_unreliable_as_a_whole():
    # nf/ runs: pcc 96/125, focus 123/125, ANTs+Sobel 36/36 timepoints were unreliable
    # because the bead thresholds judged their correlation scores
    from biahub.core.transform import Transform
    from biahub.estimate_transform import transform_entries
    from biahub.registration.engine import SeriesResult
    from biahub.registration.engine.fallback import flag_series

    rng = np.random.default_rng(0)
    scores = 0.1 + rng.permutation(np.linspace(-0.003, 0.003, 40))  # a run's spread
    scores[17] = 0.01  # one timepoint that really failed
    result = SeriesResult(scores=dict(enumerate(scores)))
    thresholds = _run_settings("correlation").fallback_thresholds()
    flagged = flag_series(result, floor=thresholds.floor, hard_fail=thresholds.hard_fail)
    entries = transform_entries(
        result, list(range(40)), [Transform.identity(3)] * 40, thresholds.hard_fail
    )
    assert flagged == [17]
    assert [t for t, e in enumerate(entries) if e.status == "unreliable"] == [17]
