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


def test_select_flagged_caps_to_the_worst_n_and_reports_what_it_dropped():
    scores = np.full(20, 0.90)
    scores[[1, 5, 10]] = [0.05, 0.20, 0.01]
    flagged, _ = select_flagged(scores, label="test pass", max_timepoints=2)
    # Worst two by score: t=10 (0.01), t=1 (0.05); t=5 (0.20) is dropped.
    assert sorted(flagged) == [1, 10]


def test_select_flagged_returns_empty_when_nothing_flagged():
    scores = np.full(10, 0.9)
    flagged, stats = select_flagged(scores, label="test pass")
    assert flagged == []
    assert stats["median"] == pytest.approx(0.9)
