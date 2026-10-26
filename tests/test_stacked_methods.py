"""Stacked methods: a registration as blocks, coarse to fine, each with its own settings."""

import pytest

from pydantic import ValidationError

from biahub.settings import EstimateTransformSettings
from biahub.utils.config import model_to_yaml, yaml_to_model


def _settings(method, **kwargs):
    return EstimateTransformSettings(
        moving={"channel": "GFP"},
        reference={"frame": "cross", "channel": "Phase3D"},
        method=method,
        **kwargs,
    )


PCC_THEN_ANTS = [
    {"phase_cross_corr": {"preprocess": [{"clip_percentiles": {}}], "maximum_shift": 1.5}},
    {"ants": {"preprocess": [{"sobel": {}}], "transform": {"type": "similarity"}}},
]


def test_a_stack_is_read_and_written_in_the_form_it_was_given(tmp_path):
    settings = _settings(PCC_THEN_ANTS)
    assert settings.is_stack and settings.method_names == ["phase_cross_corr", "ants"]
    assert settings.final_method == "ants"
    first, second = settings.method
    assert [b.kind for b in first.preprocess] == ["clip_percentiles"]
    assert (first.type, second.type) == (None, "similarity")
    path = tmp_path / "stack.yml"
    model_to_yaml(settings, path)
    assert yaml_to_model(path, EstimateTransformSettings) == settings
    assert "phase_cross_corr" in path.read_text()  # still `- <method>: {...}`


def test_each_block_becomes_an_ordinary_single_method_config():
    settings = _settings(
        PCC_THEN_ANTS, transform={"type": "euclidean", "seed_from": "previous_timepoint"}
    )
    pcc, ants = settings.stack_settings()
    assert (pcc.method, ants.method) == ("phase_cross_corr", "ants")
    assert pcc.phase_cross_corr.maximum_shift == 1.5
    assert [b.kind for b in pcc.preprocess] == ["clip_percentiles"]
    assert [b.kind for b in ants.preprocess] == ["sobel"]
    # a block's transform type, else the run's; the rest of the run is shared
    assert (pcc.transform.type, ants.transform.type) == ("euclidean", "similarity")
    assert pcc.transform.seed_from == ants.transform.seed_from == "previous_timepoint"
    assert not pcc.is_stack


def test_the_same_method_can_appear_twice_with_its_own_settings():
    loose = {"bead_matching": {"hungarian_match_settings": {"cost_threshold": 0.2}}}
    tight = {"bead_matching": {"hungarian_match_settings": {"cost_threshold": 0.05}}}
    coarse, fine = _settings([loose, tight]).stack_settings()
    assert coarse.bead_matching.hungarian_match_settings.cost_threshold == 0.2
    assert fine.bead_matching.hungarian_match_settings.cost_threshold == 0.05


def test_the_default_score_is_the_final_blocks():
    assert _settings(PCC_THEN_ANTS).effective_score_metric == "correlation"
    assert _settings([{"ants": {}}, {"bead_matching": {}}]).effective_score_metric == "overlap"


MASK = {"target": ["reference"], "radius": 0.9}


@pytest.mark.parametrize(
    "method, extra, message",
    [
        ([], {}, "at least one block"),
        ([{"elastix": {}}], {}, "elastix"),
        ([{"ants": {}, "bead_matching": {}}], {}, "one method"),
        ([{"bead_matching": {"no_such_setting": 1}}], {}, r"block 1 \(bead_matching\)"),
        # the existing checks hold per block
        (
            [{"phase_cross_corr": {"preprocess": [{"mask_circle": MASK}]}}],
            {},
            "mask blocks need",
        ),
        ([{"ants": {}}], {"ants": {}}, "belong in the blocks"),
        ([{"ants": {}}], {"preprocess": [{"sobel": {}}]}, "belong in the blocks"),
        (
            [{"ants": {}}],
            {"fallback": {"sweep": {"grid": [{"transform.type": ["affine"]}]}}},
            "sweep",
        ),
    ],
)
def test_a_stack_refuses_what_cannot_work(method, extra, message):
    with pytest.raises(ValidationError, match=message) as error:
        _settings(method, **extra)
    # only the form written is checked: no "a list is not a method name" error
    assert "method.literal[" not in str(error.value)


# --- running a stack ---------------------------------------------------------------------


@pytest.fixture
def beads_plate(tmp_path):
    """Two timepoints; GFP is Phase3D shifted (as in test_estimate_transform)."""
    import numpy as np

    from scipy.ndimage import shift as ndi_shift

    from tests.test_estimate_transform import (
        APPLIED_SHIFT_ZYX,
        SHAPE,
        _synthetic_bead_volume,
        _write_plate,
    )

    ref = _synthetic_bead_volume(np.random.default_rng(11), SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    return _write_plate(tmp_path / "beads.zarr", [(ref, mov), (ref, mov)])


PEAKS = {"threshold_abs": 100, "nms_distance": 4, "min_distance": 0, "block_size": [8, 8, 8]}
BEADS_BLOCK = {
    "bead_matching": {"source_peaks_settings": PEAKS, "target_peaks_settings": PEAKS}
}


def test_a_stack_runs_its_blocks_in_order_each_from_the_one_before(monkeypatch):
    import numpy as np

    from biahub.core.transform import Transform
    from biahub.registration.engine import build_estimator
    from biahub.registration.estimators import ChainedEstimator

    settings = _settings(PCC_THEN_ANTS)
    estimator, score_fn, seed = build_estimator(settings, (16, 64, 64))
    assert isinstance(estimator, ChainedEstimator) and len(estimator.stages) == 2
    pcc, ants = estimator.stages
    assert type(pcc.method).__name__ == "PCCEstimator"  # wrapped by its block's preprocess
    assert type(ants.method).__name__ == "AntsEstimator"

    first = Transform.from_translation([1.0, 2.0, 3.0])
    second = Transform.from_translation([1.0, 2.5, 3.0])
    seen = []
    monkeypatch.setattr(pcc, "estimate", lambda m, r, seed=None: seen.append(seed) or first)
    monkeypatch.setattr(ants, "estimate", lambda m, r, seed=None: seen.append(seed) or second)
    monkeypatch.setattr(estimator, "score_fn", lambda t, m, r: 0.9 if t is second else 0.5)
    vol = np.zeros((16, 64, 64), np.float32)
    assert estimator.estimate(vol, vol, seed=seed) is second
    assert seen[0] is seed and seen[1] is first


def test_the_stack_uses_the_seed_when_its_first_block_does():
    from biahub.registration.engine import build_estimator

    assert not build_estimator(_settings(PCC_THEN_ANTS), (16, 64, 64))[0].uses_seed
    assert build_estimator(_settings([{"ants": {}}, BEADS_BLOCK]), (16, 64, 64))[0].uses_seed


def test_a_stack_estimates_end_to_end_and_names_its_blocks(beads_plate, tmp_path):
    import numpy as np

    from tests.test_estimate_transform import APPLIED_SHIFT_ZYX, _inverse_translations, _run

    settings = _settings([{"phase_cross_corr": {}}, BEADS_BLOCK])
    assert settings.effective_score_metric == "overlap"  # the final block's, beads
    config = tmp_path / "stack.yml"
    model_to_yaml(settings, config)
    output = tmp_path / "out" / "transforms.yml"
    _run(beads_plate, config, output)

    for row in _inverse_translations(output):
        np.testing.assert_allclose(row, APPLIED_SHIFT_ZYX, atol=0.5)
    from biahub.settings import load_transform_settings

    assert load_transform_settings(output).method == "phase_cross_corr > bead_matching"


def test_a_stack_plans_for_its_heaviest_and_interactive_blocks(beads_plate, tmp_path):
    from biahub.registration import engine

    ants_alone = engine.init_run(
        beads_plate, beads_plate, _settings([{"ants": {}}]), tmp_path / "a"
    )
    stack = engine.init_run(
        beads_plate, beads_plate, _settings([BEADS_BLOCK, {"ants": {}}]), tmp_path / "b"
    )
    for phase in ("estimate", "repair"):  # the heaviest block's CPUs and memory
        for key in ("cpus", "mem_gb"):
            assert stack["resources"][phase][key] == ants_alone["resources"][phase][key]
    # time for each block, run one after another
    assert stack["resources"]["estimate"]["time_minutes"] == (
        2 * ants_alone["resources"]["estimate"]["time_minutes"]
    )
    manual = engine.init_run(
        beads_plate,
        beads_plate,
        _settings([{"manual": {}}, {"ants": {}}], time_indices=0),
        tmp_path / "c",
    )
    assert manual["interactive"]


# --- what each block did ------------------------------------------------------------------


class _Fixed:
    """A stage that returns a fixed transform, or fails."""

    def __init__(self, transform=None, error=None):
        self.transform, self.error = transform, error

    def estimate(self, mov, ref, seed=None):
        from biahub.registration.estimators import EstimationError

        if self.error:
            raise EstimationError(self.error)
        return self.transform


def test_a_stack_remembers_each_blocks_transform_score_and_the_winner():
    import numpy as np

    from biahub.core.transform import Transform
    from biahub.registration.estimators import ChainedEstimator

    coarse = Transform.from_translation([0.0, 2.0, 0.0])
    fine = Transform.from_translation([0.0, 2.5, 0.0])
    scores = {id(coarse): 0.6, id(fine): 0.9}
    chain = ChainedEstimator(
        [_Fixed(coarse), _Fixed(fine), _Fixed(error="too few matches")],
        score_fn=lambda t, m, r: scores[id(t)],
        names=["1:pcc", "2:ants", "3:bead_matching"],
    )
    vol = np.zeros((4, 8, 8), np.float32)
    assert chain.estimate(vol, vol) is fine
    first, second, third = chain.last_stages
    assert [s["block"] for s in chain.last_stages] == ["1:pcc", "2:ants", "3:bead_matching"]
    assert first["matrix"] == coarse.to_list() and first["score"] == 0.6
    assert (second["score"], second["won"], first["won"]) == (0.9, True, False)
    assert third["matrix"] is None and third["error"] == "too few matches"

    # a block that fails after a result keeps the result; later blocks did not run
    chain = ChainedEstimator(
        [_Fixed(coarse), _Fixed(error="no overlap"), _Fixed(fine)],
        score_fn=lambda t, m, r: scores[id(t)],
        names=["1:a", "2:b", "3:c"],
    )
    assert chain.estimate(vol, vol) is coarse
    assert chain.last_stages[2]["error"] == "not run: an earlier block failed"

    # the method's own internal chains (no names) record nothing
    internal = ChainedEstimator([_Fixed(coarse)], score_fn=lambda t, m, r: 0.5)
    internal.estimate(vol, vol)
    assert internal.last_stages is None


def test_every_timepoint_records_its_blocks_and_the_report_sums_them(beads_plate, tmp_path):
    import json

    from tests.test_estimate_transform import _run

    config = tmp_path / "stack.yml"
    model_to_yaml(_settings([{"phase_cross_corr": {}}, BEADS_BLOCK]), config)
    output = tmp_path / "out" / "transforms.yml"
    _run(beads_plate, config, output)

    run = output.with_suffix("")
    record = json.loads((run / "timepoints" / "0.json").read_text())
    blocks = record["blocks"]
    assert [b["block"] for b in blocks] == ["1:phase_cross_corr", "2:bead_matching"]
    assert all(len(b["matrix"]) == 4 for b in blocks)
    assert sum(b["won"] for b in blocks) == 1

    summary = json.loads((run / "estimate_transform_report.json").read_text())["blocks"]
    assert set(summary) == {"1:phase_cross_corr", "2:bead_matching"}
    assert sum(s["won"] for s in summary.values()) == 2  # two timepoints, one winner each
    assert all("median_score" in s and "failed" in s for s in summary.values())


class _Shift:
    """A block that moves its start by `d` voxels in x."""

    def __init__(self, d):
        self.d = d

    def estimate(self, mov, ref, seed=None):
        from biahub.core.transform import Transform

        start = seed if seed is not None else Transform.identity(3)
        return Transform.from_translation([0.0, 0.0, self.d]) @ start


def test_propagated_blocks_are_the_winning_runs_and_empty_frames_have_none():
    # With seed_from previous_timepoint the chain runs twice per timepoint (from the
    # previous result and from the input seed); the record kept the second run's blocks
    # even when the first won, and an empty frame kept the previous timepoint's.
    import numpy as np

    from biahub.core.transform import Transform
    from biahub.registration.engine import CrossChannel, estimate_propagated
    from biahub.registration.estimators import ChainedEstimator

    def score(t, mov, ref):  # best near x = 3
        return -abs(t.translation[2] - 3)

    chain = ChainedEstimator([_Shift(1.0), _Shift(1.0)], score_fn=score, names=["1:a", "2:b"])
    mov = np.random.default_rng(0).random((4, 4, 8, 8)).astype(np.float32) + 1
    mov[2] = 0  # an empty frame
    ref = mov.copy()
    seen = {}

    def on_timepoint(t, result):
        seen[t] = (result.transforms.get(t), chain.last_stages)

    estimate_propagated(
        mov,
        CrossChannel(ref),
        chain,
        Transform.identity(3),
        score,
        [0, 1, 2, 3],
        on_timepoint=on_timepoint,
    )
    for t in (1, 3):
        transform, stages = seen[t]
        won = [s for s in stages if s["won"]]
        assert len(won) == 1
        np.testing.assert_allclose(won[0]["matrix"], transform.matrix)  # the chosen run's
    assert seen[2][1] is None  # the empty frame was not estimated


def test_bead_metrics_belong_to_the_run_and_need_a_final_beads_block():
    # The report and the score are the run's, computed by the final block: a stack ending
    # in beads may report bead metrics; one ending elsewhere may not score with them.
    stack = _settings(
        [{"phase_cross_corr": {"local": {}}}, BEADS_BLOCK], score={"report": ["overlap"]}
    )
    assert stack.stack_settings()[0].method == "phase_cross_corr"
    with pytest.raises(ValidationError, match="final block"):
        _settings([BEADS_BLOCK, {"ants": {}}], score={"metric": "overlap"})


def test_a_focus_only_stack_keeps_the_across_gap_rule_and_a_mixed_one_does_not():
    from biahub.registration.estimators import ChainedEstimator

    class _Focus:
        holds_xy_across_gaps = True

    class _Other:
        pass

    assert ChainedEstimator([_Focus(), _Focus()]).holds_xy_across_gaps
    assert not ChainedEstimator([_Focus(), _Other()]).holds_xy_across_gaps


def test_a_stack_gets_time_for_each_block(beads_plate, tmp_path):
    from biahub.registration import engine

    one = engine.init_run(beads_plate, beads_plate, _settings([{"ants": {}}]), tmp_path / "a")
    two = engine.init_run(
        beads_plate, beads_plate, _settings([{"ants": {}}, {"ants": {}}]), tmp_path / "b"
    )
    minutes = lambda plan: plan["resources"]["estimate"]["time_minutes"]  # noqa: E731
    assert minutes(two) == 2 * minutes(one)


def test_every_method_has_a_family_and_says_what_it_estimates():
    from typing import get_args

    from biahub.settings import METHOD_FAMILIES, EstimationMethod

    assert set(METHOD_FAMILIES) == set(get_args(EstimationMethod))
    assert {family for family, _ in METHOD_FAMILIES.values()} == {
        "landmark-based",
        "intensity-based",
        "interactive",
    }


@pytest.mark.parametrize(
    "old, new",
    [
        ("beads", "bead_matching"),
        ("nuclei", "nuclei_matching"),
        ("phase-cross-corr", "phase_cross_corr"),
        ("focus-finding", "focus_finding"),
    ],
)
def test_an_old_method_name_says_its_new_one(old, new):
    with pytest.raises(ValidationError, match=f"is now '{new}'"):
        _settings(old)
    with pytest.raises(ValidationError, match=f"is now '{new}'"):
        _settings([{old: {}}])


def test_every_method_is_named_like_its_settings_section():
    # YAML values and keys follow one style: a method's name is its settings section
    from typing import get_args

    from biahub.settings import METHOD_BLOCKS, EstimationMethod

    assert {m: METHOD_BLOCKS[m] for m in get_args(EstimationMethod)} == {
        m: m for m in get_args(EstimationMethod) if m in METHOD_BLOCKS
    }
