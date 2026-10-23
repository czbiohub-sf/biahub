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
    {"phase-cross-corr": {"preprocess": [{"clip_percentiles": {}}], "maximum_shift": 1.5}},
    {"ants": {"preprocess": [{"sobel": {}}], "transform": {"type": "similarity"}}},
]


def test_a_stack_is_read_and_written_in_the_form_it_was_given(tmp_path):
    settings = _settings(PCC_THEN_ANTS)
    assert settings.is_stack and settings.method_names == ["phase-cross-corr", "ants"]
    assert settings.final_method == "ants"
    first, second = settings.method
    assert [b.kind for b in first.preprocess] == ["clip_percentiles"]
    assert (first.type, second.type) == (None, "similarity")
    path = tmp_path / "stack.yml"
    model_to_yaml(settings, path)
    assert yaml_to_model(path, EstimateTransformSettings) == settings
    assert "phase-cross-corr" in path.read_text()  # still `- <method>: {...}`


def test_each_block_becomes_an_ordinary_single_method_config():
    settings = _settings(
        PCC_THEN_ANTS, transform={"type": "euclidean", "seed_from": "previous_timepoint"}
    )
    pcc, ants = settings.stack_settings()
    assert (pcc.method, ants.method) == ("phase-cross-corr", "ants")
    assert pcc.phase_cross_corr.maximum_shift == 1.5
    assert [b.kind for b in pcc.preprocess] == ["clip_percentiles"]
    assert [b.kind for b in ants.preprocess] == ["sobel"]
    # a block's transform type, else the run's; the rest of the run is shared
    assert (pcc.transform.type, ants.transform.type) == ("euclidean", "similarity")
    assert pcc.transform.seed_from == ants.transform.seed_from == "previous_timepoint"
    assert not pcc.is_stack


def test_the_same_method_can_appear_twice_with_its_own_settings():
    loose = {"beads": {"hungarian_match_settings": {"cost_threshold": 0.2}}}
    tight = {"beads": {"hungarian_match_settings": {"cost_threshold": 0.05}}}
    coarse, fine = _settings([loose, tight]).stack_settings()
    assert coarse.beads.hungarian_match_settings.cost_threshold == 0.2
    assert fine.beads.hungarian_match_settings.cost_threshold == 0.05


def test_the_default_score_is_the_final_blocks():
    assert _settings(PCC_THEN_ANTS).effective_score_metric == "correlation"
    assert _settings([{"ants": {}}, {"beads": {}}]).effective_score_metric == "overlap"


MASK = {"target": ["reference"], "radius": 0.9}


@pytest.mark.parametrize(
    "method, extra, message",
    [
        ([], {}, "at least one block"),
        ([{"elastix": {}}], {}, "elastix"),
        ([{"ants": {}, "beads": {}}], {}, "one method"),
        ([{"beads": {"no_such_setting": 1}}], {}, r"block 1 \(beads\)"),
        # the existing checks hold per block
        (
            [{"phase-cross-corr": {"preprocess": [{"mask_circle": MASK}]}}],
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
BEADS_BLOCK = {"beads": {"source_peaks_settings": PEAKS, "target_peaks_settings": PEAKS}}


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

    settings = _settings([{"phase-cross-corr": {}}, BEADS_BLOCK])
    assert settings.effective_score_metric == "overlap"  # the final block's, beads
    config = tmp_path / "stack.yml"
    model_to_yaml(settings, config)
    output = tmp_path / "out" / "transforms.yml"
    _run(beads_plate, config, output)

    for row in _inverse_translations(output):
        np.testing.assert_allclose(row, APPLIED_SHIFT_ZYX, atol=0.5)
    from biahub.settings import load_transform_settings

    assert load_transform_settings(output).method == "phase-cross-corr > beads"


def test_a_stack_plans_for_its_heaviest_and_interactive_blocks(beads_plate, tmp_path):
    from biahub.registration import engine

    ants_alone = engine.init_run(
        beads_plate, beads_plate, _settings([{"ants": {}}]), tmp_path / "a"
    )
    stack = engine.init_run(
        beads_plate, beads_plate, _settings([BEADS_BLOCK, {"ants": {}}]), tmp_path / "b"
    )
    assert stack["resources"] == ants_alone["resources"]  # ANTs needs the most
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
        names=["1:pcc", "2:ants", "3:beads"],
    )
    vol = np.zeros((4, 8, 8), np.float32)
    assert chain.estimate(vol, vol) is fine
    first, second, third = chain.last_stages
    assert [s["block"] for s in chain.last_stages] == ["1:pcc", "2:ants", "3:beads"]
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
    model_to_yaml(_settings([{"phase-cross-corr": {}}, BEADS_BLOCK]), config)
    output = tmp_path / "out" / "transforms.yml"
    _run(beads_plate, config, output)

    run = output.with_suffix("")
    record = json.loads((run / "timepoints" / "0.json").read_text())
    blocks = record["blocks"]
    assert [b["block"] for b in blocks] == ["1:phase-cross-corr", "2:beads"]
    assert all(len(b["matrix"]) == 4 for b in blocks)
    assert sum(b["won"] for b in blocks) == 1

    summary = json.loads((run / "estimate_transform_report.json").read_text())["blocks"]
    assert set(summary) == {"1:phase-cross-corr", "2:beads"}
    assert sum(s["won"] for s in summary.values()) == 2  # two timepoints, one winner each
    assert all("median_score" in s and "failed" in s for s in summary.values())
