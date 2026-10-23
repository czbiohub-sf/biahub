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
