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


def test_the_engine_says_stacks_are_not_run_yet():
    import click

    from biahub.registration.engine import build_estimator

    with pytest.raises(click.UsageError, match="stack"):
        build_estimator(_settings(PCC_THEN_ANTS))
