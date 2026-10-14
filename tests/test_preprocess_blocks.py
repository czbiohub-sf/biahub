"""Preprocessing blocks: named steps, each on a list of target volumes, before a method."""

import numpy as np
import pytest
import yaml

from pydantic import TypeAdapter, ValidationError

from biahub.registration.preprocess import apply_image_blocks, build_mask
from biahub.settings import PreprocessBlocks

BLOCKS = TypeAdapter(PreprocessBlocks)


def _parse(text):
    return BLOCKS.validate_python(yaml.safe_load(text))


# --- the config form ----------------------------------------------------------------------


def test_blocks_are_read_and_written_as_name_and_parameters():
    text = """
- clip_percentiles: {target: [moving], low: 2.0, high: 99.0}
- sobel: {}
- mask_circle: {target: [reference], radius: 0.9}
- mask_box: {target: [reference], box: [all, [10, 50], [10, 50]]}
"""
    blocks = _parse(text)
    assert [type(b).__name__ for b in blocks] == [
        "ClipPercentiles",
        "Sobel",
        "MaskCircle",
        "MaskBox",
    ]
    assert blocks[1].target == ["moving", "reference"]  # image blocks default to both
    # written back in the form it was read in, so a run's resolved settings read the same
    dumped = BLOCKS.dump_python(blocks)
    assert dumped[0] == {"clip_percentiles": {"target": ["moving"], "low": 2.0, "high": 99.0}}
    assert dumped[3] == {
        "mask_box": {"target": ["reference"], "box": ["all", [10, 50], [10, 50]]}
    }
    assert BLOCKS.validate_python(dumped) == blocks


@pytest.mark.parametrize(
    "text, message",
    [
        ("- sobel: {target: []}", "at least one"),
        ("- sobel: {target: [moving, moving]}", "more than once"),
        ("- sobel: {target: [phase]}", "moving"),
        ("- mask_circle: {radius: 0.9}", "target"),  # a mask names its grid
        ("- mask_box: {target: [moving, reference], box: [all, all, all]}", "one target"),
        ("- mask_box: {target: [reference], box: [all, [50, 10], all]}", "start"),
        ("- mask_circle: {target: [reference], radius: 1.5}", "radius"),
        ("- clip_percentiles: {low: 99, high: 1}", "low"),
        ("- blur: {}", "blur"),
        ("- {sobel: {}, clip_percentiles: {}}", "one block"),
    ],
)
def test_blocks_refuse_what_cannot_work(text, message):
    with pytest.raises(ValidationError, match=message):
        _parse(text)


# --- what the blocks do -------------------------------------------------------------------


def test_image_blocks_apply_in_order_to_their_targets_only():
    rng = np.random.default_rng(0)
    volume = rng.normal(100, 10, size=(4, 16, 16)).astype(np.float32)
    volume[0, 0, 0] = 1e6  # a hot pixel
    blocks = _parse("- clip_percentiles: {target: [moving], low: 1, high: 99}")

    moving = apply_image_blocks(volume, blocks, "moving")
    assert moving.max() < 1e3 and moving.max() == pytest.approx(np.percentile(volume, 99))
    np.testing.assert_array_equal(apply_image_blocks(volume, blocks, "reference"), volume)

    sobel_after_clip = apply_image_blocks(
        volume, _parse("- clip_percentiles: {}\n- sobel: {}"), "moving"
    )
    from skimage import filters

    clipped = np.clip(volume, *np.percentile(volume, [1.0, 99.5]))
    np.testing.assert_allclose(sobel_after_clip, filters.sobel(clipped), rtol=1e-5)


def test_masks_combine_and_are_none_without_a_mask_block():
    shape = (3, 40, 60)
    assert build_mask(shape, _parse("- sobel: {}"), "reference") is None

    circle = build_mask(
        shape, _parse("- mask_circle: {target: [reference], radius: 1.0}"), "reference"
    )
    assert circle.shape == shape and circle[:, 20, 30].all() and not circle[:, 0, 0].any()
    assert (circle[0] == circle[2]).all()  # the same circle on every z slice
    # radius is a fraction of the half-width of the shorter axis, centred on (y, x)
    assert circle[1, 20, 30 + 20].all() and not circle[1, 20, 30 + 21].any()

    both = build_mask(
        shape,
        _parse(
            "- mask_circle: {target: [reference], radius: 1.0}\n"
            "- mask_box: {target: [reference], box: [[1, 2], all, [30, 60]]}"
        ),
        "reference",
    )
    expected = circle.copy()
    expected[[0, 2]] = False
    expected[:, :, :30] = False
    np.testing.assert_array_equal(both, expected)
    # the moving volume is not masked by reference blocks
    assert (
        build_mask(
            shape, _parse("- mask_circle: {target: [reference], radius: 1.0}"), "moving"
        )
        is None
    )


def test_a_box_outside_its_volume_is_refused():
    blocks = _parse("- mask_box: {target: [reference], box: [all, [0, 100], all]}")
    with pytest.raises(ValueError, match="outside"):
        build_mask((3, 40, 60), blocks, "reference")


# --- applied before the method --------------------------------------------------------------


class _Recording:
    """A method that records what it was given (and takes a competitor, like beads)."""

    uses_seed = True
    last_winner = "hungarian"

    def __init__(self):
        self.seen = {}

    def estimate(self, mov, ref, seed=None, competitor=None, ref_mask=None, mov_mask=None):
        self.seen = dict(
            mov=mov,
            ref=ref,
            seed=seed,
            competitor=competitor,
            ref_mask=ref_mask,
            mov_mask=mov_mask,
        )
        from biahub.core.transform import Transform

        return Transform.identity(3)


def test_the_method_gets_prepared_volumes_and_masks_and_keeps_its_interface():
    import inspect

    from biahub.registration.preprocess import PreprocessedEstimator

    blocks = _parse(
        "- clip_percentiles: {target: [moving], low: 0, high: 50}\n"
        "- mask_box: {target: [reference], box: [all, [0, 8], all]}"
    )
    inner = _Recording()
    wrapped = PreprocessedEstimator(inner, blocks)
    mov = np.arange(4 * 16 * 16, dtype=np.float32).reshape(4, 16, 16)
    ref = mov.copy()

    wrapped.estimate(mov, ref, seed="s", competitor="c")
    assert inner.seen["mov"].max() == pytest.approx(np.percentile(mov, 50))
    np.testing.assert_array_equal(inner.seen["ref"], ref)  # no image block on the reference
    assert inner.seen["ref_mask"][:, :8].all() and not inner.seen["ref_mask"][:, 8:].any()
    assert inner.seen["mov_mask"] is None
    assert (inner.seen["seed"], inner.seen["competitor"]) == ("s", "c")
    # the engine reads the method's signature and attributes through the wrapper
    assert "competitor" in inspect.signature(wrapped.estimate).parameters
    assert (wrapped.uses_seed, wrapped.last_winner) == (True, "hungarian")


@pytest.mark.parametrize("method", ["beads", "phase-cross-corr", "focus-finding", "manual"])
def test_mask_blocks_are_refused_for_methods_that_cannot_use_them(method, tmp_path):
    from biahub.settings import EstimateTransformSettings

    with pytest.raises(ValidationError, match="mask blocks need"):
        EstimateTransformSettings(
            moving={"channel": "GFP"},
            reference={"frame": "cross", "channel": "Phase3D"},
            method=method,
            preprocess=[{"mask_circle": {"target": ["reference"], "radius": 0.9}}],
        )
