"""The score: one metric for every decision, with its own blocks, independent of the method's."""

import numpy as np
import pytest

from pydantic import ValidationError
from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform
from biahub.settings import EstimateTransformSettings


def _settings(method="ants", **kwargs):
    return EstimateTransformSettings(
        moving={"channel": "GFP"},
        reference={"frame": "cross", "channel": "Phase3D"},
        method=method,
        **kwargs,
    )


def test_the_metric_defaults_to_the_methods_and_can_be_set():
    assert _settings("ants").effective_score_metric == "correlation"
    assert _settings("beads").effective_score_metric == "overlap"
    chosen = _settings("ants", score={"metric": "mutual_information"})
    assert chosen.effective_score_metric == "mutual_information"


def test_the_old_score_metric_key_points_to_score_metric():
    with pytest.raises(ValidationError, match="score: {metric: correlation}"):
        _settings("ants", score_metric="correlation")


@pytest.mark.parametrize(
    "method, score, message",
    [
        ("beads", {"preprocess": [{"sobel": {}}]}, "intensity"),  # beads' overlap: raw beads
        ("ants", {"metric": "overlap", "preprocess": [{"sobel": {}}]}, "intensity"),
        (
            "ants",
            {"preprocess": [{"mask_circle": {"target": ["reference"], "radius": 0.9}}]},
            "image blocks",
        ),
    ],
)
def test_score_blocks_are_image_blocks_for_intensity_metrics(method, score, message):
    with pytest.raises(ValidationError, match=message):
        _settings(method, score=score)


def _pair():
    from tests.test_registration_estimators import _synthetic_blob_volume

    rng = np.random.default_rng(3)
    ref = _synthetic_blob_volume(rng, (24, 40, 40))
    mov = ndi_shift(ref, shift=(1, -2, 3), order=1, mode="constant") * 0.5 + 7.0
    return mov.astype(np.float32), ref.astype(np.float32)


def test_score_blocks_run_after_the_warp_inside_the_raw_overlap():
    # correlation with a sobel block is exactly the Sobel correlation ANTs scored with:
    # warp, find the overlap on the raw volumes, filter, compare.
    from biahub.registration.engine import _score_fn
    from biahub.registration.metrics import correlation_score

    mov, ref = _pair()
    transform = Transform.from_translation([-1.0, 2.0, -3.0])
    with_sobel = _score_fn(_settings("ants", score={"preprocess": [{"sobel": {}}]}))
    plain = _score_fn(_settings("ants"))

    assert with_sobel(transform, mov, ref) == correlation_score(
        transform, mov, ref, sobel_filter=True
    )
    assert plain(transform, mov, ref) == correlation_score(transform, mov, ref)
    off = Transform.from_translation(
        [-1.0, 1.0, -2.0]
    )  # misaligned: edges and intensities differ
    assert plain(off, mov, ref) != with_sobel(off, mov, ref)


def test_the_methods_blocks_do_not_change_the_score():
    from biahub.registration.engine import _score_fn

    mov, ref = _pair()
    transform = Transform.from_translation([-1.0, 2.0, -3.0])
    plain = _score_fn(_settings("ants"))
    method_blocks = _score_fn(_settings("ants", preprocess=[{"sobel": {}}]))
    assert method_blocks(transform, mov, ref) == plain(transform, mov, ref)


def test_report_metrics_default_to_none_and_bead_metrics_need_beads():
    assert _settings("ants").score.report == []
    _settings("beads", score={"report": ["overlap", "mutual_information"]})
    with pytest.raises(ValidationError, match="bead"):
        _settings("ants", score={"report": ["overlap"]})


def test_score_blocks_do_not_see_the_edge_of_the_empty_padding():
    # Shifting the moving volume back in z leaves an empty slab in the warped volume. Its
    # edge is not content: with a sobel block the exact transform must still score like a
    # perfect match (DragonFly: the edge made every z-shifted transform score below identity).
    from biahub.registration.engine import _score_fn
    from tests.test_registration_estimators import _synthetic_blob_volume

    rng = np.random.default_rng(6)
    # dim structures on a high background, as fluorescence: the jump from background to the
    # empty padding is far larger than any edge of the content
    ref = 0.02 * _synthetic_blob_volume(rng, (24, 40, 40)) + 250.0
    mov = ndi_shift(ref, (3, 0, 0), order=0, mode="constant", cval=0.0)
    exact = Transform.from_translation([-3.0, 0.0, 0.0])
    plain = _score_fn(_settings("ants"))(exact, mov, ref)
    edges = _score_fn(_settings("ants", score={"preprocess": [{"sobel": {}}]}))(
        exact, mov, ref
    )
    assert plain > 0.99
    assert edges > 0.99
