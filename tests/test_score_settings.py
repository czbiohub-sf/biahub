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


def test_a_clip_score_block_takes_its_percentiles_inside_the_overlap():
    # Percentiles over the whole warped volume counted its empty padding, so the exact
    # transform scored lower the more padding it left.
    from biahub.registration.engine import _score_fn
    from tests.test_registration_estimators import _synthetic_blob_volume

    rng = np.random.default_rng(7)
    ref = _synthetic_blob_volume(rng, (24, 40, 40)) + 100.0
    score = _score_fn(
        _settings(
            "ants", score={"preprocess": [{"clip_percentiles": {"low": 20, "high": 80}}]}
        )
    )
    for dz in (2, 8):
        mov = ndi_shift(ref, (dz, 0, 0), order=0, mode="constant", cval=0.0)
        assert score(Transform.from_translation([-dz, 0.0, 0.0]), mov, ref) > 0.999


def test_the_padding_margin_covers_diagonal_neighbours():
    # Sobel reaches the 26 neighbours of a voxel; a margin of face neighbours only left
    # voxels by a rotated padding edge in the score.
    from skimage import filters

    from biahub.registration.metrics import SOBEL_FOOTPRINT, _away_from_padding

    flat = np.full((16, 48, 48), 100.0, dtype=np.float32)
    angle = np.deg2rad(10)
    matrix = np.eye(4)
    matrix[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    matrix[1:3, 3] = [4.0, -3.0]
    warped = Transform(matrix).apply(flat, reference=flat, order=0)
    kept = _away_from_padding(warped != 0, SOBEL_FOOTPRINT)
    assert kept.any()
    assert np.abs(filters.sobel(warped)[kept]).max() == 0


def test_report_metrics_share_one_warp(monkeypatch):
    # Every report metric used to warp the volume again (~100 s each on a 2048 x 2048 pair).
    from biahub.registration import engine
    from biahub.registration.metrics import correlation_score, normalized_mutual_information

    mov, ref = _pair()
    transform = Transform.from_translation([-1.0, 2.0, -3.0])
    settings = _settings(
        "ants", score={"report": ["correlation", "mutual_information", "gradient_correlation"]}
    )
    expected = {
        "correlation": correlation_score(transform, mov, ref),
        "mutual_information": normalized_mutual_information(transform, mov, ref),
    }
    warps = []
    real = Transform.apply
    monkeypatch.setattr(
        Transform, "apply", lambda self, *a, **k: warps.append(1) or real(self, *a, **k)
    )
    report = engine._report_metrics(settings, transform, 0, mov[None], ref[None])
    assert (
        len(warps) == 2
    )  # one for the result, one for the unregistered pair -- not per metric
    assert {k: report[k] for k in expected} == expected


def test_the_report_compares_with_the_unregistered_pair_and_warns_when_it_scores_higher(
    capsys,
):
    from biahub.registration import engine
    from biahub.registration.metrics import correlation_score

    mov, ref = _pair()
    settings = _settings("ants", score={"report": ["correlation"]})
    good = Transform.from_translation([-1.0, 2.0, -3.0])  # undoes the pair's shift
    report = engine._report_metrics(settings, good, 0, mov[None], ref[None], score=0.99)
    identity = Transform.identity(3)
    assert report["unregistered_correlation"] == correlation_score(identity, mov, ref)
    assert report["unregistered_score"] == engine._score_fn(settings)(identity, mov, ref)
    assert "unregistered pair scores" not in capsys.readouterr().out

    engine._report_metrics(settings, good, 0, mov[None], ref[None], score=-1.0)
    assert (
        "unregistered pair scores" in capsys.readouterr().out
    )  # a score that ranks it higher

    # an identity result (a stabilization reference frame) is the unregistered pair: the
    # tie is expected, not a sign of a poor metric
    identity_score = report["unregistered_score"]
    engine._report_metrics(settings, identity, 0, mov[None], ref[None], score=identity_score)
    assert "unregistered pair scores" not in capsys.readouterr().out


def test_scores_with_a_padding_margin_work_on_2d_images():
    from biahub.registration.metrics import (
        correlation_score,
        gradient_correlation,
        normalized_mutual_information,
    )

    image = np.random.default_rng(8).random((40, 40)).astype(np.float32) + 1.0
    identity = Transform.identity(2)
    assert gradient_correlation(identity, image, image) == pytest.approx(1.0)
    assert correlation_score(identity, image, image, sobel_filter=True) == pytest.approx(1.0)
    assert np.isfinite(normalized_mutual_information(identity, image, image, sobel=True))


@pytest.mark.parametrize(
    "blocks, footprint",
    [
        ([{"sobel": {}}], 1),
        # each volume filtered once, as with one block on both
        ([{"sobel": {"target": ["moving"]}}, {"sobel": {"target": ["reference"]}}], 1),
        ([{"sobel": {}}, {"sobel": {"target": ["moving"]}}], 2),
    ],
)
def test_the_score_margin_is_the_reach_on_the_most_filtered_volume(blocks, footprint):
    from biahub.registration.preprocess import blocks_footprint

    assert blocks_footprint(
        _settings("ants", score={"preprocess": blocks}).score.preprocess
    ) == (footprint)
