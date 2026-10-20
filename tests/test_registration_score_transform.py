import numpy as np

from biahub.core.transform import Transform
from biahub.registration.metrics import score_transform
from biahub.settings import BeadsMatchSettings


def test_score_transform_warps_mov_then_scores_peak_overlap(monkeypatch):
    mov = np.zeros((5, 5, 5))
    ref = np.zeros((5, 5, 5))
    transform = Transform.from_translation([1.0, 0.0, 0.0])
    seen = {}

    def _fake_peaks_from_beads(
        mov, ref, mov_peaks_settings, ref_peaks_settings, verbose=False
    ):
        seen["mov"] = mov
        seen["ref"] = ref
        return np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]])

    monkeypatch.setattr("biahub.registration.metrics.peaks_from_beads", _fake_peaks_from_beads)

    score = score_transform(transform, mov, ref, BeadsMatchSettings())

    assert score == 1.0
    # The warped array (not the raw `mov`) is what gets passed on for peak detection.
    np.testing.assert_array_equal(seen["ref"], ref)
    assert seen["mov"].shape == mov.shape


def test_score_transform_returns_nan_when_not_enough_peaks_detected(monkeypatch):
    mov = np.zeros((5, 5, 5))
    ref = np.zeros((5, 5, 5))
    transform = Transform.from_translation([0.0, 0.0, 0.0])

    monkeypatch.setattr(
        "biahub.registration.metrics.peaks_from_beads",
        lambda *args, **kwargs: None,
    )

    score = score_transform(transform, mov, ref, BeadsMatchSettings())

    assert np.isnan(score)
