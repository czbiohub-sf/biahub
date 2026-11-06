import numpy as np

from biahub.core.transform import Transform
from biahub.registration.metrics import score_transform
from biahub.settings import BeadsMatchSettings

PEAKS = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])


def test_score_transform_warps_mov_then_scores_peak_overlap(monkeypatch):
    mov = np.zeros((5, 5, 5))
    ref = np.zeros((5, 5, 5))
    transform = Transform.from_translation([1.0, 0.0, 0.0])
    seen = []

    def _fake_detect(volume, settings):
        seen.append(volume)
        return PEAKS.copy()

    monkeypatch.setattr(
        "biahub.registration.methods.landmark.bead_matching._detect", _fake_detect
    )

    score = score_transform(transform, mov, ref, BeadsMatchSettings())

    assert score == 1.0
    # the moving volume is detected once warped onto the reference, the reference as it is
    warped, reference = seen
    assert warped is not mov and warped.shape == mov.shape
    assert reference is ref


def test_score_transform_returns_nan_when_not_enough_peaks_detected(monkeypatch):
    mov = np.zeros((5, 5, 5))
    ref = np.zeros((5, 5, 5))
    monkeypatch.setattr(
        "biahub.registration.methods.landmark.bead_matching._detect",
        lambda volume, settings: PEAKS[:1].copy(),  # one bead: too few to pair
    )

    score = score_transform(Transform.identity(3), mov, ref, BeadsMatchSettings())

    assert np.isnan(score)


def test_a_job_warps_and_detects_each_volume_once_per_transform(monkeypatch):
    # every bead score, report metric and record metric re-warped and re-detected: on a
    # synthetic bead job, 20 detections and 7 warps for 2 distinct transforms
    from biahub.registration.metrics import bead_alignment_metrics

    calls = []
    monkeypatch.setattr(
        "biahub.registration.methods.landmark.bead_matching._detect",
        lambda volume, settings: calls.append(volume.shape) or PEAKS.copy(),
    )
    mov, ref = np.zeros((5, 5, 5)), np.zeros((5, 5, 5))
    settings = BeadsMatchSettings()
    near, far = Transform.identity(3), Transform.from_translation([0.0, 0.0, 1.0])
    for transform in (near, far, near, far, near):
        bead_alignment_metrics(transform, mov, ref, settings)
    assert len(calls) == 3  # the reference once, the moving volume once per transform
