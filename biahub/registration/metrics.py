"""Alignment metrics: how well a transform brings `mov` onto `ref`.

Two families, because the pipeline registers across modalities:

- Bead metrics (`bead_alignment_metrics`, `residual_score`) need beads visible in both
  channels. They complement `beads.overlap_score`, which is a bead-count ratio quantized
  at ~1/N per bead, with continuous residuals in voxels.
- Intensity metrics work on any pair of channels. Plain correlation is meaningless
  between, say, phase and fluorescence (the correct transform can score ~0.1), so
  `normalized_mutual_information` -- the cross-modal similarity ANTs' Mattes metric
  approximates -- is the choice when no beads are available, and
  `gradient_correlation` is a cheaper edge-based alternative.

Every function takes the engine's forward transform and warps with `Transform.apply`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from skimage import filters

from biahub.core.transform import Transform
from biahub.registration.beads import overlap_score, peaks_from_beads
from biahub.registration.phase_cross_correlation import phase_cross_corr
from biahub.settings import BeadsMatchSettings


@dataclass
class BeadAlignmentMetrics:
    n_mov_peaks: int
    n_ref_peaks: int
    n_matched: int  # reference peaks with a warped moving peak within `radius`
    overlap: float  # beads.overlap_score
    median_residual: float  # voxels, over matched pairs; nan when none
    rms_residual: float

    def to_dict(self) -> dict:
        return asdict(self)


def bead_alignment_metrics(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
) -> BeadAlignmentMetrics | None:
    """Warp, re-detect beads, and measure how the two peak sets line up.

    Returns None when peaks cannot be detected on either side.
    """
    radius = beads_match_settings.qc_settings.score_centroid_mask_radius
    ref = np.asarray(ref)
    warped = transform.apply(np.asarray(mov), reference=ref)
    peaks = peaks_from_beads(
        mov=warped,
        ref=ref,
        mov_peaks_settings=beads_match_settings.source_peaks_settings,
        ref_peaks_settings=beads_match_settings.target_peaks_settings,
        verbose=False,
    )
    if peaks is None:
        return None
    mov_peaks, ref_peaks = (np.asarray(p, dtype=float) for p in peaks)
    if len(mov_peaks) == 0 or len(ref_peaks) == 0:
        return None
    distances, _ = cKDTree(mov_peaks).query(ref_peaks, distance_upper_bound=radius)
    matched = distances[np.isfinite(distances)]
    return BeadAlignmentMetrics(
        n_mov_peaks=int(len(mov_peaks)),
        n_ref_peaks=int(len(ref_peaks)),
        n_matched=int(len(matched)),
        overlap=float(overlap_score(mov_peaks, ref_peaks, radius=radius, verbose=False)),
        median_residual=float(np.median(matched)) if len(matched) else float("nan"),
        rms_residual=float(np.sqrt(np.mean(matched**2))) if len(matched) else float("nan"),
    )


def residual_score(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    beads_match_settings: BeadsMatchSettings,
) -> float:
    """Continuous bead score in [0, 1]: overlap weighted by how tightly matched beads sit.

    `overlap * (1 - median_residual / radius)`: the same quantity `overlap_score`
    measures, but a transform whose matched beads land 0.5 voxel off outscores one whose
    beads land 4 voxels off instead of tying with it. nan when no beads are detected.
    """
    metrics = bead_alignment_metrics(transform, mov, ref, beads_match_settings)
    if metrics is None or metrics.n_matched == 0:
        return float("nan") if metrics is None else 0.0
    radius = beads_match_settings.qc_settings.score_centroid_mask_radius
    return float(metrics.overlap * max(0.0, 1.0 - metrics.median_residual / radius))


def _overlap_pair(transform: Transform, mov: ArrayLike, ref: ArrayLike):
    ref = np.asarray(ref, dtype=np.float32)
    warped = transform.apply(np.asarray(mov, dtype=np.float32), reference=ref)
    mask = (warped != 0) & (ref != 0)
    return warped, ref, mask


def normalized_mutual_information(
    transform: Transform,
    mov: ArrayLike,
    ref: ArrayLike,
    bins: int = 64,
    percentiles: tuple[float, float] = (0.5, 99.5),
    sobel: bool = False,
) -> float:
    """Compute normalized mutual information of the warped moving volume and the reference.

    2 * I(A; B) / (H(A) + H(B)) over the overlapping voxels, in [0, 1]; intensities are
    clipped to `percentiles` before binning so a few hot voxels do not collapse the
    histogram. Independent of the intensity relation between the two channels, which is
    what makes it usable across modalities; `sobel` compares edge magnitudes instead of
    intensities. nan when the volumes do not overlap.
    """
    warped, ref, mask = _overlap_pair(transform, mov, ref)
    if mask.sum() < bins:
        return float("nan")
    if sobel:
        # MI of edge magnitudes: what the fault detectors in registration-eval found most
        # reliable between fluorescence and virtual staining.
        warped, ref = filters.sobel(warped), filters.sobel(ref)
    a, b = warped[mask], ref[mask]
    a_lo, a_hi = np.percentile(a, percentiles)
    b_lo, b_hi = np.percentile(b, percentiles)
    if a_hi <= a_lo or b_hi <= b_lo:
        return float("nan")
    joint, _, _ = np.histogram2d(
        np.clip(a, a_lo, a_hi),
        np.clip(b, b_lo, b_hi),
        bins=bins,
        range=[[a_lo, a_hi], [b_lo, b_hi]],
    )
    p = joint / joint.sum()
    pa, pb = p.sum(axis=1), p.sum(axis=0)

    def entropy(q):
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())

    h_a, h_b, h_ab = entropy(pa), entropy(pb), entropy(p.ravel())
    if h_a + h_b == 0:
        return float("nan")
    return float(2.0 * (h_a + h_b - h_ab) / (h_a + h_b))


def gradient_correlation(transform: Transform, mov: ArrayLike, ref: ArrayLike) -> float:
    """Pearson correlation of Sobel gradient magnitudes over the overlap, in [-1, 1].

    Edges exist in every modality even when intensities do not correlate; cheaper than
    mutual information and adequate when both channels show the same structures.
    """
    warped, ref, mask = _overlap_pair(transform, mov, ref)
    if mask.sum() < 2:
        return float("nan")
    a = filters.sobel(warped)[mask]
    b = filters.sobel(ref)[mask]
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def residual_shift(transform: Transform, mov: ArrayLike, ref: ArrayLike) -> float:
    """Remaining misalignment after `transform`, in voxels, as a displacement not a similarity.

    Phase cross-correlation between the Sobel edges of the warped moving volume and the
    reference; the norm of the recovered shift. A displacement stays interpretable when
    similarity metrics are depressed by defocus or bleaching (registration-eval's
    `mch_eshift`), and 0 means aligned regardless of modality.
    """
    warped, ref, mask = _overlap_pair(transform, mov, ref)
    if mask.sum() < 8:
        return float("nan")
    shift, _ = phase_cross_corr(
        filters.sobel(ref).astype(np.float32), filters.sobel(warped).astype(np.float32)
    )
    return float(np.linalg.norm(shift))
