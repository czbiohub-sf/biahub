"""Torch + MONAI replacements for the dexp surface used by ``stitch.tile``.

Upstream ``ahillsley/stitching@jen:stitch/stitch/tile.py`` imports three names
from the ``dexp`` package:

* ``dexp.processing.registration.model.translation_registration_model.TranslationRegistrationModel``
* ``dexp.processing.registration.translation_nd.register_translation_nd``
* ``dexp.processing.utils.linear_solver.linsolve``

The surface used at those call sites is narrow:

* ``TranslationRegistrationModel`` is read for ``shift_vector`` (numpy-like)
  and ``confidence`` (scalar), and has ``shift_vector += ...`` applied.
* ``register_translation_nd(a, b)`` runs an N-D phase correlation and returns
  a model with an integer ``shift_vector`` and a scalar ``confidence``.
* ``linsolve(A, y, tolerance=, x0=, maxiter=, order_error=1, order_reg=1,
  alpha_reg=0, ...)`` is called twice with ``alpha_reg=0``, i.e. plain least
  squares over a sparse incidence matrix.

This module replaces those with torch + MONAI:

* Phase correlation via :func:`torch.fft.fftn`. Runs on GPU when available.
* Confidence via :class:`monai.losses.LocalNormalizedCrossCorrelationLoss`
  applied to the overlapping strips after alignment — a physically meaningful
  similarity score in ``[-1, 1]`` rather than dexp's peak-to-mean ratio.
* ``linsolve`` via :func:`torch.linalg.lstsq` on a densified incidence matrix
  (biahub-scale stitching has O(10-100) tiles, densifying is negligible).

Device selection is automatic (CUDA if available, else CPU); override with the
``BIAHUB_STITCH_DEVICE`` environment variable (e.g. ``cpu`` or ``cuda:1``).
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from monai.losses import LocalNormalizedCrossCorrelationLoss


def _device() -> torch.device:
    override = os.environ.get("BIAHUB_STITCH_DEVICE")
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TranslationRegistrationModel:
    shift_vector: np.ndarray
    confidence: float = 0.0


def _phase_correlation_shift(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the integer translation that best aligns ``b`` to ``a``.

    Implements magnitude-normalized phase correlation. Positive entries point
    toward larger indices; wrapped entries beyond the midpoint are converted
    to signed offsets.
    """
    Fa = torch.fft.fftn(a)
    Fb = torch.fft.fftn(b)
    eps = torch.finfo(Fa.real.dtype).eps
    cross = Fa * Fb.conj()
    cross = cross / torch.clamp(cross.abs(), min=eps)
    corr = torch.fft.ifftn(cross).real

    peak = torch.argmax(corr.flatten())
    idx = torch.unravel_index(peak, corr.shape)
    shape = torch.tensor(corr.shape, device=corr.device)
    shift = torch.stack(idx).to(torch.float32)
    wrap = shift > (shape // 2)
    shift = torch.where(wrap, shift - shape.to(shift.dtype), shift)
    return shift


def _lncc_confidence(a: torch.Tensor, b: torch.Tensor, shift: torch.Tensor) -> float:
    """Local normalized cross-correlation of ``a`` and ``b`` aligned by ``shift``.

    Uses integer-pixel alignment (cropping the overlapping region post-shift)
    to avoid introducing interpolation artifacts into the confidence score.
    Returns a scalar in ``[0, 1]`` (squared local cross-correlation averaged
    over the overlap region); higher is a better alignment.
    """
    ndim = a.ndim
    s = shift.round().to(torch.long).tolist()

    a_slice, b_slice = [], []
    for axis in range(ndim):
        d = int(s[axis])
        n = a.shape[axis]
        if d >= 0:
            a_slice.append(slice(d, n))
            b_slice.append(slice(0, n - d))
        else:
            a_slice.append(slice(0, n + d))
            b_slice.append(slice(-d, n))

    a_aligned = a[tuple(a_slice)]
    b_aligned = b[tuple(b_slice)]

    if any(size == 0 for size in a_aligned.shape):
        return 0.0

    # LNCC wants (N, C, *spatial). Add batch + channel.
    a_batched = a_aligned.unsqueeze(0).unsqueeze(0).float()
    b_batched = b_aligned.unsqueeze(0).unsqueeze(0).float()

    kernel_size = min(9, *(s - (s + 1) % 2 for s in a_aligned.shape))
    if kernel_size < 3:
        # overlap too thin for a windowed correlation; fall back to global cc^2
        a_flat = a_batched.flatten()
        b_flat = b_batched.flatten()
        a_flat = a_flat - a_flat.mean()
        b_flat = b_flat - b_flat.mean()
        denom = (a_flat * a_flat).sum() * (b_flat * b_flat).sum()
        if denom <= 0:
            return 0.0
        cc = (a_flat * b_flat).sum() / torch.sqrt(denom)
        return float(cc * cc)

    lncc = LocalNormalizedCrossCorrelationLoss(
        spatial_dims=ndim, kernel_size=kernel_size, kernel_type="rectangular"
    )
    # MONAI's LNCC returns ``-mean(cc^2)`` so a perfectly aligned pair gives
    # ``loss == -1``. Negate to get a [0, 1] "higher is better" confidence.
    with torch.no_grad():
        loss = lncc(a_batched, b_batched)
    return float(-loss.item())


def register_translation_nd(
    image_a: np.ndarray, image_b: np.ndarray
) -> TranslationRegistrationModel:
    """N-D phase-correlation translation from ``image_a`` to ``image_b``.

    Uses :func:`torch.fft` for the shift estimate (runs on GPU when available)
    and MONAI's local normalized cross-correlation for the confidence score.
    The returned ``shift_vector`` is a numpy ``float32`` array to preserve the
    caller's ``+=`` usage pattern. ``confidence`` is in ``[0, 1]`` (squared
    local cross-correlation averaged over the overlap); higher means better
    alignment of the overlapping strips.
    """
    device = _device()
    a = torch.as_tensor(np.asarray(image_a), dtype=torch.float32, device=device)
    b = torch.as_tensor(np.asarray(image_b), dtype=torch.float32, device=device)

    shift = _phase_correlation_shift(a, b)
    confidence = _lncc_confidence(a, b, shift)

    return TranslationRegistrationModel(
        shift_vector=shift.cpu().numpy().astype(np.float32),
        confidence=confidence,
    )


def linsolve(
    A,
    y,
    *,
    tolerance: float = 1e-5,
    x0=None,
    maxiter: int = 10**8,
    order_error: int = 1,
    order_reg: int = 1,
    alpha_reg: float = 0.0,
    **_: Any,
) -> np.ndarray:
    """Least-squares solve of ``A @ x = y`` via :func:`torch.linalg.lstsq`.

    ``A`` is typically a ``scipy.sparse`` matrix of size
    ``(N_edges + 1, N_tiles)``; at biahub stitching scale (O(100) tiles)
    densifying is cheap. Only the unregularized case (``alpha_reg == 0``) is
    implemented; ``tolerance``/``maxiter``/``x0`` are accepted for call-site
    compatibility and unused (torch's direct LAPACK solver converges in one
    step, no warm start needed).
    """
    if alpha_reg != 0:
        raise NotImplementedError(
            "linsolve shim only implements the unregularized (alpha_reg=0) case."
        )
    device = _device()
    A_dense = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
    A_t = torch.as_tensor(A_dense, dtype=torch.float64, device=device)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.float64, device=device)
    result = torch.linalg.lstsq(A_t, y_t.unsqueeze(-1)).solution.squeeze(-1)
    return result.cpu().numpy()
