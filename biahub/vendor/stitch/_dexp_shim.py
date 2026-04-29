"""Torch port of the dexp surface used by ``stitch.tile``.

Replaces three names imported by ``ahillsley/stitching@jen:stitch/stitch/tile.py``:

* ``TranslationRegistrationModel``: dataclass with ``shift_vector`` (numpy)
  + ``confidence`` (scalar). Caller mutates ``shift_vector`` via ``+=``.
* ``register_translation_nd(a, b)``: N-D phase correlation, returns model.
* ``linsolve(A, y, order_error=1, alpha_reg=0)``: L1 minimization over a
  sparse incidence matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np
import scipy.ndimage as ndi
import torch

from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass
class TranslationRegistrationModel:
    shift_vector: np.ndarray
    confidence: float = 0.0


def _sobel_magnitude(image: NDArray, *, log_compression: bool = True) -> np.ndarray:
    """L1 Sobel-magnitude edge response, matching ``dexp.processing.filters.sobel_filter``.

    With ``exponent=1, normalise_input=False`` (the dexp call site in
    ``register_translation_nd``), this is ``sum_axis |sobel_axis(log1p(x))|``.
    """
    image = image.astype(np.float32, copy=True)
    if log_compression:
        image = np.log1p(image)
    out = np.zeros_like(image)
    for axis in range(image.ndim):
        out += np.abs(ndi.sobel(image, axis=axis))
    return out


def _preprocess(image: NDArray, *, denoise_sigma: float = 1.5) -> np.ndarray:
    """Apply dexp's pre-FFT pipeline: gaussian denoise + log1p + sobel."""
    image = image.astype(np.float32, copy=True)
    if denoise_sigma > 0:
        image = ndi.gaussian_filter(image, sigma=denoise_sigma)
    image = np.log1p(image)
    image = _sobel_magnitude(image, log_compression=True)
    return image


def _hanning_window(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Outer product of per-axis ``hanning(s)**0.5``; matches dexp's ``window=0.5``."""
    axes = [
        torch.hann_window(s, periodic=False, device=device, dtype=torch.float32) ** 0.5
        for s in shape
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    return reduce(torch.mul, grids)


def _phase_correlation(
    a: torch.Tensor, b: torch.Tensor, *, epsilon: float = 1e-6
) -> torch.Tensor:
    """Magnitude-normalized phase correlation, fftshift-centered."""
    window = _hanning_window(tuple(a.shape), a.device)
    a = a * window
    b = b * window
    Fa = torch.fft.fftn(a)
    Fb = torch.fft.fftn(b)
    R = Fa * Fb.conj()
    R = R / (R.abs() + epsilon)
    corr = torch.fft.ifftn(R).real
    return torch.fft.fftshift(corr)


def _shift_and_confidence(
    correlation: NDArray,
    *,
    max_range_ratio: float = 0.9,
    decimate: int = 16,
    quantile: float = 0.999,
    sigma: float = 1.5,
) -> tuple[np.ndarray, float]:
    """Replicate dexp's argmax + confidence pipeline on a fftshift'd correlation.

    Crops to ``max_range_ratio``, estimates the noise floor from a corner
    region, clips, gaussian-smooths, then takes argmax and the
    ``(peak - bg_peak)/(eps + peak)`` confidence.
    """
    max_ranges = tuple(int(0.5 * max_range_ratio * s) for s in correlation.shape)
    center = tuple(s // 2 for s in correlation.shape)

    empty_region = correlation[
        tuple(slice(0, c - r) for c, r in zip(center, max_ranges, strict=True))
    ]
    flat = empty_region.ravel()[::decimate].astype(np.float32)
    if flat.size == 0:
        noise_floor = float(correlation.mean())
    else:
        noise_floor = float(np.quantile(flat, q=quantile))
        if not np.isfinite(noise_floor):
            noise_floor = float(flat.mean())

    cropped = correlation[
        tuple(
            slice(max(c - r, 0), min(c + r, s))
            for c, r, s in zip(center, max_ranges, correlation.shape, strict=True)
        )
    ]
    cropped = np.maximum(cropped, noise_floor) - noise_floor

    if sigma > 0:
        cropped = ndi.gaussian_filter(cropped, sigma=sigma, mode="wrap")

    rough_shift = np.unravel_index(int(np.argmax(cropped)), cropped.shape)
    max_correlation = float(cropped[rough_shift])
    shift_vector = np.array(
        [int(rs) - r for rs, r in zip(rough_shift, max_ranges, strict=True)],
        dtype=np.float32,
    )

    masked = cropped.copy()
    mask_size = tuple(max(8, int(s**0.9) // 8) for s in masked.shape)
    masked[
        tuple(slice(rs - s, rs + s) for rs, s in zip(rough_shift, mask_size, strict=True))
    ] = 0
    background_max = float(masked.max())
    confidence = (max_correlation - background_max) / (1e-6 + max_correlation)

    return shift_vector, confidence


def register_translation_nd(
    image_a: NDArray, image_b: NDArray
) -> TranslationRegistrationModel:
    """Estimate the integer N-D translation from ``image_a`` to ``image_b``.

    Port of ``dexp.processing.registration.translation_nd.register_translation_nd``
    with upstream defaults baked in (``denoise_input_sigma=1.5,
    log_compression=True, edge_filter=True, max_range_ratio=0.9,
    quantile=0.999, sigma=1.5``). The FFT runs on GPU when available;
    preprocessing and post-FFT noise-floor logic use scipy.ndimage and numpy.

    Parameters
    ----------
    image_a, image_b : numpy.ndarray
        Input volumes of identical shape and dtype. Any numeric dtype is
        accepted; values are converted to ``float32`` internally.

    Returns
    -------
    TranslationRegistrationModel
        ``shift_vector`` is a length-``ndim`` numpy ``float32`` array of
        signed integer shifts (in pixels) such that
        ``image_a[shift] ≈ image_b[0]``. ``confidence`` is in ``[0, 1)``,
        defined as ``(peak - bg_max) / (eps + peak)`` after masking the
        peak region — a sharper, better-isolated peak gives a higher score.

    Raises
    ------
    ValueError
        If ``image_a.dtype != image_b.dtype``.
    """
    if image_a.dtype != image_b.dtype:
        raise ValueError("image_a and image_b must share a dtype")

    pre_a = _preprocess(image_a)
    pre_b = _preprocess(image_b)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ta = torch.as_tensor(pre_a, dtype=torch.float32, device=device)
    tb = torch.as_tensor(pre_b, dtype=torch.float32, device=device)
    correlation = _phase_correlation(ta, tb).cpu().numpy()

    shift, confidence = _shift_and_confidence(correlation)
    return TranslationRegistrationModel(shift_vector=shift, confidence=confidence)


def linsolve(
    A: NDArray,
    y: NDArray,
    *,
    tolerance: float = 1e-6,
    x0: NDArray | None = None,
    maxiter: int = 10**12,
    maxfun: int = 10**12,
    order_error: float = 1,
    order_reg: float = 1,
    alpha_reg: float = 0.0,
    **_: Any,
) -> np.ndarray:
    """Solve ``min_x |A x - y|_p + alpha |x|_q`` via L-BFGS-B.

    Port of ``dexp.processing.utils.linear_solver.linsolve``. The stitch
    call site uses ``order_error=1, alpha_reg=0``, i.e. plain L1 regression.
    L2 least-squares gives systematically different answers here because
    noisy / low-confidence edges contribute outlier rows that L2 won't
    down-weight.

    Parameters
    ----------
    A : NDArray
        Coefficient matrix, shape ``(n_constraints, n_unknowns)``. Sparse
        matrices are densified internally (the stitch problem is small).
    y : NDArray
        Right-hand-side vector, shape ``(n_constraints,)``.
    tolerance : float, optional
        Convergence tolerance passed to L-BFGS-B as both ``tol`` and
        ``gtol``. Default ``1e-6``.
    x0 : NDArray, optional
        Warm-start vector of length ``n_unknowns``. Defaults to zeros.
    maxiter, maxfun : int, optional
        L-BFGS-B iteration / function-evaluation limits.
    order_error : float, optional
        ``p`` in the residual norm ``|A x - y|_p``. Default 1 (L1).
    order_reg : float, optional
        ``q`` in the regularization norm ``|x|_q``. Default 1 (L1).
    alpha_reg : float, optional
        Regularization strength. Default 0 (no regularization).

    Returns
    -------
    numpy.ndarray
        Optimal ``x`` of length ``n_unknowns``. On convergence failure,
        ``x0`` is returned (matching dexp's behavior).
    """
    A_dense = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
    y_arr = np.asarray(y, dtype=np.float64)
    A_arr = A_dense.astype(np.float64)

    if x0 is None:
        x0 = np.zeros(A_arr.shape[1], dtype=np.float64)
    else:
        x0 = np.asarray(x0, dtype=np.float64)

    beta = (1.0 / y_arr.shape[0]) ** (1.0 / order_error)
    alpha = (1.0 / x0.shape[0]) ** (1.0 / order_reg)

    def fun(x):
        residual = beta * float(np.linalg.norm(A_arr @ x - y_arr, ord=order_error))
        if alpha_reg == 0:
            return residual
        return residual + (alpha_reg * alpha) * float(np.linalg.norm(x, ord=order_reg))

    result = minimize(
        fun,
        x0,
        method="L-BFGS-B",
        tol=tolerance,
        options={
            "maxiter": int(maxiter),
            "maxfun": int(maxfun),
            "gtol": tolerance,
            "eps": 1e-5,  # dexp comment: minimization sometimes won't converge without this
        },
    )
    if not result.success:
        # dexp falls back to x0 on convergence failure rather than raising
        return x0
    return result.x
