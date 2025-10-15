from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


def parameterized_1d_gaussian(bg: float, amp: float, mu: float, sigma: float) -> Callable:
    """Parameterized 1D Gaussian.

    Parameters
    ----------
    bg :
        Background
    amp :
        Amplitude
    mu :
        Center
    sigma :
        Standard deviation.

    Returns
    -------
    Callable
        Parameterized 1D Gaussian
    """
    return lambda x: amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + bg


def evaluate_1d_gaussian(
    x: ArrayLike, bg: float, amp: float, mu: float, sigma: float
) -> ArrayLike:
    """Sample 1D Gaussian.

    Evaluate a 1D Gaussian with the given parameters at positions `x`.

    Parameters
    ----------
    x :
        Positions where the Gaussian function is evaluated.
    bg :
        Background value
    amp :
        Gaussian amplitude
    mu :
        Gaussian center
    sigma :
        Gaussian standard deviation

    Returns
    -------
    ArrayLike
        Sampled values
    """
    return parameterized_1d_gaussian(bg=bg, amp=amp, mu=mu, sigma=sigma)(x)
