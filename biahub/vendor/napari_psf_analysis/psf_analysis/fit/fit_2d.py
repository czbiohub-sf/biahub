from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


def parameterized_2d_gaussian(
    bg: float, amp: float, mu_y: float, mu_x: float, cyy: float, cyx: float, cxx: float
) -> Callable:
    """Parameterized 2D Gaussian.

    Parameters
    ----------
    bg :
        Background
    amp :
        Amplitude
    mu_y :
        Center along y-axis
    mu_x :
        Center along x-axis
    cyy :
        Covariance matrix entry cyy.
    cyx :
        Covariance matrix entry cyx.
    cxx :
        Covariance matrix entry cxx.

    Returns
    -------
    Callable
        Parameterized 2D Gaussian
    """

    def _gauss_2d(coords: ArrayLike):
        cov_inv = np.linalg.inv(np.array([[cyy, cyx], [cyx, cxx]]))
        exponent = -0.5 * (
            cov_inv[0, 0] * (coords[:, 0] - mu_y) ** 2
            + 2 * cov_inv[0, 1] * (coords[:, 0] - mu_y) * (coords[:, 1] - mu_x)
            + cov_inv[1, 1] * (coords[:, 1] - mu_x) ** 2
        )

        return amp * np.exp(exponent) + bg

    return _gauss_2d


def evaluate_2d_gaussian(
    x: ArrayLike,
    bg: float,
    amp: float,
    mu_y: float,
    mu_x: float,
    cyy: float,
    cyx: float,
    cxx: float,
) -> ArrayLike:
    """Sample 2D Gaussian.

    Evaluate a 2D Gaussian with the given parameters at positions `x`.

    Parameters
    ----------
    x :
        Positions where the Gaussian function is evaluated.
    bg :
        Background value
    amp :
        Gaussian amplitude
    mu_y :
        Center along y-axis
    mu_x :
        Center along x-axis
    cyy :
        Covariance matrix entry cyy.
    cyx :
        Covariance matrix entry cyx.
    cxx :
        Covariance matrix entry cxx.

    Returns
    -------
    ArrayLike
        Sampled values
    """
    return parameterized_2d_gaussian(
        bg=bg, amp=amp, mu_y=mu_y, mu_x=mu_x, cyy=cyy, cyx=cyx, cxx=cxx
    )(x)
