from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


def parameterized_3d_gaussian(
    bg: float,
    amp: float,
    mu_z: float,
    mu_y: float,
    mu_x: float,
    czz: float,
    czy: float,
    czx: float,
    cyy: float,
    cyx: float,
    cxx: float,
) -> Callable:
    """Parameterized 3D Gaussian.

    Parameters
    ----------
    bg :
        Background
    amp :
        Amplitude
    mu_z :
        Center along z-axis
    mu_y :
        Center along y-axis
    mu_x :
        Center along x-axis
    czz :
        Covariance matrix entry czz.
    czy :
        Covariance matrix entry czy.
    czx :
        Covariance matrix entry czx.
    cyy :
        Covariance matrix entry cyy.
    cyx :
        Covariance matrix entry cyx.
    cxx :
        Covariance matrix entry cxx.

    Returns
    -------
    Callable
        Parameterized 3D Gaussian
    """

    def _gauss_3d(coords: ArrayLike):
        cov_inv = np.linalg.inv(np.array([[czz, czy, czx], [czy, cyy, cyx], [czx, cyx, cxx]]))

        exponent = -0.5 * (
            cov_inv[0, 0] * (coords[:, 0] - mu_z) ** 2
            + 2 * cov_inv[0, 1] * (coords[:, 0] - mu_z) * (coords[:, 1] - mu_y)
            + 2 * cov_inv[0, 2] * (coords[:, 0] - mu_z) * (coords[:, 2] - mu_x)
            + cov_inv[1, 1] * (coords[:, 1] - mu_y) ** 2
            + 2 * cov_inv[1, 2] * (coords[:, 1] - mu_y) * (coords[:, 2] - mu_x)
            + cov_inv[2, 2] * (coords[:, 2] - mu_x) ** 2
        )

        return amp * np.exp(exponent) + bg

    return _gauss_3d


def evaluate_3d_gaussian(
    x: ArrayLike,
    bg: float,
    amp: float,
    mu_z: float,
    mu_y: float,
    mu_x: float,
    czz: float,
    czy: float,
    czx: float,
    cyy: float,
    cyx: float,
    cxx: float,
) -> ArrayLike:
    """Sample 3D Gaussian.

    Evaluate a 3D Gaussian with the given parameters at positions `x`.

    Parameters
    ----------
    x :
        Positions where the Gaussian function is evaluated.
    bg :
        Background value
    amp :
        Gaussian amplitude
    mu_z :
        Center along z-axis
    mu_y :
        Center along y-axis
    mu_x :
        Center along x-axis
    czz :
        Covariance matrix entry czz.
    czy :
        Covariance matrix entry czy.
    czx :
        Covariance matrix entry czx.
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
    return parameterized_3d_gaussian(
        bg=bg,
        amp=amp,
        mu_z=mu_z,
        mu_y=mu_y,
        mu_x=mu_x,
        czz=czz,
        czy=czy,
        czx=czx,
        cyy=cyy,
        cyx=cyx,
        cxx=cxx,
    )(x)
