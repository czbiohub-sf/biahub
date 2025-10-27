from typing import Tuple

import numpy as np

from numpy.typing import ArrayLike
from skimage.measure import centroid


def sigma(fwhm: float) -> float:
    """Compute sigma from full width half maximum.

    Parameters
    ----------
    fwhm :
        full width half maximum

    Returns
    -------
    float
        sigma
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def fwhm(sigma: float) -> float:
    """Full width at half maximum.

    Compute full width at half maximum (FWHM) for the provided standard
    deviation of `sigma`.

    Parameters
    ----------
    sigma :
        standard deviation

    Returns
    -------
    float
        full width at half maximum
    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def compute_cov_matrix(img_data: ArrayLike, spacing: Tuple[float, ...]) -> ArrayLike:
    """Weighted covariance matrix.

    Given some image data the intensity weighted covariance matrix over the
    spatial dimensions is computed.

    Parameters
    ----------
    img_data :
        n-dimensional image
    spacing :
        pixel spacing

    Returns
    -------
        The covariance matrix of the image data weighted by the intensity.
    """
    extends = [np.arange(dim_size) * s for dim_size, s in zip(img_data.shape, spacing)]

    grids = np.meshgrid(*extends, indexing="ij")

    m = np.stack([g.ravel() for g in grids])

    return np.cov(m, fweights=img_data.ravel())


def estimate_from_data(sample: ArrayLike, data: ArrayLike, sample_spacing: Tuple[float, ...]):
    """Estimate Gaussian fit parameters from data.

    Given some sample and data initialization parameters for a Gaussian fit
    are estimated. The following parameters are computed:
    * background: Is the median over all `data`.
    * Amplitude: Maximum of `sample`.
    * mean: Weighted centroid of the background subtracted sample.
    * sigma: Standard deviation estimated from the covariance matrix.

    Parameters
    ----------
    sample :
        A subset of data for which the estimates are computed.
    data :
        The whole data, used to get a better estimate for the background.
    sample_spacing :
        Pixel spacing of `sample`.

    Returns
    -------
    background:
        Estimated data background
    amplitude:
        Estimated Gaussian amplitude
    mu:
        Weighed centroid of the sample
    sigma:
        Standard deviation of the sample
    """
    bg = np.median(data).astype(np.uint16)
    amp = sample.max() - bg

    z_sample_no_bg = np.clip(sample - bg, 0, data.max())

    mu = centroid(z_sample_no_bg) * np.array(sample_spacing)
    cov_matrix = compute_cov_matrix(img_data=z_sample_no_bg, spacing=sample_spacing)
    if cov_matrix.ndim == 0:
        sigma = np.sqrt(cov_matrix)
    else:
        sigma = np.sqrt(np.diag(cov_matrix))
    return bg, amp, mu, sigma
