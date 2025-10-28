from typing import Tuple

import numpy as np

from biahub.vendor.napari_psf_analysis.image import Calibrated3DImage
from biahub.vendor.napari_psf_analysis.psf_analysis.sample import (
    Sample,
    YXSample,
    ZSample,
    ZYXSample,
)
from biahub.vendor.napari_psf_analysis.psf_analysis.utils import compute_cov_matrix


class Estimator:
    image: Calibrated3DImage = None
    sample: Sample = None
    _background: np.uint16 = None
    _amplitude: np.uint16 = None

    def __init__(self, image: Calibrated3DImage):
        self.image = image

    def get_background(self) -> np.uint16:
        if self._background is None:
            self._background = np.median(self.image.data).astype(np.uint16)

        return self._background

    def get_amplitude(self) -> np.uint16:
        if self._amplitude is None:
            self._amplitude = self.sample.image.data.max() - self.get_background()

        return self._amplitude


class ZEstimator(Estimator):
    sample: ZSample = None
    _centroid: float = None
    _sigma: float = None

    def __init__(self, image: Calibrated3DImage, z_sample: ZSample):
        super().__init__(image=image)
        self.sample = z_sample

    def get_centroid(self) -> float:
        if self._centroid is None:
            from skimage.measure import centroid

            self._centroid = centroid(self.sample.image.data)[0] * self.sample.image.spacing[0]
        return self._centroid

    def get_sigma(self) -> float:
        if self._sigma is None:
            cov_matrix = compute_cov_matrix(
                img_data=self.sample.image.data, spacing=self.sample.image.spacing
            )
            self._sigma = np.abs(np.sqrt(cov_matrix))

        return self._sigma


class YXEstimator(Estimator):
    sample: YXSample = None
    _centroid: Tuple[float, float] = None
    _sigmas: Tuple[float, float] = None

    def __init__(self, image: Calibrated3DImage, yx_sample: YXSample):
        super().__init__(image=image)
        self.sample = yx_sample

    def get_centroid(self) -> Tuple[float, float]:
        if self._centroid is None:
            from skimage.measure import centroid

            uncalibrated_centroid = centroid(self.sample.image.data)
            self._centroid = (
                uncalibrated_centroid[0] * self.sample.image.spacing[0],
                uncalibrated_centroid[1] * self.sample.image.spacing[1],
            )

        return self._centroid

    def get_sigmas(self) -> Tuple[float, float]:
        if self._sigmas is None:
            cov_matrix = compute_cov_matrix(
                img_data=self.sample.image.data, spacing=self.sample.image.spacing
            )
            self._sigmas = (
                np.abs(np.sqrt(cov_matrix[0, 0])),
                np.abs(np.sqrt(cov_matrix[1, 1])),
            )

        return self._sigmas


class ZYXEstimator(Estimator):
    sample: ZYXSample = None
    _centroid: Tuple[float, float, float] = None
    _sigmas: Tuple[float, float, float] = None

    def __init__(self, image: Calibrated3DImage, zyx_sample: ZYXSample):
        super().__init__(image=image)
        self.sample = zyx_sample

    def get_centroid(self) -> Tuple[float, float, float]:
        if self._centroid is None:
            from skimage.measure import centroid

            uncalibrated_centroid = centroid(self.sample.image.data)
            self._centroid = (
                uncalibrated_centroid[0] * self.sample.image.spacing[0],
                uncalibrated_centroid[1] * self.sample.image.spacing[1],
                uncalibrated_centroid[2] * self.sample.image.spacing[2],
            )

        return self._centroid

    def get_sigmas(self) -> Tuple[float, float, float]:
        if self._sigmas is None:
            cov_matrix = compute_cov_matrix(
                img_data=self.sample.image.data, spacing=self.sample.image.spacing
            )
            self._sigmas = (
                np.abs(np.sqrt(cov_matrix[0, 0])),
                np.abs(np.sqrt(cov_matrix[1, 1])),
                np.abs(np.sqrt(cov_matrix[2, 2])),
            )

        return self._sigmas
