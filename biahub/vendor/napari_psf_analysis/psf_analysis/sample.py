from abc import ABC, abstractmethod

import numpy as np

from numpy.typing import ArrayLike

from biahub.vendor.napari_psf_analysis.image import (
    Calibrated1DImage,
    Calibrated2DImage,
    Calibrated3DImage,
    CalibratedImage,
)


class Sample(ABC):
    image: CalibratedImage = None

    @abstractmethod
    def get_ravelled_coordinates(self) -> ArrayLike:
        pass


class ZSample(Sample):
    image: Calibrated1DImage = None

    def __init__(self, image: Calibrated1DImage):
        self.image = image

    def get_ravelled_coordinates(self) -> ArrayLike:
        return np.arange(self.image.data.shape[0]) * self.image.spacing[0]


class YXSample(Sample):
    image: Calibrated2DImage = None

    def __init__(self, image: Calibrated2DImage):
        self.image = image

    def get_ravelled_coordinates(self) -> ArrayLike:
        yy = np.arange(self.image.data.shape[0]) * self.image.spacing[0]
        xx = np.arange(self.image.data.shape[1]) * self.image.spacing[1]
        y, x = np.meshgrid(yy, xx, indexing="ij")
        return np.stack([y.ravel(), x.ravel()], -1)


class ZYXSample(Sample):
    image: Calibrated3DImage = None

    def __init__(self, image: Calibrated3DImage):
        self.image = image

    def get_ravelled_coordinates(self) -> ArrayLike:
        zz = np.arange(self.image.data.shape[0]) * self.image.spacing[0]
        yy = np.arange(self.image.data.shape[1]) * self.image.spacing[1]
        xx = np.arange(self.image.data.shape[2]) * self.image.spacing[2]
        z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
        return np.stack([z.ravel(), y.ravel(), x.ravel()], -1)
