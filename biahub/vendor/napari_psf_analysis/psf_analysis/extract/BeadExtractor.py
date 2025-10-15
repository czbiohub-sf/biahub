import logging

from typing import List, Tuple

import numpy as np

from numpy.typing import ArrayLike

from biahub.vendor.napari_psf_analysis.image import Calibrated3DImage

logger = logging.getLogger(__name__)


class BeadExtractor:
    _image: Calibrated3DImage = None
    _patch_size: Tuple[int, int, int] = None
    _margins: ArrayLike = None

    def __init__(self, image: Calibrated3DImage, patch_size: Tuple[int, int, int]):
        self._image = image
        self._patch_size = patch_size
        self._margins = np.array(patch_size) / np.array(image.spacing)

    def _in_margins(self, point: Tuple[int, int, int]) -> bool:
        lower_bounds = self._margins / 2
        upper_bounds = np.array(self._image.data.shape) - self._margins / 2

        out_of_bounds = np.any(point < lower_bounds) or np.any(point >= upper_bounds)

        if out_of_bounds:
            logger.warning(
                "Discarded point ({}, {}, {}). Too close to image border.".format(
                    np.round(point[2]),
                    np.round(point[1]),
                    np.round(point[0]),
                )
            )
            return False
        else:
            return True

    def extract_beads(self, points: ArrayLike) -> List[Calibrated3DImage]:
        beads = []
        for point in points:
            if self._in_margins(point):
                closest_peak = self._find_closest_peak(point)
                bead_data = self._extract_rough_crop(closest_peak)
                bead = Calibrated3DImage(data=bead_data, spacing=self._image.spacing)
                bead.offset = tuple(np.array(closest_peak) - np.array(bead_data.shape) // 2)
                beads.append(bead)

        return beads

    def _create_slices(self, point: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
        slices = []
        for coordinate, margin in zip(point, self._margins):
            start = int(coordinate - margin // 2)
            end = start + int(margin)
            slices.append(slice(start, end))

        return tuple(slices)

    def _extract_rough_crop(self, point: Tuple[int, int, int]):
        z_slice, y_slice, x_slice = self._create_slices(point)
        return self._image.data[z_slice, y_slice, x_slice]

    def _find_closest_peak(self, point: Tuple[int, int, int]) -> Tuple[int, int, int]:
        crop = self._extract_rough_crop(point)
        peak_offset = self._compute_peak_offset(crop)
        return (
            int(point[0] + peak_offset[0]),
            int(point[1] + peak_offset[1]),
            int(point[2] + peak_offset[2]),
        )

    def _compute_peak_offset(self, crop: ArrayLike) -> Tuple[int, int, int]:
        from skimage.filters import gaussian

        peak_coordinates = np.unravel_index(
            np.argmax(gaussian(crop, 2, mode="constant", preserve_range=True)),
            crop.shape,
        )
        new_coords = np.array(peak_coordinates) - np.array(crop.shape) // 2
        return tuple(new_coords)
