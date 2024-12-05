from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import click
import matplotlib.pyplot as plt
import numpy as np

from iohub.ngff import open_ome_zarr
from skimage.exposure import rescale_intensity
from skimage.measure import find_contours

from biahub.analysis.AnalysisSettings import Render2DSettings
from biahub.cli.parsing import config_filepath, output_filepath
from biahub.cli.utils import yaml_to_model

if TYPE_CHECKING:
    from iohub.ngff import ImageArray
    from numpy.typing import NDArray

    from biahub.analysis.AnalysisSettings import (
        ContourChannelRender2DSettings,
        ImageChannelRender2DSettings,
    )


def render_image(image: NDArray, settings: ImageChannelRender2DSettings) -> NDArray:
    """Render an image in RGBA and scale to RGB by A."""
    image = image.astype(np.float32)
    if settings.clim is not None:
        if settings.clim_mode == "absolute":
            low, high = settings.clim
        elif settings.clim_mode == "percentile":
            low, high = np.percentile(image, settings.clim)
    else:
        low, high = np.min(image), np.max(image)
    image = rescale_intensity(image, in_range=(low, high), out_range=(0, 1))
    image = settings.lut(image, gamma=settings.gamma)
    image[..., 3] * settings.alpha
    return image[..., :3] * image[..., 3:]


@click.command()
@output_filepath()
@config_filepath()
def render_2d(config_filepath: Path, output_filepath: Path) -> None:
    """Render a 2D slice in the given LUT."""
    settings = yaml_to_model(config_filepath, Render2DSettings)

    def _slice_tczyx(array: ImageArray, channel_index: int) -> NDArray:
        crop = array[
            settings.time_index,
            channel_index,
            slice(*settings.z_range),
            slice(*settings.y_range),
            slice(*settings.x_range),
        ]
        return np.squeeze(crop)

    rendered_images: list[NDArray] = []
    labels_and_settings: list[tuple[NDArray, ContourChannelRender2DSettings]] = []
    for ch_setting in settings.channels:
        with open_ome_zarr(ch_setting.path, layout="fov") as dataset:
            ch_idx = dataset.get_channel_index(ch_setting.name)
            crop = _slice_tczyx(dataset[ch_setting.multiscal_level], ch_idx)
        crop_and_settings = (crop, ch_setting)
        if ch_setting.channel_type == "image":
            rendered_images.append(render_image(crop, settings))
        elif ch_setting.channel_type == "contour":
            labels_and_settings.append(crop_and_settings)

    # render images
    image = np.sum(rendered_images, axis=0).clip(0, 1)

    # render contours
    figure, ax = plt.subplots(figsize=settings.figsize, dpi=300)
    ax.imshow(image)
    for idx, (labels, ct_settings) in enumerate(labels_and_settings):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == 0:
                continue
            mask = labels == label
            contours = find_contours(mask, level=0.5)
            for contour in contours:
                ax.plot(
                    contour[:, 1],
                    contour[:, 0],
                    linewidth=ct_settings.linewidth,
                    color=ct_settings.lut(idx),
                )
    # Use image coordinates, not data coordinates
    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    figure.savefig(output_filepath, bbox_inches="tight", pad_inches=0)
