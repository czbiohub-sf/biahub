from pathlib import Path
from typing import Optional, Tuple

import dask.array as da
import napari
import numpy as np
import scipy.ndimage as ndi

from napari_animation import Animation
from PIL import Image, ImageDraw, ImageFont


def get_contours(labels, thickness: int, background_label: int):
    """Computes the contours of a 2D label image.

    Parameters
    ----------
    labels : array of integers
        An input labels image.
    thickness : int
        It controls the thickness of the inner boundaries. The outside thickness is always 1.
        The final thickness of the contours will be `thickness + 1`.
    background_label : int
        That label is used to fill everything outside the boundaries.

    Returns
    -------
        A new label image in which only the boundaries of the input image are kept.
    """
    struct_elem = ndi.generate_binary_structure(labels.ndim, 1)

    thick_struct_elem = ndi.iterate_structure(struct_elem, thickness).astype(bool)

    dilated_labels = ndi.grey_dilation(labels, footprint=struct_elem)
    eroded_labels = ndi.grey_erosion(labels, footprint=thick_struct_elem)
    not_boundaries = dilated_labels == eroded_labels

    contours = labels.copy()
    contours = da.where(not_boundaries, background_label, contours)

    return contours


def suggest_contrast_limits(intensity_array):
    """
    Suggest contrast limits for an array of pixel intensities.

    Parameters:
    intensity_array (numpy array): A flattened array of pixel intensity values (0-255).

    Returns:
    tuple: Suggested lower and upper contrast limits (1st percentile, 99th percentile).
    """
    if intensity_array.size == 0:
        raise ValueError("The intensity array is empty.")

    lower_limit = np.percentile(intensity_array, 1)
    upper_limit = np.percentile(intensity_array, 99)

    return lower_limit, upper_limit


def simple_recording(
    viewer: napari.Viewer,
    output_path: Path,
    loop_axes: list[Tuple[int, Tuple[Optional[int], Optional[int]]]],
    capture_factors: list[int],
    fps: int = 60,
    buffer_frames: int = 10,
) -> None:
    """
    Generate a recording looping over specified axes in the given sequence.

    Parameters:
    - viewer: napari.Viewer
        The napari viewer instance.
    - output_path: Path
        Path to save the final animation file.
    - loop_axes: List[Tuple[int, Tuple[Optional[int], Optional[int]]]]
        List of axes and their ranges (axis_index, (min_value, max_value)).
        Example: [(0, (None, None))] for looping over time, [(1, (None, None))] for channels, [(2, (None, None))] for z-axis.
        Use None for min/max values to automatically infer them based on the layer data shape.
    - capture_factors: List[int]
        List of time scaling factors that correspond to each axis in `loop_axes`. Must be the same length as `loop_axes`.
    - fps: int, default=60
        Frames per second for the final output.

    Usage:
    Record a time-lapse video looping over time, assuming the input is TCZYX:
    - simple_recording(viewer, Path("output.mp4"), [(0, (None, None))], [1], fps=60)
    """
    viewer.dims.set_point(0, 0)
    # Check that capture_factors matches the length of loop_axes
    if len(loop_axes) != len(capture_factors):
        raise ValueError("The length of capture_factors must match the length of loop_axes.")

    # Reset viewer to the starting point of the first axis
    animation = Animation(viewer)
    animation.capture_keyframe()

    # Loop over specified axes in sequence
    for (axis, (min_val, max_val)), capture_factor in zip(loop_axes, capture_factors):
        # Automatically infer min value if None is provided
        if min_val is None:
            min_val = 0

        # Automatically infer max value if None is provided
        if max_val is None:
            max_val = viewer.layers[0].data.shape[axis] - 1

        # Loop over the axis and capture keyframes at each point
        for i in range(min_val, max_val + 1):
            viewer.dims.set_point(axis, i)
            animation.capture_keyframe(i * capture_factor)

        # Capture buffer frames for smooth transition
        for _ in range(buffer_frames):
            animation.capture_keyframe(i * capture_factor)

    # Save the animation
    animation.animate(output_path, fps=fps, canvas_only=True)


def add_scale_bar(
    image_path: str,
    output_path: str,
    pixel_size: float,
    bar_length: float,
    bar_height: int = 5,
    bar_color: str = 'white',
    label_color: str = 'white',
    position: Tuple[int, int] = (-50, -50),
    font_size: int = 15,
) -> None:
    """
    Add a scale bar to the given image.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_path : str
        Path to save the image with scale bar.
    pixel_size : float
        Size of one pixel in units (e.g., micrometers).
    bar_length : float
        Length of the scale bar in the same units as pixel_size.
    bar_height : int, optional
        Height of the scale bar in pixels. Default is 5 pixels.
    bar_color : str, optional
        Color of the scale bar. Default is 'white'.
    label_color : str, optional
        Color of the label text. Default is 'white'.
    position : tuple of int, optional
        Tuple (x, y) indicating the top-left position of the scale bar. Default is (10, 10).

    Examples
    --------
    >>> add_scale_bar("input.jpg", "output_with_scale.jpg", 0.5, 100)
    """

    # Calculate the length of the scale bar in pixels
    bar_length_pixels = int(bar_length / pixel_size)

    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    if position[0] < 0:
        position = (image.width + position[0] - bar_length_pixels, position[1])

    if position[1] < 0:
        position = (position[0], image.height + position[1])

    # Draw the scale bar
    bar_top_left = position
    bar_bottom_right = (position[0] + bar_length_pixels, position[1] + bar_height)
    draw.rectangle([bar_top_left, bar_bottom_right], fill=bar_color)

    # Draw the scale bar label (optional, but good to have)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    label_position = (bar_top_left[0], bar_top_left[1] - 20)
    draw.text(label_position, f"{bar_length} units", fill=label_color, font=font)

    # Save the image
    image.save(output_path)
