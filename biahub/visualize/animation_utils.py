from enum import Enum, auto
from pathlib import Path
from typing import Optional

import dask.array as da
import napari
import numpy as np
import scipy.ndimage as ndi

from napari_animation import Animation


class ElementPosition(Enum):
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()


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

    Parameters
    ----------
    intensity_array : numpy array
        A flattened array of pixel intensity values (0-255).

    Returns
    -------
    tuple
        Suggested lower and upper contrast limits (1st percentile, 99th percentile).
    """
    if intensity_array.size == 0:
        raise ValueError("The intensity array is empty.")

    lower_limit = np.percentile(intensity_array, 1)
    upper_limit = np.percentile(intensity_array, 99)

    return lower_limit, upper_limit


def _clear_overlays(viewer: napari.Viewer, layer_name: str = "scale_bar") -> None:
    """
    Remove all overlay layers with the specified name from the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    layer_name : str, default="scale_bar"
        Name of the layers to remove.
    """
    # Create a list of layers to remove (to avoid modifying list while iterating)
    layers_to_remove = [layer for layer in viewer.layers if layer.name == layer_name]
    for layer in layers_to_remove:
        viewer.layers.remove(layer)


def _clear_dim_callbacks(viewer: napari.Viewer) -> None:
    """
    Clear all custom callbacks from the viewer's dimension events.
    Preserves napari's internal callbacks.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    """
    # Get all callbacks
    callbacks = viewer.dims.events.current_step.callbacks

    # Keep only internal napari callbacks (first two callbacks)
    internal_callbacks = list(callbacks)[:2]

    # Disconnect all existing callbacks
    for callback in callbacks:
        viewer.dims.events.current_step.disconnect(callback)

    # Restore internal callbacks
    for callback in internal_callbacks:
        viewer.dims.events.current_step.connect(callback)


def _create_positioned_element(
    viewer: napari.Viewer,
    position: ElementPosition,
    margin_factor: float = 0.05,
    text_size: int = 20,
    color: str = "white",
    text: Optional[str] = None,
    layer_name: str = "overlay_element",
    edge_width: int = 0,
    line_length: Optional[float] = None,
) -> None:
    """
    Base function to create positioned elements (text overlays, scale bars, etc.) in the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    position : ElementPosition
        Position of the element.
    margin_factor : float, default=0.05
        Margin from the edge as a fraction of image dimensions.
    text_size : int, default=20
        Size of the text.
    color : str, default="white"
        Color of the element and text.
    text : Optional[str], default=None
        Text to display. If None, no text is shown.
    layer_name : str, default="overlay_element"
        Name of the layer.
    edge_width : int, default=0
        Width of the line/edge. Set to 0 for invisible lines (text-only overlays).
    line_length : Optional[float], default=None
        Length of the line in physical units. If None, a minimal line is created.
    """
    # Get dimensions and scale from the first layer
    scale = viewer.layers[0].scale
    data_shape = viewer.layers[0].data.shape

    # Calculate image dimensions in physical units
    Y = data_shape[-2] * scale[-2]
    X = data_shape[-1] * scale[-1]

    # Calculate margins in physical units
    margin_y = Y * margin_factor
    margin_x = X * margin_factor

    # Calculate y position
    if position in [ElementPosition.TOP_LEFT, ElementPosition.TOP_RIGHT]:
        y_pos = margin_y
    else:  # BOTTOM positions
        y_pos = Y - margin_y

    # Calculate x positions for line
    if line_length is not None:
        if position in [ElementPosition.TOP_LEFT, ElementPosition.BOTTOM_LEFT]:
            x_start = margin_x
            x_end = x_start + line_length
        else:  # RIGHT positions
            x_end = X - margin_x
            x_start = x_end - line_length
        lines = np.array([[y_pos, x_start], [y_pos, x_end]])
    else:
        # For text-only overlays, create a minimal line at the correct position
        if position in [ElementPosition.TOP_LEFT, ElementPosition.BOTTOM_LEFT]:
            x_pos = margin_x
        else:  # RIGHT positions
            x_pos = X - margin_x
        lines = np.array([[y_pos, x_pos], [y_pos, x_pos + 1]])

    # Map position to napari's anchor values
    anchor_map = {
        ElementPosition.TOP_LEFT: "upper_left",
        ElementPosition.TOP_RIGHT: "upper_right",
        ElementPosition.BOTTOM_LEFT: "lower_left",
        ElementPosition.BOTTOM_RIGHT: "lower_right",
    }

    # Add the shape layer
    text_parameters = (
        {
            "text": "label",
            "size": text_size,
            "color": [color],
            "anchor": anchor_map[position],
        }
        if text is not None
        else {}
    )

    properties = {"label": [text]} if text is not None else {}

    return viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=edge_width,
        edge_color=[color],
        properties=properties,
        text=text_parameters,
        name=layer_name,
    )


def add_scale_bar(
    viewer: napari.Viewer,
    scale_bar_length: float,
    position: ElementPosition = ElementPosition.BOTTOM_RIGHT,
    margin_factor: float = 0.05,
    line_width: int = 5,
    text_size: Optional[int] = None,
    color: str = "white",
) -> napari.layers.Shapes:
    """
    Add a scale bar to the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    scale_bar_length : float
        Length of scale bar in micrometers.
    position : ElementPosition, default=ElementPosition.BOTTOM_RIGHT
        Position of the scale bar.
    margin_factor : float, default=0.05
        Margin from the edge as a fraction of image dimensions.
    line_width : int, default=5
        Width of the scale bar line in pixels.
    text_size : Optional[int], default=None
        Size of the scale bar text. If None, no text is shown.
    color : str, default="white"
        Color of the scale bar and text.

    Returns
    -------
    napari.layers.Shapes
        The scale bar layer.

    Usage
    -----
    scale_bar = add_scale_bar(viewer, 100, position=ElementPosition.BOTTOM_RIGHT, margin_factor=0.05, line_width=5, text_size=20, color="white")
    """
    text = f"{scale_bar_length}µm" if text_size is not None else None

    return _create_positioned_element(
        viewer=viewer,
        position=position,
        margin_factor=margin_factor,
        text_size=text_size or 14,
        color=color,
        text=text,
        layer_name="scale_bar",
        edge_width=line_width,
        line_length=scale_bar_length,
    )


def add_text_overlay(
    viewer: napari.Viewer,
    time_axis: Optional[int] = 0,  # None to disable time display
    z_axis: Optional[int] = 1,  # None to disable z display
    position: ElementPosition = ElementPosition.TOP_LEFT,
    margin_factor: float = 0.05,
    text_size: int = 20,
    color: str = "white",
    layer_name: str = "time_z_overlay",
) -> None:
    """
    Add a text overlay to the viewer showing time and/or z position.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    time_axis : Optional[int], default=0
        Index of the time axis. Set to None to disable time display.
    z_axis : Optional[int], default=1
        Index of the z axis. Set to None to disable z display.
    position : ElementPosition, default=ElementPosition.TOP_LEFT
        Position of the text overlay.
    margin_factor : float, default=0.05
        Margin from the edge as a fraction of image dimensions.
    text_size : int, default=20
        Size of the text.
    color : str, default="white"
        Color of the text.
    layer_name : str, default="time_z_overlay"
        Name of the overlay layer.

    Returns
    -------
    napari.layers.Shapes
        The text overlay layer.

    Usage
    -----
    # Show both time and z
    text_overlay = add_text_overlay(viewer, time_axis=0, z_axis=1)

    # Show only time
    text_overlay = add_text_overlay(viewer, time_axis=0, z_axis=None)

    # Show only z
    text_overlay = add_text_overlay(viewer, time_axis=None, z_axis=1)
    """
    # Clear existing overlays and callbacks
    _clear_overlays(viewer, layer_name)
    _clear_dim_callbacks(viewer)

    scale = viewer.layers[0].scale

    def update_overlay():
        if time_axis is None and z_axis is None:
            return

        parts = []
        if time_axis is not None:
            # Convert time to hours:minutes using the scale
            total_minutes = viewer.dims.current_step[time_axis] * scale[time_axis]
            hh = int(total_minutes // 60)
            mm = int(total_minutes % 60)
            parts.append(f"t = {hh}h{mm:02d}m")

        if z_axis is not None:
            zz = viewer.dims.current_step[z_axis] * scale[z_axis]
            parts.append(f"z = {zz:.2f}µm")

        text = ", ".join(parts)
        viewer.layers[layer_name].properties = {"label": [text]}

    layer = _create_positioned_element(
        viewer=viewer,
        position=position,
        margin_factor=margin_factor,
        text_size=text_size,
        color=color,
        text="",  # Initial empty text
        layer_name=layer_name,
        edge_width=0,
    )

    # Connect the update function
    event_callback = viewer.dims.events.current_step.connect(update_overlay)
    update_overlay()  # Initial update

    return layer, event_callback  # Return both the layer and the event callback


def simple_recording(
    viewer: napari.Viewer,
    output_path: Path,
    loop_axes: list[tuple[int, tuple[Optional[int], Optional[int]], Optional[float]]],
    z_focal_plane: Optional[int] = None,  # New argument to specify z
    fps: int = 60,
    buffer_duration: float = 0.5,
    default_duration: float = 5.0,
) -> None:
    """
    Generate a recording looping over specified axes in the given sequence.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    output_path : Path
        Path to save the final animation file.
    loop_axes : List[Tuple[int, Tuple[Optional[int], Optional[int]], Optional[float]]]
        List of (axis_index, (min_value, max_value), duration).
        Any value can be None:
        - None for min/max: automatically use full axis range
        - None for duration: use default_duration
    fps : int, default=60
        Frames per second for the final output.
    buffer_duration : float, default=0.5
        Duration in seconds to hold at start and end of each axis transition.
    default_duration : float, default=5.0
        Default duration in seconds when None is provided.

    Usage
    -----
    Assuming TZYX order, loop over time from t=10 to t=100 in 5 seconds
    ```python
    simple_recording(viewer, 'test.mp4', [(0, (10, 100), 5)])
    ```
    Loop over two axes t and z with 5 seconds duration each
    ```python
    simple_recording(viewer, 'test.mp4', [(0, (None, None), 5), (1, (None, None), 5)])
    ```
    """
    animation = Animation(viewer)
    buffer_frames = int(buffer_duration * fps)

    # If fixed_z is provided, set z-axis first
    if z_focal_plane is not None:
        z_axis = loop_axes[1][0]  # Assuming second tuple corresponds to z
        viewer.dims.set_current_step(z_axis, z_focal_plane)  # Explicitly fix z

    # Loop over specified axes in sequence
    for axis, (min_val, max_val), duration in loop_axes:
        axis_size = viewer.layers[0].data.shape[axis]
        # Use full range if min/max is None
        actual_min = 0 if min_val is None else min_val
        actual_max = (axis_size - 1) if max_val is None else max_val
        actual_duration = default_duration if duration is None else duration

        # Calculate frames
        n_frames = int(actual_duration * fps)
        positions = np.linspace(
            actual_min, actual_max, n_frames, dtype=int
        )  # Ensure integer steps

        # Start at initial position
        viewer.dims.set_current_step(axis, actual_min)
        animation.capture_keyframe()

        # Capture intermediate frames
        for pos in positions[1:]:
            viewer.dims.set_current_step(axis, pos)
            viewer.dims.events.current_step(value=viewer.dims.current_step)  # Force UI update
            animation.capture_keyframe(1)

        # Add buffer frames at the end of each transition
        animation.capture_keyframe(buffer_frames)

    # Save the animation
    animation.animate(output_path, fps=fps, canvas_only=True)
