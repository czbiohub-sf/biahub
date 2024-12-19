import warnings

from pathlib import Path
from typing import Annotated, Literal, NamedTuple, Optional, Union

import numpy as np
import torch

from cmap import Colormap
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


# All settings classes inherit from MyBaseModel, which forbids extra parameters to guard against typos
class MyBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProcessingSettings(MyBaseModel):
    fliplr: Optional[bool] = False
    flipud: Optional[bool] = False


class DeskewSettings(MyBaseModel):
    pixel_size_um: PositiveFloat
    ls_angle_deg: PositiveFloat
    px_to_scan_ratio: Optional[PositiveFloat] = None
    scan_step_um: Optional[PositiveFloat] = None
    keep_overhang: bool = False
    average_n_slices: PositiveInt = 3

    @field_validator("ls_angle_deg")
    @classmethod
    def ls_angle_check(cls, v):
        if v < 0 or v > 45:
            raise ValueError("Light sheet angle must be be between 0 and 45 degrees")
        return round(float(v), 2)

    @field_validator("px_to_scan_ratio")
    @classmethod
    def px_to_scan_ratio_check(cls, v):
        if v is not None:
            return round(float(v), 3)

    def __init__(self, **data):
        if data.get("px_to_scan_ratio") is None:
            if data.get("scan_step_um") is not None:
                data["px_to_scan_ratio"] = round(
                    data["pixel_size_um"] / data["scan_step_um"], 3
                )
            else:
                raise ValueError(
                    "If px_to_scan_ratio is not provided, both pixel_size_um and scan_step_um must be provided"
                )
        super().__init__(**data)


class RegistrationSettings(MyBaseModel):
    source_channel_names: list[str]
    target_channel_name: str
    affine_transform_zyx: list
    keep_overhang: bool = False
    interpolation: str = "linear"
    time_indices: Union[NonNegativeInt, list[NonNegativeInt], Literal["all"]] = "all"

    @field_validator("affine_transform_zyx")
    @classmethod
    def check_affine_transform(cls, v):
        if not isinstance(v, list) or len(v) != 4:
            raise ValueError("The input array must be a list of length 3.")

        for row in v:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError("Each row of the array must be a list of length 3.")

        try:
            # Try converting the list to a 3x3 ndarray to check for valid shape and content
            np_array = np.array(v)
            if np_array.shape != (4, 4):
                raise ValueError("The array must be a 3x3 ndarray.")
        except ValueError:
            raise ValueError("The array must contain valid numerical values.")

        return v


class PsfFromBeadsSettings(MyBaseModel):
    axis0_patch_size: PositiveInt = 101
    axis1_patch_size: PositiveInt = 101
    axis2_patch_size: PositiveInt = 101


class DeconvolveSettings(MyBaseModel):
    regularization_strength: PositiveFloat = 0.001


class CharacterizeSettings(MyBaseModel):
    block_size: list[NonNegativeInt] = (64, 64, 32)
    blur_kernel_size: NonNegativeInt = 3
    nms_distance: NonNegativeInt = 32
    min_distance: NonNegativeInt = 50
    threshold_abs: PositiveFloat = 200.0
    max_num_peaks: NonNegativeInt = 2000
    exclude_border: list[NonNegativeInt] = (5, 10, 5)
    device: str = "cuda"
    patch_size: tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None = None
    axis_labels: list[str] = ["AXIS0", "AXIS1", "AXIS2"]

    @field_validator("device")
    @classmethod
    def check_device(cls, v):
        return "cuda" if torch.cuda.is_available() else "cpu"


class ConcatenateSettings(MyBaseModel):
    concat_data_paths: list[str]
    time_indices: Union[int, list[int], Literal["all"]] = "all"
    channel_names: list[Union[str, list[str]]]
    X_slice: Union[list[int], Literal["all"]] = "all"
    Y_slice: Union[list[int], Literal["all"]] = "all"
    Z_slice: Union[list[int], Literal["all"]] = "all"
    chunks_czyx: Union[Literal[None], list[int]] = None

    @field_validator("concat_data_paths")
    @classmethod
    def check_concat_data_paths(cls, v):
        if not isinstance(v, list) or not all(isinstance(path, str) for path in v):
            raise ValueError("concat_data_paths must be a list of positions.")
        return v

    @field_validator("channel_names")
    @classmethod
    def check_channel_names(cls, v):
        if not isinstance(v, list) or not all(isinstance(name, (str, list)) for name in v):
            raise ValueError("channel_names must be a list of strings or lists of strings.")
        return v

    @field_validator("X_slice", "Y_slice", "Z_slice")
    @classmethod
    def check_slices(cls, v):
        if v != "all" and (
            not isinstance(v, list)
            or len(v) != 2
            or not all(isinstance(i, int) and i >= 0 for i in v)
        ):
            raise ValueError("Slices must be 'all' or lists of two non-negative integers.")
        return v

    @field_validator("chunks_czyx")
    @classmethod
    def check_chunk_size(cls, v):
        if v is not None and (
            not isinstance(v, list) or len(v) != 4 or not all(isinstance(i, int) for i in v)
        ):
            raise ValueError("chunks_czyx must be a list of 4 integers (C, Z, Y, X)")
        return v


class StabilizationSettings(MyBaseModel):
    stabilization_estimation_channel: str
    stabilization_type: Literal["z", "xy", "xyz"]
    stabilization_channels: list
    affine_transform_zyx_list: list
    time_indices: Union[NonNegativeInt, list[NonNegativeInt], Literal["all"]] = "all"

    @field_validator("affine_transform_zyx_list")
    @classmethod
    def check_affine_transform_zyx_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("affine_transform_list must be a list")

        for arr in v:
            arr = np.array(arr)
            if arr.shape != (4, 4):
                raise ValueError("Each element in affine_transform_list must be a 4x4 ndarray")

        return v


class StitchSettings(MyBaseModel):
    channels: Optional[list[str]] = None
    preprocessing: Optional[ProcessingSettings] = None
    postprocessing: Optional[ProcessingSettings] = None
    column_translation: Optional[list[float, float]] = None
    row_translation: Optional[list[float, float]] = None
    total_translation: Optional[dict[str, list[float, float]]] = None

    def __init__(self, **data):
        if data.get("total_translation") is None:
            if any(
                (
                    data.get("column_translation") is None,
                    data.get("row_translation") is None,
                )
            ):
                raise ValueError(
                    "If total_translation is not provided, both column_translation and row_translation must be provided"
                )
            else:
                warnings.warn(
                    "column_translation and row_translation are deprecated. Use total_translation instead.",
                    DeprecationWarning,
                )
        super().__init__(**data)


class IndexRange(NamedTuple):
    """Index range for a single axis."""

    start: NonNegativeInt
    stop: NonNegativeInt


class BaseChannelRender2DSettings(MyBaseModel):
    """Settings for rendering a single channel in 2D.
    Base model for a discriminated union."""

    path: Path
    name: str
    multiscale_level: str = "0"


PositiveFloatLE1 = Annotated[float, Field(gt=0.0, le=1.0)]


class ImageChannelRender2DSettings(BaseChannelRender2DSettings):
    """Settings for rendering an image as a bitmap in 2D."""

    lut: Colormap
    channel_type: Literal["image"]
    clim: tuple[float, float] | None
    clim_mode: Literal["absolute", "percentile"] | None
    alpha: PositiveFloatLE1 = 1.0
    gamma: PositiveFloatLE1 = 1.0

    @model_validator(mode="after")
    @classmethod
    def check_clim_mode(cls, data):
        if data.clim_mode is not None and data.clim is None:
            raise ValueError("clim_mode requires clim to be set")
        return data


class ContourChannelRender2DSettings(BaseChannelRender2DSettings):
    """Settings for rendering contours as vectors in 2D."""

    channel_type: Literal["contour"]
    linewidth: PositiveFloat = 2.0


ChannelRender2DSettings = Annotated[
    ImageChannelRender2DSettings | ContourChannelRender2DSettings,
    Field(discriminator="channel_type"),
]


class Render2DSettings(MyBaseModel):
    """Settings for rendering 2D images."""

    time_index: NonNegativeInt
    channels: list[ChannelRender2DSettings]
    z_range: IndexRange
    y_range: IndexRange
    x_range: IndexRange
    figsize: tuple[PositiveFloat, PositiveFloat] = (4, 4)
    dpi: PositiveInt = 300
