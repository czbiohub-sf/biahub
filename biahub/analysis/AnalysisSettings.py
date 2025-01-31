import warnings

from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch

from pydantic import (
    BaseModel,
    ConfigDict,
    ImportString,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    validator,
)


# All settings classes inherit from MyBaseModel, which forbids extra parameters to guard against typos
class MyBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProcessingFunctions(BaseModel):
    function: str  # Updated from ImportString for simplicity
    input_arrays: list[str]
    kwargs: Dict[str, Any] = {}


class TrackingSettings(MyBaseModel):
    z_slices: Tuple[int, int]
    tracking_config: Dict[str, Any] = {}  # Define as raw dict
    vs_projection_function: ProcessingFunctions = None
    preprocessing_functions: Dict[str, ProcessingFunctions] = {}
    tracking_functions: Dict[str, ProcessingFunctions] = {}


class ProcessingImportFuncSettings(MyBaseModel):
    processing_functions: list[ProcessingFunctions] = []


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


def get_valid_eval_args():
    """Attempt to import cellpose and retrieve valid eval arguments."""
    try:
        from cellpose import models

        return models.CellposeModel.eval.__code__.co_varnames[
            : models.CellposeModel.eval.__code__.co_argcount
        ]
    except ImportError:
        raise ImportError(
            "The 'cellpose' package is required to validate 'eval_args' in cellpose model configurations. "
            "Please install it to proceed with cellpose-related configurations."
        )


class PreprocessingFunctions(BaseModel):
    function: ImportString
    channel: str
    kwargs: Dict[str, Any] = {}


class SegmentationModel(BaseModel):
    path_to_model: str
    eval_args: Dict[str, Any]
    z_slice_2D: Optional[int] = None
    preprocessing: list[PreprocessingFunctions] = []

    @validator("eval_args", pre=True)
    def validate_eval_args(cls, value):
        # Retrieve valid arguments dynamically if cellpose is required
        valid_args = get_valid_eval_args()

        # Check that all keys in eval_args are valid arguments for cellpose_eval
        invalid_args = [arg for arg in value.keys() if arg not in valid_args]
        if invalid_args:
            raise ValueError(
                f"Invalid eval arguments provided: {invalid_args}. Allowed arguments are {valid_args}"
            )

        return value

    @validator("z_slice_2D")
    def check_z_slice_with_do_3D(cls, z_slice_2D, values):
        # Only run this check if z_slice is provided (not None) and do_3D exists in eval_args
        if z_slice_2D is not None:
            eval_args = values.get("eval_args", {})
            do_3D = eval_args.get("do_3D", None)
            if do_3D:
                raise ValueError(
                    "If 'z_slice_2D' is provided, 'do_3D' in 'eval_args' must be set to False."
                )
        return z_slice_2D


class SegmentationSettings(BaseModel):
    models: Dict[str, SegmentationModel]
    model_config = {"extra": "forbid", "protected_namespaces": ()}
