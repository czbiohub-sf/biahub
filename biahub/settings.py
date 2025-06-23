import warnings

from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
    validator,
)


# All settings classes inherit from MyBaseModel, which forbids extra parameters to guard against typos
class MyBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectPeaksSettings(MyBaseModel):
    threshold_abs: float = 110
    nms_distance: int = 16
    min_distance: int = 0
    block_size: list[int] = [8, 8, 8]


class BeadsMatchSettings(MyBaseModel):
    algorithm: Literal["hungarian", "match_descriptor", "mutual_information"] = "hungarian"
    filter_angle_threshold: float = 0
    cost_threshold: float = 0.10
    max_ratio: float = 0.8
    knn_k: int = 5
    cross_check: bool = True
    distance_metric: Literal["euclidean", "cosine", "cityblock"] = "euclidean"
    source_peaks_settings: Optional[DetectPeaksSettings] = Field(
        default_factory=DetectPeaksSettings
    )
    target_peaks_settings: Optional[DetectPeaksSettings] = Field(
        default_factory=DetectPeaksSettings
    )
    t_reference: Literal["first", "previous"] = "first"


class PhaseCrossCorrSettings(MyBaseModel):
    maximum_shift: float = 1.2
    normalization: bool = False
    t_reference: Literal["first", "previous"] = "first"


class FocusFindingSettings(MyBaseModel):
    average_across_wells: bool = False
    average_across_wells_method: Literal["mean", "median"] = "mean"
    skip_beads_fov: str = "0"
    crop_size_xy: list[int, int] = [800, 800]


class StackRegSettings(MyBaseModel):
    crop_size_xy: list[int, int] = [800, 800]
    skip_beads_fov: str = "0"
    focus_finding_settings: Optional[FocusFindingSettings] = Field(
        default_factory=FocusFindingSettings
    )
    t_reference: Literal["first", "previous"] = "first"


class EvalTransformSettings(MyBaseModel):
    validation_window_size: int = 10
    validation_tolerance: float = 1000.0
    interpolation_window_size: int = 3
    interpolation_type: Literal["linear", "cubic"] = "linear"


class AffineTransformSettings(MyBaseModel):
    transform_type: Literal["euclidean", "similarity", "affine"] = "euclidean"
    approx_transform: list = np.eye(4).tolist()

    @field_validator("approx_transform")
    @classmethod
    def check_affine_transform_zyx_list(cls, v):
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("approx_transform must be a list")
            arr = np.array(v)
            if arr.shape != (4, 4):
                raise ValueError("approx_transform must be a 4x4 array")

        return v


class AntsRegistrationSettings(MyBaseModel):
    sobel_filter: bool = False


class ManualRegistrationSettings(MyBaseModel):
    time_index: int = 0
    affine_90degree_rotation: int = 0


class EstimateRegistrationSettings(MyBaseModel):
    target_channel_name: str
    source_channel_name: str
    estimation_method: Literal["manual", "beads", "ants"] = "manual"
    beads_match_settings: Optional[BeadsMatchSettings] = Field(
        default_factory=BeadsMatchSettings
    )
    focus_finding_settings: Optional[FocusFindingSettings] = Field(
        default_factory=FocusFindingSettings
    )

    affine_transform_settings: AffineTransformSettings = Field(
        default_factory=AffineTransformSettings
    )
    eval_transform_settings: Optional[EvalTransformSettings] = Field(
        default_factory=EvalTransformSettings
    )
    ants_registration_settings: Optional[AntsRegistrationSettings] = Field(
        default_factory=AntsRegistrationSettings
    )
    manual_registration_settings: Optional[ManualRegistrationSettings] = Field(
        default_factory=ManualRegistrationSettings
    )
    verbose: bool = False


class EstimateStabilizationSettings(MyBaseModel):
    stabilization_estimation_channel: str
    stabilization_channels: list
    stabilization_type: Literal["z", "xy", "xyz"]
    stabilization_method: Literal[
        "beads", "phase-cross-corr", "focus-finding"
    ] = "focus-finding"
    beads_match_settings: Optional[BeadsMatchSettings] = Field(
        default_factory=BeadsMatchSettings
    )
    phase_cross_corr_settings: Optional[PhaseCrossCorrSettings] = Field(
        default_factory=PhaseCrossCorrSettings
    )
    stack_reg_settings: Optional[StackRegSettings] = Field(default_factory=StackRegSettings)
    focus_finding_settings: Optional[FocusFindingSettings] = Field(
        default_factory=FocusFindingSettings
    )
    affine_transform_settings: AffineTransformSettings = Field(
        default_factory=AffineTransformSettings
    )
    eval_transform_settings: Optional[EvalTransformSettings] = Field(
        default_factory=EvalTransformSettings
    )
    verbose: bool = False


class ProcessingSettings(MyBaseModel):
    fliplr: Optional[bool] = False
    flipud: Optional[bool] = False
    rot90: Optional[int] = 0


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
    offset: float = 0.0
    gain: float = 1.0
    use_robust_1d_fwhm: bool = False
    fwhm_plot_type: Literal["1D", "3D"] = "3D"

    @field_validator("device")
    @classmethod
    def check_device(cls, v):
        return "cuda" if torch.cuda.is_available() else "cpu"


class ConcatenateSettings(MyBaseModel):
    concat_data_paths: list[str]
    time_indices: Union[int, list[int], Literal["all"]] = "all"
    channel_names: list[Union[str, list[str]]]
    X_slice: Union[list, list[Union[list, Literal["all"]]], Literal["all"]] = "all"
    Y_slice: Union[list, list[Union[list, Literal["all"]]], Literal["all"]] = "all"
    Z_slice: Union[list, list[Union[list, Literal["all"]]], Literal["all"]] = "all"
    chunks_czyx: Union[Literal[None], list[int]] = None
    ensure_unique_positions: Optional[bool] = False

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
    def check_slices(cls, v, info):
        if v == "all":
            return v

        if not isinstance(v, list):
            raise ValueError("Slice must be 'all' or a list.")

        # Check if it's a list of per-path slice specifications
        if any(
            isinstance(item, list) and any(isinstance(subitem, list) for subitem in item)
            for item in v
        ):
            # This is a list of per-path slice specifications
            # Each item should be a valid slice specification
            for item in v:
                if item == "all":
                    continue

                # Check if it's a simple [start, end] format
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and all(isinstance(i, int) for i in item)
                ):
                    if not all(i >= 0 for i in item):
                        raise ValueError("Slice indices must be non-negative integers.")
                    continue

                # Check if it's a list of slice ranges or mixed format
                if isinstance(item, list):
                    for subitem in item:
                        # Subitem can be 'all'
                        if subitem == "all":
                            continue

                        # Subitem can be a single slice range [start, end]
                        if (
                            isinstance(subitem, list)
                            and len(subitem) == 2
                            and all(isinstance(i, int) for i in subitem)
                        ):
                            if not all(i >= 0 for i in subitem):
                                raise ValueError(
                                    "Slice indices must be non-negative integers."
                                )
                            continue

                        # If we get here, the subitem is invalid
                        raise ValueError(
                            "Each slice subitem must be 'all' or a list of two non-negative integers [start, end]."
                        )
                else:
                    raise ValueError(
                        "Each item in a per-path slice list must be 'all' or a valid slice specification."
                    )
            return v

        # Check if it's a simple [start, end] format
        if len(v) == 2 and all(isinstance(i, int) for i in v):
            if not all(i >= 0 for i in v):
                raise ValueError("Slice indices must be non-negative integers.")
            return v

        # Check if it's a list of slice ranges or mixed format
        for item in v:
            # Item can be 'all'
            if item == "all":
                continue

            # Item can be a single slice range [start, end]
            if (
                isinstance(item, list)
                and len(item) == 2
                and all(isinstance(i, int) for i in item)
            ):
                if not all(i >= 0 for i in item):
                    raise ValueError("Slice indices must be non-negative integers.")
                continue

            # If we get here, the item is invalid
            raise ValueError(
                "Each slice item must be 'all' or a list of two non-negative integers [start, end]."
            )

        return v

    @field_validator("chunks_czyx")
    @classmethod
    def check_chunk_size(cls, v):
        if v is not None and (
            not isinstance(v, list) or len(v) != 4 or not all(isinstance(i, int) for i in v)
        ):
            raise ValueError("chunks_czyx must be a list of 4 integers (C, Z, Y, X)")
        return v

    @model_validator(mode="after")
    def validate_slice_lengths(self):
        # Get the length of concat_data_paths
        data_paths = self.concat_data_paths
        if not data_paths:
            return self

        # Check X_slice
        x_slice = self.X_slice
        if (
            isinstance(x_slice, list)
            and x_slice != "all"
            and len(x_slice) != len(data_paths)
            and not (len(x_slice) == 2 and all(isinstance(i, int) for i in x_slice))
        ):
            raise ValueError(
                f"X_slice must be 'all', a single slice specification, or a list with the same length as concat_data_paths ({len(data_paths)})"
            )

        # Check Y_slice
        y_slice = self.Y_slice
        if (
            isinstance(y_slice, list)
            and y_slice != "all"
            and len(y_slice) != len(data_paths)
            and not (len(y_slice) == 2 and all(isinstance(i, int) for i in y_slice))
        ):
            raise ValueError(
                f"Y_slice must be 'all', a single slice specification, or a list with the same length as concat_data_paths ({len(data_paths)})"
            )

        # Check Z_slice
        z_slice = self.Z_slice
        if (
            isinstance(z_slice, list)
            and z_slice != "all"
            and len(z_slice) != len(data_paths)
            and not (len(z_slice) == 2 and all(isinstance(i, int) for i in z_slice))
        ):
            raise ValueError(
                f"Z_slice must be 'all', a single slice specification, or a list with the same length as concat_data_paths ({len(data_paths)})"
            )

        return self


class StabilizationSettings(MyBaseModel):
    stabilization_estimation_channel: str
    stabilization_type: Literal["z", "xy", "xyz"]
    stabilization_method: Literal[
        "beads", "phase-cross-corr", "focus-finding"
    ] = "focus-finding"
    stabilization_channels: list
    affine_transform_zyx_list: list
    time_indices: Union[NonNegativeInt, list[NonNegativeInt], Literal["all"]] = "all"
    output_voxel_size: list[
        PositiveFloat, PositiveFloat, PositiveFloat, PositiveFloat, PositiveFloat
    ] = [1.0, 1.0, 1.0, 1.0, 1.0]

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
    affine_transform: Optional[dict[str, list]] = None

    def __init__(self, **data):
        if not any(
            (
                data.get("total_translation"),
                data.get("affine_transform"),
                all((data.get("column_translation"), data.get("row_translation"))),
            )
        ):
            raise ValueError(
                "One of affine_transform, total_translation or (column_translation, row_translation) must be provided"
            )
        if any((data.get("column_translation"), data.get("row_translation"))):
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
            z_slice_2D = 0
        return z_slice_2D


class SegmentationSettings(BaseModel):
    models: Dict[str, SegmentationModel]
    model_config = {"extra": "forbid", "protected_namespaces": ()}
