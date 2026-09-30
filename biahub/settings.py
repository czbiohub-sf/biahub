from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationInfo,
    field_validator,
    model_validator,
)


# All settings classes inherit from MyBaseModel, which forbids extra parameters to guard against typos
class MyBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectPeaksSettings(MyBaseModel):
    threshold_abs: float = 110
    nms_distance: int = 16
    min_distance: int = 0
    block_size: list[int] = [8, 8, 8]


class ProcessingFunctions(MyBaseModel):
    function: str
    input_channels: list[str] | None = None  # Optional
    kwargs: dict[str, Any] = {}
    per_timepoint: bool | None = True


class ProcessingImportFuncSettings(MyBaseModel):
    processing_functions: list[ProcessingFunctions] = []
    # When None, preserve the OME-Zarr version of the input store.
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None


class ProcessingInputChannel(MyBaseModel):
    path: Path | None = None
    channels: dict[str, list[ProcessingFunctions]]

    @field_validator("path")
    @classmethod
    def validate_path_not_plate(cls, v):
        if v is None:
            return v
        v = Path(v)
        if v.suffix != ".zarr":
            raise ValueError("Path must be a valid OME-Zarr dataset.")
        return v


class CellposeConfig(MyBaseModel):
    """Configuration for Cellpose segmentation used as input to tracking."""

    model_type: str = "nuclei"
    diameter: float = 80
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    gpu: bool = True
    min_size: int = 500
    input_channel: str = "nuclei_prediction"
    labels_sigma: float = 5.0


class ZSlicing(MyBaseModel):
    """How to SELECT the Z-planes used for tracking.

    The ``method`` decides which of the other fields are used; fields that belong to a
    different method are simply ignored (all have defaults). The block selects a
    z-window; the actual reduction to 2D is governed separately by
    ``TrackingSettings.output_mode`` (plus any Z-projection step, e.g. ``np.mean``, in
    ``input_images``).

    Methods
    -------
    all
        Use every plane (``slice(None)``).
    central
        Use an automatically centred window (see ``central_z_slice``).
    range
        Use the explicit ``range`` ``[start, stop]`` slice (falls back to all planes
        if ``range`` is left unset).
    focus
        Detect the in-focus plane per-FOV (waveorder ``focus_from_transverse_band``
        on ``focus_channel``) and take a fixed window of ``window_size`` planes around it,
        split ``frac_below``/``frac_above``.
    """

    method: Literal["all", "central", "range", "focus"] = "all"
    range: tuple[int, int] | None = None  # method: range
    window_size: int = 48  # method: focus (fixed window size, in z-planes)
    frac_below: float = 1 / 3  # method: focus
    frac_above: float = 2 / 3  # method: focus
    focus_channel: str | None = None  # method: focus -- channel focus-finding runs on


class TrackingSettings(MyBaseModel):
    target_channel: str = "nuclei_prediction"
    fov: str = "*/*/*"
    blank_frames_path: Path | None = None
    # 2D writes an output plate with Z=1 (input must be projected); 3D keeps the
    # selected z-window. Does not itself project the data.
    output_mode: Literal["2D", "3D"] = "2D"
    # Which Z-planes to select for tracking. See ZSlicing.
    z_slicing: ZSlicing = ZSlicing()
    input_images: list[ProcessingInputChannel]
    tracking_config: dict[str, Any] = {}
    segmentation_method: Literal["foreground_contour", "cellpose"] = "foreground_contour"
    cellpose_config: CellposeConfig | None = None
    # When None, preserve the OME-Zarr version of the input store.
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None

    @field_validator("blank_frames_path")
    @classmethod
    def validate_blank_frames_path(cls, v):
        if v is None:
            return v
        return Path(v)


class EdgeGraphSettings(BaseModel):
    method: Literal["knn", "radius", "full"] = "knn"
    k: int | None = None
    radius: float | None = None

    @model_validator(mode="after")
    def set_defaults_and_validate(self) -> "EdgeGraphSettings":
        if self.method == "knn":
            if self.k is None:
                self.k = 5  # set default
            self.radius = None  # ignore
        elif self.method == "radius":
            if self.radius is None:
                self.radius = 30.0  # set default
            self.k = None  # ignore
        elif self.method == "full":
            self.k = None
            self.radius = None
        return self


class CostMatrixSettings(MyBaseModel):
    weights: dict[str, float] = {
        "dist": 0.5,
        "edge_angle": 1.0,
        "edge_length": 1.0,
        "pca_dir": 0.0,
        "pca_aniso": 0.0,
        "edge_descriptor": 0.0,
    }
    normalize: bool = False


class HungarianMatchSettings(MyBaseModel):
    distance_metric: Literal["euclidean", "cosine", "cityblock"] = "euclidean"
    cost_threshold: float = 0.10
    max_ratio: float = 0.8
    cross_check: bool = False
    edge_graph_settings: EdgeGraphSettings = EdgeGraphSettings()
    cost_matrix_settings: CostMatrixSettings = CostMatrixSettings()


class MatchDescriptorSettings(MyBaseModel):
    distance_metric: Literal["euclidean", "cosine", "cityblock"] = "euclidean"
    max_ratio: float = 0.8
    cross_check: bool = False


class SpectralMatchSettings(MyBaseModel):
    """Leordeanu-Hebert pairwise-consistency matching (GraphMatcher algorithm "spectral").

    sigma is the tolerance, in voxels, on how well a pair of candidates must preserve
    pairwise distance to reinforce each other; rel_cut drops candidates scoring below this
    fraction of the top eigenvector entry (the precision/recall dial).
    """

    sigma: float = 3.0
    rel_cut: float = 0.5
    max_iter: int = 60


class SeedCorrectionSettings(MyBaseModel):
    """Correct a per-timepoint seed by bead displacement voting before estimating.

    For a series whose geometry drifts beyond the static seed's capture range while the
    bead field is too thin for the matchers to re-acquire from scratch. The moving volume
    is warped with the seed, beads are detected densely (`vote_peaks_settings`) and each
    votes for its displacement to every reference bead within `capture_radius`; the mean
    of the densest `cluster_radius`-ball of votes is the residual drift. "voteseed"
    composes that translation into the seed; "votefit" additionally fits an affine on the
    cluster's members, kept only if it lowers the peaks' median nearest-neighbour distance.
    Candidates only compete against the unchanged seed, so a bad vote cannot make it worse.
    """

    mode: Literal["none", "voteseed", "votefit"] = "none"
    vote_peaks_settings: DetectPeaksSettings = DetectPeaksSettings(
        threshold_abs=200.0, nms_distance=8, min_distance=0, block_size=[16, 16, 16]
    )
    capture_radius: float = 80.0
    cluster_radius: float = 10.0
    min_votes: int = 3


class VoteIcpSettings(MyBaseModel):
    """Tunables for `estimation_mode: vote_icp` (see `pointcloud.vote_icp_register`).

    The capture radius starts wide enough to reach a badly drifted seed and shrinks
    geometrically each iteration (default 80 -> 40 -> 20 -> 10, then held), so early
    rounds recover the bulk misalignment and late rounds only see unambiguous neighbours.
    `vote_peaks_settings` is the dense detection on the raw moving volume for the reach
    stage; a precision stage re-runs at short range with the pipeline's own
    `source_peaks_settings` and is kept only on a strict score win.
    """

    vote_peaks_settings: DetectPeaksSettings = DetectPeaksSettings(
        threshold_abs=200.0, nms_distance=8, min_distance=0, block_size=[16, 16, 16]
    )
    initial_capture_radius: float = 80.0
    min_capture_radius: float = 10.0
    radius_decay: float = 0.5
    cluster_radius: float = 10.0
    min_votes: int = 3
    max_iterations: int = 20
    convergence_translation: float = 0.5


class FilterMatchesSettings(MyBaseModel):
    angle_threshold: float = 0
    direction_threshold: float = 0
    min_distance_quantile: float = 0.01
    max_distance_quantile: float = 0.95


class QCBeadsRegistrationSettings(MyBaseModel):
    iterations: int = 2
    score_threshold: float = 0.40
    score_centroid_mask_radius: int = 6


class BeadsMatchSettings(MyBaseModel):
    algorithm: Literal["hungarian", "match_descriptor", "spectral"] = "hungarian"
    source_peaks_settings: DetectPeaksSettings | None = Field(
        default_factory=DetectPeaksSettings
    )
    target_peaks_settings: DetectPeaksSettings | None = Field(
        default_factory=DetectPeaksSettings
    )
    match_descriptor_settings: MatchDescriptorSettings = MatchDescriptorSettings()
    hungarian_match_settings: HungarianMatchSettings = HungarianMatchSettings()
    spectral_match_settings: SpectralMatchSettings = SpectralMatchSettings()
    # Second arm of the estimate: acquire the correspondence with spectral matching, then
    # refine with `algorithm`; the higher-scoring arm wins. "on_low_score" runs it only when
    # the first arm scores below qc_settings.score_threshold.
    spectral_arm: Literal["off", "on_low_score", "always"] = "off"
    # "matching": detect -> match -> fit (the configured algorithm, plus the spectral arm);
    # "vote_icp": correspond-by-voting ICP over the peak clouds (pointcloud.vote_icp_register).
    estimation_mode: Literal["matching", "vote_icp"] = "matching"
    vote_icp_settings: VoteIcpSettings = VoteIcpSettings()
    seed_correction_settings: SeedCorrectionSettings = SeedCorrectionSettings()
    filter_matches_settings: FilterMatchesSettings = FilterMatchesSettings()
    qc_settings: QCBeadsRegistrationSettings = QCBeadsRegistrationSettings()


class PhaseCrossCorrSettings(MyBaseModel):
    normalization: Literal["magnitude", "classic"] | None = None
    maximum_shift: float = 1.2
    function_type: Literal["custom_padding", "custom"] = "custom"
    t_reference: Literal["first", "previous"] = "first"
    skip_beads_fov: str = "0"
    center_crop_xy: list[int, int] = None
    X_slice: list | list[list | Literal["all"]] | Literal["all"] = "all"
    Y_slice: list | list[list | Literal["all"]] | Literal["all"] = "all"
    Z_slice: list | list[list | Literal["all"]] | Literal["all"] = "all"


class FocusFindingSettings(MyBaseModel):
    average_across_wells: bool = False
    average_across_wells_method: Literal["mean", "median"] = "mean"
    skip_beads_fov: str = "0"
    center_crop_xy: list[int, int] = [800, 800]


class StackRegSettings(MyBaseModel):
    center_crop_xy: list[int, int] = [800, 800]
    skip_beads_fov: str = "0"
    focus_finding_settings: FocusFindingSettings | None = Field(
        default_factory=FocusFindingSettings
    )
    t_reference: Literal["first", "previous"] = "first"


class EvalTransformSettings(MyBaseModel):
    validation_window_size: int = 10
    validation_tolerance: float = 1000.0
    interpolation_window_size: int = 3
    interpolation_type: Literal["linear", "cubic"] = "linear"


class AffineTransformSettings(MyBaseModel):
    t_reference: Literal["first", "previous"] = "first"
    transform_type: Literal["euclidean", "similarity", "affine"] = "euclidean"
    approx_transform: list = np.eye(4).tolist()
    use_prev_t_transform: bool = True
    compute_approx_transform: bool = False

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
    """Settings for the ANTs registration backend.

    Field names and defaults mirror the keyword arguments of
    ``biahub.registration.methods.ants.preprocess_zyx``, which is what consumes them.

    Attributes
    ----------
    sobel_filter : bool
        Apply a Sobel filter (3D gradient magnitude) to both volumes before
        registering, so ANTs matches structural edges rather than raw
        intensity. Needed for cross-modality pairs such as fluorescence
        against a virtual-staining prediction.
    crop : bool
        Crop both volumes to their overlapping region with the LIR algorithm
        before registering.
    ref_mask_radius : float | None
        Radius of a circular mask applied to the reference channel, as a
        fraction of image width in ``(0, 1]``. ``None`` applies no mask.
    clip : bool
        Clip both volumes to hardcoded intensity limits. Those limits assume a
        **phase** reference (``np.clip(ref, 0, 0.5)``); leave this off for any
        other reference, e.g. a virtual-staining prediction whose values range
        well above 0.5.
    """

    sobel_filter: bool = False
    crop: bool = False
    ref_mask_radius: float | None = None
    clip: bool = False

    @field_validator("ref_mask_radius")
    @classmethod
    def check_ref_mask_radius(cls, v):
        # preprocess_zyx raises on this too, but only after the data is
        # loaded -- catching it at config-parse time is much cheaper.
        if v is not None and not (0 < v <= 1):
            raise ValueError(
                "ref_mask_radius must be given as a fraction of image width, i.e. (0, 1]."
            )
        return v


class ManualRegistrationSettings(MyBaseModel):
    time_index: int = 0
    affine_90degree_rotation: int = 0
    affine_fliplr: bool = False


class EstimateRegistrationSettings(MyBaseModel):
    target_channel_name: str
    source_channel_name: str
    estimation_method: Literal["manual", "beads", "ants", "phase-cross-corr"] = "manual"
    beads_match_settings: BeadsMatchSettings | None = None
    phase_cross_corr_settings: PhaseCrossCorrSettings | None = None
    focus_finding_settings: FocusFindingSettings | None = None
    affine_transform_settings: AffineTransformSettings = Field(
        default_factory=AffineTransformSettings
    )
    eval_transform_settings: EvalTransformSettings | None = None
    ants_registration_settings: AntsRegistrationSettings | None = None
    manual_registration_settings: ManualRegistrationSettings | None = None
    time_indices: NonNegativeInt | list[NonNegativeInt] | Literal["all"] = "all"
    verbose: bool = False

    @model_validator(mode="after")
    def set_defaults_and_validate(self) -> "EstimateRegistrationSettings":
        if self.estimation_method == "manual" and self.manual_registration_settings is None:
            self.manual_registration_settings = ManualRegistrationSettings()
        elif self.estimation_method == "beads" and self.beads_match_settings is None:
            self.beads_match_settings = BeadsMatchSettings()
        elif self.estimation_method == "ants" and self.ants_registration_settings is None:
            self.ants_registration_settings = AntsRegistrationSettings()
        elif (
            self.estimation_method == "phase-cross-corr"
            and self.phase_cross_corr_settings is None
        ):
            self.phase_cross_corr_settings = PhaseCrossCorrSettings()
        return self


class EstimateStabilizationSettings(MyBaseModel):
    stabilization_estimation_channel: str
    stabilization_channels: list
    stabilization_type: Literal["z", "xy", "xyz"]
    stabilization_method: Literal["beads", "phase-cross-corr", "focus-finding"] = (
        "focus-finding"
    )
    beads_match_settings: BeadsMatchSettings | None = None
    phase_cross_corr_settings: PhaseCrossCorrSettings | None = None
    stack_reg_settings: StackRegSettings | None = None
    focus_finding_settings: FocusFindingSettings | None = None
    affine_transform_settings: AffineTransformSettings = Field(
        default_factory=AffineTransformSettings
    )
    eval_transform_settings: EvalTransformSettings | None = None
    verbose: bool = False

    @model_validator(mode="after")
    def set_defaults_and_validate(self) -> "EstimateStabilizationSettings":
        if self.stabilization_method == "beads" and self.beads_match_settings is None:
            self.beads_match_settings = BeadsMatchSettings()
        elif (
            self.stabilization_method == "phase-cross-corr"
            and self.phase_cross_corr_settings is None
        ):
            self.phase_cross_corr_settings = PhaseCrossCorrSettings()
        elif self.stabilization_method == "focus-finding" and self.stabilization_type == "xyz":
            if self.focus_finding_settings is None:
                self.focus_finding_settings = FocusFindingSettings()
            if self.stack_reg_settings is None:
                self.stack_reg_settings = StackRegSettings()
        elif self.stabilization_method == "focus-finding" and self.stabilization_type == "z":
            if self.focus_finding_settings is None:
                self.focus_finding_settings = FocusFindingSettings()
        elif self.stabilization_method == "focus-finding" and self.stabilization_type == "xy":
            if self.stack_reg_settings is None:
                self.stack_reg_settings = StackRegSettings()

        return self


class FlatFieldCorrectionSettings(MyBaseModel):
    channel_names: list[str] | None = None
    # When None, preserve the OME-Zarr version of the input store.
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None


class ProcessingSettings(MyBaseModel):
    fliplr: bool | None = False
    flipud: bool | None = False
    rot90: int | None = 0


class DeskewSettings(MyBaseModel):
    pixel_size_um: PositiveFloat
    ls_angle_deg: PositiveFloat
    px_to_scan_ratio: PositiveFloat | None = None
    scan_step_um: PositiveFloat | None = None
    keep_overhang: bool = False
    overhang_fill: Literal["mean"] | float = 0
    average_n_slices: PositiveInt = 3
    device: str = "cpu"
    # When None, preserve the OME-Zarr version of the input store.
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None

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
    time_indices: NonNegativeInt | list[NonNegativeInt] | Literal["all"] = "all"
    verbose: bool = False
    # When None, preserve the OME-Zarr version of the input store.
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None

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
            raise ValueError("The array must contain valid numerical values.") from None

        return v


class PsfFromBeadsSettings(MyBaseModel):
    axis0_patch_size: PositiveInt = 101
    axis1_patch_size: PositiveInt = 101
    axis2_patch_size: PositiveInt = 101


class DeconvolveSettings(MyBaseModel):
    regularization_strength: PositiveFloat = 0.001
    # When None, preserve the OME-Zarr version of the input store.
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None


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
    # Source positions, one glob or path per source store. Optional: the CLI's
    # repeated `-i` supplies the sources (one `-i` per store) and takes
    # precedence, in which case the config holds only parameters.
    concat_data_paths: list[str] | None = None
    time_indices: int | list[int] | Literal["all"] = "all"
    # "all" takes every channel of every source, like time_indices. The list
    # form has one entry per source: "all" or the channel names to take.
    channel_names: Literal["all"] | list[str | list[str]] = "all"
    X_slice: list | list[list | Literal["all"]] | Literal["all"] = "all"
    Y_slice: list | list[list | Literal["all"]] | Literal["all"] = "all"
    Z_slice: list | list[list | Literal["all"]] | Literal["all"] = "all"
    chunks_czyx: Literal[None] | list[int] = None
    shards_ratio: list[int] | None = None
    ensure_unique_positions: bool | None = False
    # Concatenate is the migration path into v0.5 stores, so it defaults to
    # "0.5". Set to None to preserve the input store's OME-Zarr version, or
    # to "0.4" / "0.5" to force a specific output version.
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = "0.5"

    @field_validator("concat_data_paths")
    @classmethod
    def check_concat_data_paths(cls, v):
        if v is None:
            return v
        if not isinstance(v, list) or not all(isinstance(path, str) for path in v):
            raise ValueError("concat_data_paths must be a list of positions.")
        return v

    @field_validator("channel_names")
    @classmethod
    def check_channel_names(cls, v):
        if v == "all":
            return v
        if not isinstance(v, list) or not all(isinstance(name, (str, list)) for name in v):
            raise ValueError(
                "channel_names must be 'all' or a list of strings or lists of strings."
            )
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
        # Per-source lists (channel_names, X/Y/Z_slice) must be one entry per
        # source. This can only be checked here when the config names the
        # sources; when they come from the CLI's `-i` the same check runs in
        # biahub.concatenate against the resolved source count.
        data_paths = self.concat_data_paths
        if not data_paths:
            return self

        if isinstance(self.channel_names, list) and len(self.channel_names) != len(data_paths):
            raise ValueError(
                f"channel_names must be 'all' or a list with the same length as "
                f"concat_data_paths ({len(data_paths)})"
            )

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
    stabilization_type: Literal["z", "xy", "xyz", "affine"]
    stabilization_method: Literal[
        "beads", "phase-cross-corr", "focus-finding", "manual", "ants", "beads"
    ] = "focus-finding"
    stabilization_channels: list
    affine_transform_zyx_list: list
    time_indices: NonNegativeInt | list[NonNegativeInt] | Literal["all"] = "all"
    output_voxel_size: list[
        PositiveFloat, PositiveFloat, PositiveFloat, PositiveFloat, PositiveFloat
    ] = [1.0, 1.0, 1.0, 1.0, 1.0]
    # When None, preserve the OME-Zarr version of the input store.
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None

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


class StitchSettings(BaseModel):
    channels: list[str] | None = None
    total_translation: dict[str, list[float, float, float]] | None = None
    affine_transform: dict[str, list] | None = None
    # When None, preserve the OME-Zarr version of the input store.
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None

    def __init__(self, **data):
        # Adding a leading zero for zyx translation for backwards compatibility
        if "total_translation" in data:
            for key, value in data["total_translation"].items():
                if len(value) == 2:
                    data["total_translation"][key] = [0] + value

        if not any(
            (
                data.get("total_translation"),
                data.get("affine_transform"),
            )
        ):
            raise ValueError("Either affine_transform or total_translation must be provided")
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
        ) from None


class PreprocessingFunctions(BaseModel):
    function: ImportString
    channel: str
    kwargs: dict[str, Any] = {}


class SegmentationModel(BaseModel):
    path_to_model: str
    eval_args: dict[str, Any]
    z_slice_2D: int | None = None
    preprocessing: list[PreprocessingFunctions] = []

    @field_validator("eval_args", mode="before")
    @classmethod
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

    @field_validator("z_slice_2D")
    @classmethod
    def check_z_slice_with_do_3D(cls, z_slice_2D, info: ValidationInfo):
        if z_slice_2D is not None:
            eval_args = info.data.get("eval_args", {})
            do_3D = eval_args.get("do_3D", None)
            if do_3D:
                raise ValueError(
                    "If 'z_slice_2D' is provided, 'do_3D' in 'eval_args' must be set to False."
                )
            return 0  # force it to 0 as per your logic
        return z_slice_2D


class SegmentationSettings(BaseModel):
    models: dict[str, SegmentationModel]
    # When None, preserve the OME-Zarr version of the input store.
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None
    model_config = {"extra": "forbid", "protected_namespaces": ()}


# --------------------------------------------------------------------------------------
# Unified transform estimation / application settings (registration engine)
# --------------------------------------------------------------------------------------

TransformDirection = Literal["forward", "pull"]


class ChannelSettings(MyBaseModel):
    channel: str


class TransformFitSettings(MyBaseModel):
    """What kind of transform to fit and where to start.

    `seed` is a 4x4 matrix in `seed_direction`: "pull" (reference -> moving, the
    convention of every transform on disk and of the legacy `approx_transform`) or
    "forward" (moving -> reference, the engine's own convention).
    """

    type: Literal["euclidean", "similarity", "affine"] = "euclidean"
    seed: list = np.eye(4).tolist()
    seed_direction: TransformDirection = "pull"
    seed_from_shapes: bool = False

    @field_validator("seed")
    @classmethod
    def check_seed(cls, v):
        if np.asarray(v, dtype=float).shape != (4, 4):
            raise ValueError("seed must be a 4x4 matrix")
        return v


class FlagSettings(MyBaseModel):
    """Adaptive flagging line.

    A timepoint is flagged below median - k_mad * MAD AND below `floor`, or below
    `hard_fail` regardless.
    """

    k_mad: float = 2.0
    floor: float = 0.80
    hard_fail: float = 0.40


RepairCandidate = Literal["t-1", "t+1", "consensus", "seed"]


class RepairSettings(MyBaseModel):
    """Repair pass over flagged timepoints: candidate seeds in the order they are tried."""

    candidates: list[RepairCandidate] = ["t-1", "t+1", "consensus", "seed"]
    consensus_threshold: float = 0.75
    consensus_min_good: int = 5
    max_timepoints: int | None = None


class FallbackSettings(MyBaseModel):
    flag: FlagSettings = FlagSettings()
    repair: RepairSettings | None = RepairSettings()


EstimationMethod = Literal["beads", "ants", "phase-cross-corr", "manual"]
ScoreMetric = Literal[
    "overlap", "residual", "mutual_information", "correlation", "gradient_correlation"
]
DEFAULT_SCORE_METRIC: dict[str, str] = {
    "beads": "overlap",
    "ants": "correlation",
    "phase-cross-corr": "correlation",
    "manual": "gradient_correlation",
}


class EstimateTransformSettings(MyBaseModel):
    """Everything `estimate-transform` needs.

    What to align onto what, how, and what to do when a timepoint comes out badly.

    Registration and stabilization are the same estimate with a different `reference`:
    "cross" aligns `source` onto `target` at each timepoint; "first" / "previous" align the
    source channel onto its own first / previous timepoint (then `target` is omitted).
    Only the settings block of the chosen `method` is required.
    """

    source: ChannelSettings
    target: ChannelSettings | None = None
    reference: Literal["cross", "first", "previous"] = "cross"
    method: EstimationMethod
    beads: BeadsMatchSettings | None = None
    ants: AntsRegistrationSettings | None = None
    phase_cross_corr: PhaseCrossCorrSettings | None = None
    manual: ManualRegistrationSettings | None = None
    transform: TransformFitSettings = TransformFitSettings()
    time_indices: NonNegativeInt | list[NonNegativeInt] | Literal["all"] = "all"
    # None: the method's default (beads: overlap, ants / phase-cross-corr: correlation,
    # manual: gradient_correlation).
    score_metric: ScoreMetric | None = None
    fallback: FallbackSettings = FallbackSettings()
    smoothing: EvalTransformSettings | None = None
    verbose: bool = False

    @model_validator(mode="after")
    def check_consistency(self) -> "EstimateTransformSettings":
        if self.reference == "cross" and self.target is None:
            raise ValueError("reference 'cross' needs a target channel")
        if self.reference != "cross" and self.target is not None:
            raise ValueError(
                f"reference '{self.reference}' aligns the source onto itself; drop target"
            )
        defaults = {
            "beads": ("beads", BeadsMatchSettings),
            "ants": ("ants", AntsRegistrationSettings),
            "phase-cross-corr": ("phase_cross_corr", PhaseCrossCorrSettings),
            "manual": ("manual", ManualRegistrationSettings),
        }
        field, model = defaults[self.method]
        if getattr(self, field) is None:
            setattr(self, field, model())
        return self

    @property
    def target_channel(self) -> str:
        return self.target.channel if self.target is not None else self.source.channel

    @property
    def effective_score_metric(self) -> str:
        return self.score_metric or DEFAULT_SCORE_METRIC[self.method]

    @classmethod
    def from_legacy(
        cls, legacy: "EstimateRegistrationSettings | EstimateStabilizationSettings"
    ) -> "EstimateTransformSettings":
        """Convert a legacy config to the same estimate.

        `use_prev_t_transform` has no equivalent: timepoints are estimated independently
        and neighbour information enters through the repair candidates.
        """
        ats = legacy.affine_transform_settings
        fit = TransformFitSettings(
            type=ats.transform_type,
            seed=ats.approx_transform,
            seed_direction="pull",
            seed_from_shapes=ats.compute_approx_transform,
        )
        common = dict(
            transform=fit, smoothing=legacy.eval_transform_settings, verbose=legacy.verbose
        )
        if isinstance(legacy, EstimateRegistrationSettings):
            method = legacy.estimation_method
            # The same channel on both sides only makes sense as stabilization against
            # itself, which is how the legacy CLI treated it (t_reference decides which frame).
            self_reference = legacy.source_channel_name == legacy.target_channel_name
            return cls(
                source=ChannelSettings(channel=legacy.source_channel_name),
                target=None
                if self_reference
                else ChannelSettings(channel=legacy.target_channel_name),
                reference=ats.t_reference if self_reference else "cross",
                method=method,
                beads=legacy.beads_match_settings,
                ants=legacy.ants_registration_settings,
                phase_cross_corr=legacy.phase_cross_corr_settings,
                manual=legacy.manual_registration_settings,
                time_indices=legacy.time_indices,
                **common,
            )
        method = legacy.stabilization_method
        if method == "focus-finding":
            raise ValueError("focus-finding stabilization has no engine estimator yet")
        reference = (
            legacy.phase_cross_corr_settings.t_reference
            if method == "phase-cross-corr" and legacy.phase_cross_corr_settings is not None
            else ats.t_reference
        )
        return cls(
            source=ChannelSettings(channel=legacy.stabilization_estimation_channel),
            target=None,
            reference=reference,
            method=method,
            beads=legacy.beads_match_settings,
            phase_cross_corr=legacy.phase_cross_corr_settings,
            **common,
        )


class TransformSettings(MyBaseModel):
    """A transform series ready to apply: one 4x4 for every timepoint, or a single one.

    `direction` says what the matrices mean -- "forward" (moving -> reference, what the
    engine estimates) or "pull" (reference -> moving, what the legacy `register` /
    `stabilize` configs hold). Nothing here has to be guessed from a variable name.
    """

    direction: TransformDirection
    matrices: list
    time_indices: NonNegativeInt | list[NonNegativeInt] | Literal["all"] = "all"
    source_channels: list[str]
    target_channel: str | None = None
    method: str = "beads"
    voxel_size: list[float] | None = None
    keep_overhang: bool = False
    interpolation: str = "linear"
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None

    @field_validator("matrices")
    @classmethod
    def check_matrices(cls, v):
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 3 or arr.shape[1:] != (4, 4) or len(arr) == 0:
            raise ValueError("matrices must be a non-empty list of 4x4 matrices")
        return v

    def as_direction(self, direction: TransformDirection) -> list:
        if direction == self.direction:
            return [np.asarray(m, dtype=float).tolist() for m in self.matrices]
        return [np.linalg.inv(np.asarray(m, dtype=float)).tolist() for m in self.matrices]

    def to_registration_settings(self) -> "RegistrationSettings":
        """Legacy single-transform config (`register`); requires exactly one matrix."""
        (matrix,) = self.as_direction("pull")
        return RegistrationSettings(
            source_channel_names=self.source_channels,
            target_channel_name=self.target_channel or self.source_channels[0],
            affine_transform_zyx=matrix,
            keep_overhang=self.keep_overhang,
            interpolation=self.interpolation,
            time_indices=self.time_indices,
            output_ome_zarr_version=self.output_ome_zarr_version,
        )

    def to_stabilization_settings(self) -> "StabilizationSettings":
        """Legacy per-timepoint config (`stabilize`)."""
        return StabilizationSettings(
            stabilization_estimation_channel=self.target_channel or self.source_channels[0],
            stabilization_type="affine",
            stabilization_method=self.method,
            stabilization_channels=sorted(
                {
                    *self.source_channels,
                    *([self.target_channel] if self.target_channel else []),
                }
            ),
            affine_transform_zyx_list=self.as_direction("pull"),
            time_indices=self.time_indices,
            output_voxel_size=self.voxel_size or [1.0] * 5,
            output_ome_zarr_version=self.output_ome_zarr_version,
        )

    @classmethod
    def from_legacy(
        cls, legacy: "RegistrationSettings | StabilizationSettings"
    ) -> "TransformSettings":
        if isinstance(legacy, RegistrationSettings):
            return cls(
                direction="pull",
                matrices=[legacy.affine_transform_zyx],
                time_indices=legacy.time_indices,
                source_channels=legacy.source_channel_names,
                target_channel=legacy.target_channel_name,
                keep_overhang=legacy.keep_overhang,
                interpolation=legacy.interpolation,
                output_ome_zarr_version=legacy.output_ome_zarr_version,
            )
        return cls(
            direction="pull",
            matrices=legacy.affine_transform_zyx_list,
            time_indices=legacy.time_indices,
            source_channels=list(legacy.stabilization_channels),
            target_channel=legacy.stabilization_estimation_channel,
            method=legacy.stabilization_method,
            voxel_size=list(legacy.output_voxel_size),
            output_ome_zarr_version=legacy.output_ome_zarr_version,
        )


def _load_first(path, models):
    data = yaml.safe_load(open(path))
    errors = []
    for model in models:
        try:
            return model(**data)
        except Exception as e:  # noqa: BLE001 -- try the next schema, report all if none fit
            errors.append(f"{model.__name__}: {str(e).splitlines()[0]}")
    raise ValueError(
        f"{path} matches none of {[m.__name__ for m in models]}:\n  " + "\n  ".join(errors)
    )


def load_estimate_transform_settings(path) -> EstimateTransformSettings:
    """Read the unified estimate config.

    A legacy estimate-registration / estimate-stabilization config is converted on the fly.
    """
    settings = _load_first(
        path,
        (
            EstimateTransformSettings,
            EstimateRegistrationSettings,
            EstimateStabilizationSettings,
        ),
    )
    return (
        settings
        if isinstance(settings, EstimateTransformSettings)
        else EstimateTransformSettings.from_legacy(settings)
    )


def load_transform_settings(path) -> TransformSettings:
    """Read the unified transform config, or a legacy register / stabilize config."""
    settings = _load_first(
        path, (TransformSettings, RegistrationSettings, StabilizationSettings)
    )
    return (
        settings
        if isinstance(settings, TransformSettings)
        else TransformSettings.from_legacy(settings)
    )
