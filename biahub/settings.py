from pathlib import Path
from typing import Any, Literal

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
    """Settings for pairwise-consistency (Leordeanu-Hebert) spectral matching.

    Attributes
    ----------
    sigma : float
        Tolerance in voxels on pairwise-distance agreement between two candidate
        correspondences. Roughly the bead-localisation uncertainty.
    rel_cut : float
        Keep candidates scoring above this fraction of the top eigenvector entry.
        Higher is stricter: measured on real data, 0.5 gives recall ~0.90 at precision
        ~0.74, and 0.7 gives recall ~0.64 at precision ~0.83.
    max_iter : int
        Power-iteration steps for the principal eigenvector.

    These defaults are deliberately left where they are, despite a sweep appearing to beat
    them. Over 24 stratified timepoints on two datasets, sigma=5.0/rel_cut=0.65 dominated
    3.0/0.5 on every summary statistic when spectral matching ran ALONE -- mean 0.872 vs
    0.856, worst case 0.720 vs 0.667. Rerunning the full cascade end-to-end with the tuned
    pair then showed no improvement at all: 17 timepoints better, 21 worse, mean delta
    -0.002, Wilcoxon p=0.20, and the run minimum fell from 0.720 to 0.708.

    The reason is that spectral only has to land the transform inside the basin that the
    subsequent hungarian refinement converges from; once it does, its own precision is
    discarded. 106 of 144 timepoints came out bit-identical under the two settings. So a
    single-pass spectral benchmark ranks these parameters for a job the cascade does not ask
    them to do, and tuning them against it is measuring the wrong thing.

    They do still matter where spectral matching is used on its own -- see
    DEFAULT_SWEEP_GRID in biahub.registration.beads, whose ranges come from that same
    stratified sweep.
    """

    sigma: float = 3.0
    rel_cut: float = 0.5
    max_iter: int = 60


class FilterMatchesSettings(MyBaseModel):
    angle_threshold: float = 0
    direction_threshold: float = 0
    min_distance_quantile: float = 0.01
    max_distance_quantile: float = 0.95


class QCBeadsRegistrationSettings(MyBaseModel):
    iterations: int = 2
    score_threshold: float = 0.40
    score_centroid_mask_radius: int = 6


class SweepFallbackSettings(MyBaseModel):
    """Last-resort per-timepoint grid search over the matching parameters.

    The other arms all reuse one parameter set for the whole timelapse. That set is chosen
    to be good on average, and a handful of timepoints are simply not well served by it.
    This re-tunes those timepoints individually.

    Off by default because it is the most expensive thing in the estimator: one ANTs warp
    plus peak re-detection per surviving trial, roughly 18 s each. Gating it on score is
    what makes it affordable -- it fires on the few percent of timepoints that need it.

    Measured on 15 flagged timepoints across two datasets, with the default grid: 8 improved,
    mean gain +0.041 over all 15 and +0.077 over those it helped. It cannot regress --
    estimate() keeps the sweep result only if it strictly beats the incumbent -- so the
    honest description is a modest rescue for a real cost, worth enabling when a few bad
    timepoints matter more than runtime. Note it found nothing at 7 of the 15, so it is a
    tail-risk reducer rather than a general improvement.

    Attributes
    ----------
    mode : Literal["off", "on_low_score"]
        "on_low_score" runs the sweep when every other arm came in below score_threshold.
    mode : Literal["off", "on_flagged", "on_low_score"]
        "on_flagged" is the recommended setting: the sweep runs as a post-pass after the
        repair pass, on the timepoints the run's OWN adaptive median-2*MAD line flags.

        "on_low_score" is the legacy behaviour -- the sweep runs inside estimate(), gated on
        the fixed score_threshold below. It is kept for reproducing earlier runs and is NOT
        recommended, because a fixed gate does not transfer between datasets. Measured with
        the gate at 0.75: on a run whose median was 0.870 it fired 0 times out of 144, and on
        a run whose median was 0.753 -- where 0.75 sits at the median -- it fired 556 times,
        stopped being a fallback, and pushed the job past its 24 h walltime.

        The reason the adaptive gate has to live in a post-pass is that inside estimate() the
        run-wide distribution does not exist yet: in independent mode the other timepoints are
        still being computed in other SLURM jobs.
    score_threshold : float
        Only used by the legacy "on_low_score" mode. Calibrated against two runs with medians
        0.870 and 0.875: 0.70 fired on 0 and 2 timepoints (dead code), 0.75 on 5 and 5 (the
        tail), 0.80 on 26 and 29 (~15%, hours of compute). Ignored by "on_flagged", which
        derives its own line per dataset.
    max_timepoints : int | None
        Cap on how many flagged timepoints to sweep, since each costs a full grid search.
        When the cap bites, the worst are swept and the skipped ones are logged by name.
    grid : dict[str, list] | list[dict[str, list]] | None
        Parameter grid; None uses DEFAULT_SWEEP_GRID in biahub.registration.beads. A list of
        dicts is searched as the union of their cross products, which is how the hungarian
        and spectral parameter sets are swept without paying for their cross product -- each
        is inert while the other matcher is in use. Keys are validated against that module's
        setter table, so a misspelled key raises instead of silently reading as a flat axis.
    """

    mode: Literal["off", "on_flagged", "on_low_score"] = "off"
    score_threshold: float = 0.75
    max_timepoints: int | None = 25
    # Which timepoints the sweep is allowed to touch, once the repair pass has run.
    #
    #   "still_flagged"         everything the adaptive line flags on the POST-repair scores.
    #                           Because repair raises the median and shrinks the MAD, that
    #                           line rises -- so this set is repair's failures PLUS timepoints
    #                           that only became outliers relative to a now-healthier run.
    #   "repair_failures_only"  strictly the timepoints repair attempted and did not lift.
    #
    # Default is "still_flagged", against the intuition that the sweep should only clean up
    # after repair, because the measurements point the other way: the sweep gains +0.058 on
    # timepoints scoring 0.72-0.78 and only +0.006 below 0.72, and found nothing at all across
    # 28 trials on the worst timepoint tested. Repair's failures are the collapsed fits near
    # zero -- exactly where sweeping does not help -- while the timepoints the rising line
    # newly catches sit in the 0.66-0.73 band, which is the sweep's useful range. Restricting
    # to repair failures therefore spends the budget where it cannot pay off and skips where
    # it can.
    scope: Literal["still_flagged", "repair_failures_only"] = "still_flagged"
    grid: dict[str, list] | list[dict[str, list]] | None = None


class RepairPassSettings(MyBaseModel):
    """Post-estimation reseeding of flagged timepoints from their neighbours.

    The other half of the fallback, and it necessarily runs after the whole series exists
    rather than inside estimate(): it seeds from t-1 AND t+1, and in independent mode t+1 is
    still being computed in another shard while t is estimated. Reaching forwards is the
    point -- propagation's built-in fallback can only reach back to t-1, and a timepoint that
    failed because the sample jumped between t-1 and t is often fine from t+1.

    It targets a different failure class from the sweep, which is why running both and keeping
    the higher score is worth more than either alone:

        sweep    right basin, suboptimal correspondence. Measured +0.058 on timepoints
                 scoring 0.72-0.78, but only +0.006 on those below 0.72.
        repair   wrong basin, or no usable transform at all -- which no amount of
                 re-weighting the cost matrix fixes. At one real timepoint propagation gave
                 0.429 where a reseed reached 0.778.

    Gated on the run's own adaptive median-2*MAD line, not a second fixed threshold, so it
    repairs exactly what the QC report flags. Every candidate is scored and accepted only if
    it strictly beats the incumbent, so the pass cannot make a run worse.

    Attributes
    ----------
    mode : Literal["off", "on_flagged"]
        "on_flagged" repairs the timepoints the adaptive threshold flags.
    try_config_seed : bool
        Also try the static config approx_transform as a seed. Worth keeping for the case
        both neighbours are themselves flagged, which is what a cluster of failures looks
        like.
    max_timepoints : int | None
        Safety cap on how many timepoints to repair, since each costs up to one full
        estimate() per candidate seed. None means no cap. When the cap bites, the worst
        timepoints are repaired and the skipped ones are logged by name rather than
        silently dropped.
    """

    mode: Literal["off", "on_flagged"] = "off"
    try_config_seed: bool = True
    max_timepoints: int | None = 25
    # Seed a flagged timepoint from the run's own consensus geometry -- the element-wise
    # median transform over the timepoints that scored well -- as well as from its
    # neighbours. This repairs rather than discards a timepoint whose fit collapsed.
    #
    # It is what makes the geometry check actionable instead of merely diagnostic. Measured on
    # one dataset, distance-from-consensus separated failed from good timepoints with AUC
    # 1.000, and 15 failures were reflections (negative determinant) -- fits that collapsed
    # outright. Those cannot be rescued by re-tuning matching parameters, but the run's own
    # agreed geometry is a sound starting point for them.
    use_consensus_seed: bool = True
    # Only good timepoints define the consensus; including failures would contaminate the
    # reference used to detect them.
    consensus_score_threshold: float = 0.75
    # Frobenius distance from the consensus linear part above which a timepoint's own linear
    # part is treated as broken, so the consensus seeds are tried first. Measured: good
    # timepoints sit at 0.021 and failures at 0.72, so anything in between separates them.
    consensus_linear_tolerance: float = 0.25
    # How many robust spreads (MAD) a flagged timepoint's translation may sit from the
    # consensus and still be considered worth keeping. Beyond this, the full-consensus seed is
    # tried first instead. Measured need: collapsed fits on one dataset had translations
    # thousands of voxels out, one reaching -15000 in x, so a broken linear part does not
    # imply an intact translation.
    consensus_translation_tolerance: float = 10.0


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
    filter_matches_settings: FilterMatchesSettings = FilterMatchesSettings()
    qc_settings: QCBeadsRegistrationSettings = QCBeadsRegistrationSettings()
    # Extra arm in estimate(): acquire the correspondence with spectral matching, then
    # refine with the configured algorithm. estimate() keeps whichever arm scores higher,
    # so enabling this cannot make the result worse than leaving it off.
    #
    #   "off"           existing behaviour, unchanged
    #   "on_low_score"  only when the other arms fall below qc_settings.score_threshold.
    #                   Cheap, but note it will rarely fire in practice: measured scores
    #                   sit at 0.6-1.0 against a 0.40 threshold.
    #   "always"        run it at every timepoint. This is what the benchmarked variant D
    #                   does, and it is not merely a rescue: with a good initial transform
    #                   the spectral cascade still won 93/144 and 117/240 timepoints on two
    #                   real datasets, so gating it on failure gives up most of the gain.
    #                   Costs one extra optimize_transform pair per timepoint.
    #
    # Defaults to "always" (i.e. variant D) on the benchmark below. Set "off" to restore
    # the previous behaviour exactly.
    spectral_arm: Literal["off", "on_low_score", "always"] = "always"
    sweep_fallback_settings: SweepFallbackSettings = SweepFallbackSettings()
    repair_pass_settings: RepairPassSettings = RepairPassSettings()


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
    # Defaults to False (independent per timepoint) rather than propagation. Propagation
    # exists to supply each timepoint with a good initial transform, but the spectral
    # cascade is insensitive to its initial transform -- measured identical results from a
    # seed scoring 0.826 and one scoring 0.000 -- so propagation stops paying for itself
    # while keeping three drawbacks: it is serial (~3-10 h against ~20 min), it cannot be
    # resumed cheaply, and a single failure propagates forward as a cluster. Set True to
    # restore it.
    use_prev_t_transform: bool = False
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
    ``biahub.registration.ants.preprocess_czyx``, which is what consumes them.

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
        # preprocess_czyx raises on this too, but only after the data is
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
    estimation_method: Literal["manual", "beads", "ants"] = "manual"
    beads_match_settings: BeadsMatchSettings | None = None
    focus_finding_settings: FocusFindingSettings | None = None
    affine_transform_settings: AffineTransformSettings = Field(
        default_factory=AffineTransformSettings
    )
    eval_transform_settings: EvalTransformSettings | None = None
    ants_registration_settings: AntsRegistrationSettings | None = None
    manual_registration_settings: ManualRegistrationSettings | None = None
    # One visible field to choose the beads strategy, rather than requiring the user to know
    # that two low-level flags -- affine_transform_settings.use_prev_t_transform and
    # beads_match_settings.spectral_arm, in two different blocks -- combine to produce it:
    #
    #   beads_strategy          use_prev_t_transform   spectral_arm
    #   propagate                       True              off
    #   propagate_spectral              True              always
    #   independent                     False             off
    #   independent_spectral            False             always     <- default
    #
    # Benchmarked on 2025_09_17 (144 t) and 2025_09_18 (240 t), scoring warped beads against
    # the reference:
    #
    #   propagate              median 0.864/0.875   min 0.000/0.667   1/0  below 0.40
    #   independent            median ~0.826        min 0.000         6/33 below 0.40
    #   independent_spectral   median 0.870/0.875   min 0.720/0.600   0/0  below 0.40
    #
    # independent_spectral is the default because the spectral cascade is insensitive to its
    # initial transform -- measured identical results from a seed scoring 0.826 and one
    # scoring 0.000 -- so propagation no longer earns its serial cost (~3-10 h against
    # ~20 min), its lack of cheap resume, or its tendency to turn one failure into a cluster.
    #
    # Set to None to drive use_prev_t_transform and spectral_arm directly instead.
    beads_strategy: (
        Literal["propagate", "propagate_spectral", "independent", "independent_spectral"]
        | None
    ) = "independent_spectral"
    verbose: bool = False

    @model_validator(mode="after")
    def set_defaults_and_validate(self) -> "EstimateRegistrationSettings":
        if self.estimation_method == "manual" and self.manual_registration_settings is None:
            self.manual_registration_settings = ManualRegistrationSettings()
        elif self.estimation_method == "beads" and self.beads_match_settings is None:
            self.beads_match_settings = BeadsMatchSettings()
        elif self.estimation_method == "ants" and self.ants_registration_settings is None:
            self.ants_registration_settings = AntsRegistrationSettings()

        # Expand the strategy into the two flags that actually drive estimate(). Runs after
        # the block above so beads_match_settings exists. Skipped when beads_strategy is
        # None, which leaves the low-level flags untouched for advanced use.
        if self.beads_strategy is not None and self.beads_match_settings is not None:
            propagate, spectral = {
                "propagate": (True, "off"),
                "propagate_spectral": (True, "always"),
                "independent": (False, "off"),
                "independent_spectral": (False, "always"),
            }[self.beads_strategy]

            # An explicit low-level flag always outranks the DEFAULT strategy. Without this,
            # the default "independent_spectral" would silently flip a config that says
            # use_prev_t_transform: true to false -- turning a propagation run into an
            # independent one with no diagnostic, which every pre-existing config in the
            # wild would hit, since they all set that flag directly and name no strategy.
            strategy_explicit = "beads_strategy" in self.model_fields_set
            prop_explicit = (
                "use_prev_t_transform" in self.affine_transform_settings.model_fields_set
            )
            spectral_explicit = "spectral_arm" in self.beads_match_settings.model_fields_set

            # Naming both a strategy and a flag that contradicts it is a config error, not a
            # precedence question: one of the two is not what the author meant.
            if strategy_explicit:
                conflicts = []
                if (
                    prop_explicit
                    and self.affine_transform_settings.use_prev_t_transform != propagate
                ):
                    conflicts.append(
                        f"use_prev_t_transform="
                        f"{self.affine_transform_settings.use_prev_t_transform} "
                        f"(strategy implies {propagate})"
                    )
                if spectral_explicit and self.beads_match_settings.spectral_arm != spectral:
                    conflicts.append(
                        f"spectral_arm={self.beads_match_settings.spectral_arm!r} "
                        f"(strategy implies {spectral!r})"
                    )
                if conflicts:
                    raise ValueError(
                        f"beads_strategy={self.beads_strategy!r} contradicts "
                        + " and ".join(conflicts)
                        + ". Set beads_strategy to null to drive the flags directly, or "
                        "remove the conflicting flag."
                    )

            if strategy_explicit or not prop_explicit:
                self.affine_transform_settings.use_prev_t_transform = propagate
            if strategy_explicit or not spectral_explicit:
                self.beads_match_settings.spectral_arm = spectral
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
    concat_data_paths: list[str]
    time_indices: int | list[int] | Literal["all"] = "all"
    channel_names: list[str | list[str]]
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
