"""One-shot conversion of the retired registration / stabilization configs.

`estimate-registration` and `estimate-stabilization` configs become an
`EstimateTransformSettings` for `estimate-transform`; `register` and `stabilize` configs
become a `TransformSettings` for `apply-transform`. The retired schemas are defined here
and nowhere else: the rest of biahub reads only the unified models, and its loaders point
at this command when they meet a legacy file.

What has no equivalent and is dropped, with a note printed at conversion time:
`use_prev_t_transform` (timepoints are estimated independently; neighbours enter through
the repair candidates), `eval_transform_settings` (the engine flags and repairs outlying
timepoints instead of smoothing the series afterwards), `average_across_wells` and
`skip_beads_fov` (per-plate orchestration, not part of the estimate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import click
import numpy as np
import yaml

from pydantic import Field, NonNegativeInt, PositiveFloat, field_validator, model_validator

from biahub.cli.parsing import config_filepath, output_filepath
from biahub.settings import (
    AffineTransformSettings,
    AntsRegistrationSettings,
    BeadsMatchSettings,
    ChannelSettings,
    EstimateTransformSettings,
    FocusSettings,
    ManualRegistrationSettings,
    MyBaseModel,
    PhaseCrossCorrSettings,
    ReferenceSettings,
    TransformEntry,
    TransformFitSettings,
    TransformSettings,
)
from biahub.utils.config import model_to_yaml

# --------------------------------------------------------------------------------------
# Retired schemas, kept verbatim so old files still parse
# --------------------------------------------------------------------------------------


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
    def set_defaults_and_validate(self) -> EstimateRegistrationSettings:
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
    def set_defaults_and_validate(self) -> EstimateStabilizationSettings:
        if self.stabilization_method == "beads" and self.beads_match_settings is None:
            self.beads_match_settings = BeadsMatchSettings()
        elif (
            self.stabilization_method == "phase-cross-corr"
            and self.phase_cross_corr_settings is None
        ):
            self.phase_cross_corr_settings = PhaseCrossCorrSettings()
        elif self.stabilization_method == "focus-finding":
            if "z" in self.stabilization_type and self.focus_finding_settings is None:
                self.focus_finding_settings = FocusFindingSettings()
            if "xy" in self.stabilization_type and self.stack_reg_settings is None:
                self.stack_reg_settings = StackRegSettings()
        return self


class RegistrationSettings(MyBaseModel):
    source_channel_names: list[str]
    target_channel_name: str
    affine_transform_zyx: list
    keep_overhang: bool = False
    interpolation: str = "linear"
    time_indices: NonNegativeInt | list[NonNegativeInt] | Literal["all"] = "all"
    verbose: bool = False
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None

    @field_validator("affine_transform_zyx")
    @classmethod
    def check_affine_transform(cls, v):
        if np.asarray(v, dtype=float).shape != (4, 4):
            raise ValueError("affine_transform_zyx must be a 4x4 matrix")
        return v


class StabilizationSettings(MyBaseModel):
    stabilization_estimation_channel: str
    stabilization_type: Literal["z", "xy", "xyz", "affine"]
    stabilization_method: Literal[
        "beads", "phase-cross-corr", "focus-finding", "manual", "ants"
    ] = "focus-finding"
    stabilization_channels: list
    affine_transform_zyx_list: list
    time_indices: NonNegativeInt | list[NonNegativeInt] | Literal["all"] = "all"
    output_voxel_size: list[
        PositiveFloat, PositiveFloat, PositiveFloat, PositiveFloat, PositiveFloat
    ] = [1.0, 1.0, 1.0, 1.0, 1.0]
    output_ome_zarr_version: Literal["0.4", "0.5"] | None = None

    @field_validator("affine_transform_zyx_list")
    @classmethod
    def check_affine_transform_zyx_list(cls, v):
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 3 or arr.shape[1:] != (4, 4):
            raise ValueError("affine_transform_zyx_list must be a list of 4x4 matrices")
        return v


LEGACY_MODELS = (
    EstimateRegistrationSettings,
    EstimateStabilizationSettings,
    RegistrationSettings,
    StabilizationSettings,
)
LegacySettings = (
    EstimateRegistrationSettings
    | EstimateStabilizationSettings
    | RegistrationSettings
    | StabilizationSettings
)

# --------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------


def estimate_settings_from_legacy(
    legacy: EstimateRegistrationSettings | EstimateStabilizationSettings,
) -> tuple[EstimateTransformSettings, list[str]]:
    """Convert a legacy estimate-* config to the same estimate; also return what was dropped."""
    notes = []
    ats = legacy.affine_transform_settings
    if ats.use_prev_t_transform:
        notes.append(
            "use_prev_t_transform dropped: timepoints are estimated independently and "
            "neighbours enter through fallback.repair.candidates"
        )
    if legacy.eval_transform_settings is not None:
        notes.append(
            "eval_transform_settings dropped: the engine flags and repairs outlying "
            "timepoints (fallback) instead of smoothing the series afterwards"
        )
    fit = TransformFitSettings(
        type=ats.transform_type,
        seed=ats.approx_transform,
        seed_direction="pull",
        seed_from_shapes=ats.compute_approx_transform,
    )
    if isinstance(legacy, EstimateRegistrationSettings):
        # The same channel on both sides only makes sense as stabilization against
        # itself, which is how the legacy CLI treated it (t_reference picks the frame).
        self_reference = legacy.source_channel_name == legacy.target_channel_name
        settings = EstimateTransformSettings(
            moving=ChannelSettings(channel=legacy.source_channel_name),
            reference=ReferenceSettings(frame=ats.t_reference)
            if self_reference
            else ReferenceSettings(frame="cross", channel=legacy.target_channel_name),
            method=legacy.estimation_method,
            beads=legacy.beads_match_settings,
            ants=legacy.ants_registration_settings,
            phase_cross_corr=legacy.phase_cross_corr_settings,
            manual=legacy.manual_registration_settings,
            transform=fit,
            time_indices=legacy.time_indices,
            verbose=legacy.verbose,
        )
        return settings, notes

    method = legacy.stabilization_method
    focus_finding = None
    reference = ats.t_reference
    if method == "focus-finding":
        stack_reg = legacy.stack_reg_settings
        focus = legacy.focus_finding_settings or (
            stack_reg.focus_finding_settings if stack_reg is not None else None
        )
        # One crop in the unified method: stackreg's when only xy is stabilized, else
        # the focus block's (legacy used each block's own crop for its own axes).
        blocks = (
            (stack_reg, focus) if legacy.stabilization_type == "xy" else (focus, stack_reg)
        )
        crop = next(
            b.center_crop_xy for b in (*blocks, FocusFindingSettings()) if b is not None
        )
        focus_finding = FocusSettings(axes=legacy.stabilization_type, center_crop_xy=crop)
        if stack_reg is not None:
            reference = stack_reg.t_reference
        if any(
            block is not None and (block.average_across_wells or block.skip_beads_fov != "0")
            for block in (focus, stack_reg)
            if hasattr(block, "average_across_wells")
        ):
            notes.append(
                "average_across_wells / skip_beads_fov dropped: per-plate orchestration, "
                "not part of the estimate"
            )
    elif method == "phase-cross-corr" and legacy.phase_cross_corr_settings is not None:
        reference = legacy.phase_cross_corr_settings.t_reference
    settings = EstimateTransformSettings(
        moving=ChannelSettings(channel=legacy.stabilization_estimation_channel),
        reference=ReferenceSettings(frame=reference),
        method=method,
        beads=legacy.beads_match_settings,
        phase_cross_corr=legacy.phase_cross_corr_settings,
        focus_finding=focus_finding,
        transform=fit,
        verbose=legacy.verbose,
    )
    return settings, notes


def transform_settings_from_legacy(
    legacy: RegistrationSettings | StabilizationSettings,
) -> tuple[TransformSettings, list[str]]:
    """Convert a legacy register / stabilize config to a pull-direction transform series.

    How the series is applied (timepoints, canvas, interpolation, output version) is no
    longer stored with the transforms: those are `apply-transform` options, noted here.
    """
    notes = []
    if legacy.time_indices != "all":
        notes.append(
            f"time_indices {legacy.time_indices!r} dropped: pass --time-indices to apply-transform"
        )
    if legacy.output_ome_zarr_version is not None:
        notes.append(
            "output_ome_zarr_version dropped: pass --ome-zarr-version to apply-transform"
        )
    if isinstance(legacy, RegistrationSettings):
        if legacy.keep_overhang:
            notes.append("keep_overhang dropped: pass --keep-overhang to apply-transform")
        if legacy.interpolation != "linear":
            notes.append(
                f"interpolation {legacy.interpolation!r} dropped: pass --interpolation to apply-transform"
            )
        settings = TransformSettings(
            direction="pull",
            moving_channels=legacy.source_channel_names,
            reference_channel=legacy.target_channel_name,
            transforms=[TransformEntry(matrix=legacy.affine_transform_zyx)],
        )
        return settings, notes
    # stabilize transformed every listed channel onto the estimation channel's own grid,
    # so the estimation channel is one of the moving channels and there is no reference.
    settings = TransformSettings(
        direction="pull",
        moving_channels=sorted(
            {*legacy.stabilization_channels, legacy.stabilization_estimation_channel}
        ),
        reference_channel=None,
        method=legacy.stabilization_method,
        voxel_size=list(legacy.output_voxel_size),
        transforms=[
            TransformEntry(t=t, matrix=matrix)
            for t, matrix in enumerate(legacy.affine_transform_zyx_list)
        ],
    )
    return settings, notes


def load_legacy_settings(path: Path) -> LegacySettings:
    """Parse a legacy config with whichever retired schema fits (they share no required key)."""
    data = yaml.safe_load(Path(path).read_text())
    errors = []
    for model in LEGACY_MODELS:
        try:
            return model(**data)
        except Exception as e:  # noqa: BLE001 -- try the next schema, report all if none fit
            errors.append(f"{model.__name__}: {str(e).splitlines()[0]}")
    raise ValueError(
        f"{path} is not a legacy estimate-registration / estimate-stabilization / register / "
        "stabilize config:\n  " + "\n  ".join(errors)
    )


def convert_settings(
    legacy: LegacySettings,
) -> tuple[EstimateTransformSettings | TransformSettings, list[str]]:
    if isinstance(legacy, (EstimateRegistrationSettings, EstimateStabilizationSettings)):
        return estimate_settings_from_legacy(legacy)
    return transform_settings_from_legacy(legacy)


def convert_settings_file(config_filepath: Path, output_filepath: Path) -> None:
    legacy = load_legacy_settings(config_filepath)
    settings, notes = convert_settings(legacy)
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    model_to_yaml(settings, output_filepath)
    command = (
        "estimate-transform"
        if isinstance(settings, EstimateTransformSettings)
        else "apply-transform"
    )
    click.echo(
        f"{config_filepath} ({type(legacy).__name__}) -> {output_filepath} "
        f"({type(settings).__name__}, for `biahub {command}`)"
    )
    for note in notes:
        click.echo(f"  note: {note}")


@click.command("convert-settings")
@config_filepath()
@output_filepath()
def convert_settings_cli(config_filepath: str, output_filepath: str) -> None:
    """Convert a retired registration / stabilization config to the unified one.

    estimate-registration / estimate-stabilization -> estimate-transform;
    register / stabilize -> apply-transform.

    >> biahub convert-settings -c estimate-registration.yml -o estimate-transform.yml
    """
    convert_settings_file(Path(config_filepath), Path(output_filepath))
