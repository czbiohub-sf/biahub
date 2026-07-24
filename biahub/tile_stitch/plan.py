"""Serializable static programs and per-unit bindings for tile stitching."""


import pickle

from dataclasses import dataclass
from pathlib import Path

from waveorder.api.tile_stitch import (
    InputTile,
    OutputTile,
    TileStitchPlan,
    TileStitchSettings,
)

from biahub.settings import MonarchConfig


@dataclass(frozen=True, slots=True)
class StitchProgram:
    """Geometry, science settings, and engine configuration shared by a run.

    Parameters
    ----------
    settings : TileStitchSettings
        Reconstruction, tiling, and blending settings.
    input_tiles : tuple[InputTile, ...]
        Reconstruction tile geometry.
    output_tiles : tuple[OutputTile, ...]
        Output chunk geometry.
    output_to_inputs : dict[int, tuple[int, ...]]
        Contributor tile IDs for every output tile.
    input_order : tuple[int, ...]
        Engine-declared reconstruction order.
    tile_dims : tuple[str, ...]
        Ordered spatial dimensions.
    leading_shape : tuple[int, ...]
        Output axes excluded from spatial tile geometry.
    monarch : MonarchConfig
        Actor-mesh execution settings.
    """

    settings: TileStitchSettings
    input_tiles: tuple[InputTile, ...]
    output_tiles: tuple[OutputTile, ...]
    output_to_inputs: dict[int, tuple[int, ...]]
    input_order: tuple[int, ...]
    tile_dims: tuple[str, ...]
    leading_shape: tuple[int, ...]
    monarch: MonarchConfig


@dataclass(frozen=True, slots=True)
class StitchWorkUnit:
    """Input and output binding for one position and timepoint.

    Parameters
    ----------
    input_path : str
        Input OME-Zarr position.
    output_path : str
        Destination OME-Zarr position.
    channel_idx : int
        Input channel index.
    timepoint : int
        Input and output time index.
    output_channel_index : int
        Destination channel slot.
    """

    input_path: str
    output_path: str
    channel_idx: int
    timepoint: int
    output_channel_index: int = 0


def write_program(
    program: StitchProgram,
    run_dir: str | Path,
    filename: str = "program.pkl",
) -> str:
    """Serialize one static stitch program.

    Parameters
    ----------
    program : StitchProgram
        Static run definition.
    run_dir : str or pathlib.Path
        Directory in which to create the program file.
    filename : str, optional
        Program filename.

    Returns
    -------
    str
        Serialized program path.
    """
    path = Path(run_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(program, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return str(path)


_PROGRAM_CACHE: dict[str, StitchProgram] = {}


def load_program(program_path: str) -> StitchProgram:
    """Load and process-cache a serialized static program.

    Parameters
    ----------
    program_path : str
        Serialized program path.

    Returns
    -------
    StitchProgram
        Deserialized static program.
    """
    if program_path not in _PROGRAM_CACHE:
        with open(program_path, "rb") as stream:
            _PROGRAM_CACHE[program_path] = pickle.load(stream)
    return _PROGRAM_CACHE[program_path]


def program_from_engine_plan(
    engine_plan: TileStitchPlan,
    *,
    settings: TileStitchSettings,
    monarch: MonarchConfig,
    leading_shape: tuple[int, ...] = (1, 1),
) -> StitchProgram:
    """Freeze reusable waveorder geometry into a static stitch program.

    Parameters
    ----------
    engine_plan : TileStitchPlan
        Geometry and dependencies produced by waveorder.
    settings : TileStitchSettings
        Reconstruction, tiling, and blend settings.
    monarch : MonarchConfig
        Actor-mesh execution settings.
    leading_shape : tuple[int, ...], optional
        Non-spatial output shape.

    Returns
    -------
    StitchProgram
        Serializable static program.
    """
    return StitchProgram(
        settings=settings,
        input_tiles=tuple(engine_plan.input_tiles),
        output_tiles=tuple(engine_plan.output_tiles),
        output_to_inputs={
            output_id: tuple(input_ids)
            for output_id, input_ids in engine_plan.output_to_inputs.items()
        },
        input_order=tuple(engine_plan.input_order),
        tile_dims=tuple(engine_plan.tile_dims),
        leading_shape=tuple(leading_shape),
        monarch=monarch,
    )
