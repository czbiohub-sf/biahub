"""Tile a fixed set of movies (e.g. channels) into one grid movie.

Each entry in ``movs`` becomes one tile, with its own crop box, rotation, frame
step, legend and timestamp. Optionally batch-render one grid per FOV by listing
``fov_list`` and embedding ``fov_placeholder`` (default ``{fov}``) in the movie
paths, legends and output filename.
"""

from copy import deepcopy
from pathlib import Path

import click

from biahub.cli.parsing import config_filepath
from biahub.cli.utils import yaml_to_model
from biahub.settings import GridMoviesSettings
from biahub.visualize.video_utils import TileSpec, tile_videos


def get_unique_output_path(base_path: Path) -> Path:
    """Append ``_2``, ``_3``, ... to avoid overwriting an existing file."""
    if not base_path.exists():
        return base_path
    suffix = 2
    while True:
        candidate = base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _tiles_from_settings(settings: GridMoviesSettings) -> list[TileSpec]:
    """Build TileSpecs from the ``movs`` list, skipping missing files."""
    tiles = []
    for mov in settings.movs:
        path = Path(mov.mov_path)
        if not path.exists():
            print(f"  !! Skipping missing video: {path}")
            continue
        tiles.append(
            TileSpec(
                path=path,
                legend=mov.legend,
                crop_box=tuple(mov.crop_box) if mov.crop_box else None,
                rotation=mov.rotation,
                frame_step=mov.frame_step,
                time_sampling_min=mov.frame_time_sampling_min,
                legend_font_size=mov.legend_font_size,
                timestamp_font_size=mov.timestamp_font_size,
            )
        )
    return tiles


def _substitute_fov(settings: GridMoviesSettings, fov: str) -> GridMoviesSettings:
    """Return a copy of ``settings`` with ``fov_placeholder`` filled in for one FOV.

    Paths and the output filename use the ``a_b_c`` form of the FOV id; legends
    keep the human-readable ``a/b/c`` form.
    """
    token = settings.fov_placeholder
    fov_path = fov.replace("/", "_")
    out = deepcopy(settings)
    out.output_filename = out.output_filename.replace(token, fov_path)
    for mov in out.movs:
        mov.mov_path = mov.mov_path.replace(token, fov_path)
        if mov.legend:
            mov.legend = mov.legend.replace(token, fov)
    return out


def grid_movies(settings: GridMoviesSettings) -> None:
    """Tile the configured movies into a single grid movie."""
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles = _tiles_from_settings(settings)
    if not tiles:
        raise RuntimeError("No videos could be opened successfully.")

    out_file = get_unique_output_path(output_dir / settings.output_filename)
    tile_videos(
        tiles,
        grid=tuple(settings.grid),
        out_file=out_file,
        fps=settings.fps,
        max_frames=settings.max_frames,
        quality=settings.quality,
    )


@click.command("grid-movies")
@config_filepath()
def grid_movies_cli(config_filepath: Path):
    """Tile a set of movies (channels) into a grid, optionally one per FOV."""
    settings = yaml_to_model(config_filepath, GridMoviesSettings)

    if settings.fov_list:
        for fov in settings.fov_list:
            print(f"\n=== FOV {fov} ===")
            grid_movies(_substitute_fov(settings, fov))
    else:
        grid_movies(settings)


if __name__ == "__main__":
    grid_movies_cli()
