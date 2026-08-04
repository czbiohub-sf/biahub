"""Tile per-FOV MP4s into per-well grid movies.

Given a directory of ``<prefix>_<well>_...mp4`` files, group them by well id and
lay each well's FOVs out on a grid, producing one (or more) movies per well.
All tiles in a run share the same rotation, frame step and timestamp sampling;
each tile is labelled with its source file stem.
"""

from math import ceil
from pathlib import Path

import click

from biahub.cli.parsing import config_filepath
from biahub.cli.utils import yaml_to_model
from biahub.settings import GridPerWellSettings
from biahub.visualize.video_utils import TileSpec, tile_videos


def get_well_id(stem: str) -> str:
    """Parse the well id from a ``prefix_well_...`` filename stem."""
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse well id from filename: {stem}")
    return parts[1]


def grid_per_well(settings: GridPerWellSettings) -> None:
    """Tile per-FOV movies into per-well grids."""
    mov_path = Path(settings.mov_path)
    output_path = Path(settings.output_path)
    grid = tuple(settings.grid)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(mov_path.glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No .mp4 files found in {mov_path}")

    by_well: dict = {}
    for f in files:
        by_well.setdefault(get_well_id(f.stem), []).append(f)
    print(f"Found wells: {sorted(by_well.keys())}")

    grid_size = grid[0] * grid[1]
    for well_id, well_files in sorted(by_well.items()):
        well_files = sorted(well_files)
        num_groups = ceil(len(well_files) / grid_size)
        print(f"\n[Well {well_id}] {len(well_files)} file(s) -> {num_groups} video(s)")

        for group_idx in range(num_groups):
            group = well_files[group_idx * grid_size : (group_idx + 1) * grid_size]
            tiles = [
                TileSpec(
                    path=f,
                    legend=f.stem if settings.add_legend else None,
                    rotation=settings.rotation,
                    frame_step=settings.frame_step,
                    time_sampling_min=settings.time_sampling_min,
                    legend_font_size=settings.legend_font_size,
                    timestamp_font_size=settings.timestamp_font_size,
                )
                for f in group
            ]
            tile_videos(
                tiles,
                grid=grid,
                out_file=output_path / f"{well_id}_{group_idx}.mp4",
                fps=settings.fps,
                max_frames=settings.max_frames,
                quality=settings.quality,
            )


@click.command("grid-per-well")
@config_filepath()
def grid_per_well_cli(config_filepath: Path):
    """Tile per-FOV MP4s into per-well grid movies."""
    settings = yaml_to_model(config_filepath, GridPerWellSettings)
    grid_per_well(settings)


if __name__ == "__main__":
    grid_per_well_cli()
