import datetime
import itertools

from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Union

import click

from iohub.ngff import Position, open_ome_zarr
from skimage.transform import downscale_local_mean
from slurmkit import SlurmParams, slurm_function, submit_function


class CreatePyramids(Protocol):
    def __call__(
        self,
        fov: Position,
        iterator: Sequence[Union[tuple[int, int], int]],
        levels: int,
        dependencies: Optional[List[Optional[int]]],
    ) -> None:
        ...


def no_pyramids(*args, **kwargs) -> None:
    pass


@slurm_function
def pyramid(t: int, c: int, fov: Position, levels: int) -> None:
    factors = (2,) * (fov.data.ndim - 2)

    # Add click.echo messages
    array = fov["0"][t, c]
    for level in range(1, levels):
        array = downscale_local_mean(array, factors)
        fov[str(level)][t, c] = array


def create_pyramids(
    fov: Position,
    iterator: Sequence[Union[tuple[int, int], int]],
    levels: int,
    dependencies: Optional[List[Optional[int]]],
) -> None:
    """Creates additional levels of multi-scales pyramid."""

    if dependencies is None:
        dependencies = [None] * len(iterator)

    elif len(dependencies) != len(iterator):
        raise ValueError(
            f"Number of dependencies ({len(dependencies)}) must match iterator length ({len(iterator)})."
        )

    params = SlurmParams(
        partition="preempted",
        cpus_per_task=16,
        mem="128G",
        time=datetime.timedelta(minutes=30),
        output="slurm_output/pyramid-%j.out",
    )

    if "1" not in fov.array_keys():
        fov.initialize_pyramid(levels=levels)

    pyramid_func = pyramid(fov=fov, levels=levels)
    try:
        for i, (t, c) in enumerate(iterator):  # type: ignore[misc]
            submit_function(pyramid_func, params, t=t, c=c, dependencies=dependencies[i])

    except TypeError:
        for i, t in enumerate(iterator):
            for c in range(fov.data.shape[1]):
                submit_function(pyramid_func, params, t=t, c=c, dependencies=dependencies[i])


@click.command("pyramid")
# can't import from parsing due to circular import
@click.argument("paths", type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.option(
    "--levels",
    "-l",
    type=int,
    default=4,
    show_default=True,
    help="Number of down-sampling levels.",
)
def pyramid_cli(paths: Sequence[Path], levels: int) -> None:
    """Creates additional levels of multi-scales pyramid."""

    for path in paths:
        fov = open_ome_zarr(path, layout="fov", mode="a")

        iterator = list(itertools.product(range(fov.data.shape[0]), range(fov.data.shape[1])))

        create_pyramids(fov, iterator, levels, dependencies=None)


if __name__ == "__main__":
    pyramid_cli()
