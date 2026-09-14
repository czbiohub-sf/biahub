import glob

from pathlib import Path
from typing import Annotated, Literal

import typer

from iohub.cli import OptionEatAll
from natsort import natsorted
from typer.core import TyperOption


def _validate_and_process_paths(value: list[Path]) -> list[Path]:
    """Sort input positions and reject plate roots."""
    from iohub.ngff import Plate, open_ome_zarr

    input_paths = [path for path in natsorted(value) if path.is_dir()]
    with open_ome_zarr(input_paths[0], mode="r") as dataset:
        if isinstance(dataset, Plate):
            raise ValueError(
                "Please supply a single position instead of an HCS plate. Likely fix: "
                "replace 'input.zarr' with 'input.zarr/0/0/0'"
            )
    return input_paths


def _validate_and_process_config_paths(value: list[Path]) -> list[Path]:
    matched_paths = []
    for pattern in value:
        expanded = glob.glob(str(pattern))
        if not expanded:
            raise typer.BadParameter(f"No files matched pattern: {pattern}")
        matched_paths.extend(expanded)

    validated = []
    for path in natsorted(map(Path, matched_paths)):
        if not path.exists():
            raise typer.BadParameter(f"Path does not exist: {path}")
        if not path.is_file():
            raise typer.BadParameter(f"Expected a file, not a directory: {path}")
        if path.suffix.lower() not in [".yml", ".yaml"]:
            raise typer.BadParameter(f"Expected a .yml file, got: {path}")
        validated.append(path)
    return validated


InputPositionDirpaths = Annotated[
    list[Path],
    typer.Option(
        "--input-position-dirpaths",
        "-i",
        callback=_validate_and_process_paths,
        help=(
            'Paths to input positions, for example: "input.zarr/0/0/0", '
            '"input.zarr/0/0/[0-9]", or "input.zarr/*/*/*"'
        ),
    ),
]

SourcePositionDirpaths = Annotated[
    list[Path],
    typer.Option(
        "--source-position-dirpaths",
        "-s",
        callback=_validate_and_process_paths,
        help=(
            'Paths to source positions, for example: "source.zarr/0/0/0" '
            'or "source.zarr/*/*/*"'
        ),
    ),
]

TargetPositionDirpaths = Annotated[
    list[Path],
    typer.Option(
        "--target-position-dirpaths",
        "-t",
        callback=_validate_and_process_paths,
        help=(
            'Paths to target positions, for example: "target.zarr/0/0/0" '
            'or "target.zarr/*/*/*"'
        ),
    ),
]

ConfigFilepaths = Annotated[
    list[Path],
    typer.Option(
        "--config-filepaths",
        "-c",
        callback=_validate_and_process_config_paths,
        help="Paths to YAML configuration files. All must be existing files with .yml extension.",
    ),
]

ConfigFilepath = Annotated[
    Path,
    typer.Option(
        "--config-filepath",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to YAML configuration file.",
    ),
]

OutputDirpath = Annotated[
    Path,
    typer.Option(
        "--output-dirpath",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Path to output directory",
    ),
]

OutputFilepath = Annotated[
    Path,
    typer.Option(
        "--output-filepath",
        "-o",
        file_okay=True,
        dir_okay=False,
        help="Path to output file",
    ),
]

SbatchFilepath = Annotated[
    Path | None,
    typer.Option(
        "--sbatch-filepath",
        "-sb",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help=(
            "SBATCH filepath that contains slurm parameters to overwrite defaults. "
            "For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU."
        ),
    ),
]

SbatchFilepathPreprocess = Annotated[
    Path | None,
    typer.Option(
        "--sbatch-filepath-preprocess",
        "-sb-preprocess",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help=(
            "SBATCH filepath that contains slurm parameters to overwrite defaults. "
            "For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU."
        ),
    ),
]

SbatchFilepathPredict = Annotated[
    Path | None,
    typer.Option(
        "--sbatch-filepath-predict",
        "-sb-predict",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help=(
            "SBATCH filepath that contains slurm parameters to overwrite defaults. "
            "For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU."
        ),
    ),
]


_GREEDY_OPTION_NAMES = {
    "input_position_dirpaths",
    "source_position_dirpaths",
    "target_position_dirpaths",
    "config_filepaths",
}


def install_eat_all_options(command) -> None:
    """Install iohub's greedy parser on biahub list options."""
    for param in command.params:
        if isinstance(param, TyperOption) and param.name in _GREEDY_OPTION_NAMES:
            param.__class__ = OptionEatAll


def sbatch_to_submitit(filepath: str) -> dict:
    """Read a text configuration file and return a dictionary of parameters.

    Which can be passed to the submitit executor. This file can contain parameters
    starting with #SBATCH to configure SLURM jobs or parameters starting with #LOCAL
    to configure local jobs. The submitit executor will only apply valid parameters
    and will, for example, ignore local parameters when running on SLURM.

    Parameters
    ----------
    value : Path
        Path to sbatch file

    Returns
    -------
    dict
        Dictionary of slurm parameters

    Example:

    --- sbatch_file.sh ---
    #SBATCH --mem-per-cpu=16G
    #SBATCH --time=1:00:00
    #LOCAL --cpus-per-task=1
    ---

    >>> dict = sbatch_to_submitit(Path("sbatch_file.sh"))
    >>> print(dict)
    {'slurm_mem_per_cpu': '16G', 'slurm_time': '1:00:00', 'cpus_per_task': 1}
    """
    with open(filepath) as f:
        sbatch_file = f.readlines()

    keywords = ["SBATCH", "LOCAL"]
    sbatch_dict = {}
    for line in sbatch_file:
        for keyword in keywords:
            if line.startswith(f"#{keyword} --"):
                line = line.strip(f"#{keyword} --").strip()
                key, value = line.split("=", 1)
                key = key.replace("-", "_").strip()
                try:
                    value = int(value.strip())
                except ValueError:
                    # If conversion to int fails, keep it as a string
                    value = value.strip()
                if keyword == "SBATCH":
                    sbatch_dict["slurm_" + key] = value
                elif keyword == "LOCAL":
                    sbatch_dict[key] = value

    return sbatch_dict


local = Annotated[
    bool,
    typer.Option(
        "--local",
        "-l",
        help="Run jobs locally instead of submitting to SLURM.",
    ),
]

cluster = Annotated[
    Literal["slurm", "local", "debug"],
    typer.Option(
        "--cluster",
        case_sensitive=False,
        show_default=True,
        help=(
            "Execution cluster: 'slurm' submits to a Slurm cluster, "
            "'local' runs jobs as subprocesses on this machine, "
            "'debug' runs jobs in-process in the foreground."
        ),
    ),
]

init_only = Annotated[
    bool,
    typer.Option(
        "--init",
        help="Only initialize the output store and exit; skip per-position processing.",
    ),
]

monitor = Annotated[
    bool,
    typer.Option(
        "--monitor",
        "-m",
        help="Monitor of submitted SLURM jobs.",
    ),
]

resume = Annotated[
    bool,
    typer.Option(
        "--resume/--no-resume",
        show_default=True,
        help=(
            "Skip the (time, channel) units this position already finished in an "
            "earlier attempt instead of recomputing the whole position. For retrying "
            "a run that was interrupted, e.g. by Slurm preemption. A finished unit is "
            "skipped without re-deriving it, so pass --no-resume (or use a fresh "
            "output store) when the settings changed."
        ),
    ),
]

num_processes = Annotated[
    int,
    typer.Option(
        "--num-processes",
        "-j",
        help="Number of parallel processes",
    ),
]
