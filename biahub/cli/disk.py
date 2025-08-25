import os
import shutil
import subprocess

from datetime import datetime
from pathlib import Path

import click

from humanize import naturalsize


def get_dir_size_du(path: str) -> int:
    """
    Use `du -sb` to get the total size in bytes of a directory or file.
    Follows symlinks to measure the actual data.
    """
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"[get_dir_size_du] Path does not exist: {resolved_path.as_posix()}"
        )

    try:
        result = subprocess.run(
            ['du', '-sb', resolved_path.as_posix()], capture_output=True, check=True, text=True
        )
        size_bytes = int(result.stdout.strip().split()[0])
        return size_bytes
    except subprocess.CalledProcessError as e:

        raise RuntimeError(
            f"[get_dir_size_du] Failed to run du on {resolved_path}: {e.stderr.strip()}"
        )


def check_disk_space_with_du(
    input_path: str, output_path: str, margin: float = 1.1, verbose: bool = True
) -> bool:
    """
    Check if there's enough disk space at the partition of `output_path` to store `input_path`.

    Args:
        input_path (str): File or directory to estimate size.
        output_path (str): Directory where output will be saved.
        margin (float): Safety factor (e.g., 1.1 = 10% extra).
        verbose (bool): Whether to print diagnostics.

    Returns:
        bool: True if there is enough space, False otherwise.
    """
    input_size = get_dir_size_du(input_path)
    required_space = input_size * margin
    available_space = shutil.disk_usage(os.path.abspath(output_path)).free

    if verbose:
        # Calculate humanized sizes once
        input_size_human = naturalsize(input_size, binary=True)
        required_space_human = naturalsize(required_space, binary=True)
        available_space_human = naturalsize(available_space, binary=True)

        click.echo("...........................................")
        click.echo(f"Input Size: {input_size_human}")
        click.echo(f"Estimated Ouput Size ({margin:.3f}x): {required_space_human}")
        click.echo(f"Available Space: {available_space_human}")
        click.echo("...........................................")
        # save as a txt file wih datetime
        report_path = (
            Path(output_path)
            / f"disk_space_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(report_path, 'w') as report_file:
            report_file.write(
                f"Input Size: {input_size_human}\n"
                f"Estimated Output Size ({margin:.3f}x): {required_space_human}\n"
                f"Available Space: {available_space_human}\n"
            )
    return available_space >= required_space


@click.command("check-disk-space")
@click.option(
    "--input-path",
    "-i",
    type=str,
    required=True,
    help="Path to SLURM log directory.",
)
@click.option(
    "--output-path",
    "-o",
    type=str,
    required=True,
    help="Output directory for the report CSV file.",
)
@click.option(
    "--margin",
    type=float,
    default=1.1,
    help="Safety margin for disk space check (default: 1.1, i.e., 10% extra space).",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=True,
    help="Print detailed diagnostics.",
)
def check_disk_space_cli(input_path: str, output_path: str, margin: float, verbose: bool):
    """
    CLI command to check disk space using `du -sb`.
    """
    enough_space = check_disk_space_with_du(
        input_path=input_path,
        output_path=output_path,
        margin=margin,
        verbose=verbose,
    )
    if enough_space:
        click.echo("Disk space check passed. Good to go!")
    else:
        click.echo("Disk space check failed. Not enough space available.")


if __name__ == "__main__":
    check_disk_space_cli()
