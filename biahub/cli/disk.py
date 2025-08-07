import subprocess
import shutil
import os
import click
from pathlib import Path

def resolve_symlink(path: str) -> str:
    """Resolve symlinks to get the actual target path."""
    return os.path.realpath(path)

def get_dir_size_du(path: str) -> int:
    """
    Use `du -sb` to get the total size in bytes of a directory or file.
    Follows symlinks to measure the actual data.
    """
    resolved_path = resolve_symlink(path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"[get_dir_size_du] Path does not exist: {resolved_path}")
    
    try:
        result = subprocess.run(
            ['du', '-sb', resolved_path],
            capture_output=True,
            check=True,
            text=True
        )
        size_bytes = int(result.stdout.strip().split()[0])
        return size_bytes
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[get_dir_size_du] Failed to run du on {resolved_path}: {e.stderr.strip()}")


def check_disk_space_with_du(input_path: str, output_path: str, margin: float = 1.1, verbose: bool = True) -> bool:
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
        click.echo(f"...........................................")
        click.echo(f"Input Size: {input_size / 1e12:.3f} TB")
        click.echo(f"Estimated Ouput Size ({margin:.3f}x): {required_space / 1e12:.3f} TB")
        click.echo(f"Available Space: {available_space / 1e12:.3f} TB")
        click.echo(f"...........................................")


    return available_space >= required_space

if __name__ == "__main__":
    # Example usage
    input_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/0-reconstruct/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr")
    output_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/test/1-preprocess/label-free/0-reconstruct/test.zarr")
    os.makedirs(output_path.parent, exist_ok=True)
    
    if check_disk_space_with_du(input_path, output_path):
        print("Sufficient disk space available.")
    else:
        print("Insufficient disk space available.")
