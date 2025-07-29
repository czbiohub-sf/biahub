import subprocess
import shutil
import os

import os
import subprocess

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
    available_space = shutil.disk_usage(os.path.abspath(output_path.parent)).free

    if verbose:
        print(f"[check_disk_space_with_du] Input size: {input_size / 1e9:.2f} GB")
        print(f"[check_disk_space_with_du] Required with margin ({margin:.2f}x): {required_space / 1e9:.2f} GB")
        print(f"[check_disk_space_with_du] Available at output: {available_space / 1e9:.2f} GB")

    return available_space >= required_space
