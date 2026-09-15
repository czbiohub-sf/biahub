import os
import re
import subprocess

from pathlib import Path
from typing import Annotated

import imageio_ffmpeg
import typer


def detect_crop_params(file_path):
    """Detect crop parameters using ffmpeg cropdetect."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-i",
        file_path,
        "-vf",
        "cropdetect=24:16:0",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # Extract the crop parameters from the output
    crop_line = None
    for line in result.stderr.splitlines():
        if "crop=" in line:
            crop_line = line

    if crop_line:
        # Get the last crop= line and extract the parameters using regex
        crop_params = re.search(r"crop=(\S+)", crop_line)
        if crop_params:
            return crop_params.group(1)

    return None


def process_video(file_path, output_dir):
    """Process a single video: detect crop parameters and apply cropping."""
    filename = os.path.basename(file_path)
    filename_no_ext = os.path.splitext(filename)[0]

    # Detect crop parameters
    crop_params = detect_crop_params(file_path)

    if crop_params:
        # Define the output path
        output_path = os.path.join(output_dir, f"{filename_no_ext}_cropped.mp4")

        # Apply cropping using imageio-ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-i",
            file_path,
            "-vf",
            f"crop={crop_params}",
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            output_path,
        ]
        subprocess.run(command)
        typer.echo(f"Processed {filename_no_ext}")
    else:
        typer.echo(f"Could not determine crop parameters for {filename_no_ext}")


cli = typer.Typer(add_completion=False)


@cli.command("crop-background")
def main(
    input_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False),
    ],
    output_dir: Annotated[Path, typer.Argument()],
):
    """Batch process videos in VIDEO-DIR and save the output to OUTPUT-DIR."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each mp4 file in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".mp4"):
            file_path = os.path.join(input_dir, file_name)
            process_video(file_path, output_dir)


if __name__ == "__main__":
    cli()
