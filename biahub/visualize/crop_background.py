import os
import re
import subprocess
from pathlib import Path

import click
import imageio_ffmpeg


def detect_crop_params(file_path: str | Path) -> str | None:
    """
    Detect crop parameters using ffmpeg cropdetect.

    This function uses ffmpeg's cropdetect filter to automatically detect
    the optimal crop parameters for removing black borders from a video file.

    Parameters
    ----------
    file_path : str | Path
        Path to the input video file.

    Returns
    -------
    str | None
        Crop parameters string in the format "width:height:x:y" if detected,
        None otherwise.
    """
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
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

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


def process_video(file_path: str | Path, output_dir: str | Path) -> None:
    """
    Process a single video: detect crop parameters and apply cropping.

    This function detects optimal crop parameters for a video file and applies
    the cropping to remove black borders, saving the result as a new cropped video.

    Parameters
    ----------
    file_path : str | Path
        Path to the input video file.
    output_dir : str | Path
        Directory where the cropped video will be saved.

    Returns
    -------
    None
        The cropped video is saved to the output directory.
    """
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
        click.echo(f"Processed {filename_no_ext}")
    else:
        click.echo(f"Could not determine crop parameters for {filename_no_ext}")


@click.command("crop-background")
@click.argument("input-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output-dir", type=click.Path())
def main(input_dir: str, output_dir: str) -> None:
    """
    Batch process videos in a directory and save cropped outputs.

    This command-line tool processes all MP4 video files in the input directory,
    detects optimal crop parameters to remove black borders, and saves cropped
    versions to the output directory.

    Parameters
    ----------
    input_dir : str
        Path to the input directory containing video files.
    output_dir : str
        Path to the output directory where cropped videos will be saved.

    Returns
    -------
    None
        Cropped videos are saved to the output directory.

    Examples
    --------
    >> biahub crop-background ./videos ./cropped_videos
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each mp4 file in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".mp4"):
            file_path = os.path.join(input_dir, file_name)
            process_video(file_path, output_dir)


if __name__ == "__main__":
    main()
