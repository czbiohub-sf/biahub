import os
import re
import subprocess

import click


def detect_crop_params(file_path):
    """Detect crop parameters using ffmpeg cropdetect."""
    command = ['ffmpeg', '-i', file_path, '-vf', 'cropdetect=24:16:0', '-f', 'null', '-']
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    # Extract the crop parameters from the output
    crop_line = None
    for line in result.stderr.splitlines():
        if 'crop=' in line:
            crop_line = line

    if crop_line:
        # Get the last crop= line and extract the parameters using regex
        crop_params = re.search(r'crop=(\S+)', crop_line)
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

        # Apply cropping
        command = [
            'ffmpeg',
            '-i',
            file_path,
            '-vf',
            f'crop={crop_params}',
            '-c:v',
            'libx264',
            '-c:a',
            'copy',
            output_path,
        ]
        subprocess.run(command)
        click.echo(f"Processed {filename_no_ext}")
    else:
        click.echo(f"Could not determine crop parameters for {filename_no_ext}")


@click.command()
@click.argument('video_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path())
def main(video_dir, output_dir):
    """Batch process videos in VIDEO_DIR and save the output to OUTPUT_DIR."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each mp4 file in the directory
    for file_name in os.listdir(video_dir):
        if file_name.endswith(".mp4"):
            file_path = os.path.join(video_dir, file_name)
            process_video(file_path, output_dir)


if __name__ == "__main__":
    main()
