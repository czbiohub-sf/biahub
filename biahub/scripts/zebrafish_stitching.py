from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from iohub import open_ome_zarr
from natsort import natsorted
from skimage.registration import phase_cross_correlation


def _masked_phase_cross_correlation(pos_0, pos_1, overlap_percentage: float = 0.3):
    """
    Calculate the shift between two images using masked phase cross-correlation with
    overlap defined as a percentage of the image width.

    Parameters:
    - pos_0: The first image (2D array).
    - pos_1: The second image (2D array).
    - overlap_percentage: The percentage of overlap between the two images (float, 0-1).

    Returns:
    - (y, x): The calculated shift between the two images (dy, dx).
    """
    y0, x0 = pos_0.shape
    y1, x1 = pos_1.shape

    # Calculate overlap in pixels based on the percentage of the image width
    overlap = int(np.ceil(x1 * overlap_percentage))

    # Create masks for the moving (pos_1) and reference (pos_0) images
    mask_mov = np.ones_like(pos_1)
    mask_ref = np.ones_like(pos_0)

    # Mask the non-overlapping part in the moving image (pos_1)
    mask_mov[:, overlap:] = 0

    # Optional: if you want to mask part of the reference image, modify mask_ref similarly
    # mask_ref[:, : int(np.ceil(w0 / 2))] = 0  # Example masking, can be adjusted

    # Perform phase cross-correlation with the masks
    (y, x), _, _ = phase_cross_correlation(
        pos_0,
        pos_1,
        reference_mask=mask_ref,
        moving_mask=mask_mov,
        overlap_ratio=overlap_percentage,
    )

    # Adjust the x-shift by centering it based on the widths of the two images
    x = (x0 / 2 + x1 / 2) - x
    print(f"Shifting (y,x): ({y},{x})")

    return (y, x)


def compute_canvas_size_and_offsets(images: list, shifts: list):
    """
    Compute the total canvas size and determine the x and y offsets for the first image
    to avoid cropping, based on the cumulative shifts.

    :param images: List of images to be stitched.
    :param shifts: List of tuples with the (y_shift, x_shift) between each pair of images.
    :return: (total_width, total_height, x_offset, y_offset, cumulative_shifts_x, cumulative_shifts_y)
    """
    cumulative_y_shift = 0
    cumulative_x_shift = 0
    ymin = 0
    xmin = 0
    ymax = images[0].shape[-2]  # Start ymax with the height of the first image
    xmax = images[0].shape[-1]  # Start xmax with the width of the first image

    # Lists to store cumulative x and y shifts for each image
    cumulative_shifts_x = [0]  # First image starts at x=0
    cumulative_shifts_y = [0]  # First image starts at y=0

    # Track ymin, ymax, xmin, and xmax as we go through the shifts
    for i, (y_shift, x_shift) in enumerate(shifts):
        cumulative_x_shift += x_shift
        cumulative_y_shift += y_shift

        # Update the cumulative shifts for the next image
        cumulative_shifts_x.append(cumulative_x_shift)
        cumulative_shifts_y.append(cumulative_y_shift)

        # Update ymin and ymax based on the cumulative shifts
        ymin = min(ymin, cumulative_y_shift)
        ymax = max(ymax, cumulative_y_shift + images[i + 1].shape[-2])

        # Update xmin and xmax based on the cumulative shifts
        xmin = min(xmin, cumulative_x_shift)
        xmax = max(xmax, len(shifts) * images[i + 1].shape[-1] - cumulative_x_shift)

    # Calculate the offsets for the first image to avoid cropping
    x_offset = -xmin  # Shift the first image right if xmin < 0
    y_offset = -ymin  # Shift the first image down if ymin < 0

    # Calculate total canvas width and height
    total_width = xmax - xmin
    total_height = ymax - ymin

    return (
        total_height,
        total_width,
        x_offset,
        y_offset,
        cumulative_shifts_y,
        cumulative_shifts_x,
    )


def weighted_blending(image1, image2, overlap_shape: tuple, orientation: bool = False):
    """
    Blend the overlapping region between two images using weighted blending.

    Parameters:
    - image1: The first image (2D array).
    - image2: The second image (2D array).
    - overlap_shape: The shape of the overlapping region (tuple of height and width).
    - orientation: If True, blend orientation
    Returns:
    - blended_part: The blended overlapping region (2D array).
    """

    # FIXME: takes CYX but C=0,1 retardance,orientation are hardcoded.
    channels1, height1, width1 = image1.shape
    channels2, height2, width2 = image2.shape
    overlap_height, overlap_width = overlap_shape
    blended_part = np.zeros((overlap_height, overlap_width), dtype=np.float32)

    for x in range(overlap_width):
        alpha = x / overlap_width  # Weight for blending (goes from 0 to 1)
        for y in range(overlap_height):  # Blend only up to the height of the smaller image
            if orientation:
                # polar to cartesian
                px1_retardance = image1[0, height1 - overlap_height + y, x] * (1 - alpha)
                px2_retardance = image2[0, y, x] * alpha
                theta_1 = image1[1, height1 - overlap_height + y, x]
                theta_2 = image2[1, y, x]

                # cartesian blend
                x_1 = px1_retardance * np.cos(2 * theta_1)
                x_2 = px2_retardance * np.cos(2 * theta_2)
                y_1 = px1_retardance * np.sin(2 * theta_1)
                y_2 = px2_retardance * np.sin(2 * theta_2)
                blended_pixel = (0.5 * np.arctan2(y_1 + y_2, x_1 + x_2)) % np.pi
            else:
                pixel1 = image1[0, height1 - overlap_height + y, x]
                pixel2 = image2[0, y, x]
                # Weighted blend of the two pixels
                blended_pixel = (1 - alpha) * pixel1 + alpha * pixel2
            blended_part[y, x] = blended_pixel
    return blended_part


def stitch_images(
    images: list, relative_shifts: list, verbose: bool = False, orientation: bool = False
):
    """
    Stitch a list of images together based on the calculated shifts between them.
    """
    print(f'Stitching {len(images)} images...')
    # Calculate the required canvas size and the offset to place the first image in the middle
    (
        total_height,
        total_width,
        y_offset,
        x_offset,
        cumulative_shifts_y,
        cumulative_shifts_x,
    ) = compute_canvas_size_and_offsets(images, relative_shifts)
    # print(f'Canvas size: {total_height} x {total_width}, Offset: ({y_offset}, {x_offset}, Cumulative shifts: {cumulative_shifts_y}, {cumulative_shifts_x})')

    # Create a blank canvas for the final blended image
    canvas = np.zeros((int(total_height), int(total_width)), dtype=np.float32)

    # FIXME:hardcoding these orientation c=1 else c=0
    if orientation:
        c_idx = 1
    else:
        c_idx = 0

    for i in range(len(images)):
        # Get dimensions of the images
        channels1, height1, width1 = images[i - 1].shape
        channels2, height2, width2 = images[i].shape

        print(f'Placing image {i}...')
        # Get the current position based on cumulative shifts and apply offsets
        current_x = cumulative_shifts_x[i] + x_offset
        current_y = cumulative_shifts_y[i] + y_offset

        # Blend the overlapping region if this isn't the first image
        if i > 0:
            # Get the overalpping region dimensions
            overlap_width = int(
                abs(relative_shifts[i - 1][1])
            )  # Determine overlap width start from the left
            y_shift = relative_shifts[i - 1][0]

            # Set overlap height and starting points based on y_shift
            if y_shift >= 0:
                overlap_height = int(min(height1 - y_shift, height2))
                y1_start = int(y_shift)  # Overlap starts y_shift rows down in the first image
                y2_start = 0  # Starts at the top of the second image
                blend_y = current_y

            else:
                overlap_height = int(min(height1, height2 + y_shift))
                y1_start = 0  # Starts at the top of the first image
                y2_start = int(-y_shift)  # Starts -y_shift rows down in the second image
                blend_y = current_y + max(
                    y1_start, y2_start
                )  # Adjust y based on overlap starting point

            canvas[
                int(current_y) : int(current_y + height2),
                int(i * width1 - current_x + overlap_width) : int(
                    i * width1 - current_x + overlap_width + (width2 - overlap_width)
                ),
            ] = images[i][c_idx, :, overlap_width:]
            # Blending
            blended_part = weighted_blending(
                image1=images[i - 1][:, y1_start : y1_start + overlap_height, -overlap_width:],
                image2=images[i][:, y2_start : y2_start + overlap_height, :overlap_width],
                overlap_shape=(overlap_height, overlap_width),
                orientation=orientation,
            )
            canvas[
                int(blend_y) : int(blend_y + overlap_height),
                int(i * width1 - current_x) : int(i * width1 - current_x + overlap_width),
            ] = blended_part
        else:
            # Paste the first image directly with the calculated offset
            canvas[
                int(current_y) : int(current_y) + images[i][0].shape[-2],
                int(current_x) : int(current_x) + images[i][0].shape[-1],
            ] = images[i][c_idx]

        if verbose:
            plt.figure()
            plt.imshow(canvas)
    return canvas


if __name__ == "__main__":
    # Find the shifts using State1-2 from the raw dataset
    dataset_folder = "/hpc/projects/comp.micro/zebrafish/2023_02_02_zebrafish_casper/3-stitch/x-test_stitching/intoto_casper_2_raw_toy_dataset.zarr"
    dataset_folder = Path(dataset_folder)
    input_positions_paths = [Path(path) for path in natsorted(glob(f"{dataset_folder}/*/*/*"))]
    channel_for_registration = 'State2'

    # Make a 3D CYX image stack to calculate the 2D shifts (translation)
    images = []
    for i, position_path in enumerate(input_positions_paths):
        print(f"Position: {position_path}")
        position = open_ome_zarr(position_path)
        channel_names = position.channel_names
        c_idx = channel_names.index(channel_for_registration)
        T, C, Z, Y, X = position['0'].shape
        images.append(position['0'][1, c_idx : c_idx + 1, Z // 2])

    # Calculate the relative shifts between pair of images
    # FIXME hardcoding to grab channel =0
    relative_shifts = []
    for i in range(1, len(images)):
        print(f"Shift between images {i-1} and {i}:")
        shift = _masked_phase_cross_correlation(images[i - 1][0], images[i][0], 0.2)
        relative_shifts.append(shift)

    # Stitch the images together based on the calculated shifts
    stitched_fish = stitch_images(images, relative_shifts, verbose=True, orientation=False)
    with open_ome_zarr(
        "./test_blending_retardance.zarr",
        layout='fov',
        mode="w",
        channel_names=['Orientation'],
    ) as store:
        store.create_image(
            "0",
            stitched_fish[
                np.newaxis,
                np.newaxis,
                np.newaxis,
            ],
        )

    # Orientation blending
    dataset_folder = "/hpc/projects/comp.micro/zebrafish/2023_02_02_zebrafish_casper/3-stitch/x-test_stitching/intoto_casper_2_combined_toy_dataset_ret_ori.zarr"
    channels_to_process = ['Retardance', 'Orientation']
    input_positions_paths = [Path(path) for path in natsorted(glob(f"{dataset_folder}/*/*/*"))]

    images = []
    for i, position_path in enumerate(input_positions_paths):
        print(f"Position: {position_path}")
        position = open_ome_zarr(position_path)
        channel_names = position.channel_names
        c_idx = [channel_names.index(ch) for ch in channels_to_process]
        T, C, Z, Y, X = position['0'].shape
        images.append(position['0'].oindex[0, c_idx, Z // 2])

    stitched_fish = stitch_images(images, relative_shifts, verbose=True, orientation=True)

    with open_ome_zarr(
        "./test_blending_orientation.zarr", layout='fov', mode="w", channel_names='Orientation'
    ) as store:
        position = store.create_position("0")
        position.add_image(stitched_fish)
