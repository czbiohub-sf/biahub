import logging

from pathlib import Path

import click
import numpy as np
import pandas as pd
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate, process_single_position
from numpy.typing import ArrayLike
from skimage import measure, morphology
from skimage.exposure import equalize_adapthist
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import frangi, threshold_otsu, threshold_triangle

from biahub.cli import utils
from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    local,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.utils import estimate_resources, yaml_to_model
from biahub.settings import OrganelleSegmentationSettings


def extract_features_zyx(
    labels_zyx: ArrayLike,
    intensity_zyx: ArrayLike = None,
    frangi_zyx: ArrayLike = None,
    spacing: tuple = (1.0, 1.0),
    properties: list = None,
    extra_properties: list = None,
):
    """
    Extract morphological and intensity features from labeled organelles.

    Handles both 2D (Z=1) and 3D (Z>1) data automatically
    For 2D data, processes the single Z-slice. For 3D data, performs max projection
    along Z axis before feature extraction.

    Based on:
    Lefebvre, A.E.Y.T., Sturm, G., Lin, TY. et al.
    Nellie (2025) https://doi.org/10.1038/s41592-025-02612-7

    Parameters
    ----------
    labels_zyx : ndarray
        Labeled segmentation mask with shape (Z, Y, X).
        Each unique integer value represents a different organelle instance.
    intensity_zyx : ndarray, optional
        Original intensity image with shape (Z, Y, X) for computing
        intensity-based features. If None, only morphological features computed.
    frangi_image : ndarray, optional
        Frangi vesselness response with shape (Z, Y, X) for computing
        tubularity/filament features.
    spacing : tuple
        Physical spacing in same units (e.g., µm).
        For 2D (Z=1): (pixel_size_y, pixel_size_x)
        For 3D (Z>1): (pixel_size_z, pixel_size_y, pixel_size_x)
    properties : list of str, optional
        List of standard regionprops features to compute. If None, uses default set.
        Available: 'label', 'area', 'perimeter', 'axis_major_length',
        'axis_minor_length', 'solidity', 'extent', 'orientation',
        'equivalent_diameter_area', 'convex_area', 'eccentricity',
        'mean_intensity', 'min_intensity', 'max_intensity'
    extra_properties : list of str, optional
        Additional features beyond standard regionprops. Options:
        - 'moments_hu': Hu moments (shape descriptors, 7 features)
        - 'texture': Haralick texture features (4 features: contrast, homogeneity, energy, correlation)
        - 'aspect_ratio': Major axis / minor axis ratio
        - 'circularity':  area / perimeter
        - 'frangi_intensity': Mean/min/max/sum/std of Frangi vesselness
        - 'feret_diameter_max': Maximum Feret diameter (expensive)
        - 'sum_intensity': Sum of intensity values
        - 'std_intensity': Standard deviation of intensity values

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame where each row represents one labeled object with columns
        for each computed feature. Always includes 'label' and 'channel' columns.

    Examples
    --------
    >>> # Basic morphology only
    >>> df = extract_features_zyx(labels_zyx)

    >>> # With intensity features
    >>> df = extract_features_zyx(labels_zyx, intensity_zyx=intensity)

    >>> # Custom property selection
    >>> df = extract_features_zyx(
    ...     labels_zyx,
    ...     intensity_zyx=intensity,
    ...     properties=['label', 'area', 'mean_intensity'],
    ...     extra_properties=['aspect_ratio', 'circularity']
    ... )

    >>> # Full feature set including Frangi
    >>> df = extract_features_zyx(
    ...     labels_zyx,
    ...     intensity_zyx=intensity,
    ...     frangi_image=vesselness,
    ...     extra_properties=['moments_hu', 'texture', 'frangi_intensity']
    ... )
    """

    if intensity_zyx is not None:
        assert intensity_zyx.shape == labels_zyx.shape, "Image and labels must have same shape"

    Z, _, _ = labels_zyx.shape

    # Default properties if not specified
    if properties is None:
        properties = [
            "label",
            "area",
            "perimeter",
            "axis_major_length",
            "axis_minor_length",
            "solidity",
            "extent",
            "orientation",
            "equivalent_diameter_area",
            "convex_area",
            "eccentricity",
        ]
        # Add intensity features if image provided
        if intensity_zyx is not None:
            properties.extend(["mean_intensity", "min_intensity", "max_intensity"])

    if extra_properties is None:
        extra_properties = []

    # Determine 2D vs 3D mode and set appropriate spacing
    spacing_2d = spacing if len(spacing) == 2 else spacing[-2:]

    if Z == 1:
        # Squeeze Z dimension for 2D processing
        labels_processed = labels_zyx[0]  # Shape: (Y, X)
        intensity_processed = intensity_zyx[0] if intensity_zyx is not None else None
        frangi_processed = frangi_zyx[0] if frangi_zyx is not None else None
    else:
        # Use max projection along Z for 3D -> 2D
        labels_processed = np.max(labels_zyx, axis=0)  # Shape: (Y, X)
        intensity_processed = (
            np.max(intensity_zyx, axis=0) if intensity_zyx is not None else None
        )
        frangi_processed = np.max(frangi_zyx, axis=0) if frangi_zyx is not None else None

    # Check if we have any objects to process
    if labels_processed.max() == 0:
        _logger.warning("No objects found")

    # Compute base regionprops features (those that support spacing)
    props_with_spacing = [p for p in properties if p not in ["moments_hu"]]

    try:
        props_dict = measure.regionprops_table(
            labels_processed,
            intensity_image=intensity_processed,
            properties=tuple(props_with_spacing),
            spacing=spacing_2d,
        )
        df = pd.DataFrame(props_dict)
    except Exception as e:
        _logger.warning(f"Error computing base regionprops: {e}")

    # Add Hu moments separately (without spacing)
    if "moments_hu" in properties or "moments_hu" in extra_properties:
        try:
            hu_props = measure.regionprops_table(
                labels_processed, properties=("label", "moments_hu"), spacing=(1, 1)
            )
            hu_df = pd.DataFrame(hu_props)
            # Rename columns to be clearer
            hu_rename = {f"moments_hu-{i}": f"hu_moment_{i}" for i in range(7)}
            hu_df = hu_df.rename(columns=hu_rename)
            df = df.merge(hu_df, on="label", how="left")
        except Exception as e:
            _logger.warning(f"Could not compute Hu moments: {e}")

    # Add derived metrics
    if "aspect_ratio" in extra_properties:
        minor_axis = df["axis_minor_length"].replace(0, 1)  # Avoid division by zero
        df["aspect_ratio"] = df["axis_major_length"] / minor_axis

    if "circularity" in extra_properties:
        perimeter_sq = df["perimeter"] ** 2
        df["circularity"] = np.divide(
            4 * np.pi * df["area"],
            perimeter_sq,
            out=np.ones_like(perimeter_sq),
            where=perimeter_sq != 0,
        )

    # Add expensive/iterative features
    if any(
        prop in extra_properties
        for prop in ["texture", "feret_diameter_max", "frangi_intensity"]
    ):
        regions = measure.regionprops(labels_processed, intensity_image=intensity_processed)
        extra_features = []

        for region in regions:
            features = {"label": region.label}

            # Haralick texture features
            if "texture" in extra_properties and intensity_processed is not None:
                min_r, min_c, max_r, max_c = region.bbox
                region_intensity = intensity_processed[min_r:max_r, min_c:max_c] * region.image

                # Normalize to uint8
                min_val, max_val = region_intensity.min(), region_intensity.max()
                if max_val > min_val:
                    region_uint8 = (
                        (region_intensity - min_val) / (max_val - min_val) * 255
                    ).astype(np.uint8)
                else:
                    region_uint8 = np.zeros_like(region_intensity, dtype=np.uint8)

                try:
                    glcm = graycomatrix(
                        region_uint8,
                        distances=[1],
                        angles=[0],
                        levels=256,
                        symmetric=True,
                        normed=True,
                    )
                    features["texture_contrast"] = graycoprops(glcm, "contrast")[0, 0]
                    features["texture_homogeneity"] = graycoprops(glcm, "homogeneity")[0, 0]
                    features["texture_energy"] = graycoprops(glcm, "energy")[0, 0]
                    features["texture_correlation"] = graycoprops(glcm, "correlation")[0, 0]
                except Exception:
                    features["texture_contrast"] = np.nan
                    features["texture_homogeneity"] = np.nan
                    features["texture_energy"] = np.nan
                    features["texture_correlation"] = np.nan

            # Feret diameter
            if "feret_diameter_max" in extra_properties:
                features["feret_diameter_max"] = region.feret_diameter_max

            # Frangi intensity features
            if "frangi_intensity" in extra_properties and frangi_processed is not None:
                min_r, min_c, max_r, max_c = region.bbox
                region_frangi = frangi_processed[min_r:max_r, min_c:max_c][region.image]

                if region_frangi.size > 0:
                    features["frangi_mean_intensity"] = np.mean(region_frangi)
                    features["frangi_min_intensity"] = np.min(region_frangi)
                    features["frangi_max_intensity"] = np.max(region_frangi)
                    features["frangi_sum_intensity"] = np.sum(region_frangi)
                    features["frangi_std_intensity"] = np.std(region_frangi)
                else:
                    features["frangi_mean_intensity"] = np.nan
                    features["frangi_min_intensity"] = np.nan
                    features["frangi_max_intensity"] = np.nan
                    features["frangi_sum_intensity"] = np.nan
                    features["frangi_std_intensity"] = np.nan

            extra_features.append(features)

        if extra_features:
            extra_df = pd.DataFrame(extra_features)
            df = df.merge(extra_df, on="label", how="left")

    # Add sum and std intensity if we have intensity image
    if intensity_processed is not None and (
        "sum_intensity" in extra_properties or "std_intensity" in extra_properties
    ):
        regions = measure.regionprops(labels_processed, intensity_image=intensity_processed)
        sum_std_features = []

        for region in regions:
            min_r, min_c, max_r, max_c = region.bbox
            region_pixels = intensity_processed[min_r:max_r, min_c:max_c][region.image]

            features = {"label": region.label}
            if region_pixels.size > 0:
                if "sum_intensity" in extra_properties:
                    features["sum_intensity"] = np.sum(region_pixels)
                if "std_intensity" in extra_properties:
                    features["std_intensity"] = np.std(region_pixels)
            else:
                if "sum_intensity" in extra_properties:
                    features["sum_intensity"] = np.nan
                if "std_intensity" in extra_properties:
                    features["std_intensity"] = np.nan

            sum_std_features.append(features)

        if sum_std_features:
            sum_std_df = pd.DataFrame(sum_std_features)
            df = df.merge(sum_std_df, on="label", how="left")

    return df


_logger = logging.getLogger("viscy")


def segment_zyx(
    input_zyx: ArrayLike,
    clahe_kernel_size=None,
    clahe_clip_limit=0.01,
    sigma_range=(0.5, 3.0),
    sigma_steps=5,
    auto_optimize_sigma=True,
    frangi_alpha=0.5,
    frangi_beta=0.5,
    frangi_gamma=None,
    threshold_method="otsu",
    min_object_size=10,
    apply_morphology=True,
):
    """
    Segment mitochondria from a 2D or 3D input_zyx using CLAHE preprocessing,
    Frangi filtering, and connected component labeling.

    Based on:
    Lefebvre, A.E.Y.T., Sturm, G., Lin, TY. et al.
    Nellie (2025) https://doi.org/10.1038/s41592-025-02612-7

    Parameters
    ----------
    input_zyx : ndarray
        Input image with shape (Z, Y, X) for 3D.
        If 2D, uses 2D Frangi filter. If 3D with Z=1, squeezes to 2D.
    clahe_kernel_size : int or None
        Kernel size for CLAHE (Contrast Limited Adaptive Histogram Equalization).
        If None, automatically set to max(input_zyx.shape) // 8.
    clahe_clip_limit : float
        Clipping limit for CLAHE, normalized between 0 and 1 (default: 0.01).
    sigma_range : tuple of float
        Range of sigma values to test for Frangi filter (min_sigma, max_sigma).
        Represents the scale of structures to detect.
    sigma_steps : int
        Number of sigma values to test in the range.
    auto_optimize_sigma : bool
        If True, automatically finds optimal sigma by maximizing vesselness response.
        If False, uses all sigmas in range for multi-scale filtering.
    frangi_alpha : float
        Frangi filter sensitivity to plate-j    like structures (2D) or blob-like (3D).
    frangi_beta : float
        Frangi filter sensitivity to blob-like structures (2D) or tube-like (3D).
    frangi_gamma : float or None
        Frangi filter sensitivity to background noise. If None, auto-computed.
    threshold_method : str
        Thresholding method: 'otsu', 'triangle', 'percentile', 'manual_X'.
    min_object_size : int
        Minimum object size in pixels for connected components.
    apply_morphology : bool
        If True, applies morphological closing to connect nearby structures.

    Returns
    -------
    labels : ndarray
        Instance segmentation labels with same dimensionality as input.
    vesselness : ndarray
        Filtered vesselness response with same dimensionality as input.
    optimal_sigma : float or None
        The optimal sigma value if auto_optimize_sigma=True, else None.
    """

    assert input_zyx.ndim == 3
    Z, Y, X = input_zyx.shape[-3:]

    if clahe_kernel_size is None:
        clahe_kernel_size = max(Z, Y, X) // 8

    # Apply CLAHE for contrast enhancement
    enhanced_zyx = equalize_adapthist(
        input_zyx,
        kernel_size=clahe_kernel_size,
        clip_limit=clahe_clip_limit,
    )

    # Generate sigma values
    sigmas = np.linspace(sigma_range[0], sigma_range[1], sigma_steps)

    # Auto-optimize sigma or use multi-scale
    if auto_optimize_sigma:
        optimal_sigma, vesselness = _find_optimal_sigma(
            enhanced_zyx, sigmas, frangi_alpha, frangi_beta, frangi_gamma
        )
    else:
        optimal_sigma = None
        vesselness = _multiscale_frangi(
            enhanced_zyx, sigmas, frangi_alpha, frangi_beta, frangi_gamma
        )

    # Threshold the vesselness response
    if threshold_method == "otsu":
        threshold = threshold_otsu(vesselness)
        _logger.debug(f"Otsu threshold: {threshold:.4f}")
    elif threshold_method == "triangle":
        threshold = threshold_triangle(vesselness)
        _logger.debug(f"Triangle threshold: {threshold:.4f}")
    elif threshold_method == "nellie_min":
        threshold_otsu_val = threshold_otsu(vesselness)
        threshold_triangle_val = threshold_triangle(vesselness)
        threshold = min(threshold_otsu_val, threshold_triangle_val)
        _logger.debug(
            f"Nellie-min threshold: otsu={threshold_otsu_val:.4f}, triangle={threshold_triangle_val:.4f}, using min={threshold:.4f}"
        )
    elif threshold_method == "nellie_max":
        threshold_otsu_val = threshold_otsu(vesselness)
        threshold_triangle_val = threshold_triangle(vesselness)
        threshold = max(threshold_otsu_val, threshold_triangle_val)
        _logger.debug(
            f"Nellie-max threshold: otsu={threshold_otsu_val:.4f}, triangle={threshold_triangle_val:.4f}, using max={threshold:.4f}"
        )
    elif threshold_method == "percentile":
        # Use percentile-based threshold (good for sparse features)
        threshold = np.percentile(vesselness[vesselness > 0], 95)  # Keep top 5%
        _logger.debug(f"Percentile (95th) threshold: {threshold:.4f}")
    elif threshold_method.startswith("manual_"):
        # Manual threshold: "manual_0.05" means threshold at 0.05
        threshold = float(threshold_method.split("_")[1])
        _logger.debug(f"Manual threshold: {threshold:.4f}")
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")

    binary_mask = vesselness > threshold

    _logger.debug(
        f"    Selected {binary_mask.sum()} / {binary_mask.size} pixels ({100*binary_mask.sum()/binary_mask.size:.2f}%)"
    )

    # Apply morphological operations
    if apply_morphology:
        binary_mask = morphology.binary_closing(binary_mask, footprint=morphology.ball(1))
        binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=64)

    # Label connected components
    labels = measure.label(binary_mask, connectivity=2)

    # Remove small objects
    labels = morphology.remove_small_objects(labels, min_size=min_object_size)

    if Z == 1:
        labels = labels[np.newaxis, ...]
        vesselness = vesselness[np.newaxis, ...]

    return labels, vesselness, optimal_sigma


def _find_optimal_sigma(input_zyx, sigmas, alpha, beta, gamma):
    """
    Find the optimal sigma that maximizes the vesselness response.

    Parameters
    ----------
    input_zyx : ndarray
         3D input_zyx (Z, Y, X).
    sigmas : array-like
        Sigma values to test.
    alpha, beta, gamma : float
        Frangi filter parameters.

    Returns
    -------
    optimal_sigma : float
        The sigma with the highest mean vesselness response.
    vesselness : ndarray
        The vesselness response using optimal sigma.
    """
    best_sigma = sigmas[0]
    best_score = -np.inf
    best_vesselness = None

    if input_zyx.shape[0] == 1:
        input_zyx = input_zyx[0]

    for sigma in sigmas:
        vessel = frangi(
            input_zyx,
            sigmas=[sigma],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=False,
        )

        # Score is the mean of top 10% vesselness values
        score = np.mean(
            np.partition(vessel.ravel(), -int(0.1 * vessel.size))[-int(0.1 * vessel.size) :]
        )

        if score > best_score:
            best_score = score
            best_sigma = sigma
            best_vesselness = vessel

    if input_zyx.shape[0] == 1:
        best_vesselness = best_vesselness[np.newaxis, ...]

    return best_sigma, best_vesselness


def _multiscale_frangi(input_zyx, sigmas: ArrayLike, alpha: float, beta: float, gamma: float):
    """
    Apply Frangi filter at multiple scales and return the maximum response.

    Parameters
    ----------
    input_zyx : ndarray
        3D input_zyx (Z, Y, X).
    sigmas : array-like
        Sigma values for multi-scale filtering.
    alpha, beta, gamma : float
        Frangi filter parameters.

    Returns
    -------
    vesselness : ndarray
        Maximum vesselness response across all scales.
    """
    if input_zyx.shape[0] == 1:
        input_zyx = input_zyx[0]
    vesselness = frangi(
        input_zyx,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=False,
    )
    if input_zyx.shape[0] == 1:
        vesselness = vesselness[np.newaxis, ...]
    return vesselness


def calculate_nellie_sigmas(
    min_radius_um, max_radius_um, pixel_size_um, num_sigma=5, min_step_size_px=0.2
):
    """
    Calculate sigma values following Nellie's approach.

    Parameters
    ----------
    min_radius_um : float
        Minimum structure radius in micrometers (e.g., 0.2 for diffraction limit)
    max_radius_um : float
        Maximum structure radius in micrometers (e.g., 1.0 for thick tubules)
    pixel_size_um : float
        Pixel size in micrometers
    num_sigma : int
        Target number of sigma values
    min_step_size_px : float
        Minimum step size between sigmas in pixels

    Returns
    -------
    tuple : (sigma_min, sigma_max)
        Sigma range in pixels
    """
    min_radius_px = min_radius_um / pixel_size_um
    max_radius_px = max_radius_um / pixel_size_um

    # Nellie uses radius/2 to radius/3 as sigma
    sigma_1 = min_radius_px / 2
    sigma_2 = max_radius_px / 3
    sigma_min = min(sigma_1, sigma_2)
    sigma_max = max(sigma_1, sigma_2)

    # Calculate step size with minimum constraint
    sigma_step_calculated = (sigma_max - sigma_min) / num_sigma
    sigma_step = max(min_step_size_px, sigma_step_calculated)

    sigmas = list(np.arange(sigma_min, sigma_max + sigma_step, sigma_step))

    _logger.debug(f"  Nellie-style sigmas: {sigma_min:.3f} to {sigma_max:.3f} pixels")
    _logger.debug(
        f"  Radius range: {min_radius_um:.3f}-{max_radius_um:.3f} µm = {min_radius_px:.2f}-{max_radius_px:.2f} pixels"
    )
    _logger.debug(f"  Sigma values: {[f'{s:.2f}' for s in sigmas]}")

    return (sigma_min, sigma_max)


def segment_organelles_data(
    czyx_data: ArrayLike,
    channel_kwargs_dict: dict,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Segment multiple channels of organelles using Frangi filtering with per-channel parameters.

    Parameters
    ----------
    czyx_data : ArrayLike
        Input image with shape (C, Z, Y, X).
    channel_kwargs_dict : dict
        Dictionary mapping channel index to segment_zyx kwargs.
        Keys are channel indices, values are dicts of kwargs for segment_zyx.
    spacing : tuple
        Physical spacing (Z, Y, X) in same units (e.g., µm).

    Returns
    -------
    labels_czyx : ndarray
        Instance segmentation labels with shape (C, Z, Y, X).
    """
    C, Z, Y, X = czyx_data.shape
    labels_czyx = np.zeros((C, Z, Y, X), dtype=np.uint16)

    for c_idx, segment_kwargs in channel_kwargs_dict.items():
        _logger.info(f"Segmenting channel {c_idx} with kwargs: {segment_kwargs}")

        # Extract ZYX data for this channel
        zyx_data = czyx_data[c_idx]

        # Segment this channel
        labels, vesselness, optimal_sigma = segment_zyx(zyx_data, **segment_kwargs)

        # Store labels
        labels_czyx[c_idx] = labels

        _logger.info(
            f"Channel {c_idx}: Found {labels.max()} objects, optimal_sigma={optimal_sigma}"
        )

    return labels_czyx


def extract_organelle_features_data(
    labels_zyx: ArrayLike,
    intensity_zyx: ArrayLike = None,
    frangi_zyx: ArrayLike = None,
    spacing: tuple = (1.0, 1.0, 1.0),
    properties: list = None,
    extra_properties: list = None,
    fov_name: str = None,
    t_idx: int = None,
) -> pd.DataFrame:
    """
    Extract morphological and intensity features from labeled organelles.

    Wrapper around extract_features_zyx that adds metadata columns.

    Parameters
    ----------
    labels_zyx : ArrayLike
        Labeled segmentation mask with shape (Z, Y, X).
    intensity_zyx : ArrayLike, optional
        Original intensity image with shape (Z, Y, X).
    frangi_zyx : ArrayLike, optional
        Frangi vesselness response with shape (Z, Y, X).
    spacing : tuple
        Physical spacing (Z, Y, X) in same units (e.g., µm).
    properties : list of str, optional
        List of standard regionprops features to compute.
    extra_properties : list of str, optional
        Additional features beyond standard regionprops.
    fov_name : str, optional
        Field of view identifier to add to DataFrame.
    t_idx : int, optional
        Timepoint index to add to DataFrame.

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame with features + metadata columns (fov_name, t).
    """
    # Extract features using existing function
    features_df = extract_features_zyx(
        labels_zyx=labels_zyx,
        intensity_zyx=intensity_zyx,
        frangi_zyx=frangi_zyx,
        spacing=spacing,
        properties=properties,
        extra_properties=extra_properties,
    )

    # Add metadata columns
    if fov_name is not None:
        features_df["fov_name"] = fov_name
    if t_idx is not None:
        features_df["t"] = t_idx

    return features_df


@click.command("segment-organelles")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@sbatch_filepath()
@local()
@monitor()
def segment_organelles_cli(
    input_position_dirpaths: list[str],
    config_filepath: Path,
    output_dirpath: str,
    sbatch_filepath: str | None = None,
    local: bool = False,
    monitor: bool = True,
):
    """
    Segment organelles using Frangi filtering with per-channel configuration.

    >> biahub segment-organelles \\
        -i ./input.zarr/*/*/* \\
        -c ./organelle_segment_params.yml \\
        -o ./output.zarr
    """

    # Convert string paths to Path objects
    output_dirpath = Path(output_dirpath)
    config_filepath = Path(config_filepath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    if sbatch_filepath is not None:
        sbatch_filepath = Path(sbatch_filepath)

    # Handle single position or wildcard filepath
    output_position_paths = utils.get_output_paths(input_position_dirpaths, output_dirpath)

    # Load the first position to infer dataset information
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as input_dataset:
        T, C, Z, Y, X = input_dataset.data.shape
        settings = yaml_to_model(config_filepath, OrganelleSegmentationSettings)
        scale = input_dataset.scale
        channel_names = input_dataset.channel_names

    # Map channel names to indices
    channel_kwargs_dict = {}
    output_channel_names = []

    for channel_name, channel_config in settings.channels.items():
        if channel_name not in channel_names:
            raise ValueError(f"Channel {channel_name} not found in dataset {channel_names}")

        # Get channel index
        channel_idx = channel_names.index(channel_name)

        # Extract segment_zyx kwargs from config
        segment_kwargs = channel_config.get("segment_kwargs", {})
        channel_kwargs_dict[channel_idx] = segment_kwargs

        # Output channel name
        output_channel_names.append(f"{channel_name}_labels")

        click.echo(
            f"Will segment channel {channel_name} (index {channel_idx}) with kwargs: {segment_kwargs}"
        )

    C_segment = len(channel_kwargs_dict)
    segmentation_shape = (T, C_segment, Z, Y, X)

    # Create a zarr store output to mirror the input
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=[path.parts[-3:] for path in input_position_dirpaths],
        channel_names=output_channel_names,
        shape=segmentation_shape,
        chunks=None,
        scale=scale,
        dtype=np.uint16,
    )

    # Estimate resources
    num_cpus, gb_ram_request = estimate_resources(shape=segmentation_shape, ram_multiplier=15)
    slurm_time = np.ceil(np.max([60, T * 2.0])).astype(int)
    slurm_array_parallelism = 100

    # Prepare SLURM arguments
    slurm_args = {
        "slurm_job_name": "segment-organelles",
        "slurm_mem_per_cpu": f"{gb_ram_request}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": slurm_array_parallelism,
        "slurm_time": slurm_time,
        "slurm_partition": "preempted",
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    # Run locally or submit to SLURM
    if local:
        cluster = "local"
    else:
        cluster = "slurm"

    # Prepare and submit jobs
    click.echo(f"Preparing jobs: {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for input_position_path, output_position_path in zip(
            input_position_dirpaths, output_position_paths
        ):
            jobs.append(
                executor.submit(
                    process_single_position,
                    segment_organelles_data,
                    input_position_path,
                    output_position_path,
                    input_channel_indices=[list(channel_kwargs_dict.keys())],
                    output_channel_indices=[list(range(C_segment))],
                    num_processes=np.min([20, int(num_cpus * 0.8)]),
                    channel_kwargs_dict=channel_kwargs_dict,
                    spacing=settings.spacing,
                )
            )

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


if __name__ == "__main__":
    segment_organelles_cli()
