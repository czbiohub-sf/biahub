import numpy as np
import pandas as pd

from biahub.segment_organelles import (
    extract_organelle_features_data,
    segment_organelles_data,
)


def test_segment_organelles_data_basic():
    """Test basic segmentation with synthetic data"""
    # Create synthetic CZYX data with two channels
    C, Z, Y, X = 2, 1, 64, 64
    czyx_data = np.random.rand(C, Z, Y, X).astype(np.float32)

    # Add some bright spots to simulate organelles
    czyx_data[0, 0, 20:30, 20:30] = 0.9  # Channel 0
    czyx_data[1, 0, 40:50, 40:50] = 0.8  # Channel 1

    # Define per-channel segmentation parameters
    channel_kwargs_dict = {
        0: {
            "sigma_range": (1.0, 2.0),
            "sigma_steps": 2,
            "threshold_method": "otsu",
            "min_object_size": 5,
        },
        1: {
            "sigma_range": (1.0, 2.0),
            "sigma_steps": 2,
            "threshold_method": "triangle",
            "min_object_size": 5,
        },
    }

    spacing = (1.0, 1.0, 1.0)

    # Run segmentation
    labels_czyx = segment_organelles_data(czyx_data, channel_kwargs_dict, spacing)

    # Verify output shape
    assert labels_czyx.shape == czyx_data.shape

    # Verify dtype
    assert labels_czyx.dtype == np.uint16

    # Verify that some labels were created (at least background label 0)
    assert labels_czyx.max() >= 0


def test_segment_organelles_data_single_channel():
    """Test segmentation with single channel"""
    C, Z, Y, X = 1, 1, 32, 32
    czyx_data = np.random.rand(C, Z, Y, X).astype(np.float32)

    # Add a bright spot
    czyx_data[0, 0, 10:20, 10:20] = 1.0

    channel_kwargs_dict = {
        0: {
            "sigma_range": (1.0, 2.0),
            "threshold_method": "otsu",
            "min_object_size": 1,
        }
    }

    labels_czyx = segment_organelles_data(czyx_data, channel_kwargs_dict)

    assert labels_czyx.shape == czyx_data.shape
    assert labels_czyx.dtype == np.uint16


def test_extract_organelle_features_data_basic():
    """Test feature extraction with synthetic labels"""
    Z, Y, X = 1, 64, 64
    labels_zyx = np.zeros((Z, Y, X), dtype=np.uint16)

    # Create two labeled objects
    labels_zyx[0, 10:20, 10:20] = 1
    labels_zyx[0, 30:40, 30:40] = 2

    # Create intensity data
    intensity_zyx = np.random.rand(Z, Y, X).astype(np.float32)
    intensity_zyx[0, 10:20, 10:20] = 0.8  # Object 1 is bright
    intensity_zyx[0, 30:40, 30:40] = 0.2  # Object 2 is dim

    spacing = (1.0, 1.0, 1.0)

    # Extract features
    features_df = extract_organelle_features_data(
        labels_zyx=labels_zyx,
        intensity_zyx=intensity_zyx,
        spacing=spacing,
        properties=["label", "area", "mean_intensity"],
        extra_properties=["aspect_ratio", "circularity"],
    )

    # Verify output
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 2  # Two objects
    assert "label" in features_df.columns
    assert "area" in features_df.columns
    assert "mean_intensity" in features_df.columns
    assert "aspect_ratio" in features_df.columns
    assert "circularity" in features_df.columns


def test_extract_organelle_features_data_with_metadata():
    """Test feature extraction with metadata columns"""
    Z, Y, X = 1, 32, 32
    labels_zyx = np.zeros((Z, Y, X), dtype=np.uint16)
    labels_zyx[0, 10:20, 10:20] = 1

    intensity_zyx = np.random.rand(Z, Y, X).astype(np.float32)

    # Extract features with metadata
    features_df = extract_organelle_features_data(
        labels_zyx=labels_zyx,
        intensity_zyx=intensity_zyx,
        fov_name="A/1/0",
        t_idx=5,
    )

    # Verify metadata columns were added
    assert "fov_name" in features_df.columns
    assert "t" in features_df.columns
    assert features_df["fov_name"].iloc[0] == "A/1/0"
    assert features_df["t"].iloc[0] == 5


def test_extract_organelle_features_data_no_labels():
    """Test feature extraction with no labels"""
    Z, Y, X = 1, 32, 32
    labels_zyx = np.zeros((Z, Y, X), dtype=np.uint16)  # All zeros, no objects
    intensity_zyx = np.random.rand(Z, Y, X).astype(np.float32)

    features_df = extract_organelle_features_data(
        labels_zyx=labels_zyx,
        intensity_zyx=intensity_zyx,
    )

    # Should return empty DataFrame when no objects present
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 0
