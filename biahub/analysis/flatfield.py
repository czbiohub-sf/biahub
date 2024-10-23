import numpy as np
from iohub import open_ome_zarr

def correct_illumination_zyx(
    czyx_data: np.ndarray,
    pattern_path: str,
    black_level_input: int = 90,
    black_level_pattern: int = 100,
):
    """Correct illumination of an FOV in place.

    Parameters
    ----------
    czyx_data : np.ndarray
        Input data to correct
    czyx_pattern : np.ndarray
        Illumination pattern to correct with
    black_level : int, optional
        Black level of the camera, by default 100.
    """
    with open_ome_zarr(pattern_path) as ds:
        bg_czyx = np.asarray(ds.data[0]).astype(np.float32)
    corrected_czyx = czyx_data.astype(np.float32)
    
    bg_czyx = np.where(bg_czyx < black_level_pattern, 1e10, bg_czyx)
    # bg = bg - black_level_pattern
    corrected_czyx = corrected_czyx - black_level_input
    corrected_czyx = corrected_czyx.clip(0, None)
    corrected_czyx = corrected_czyx / (bg_czyx + 1e-10)
    corrected_czyx = corrected_czyx + black_level_input

    return corrected_czyx