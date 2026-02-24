
from pathlib import Path
import numpy as np
from biahub.core.transform import Transform
import dask.array as da


def estimate_txy(
    data_txy: np.ndarray,
    t_reference: str = "previous",
    verbose: bool = False,

) -> list[Transform]:

    from pystackreg import StackReg

    sr = StackReg(StackReg.TRANSLATION)
    T_stackreg = sr.register_stack(data_txy, reference=t_reference, axis=0)

    # Swap translation directions: (x, y) -> (y, x)
    for tform in T_stackreg:
        tform[0, 2], tform[1, 2] = tform[1, 2], tform[0, 2]

    transforms_txy = np.zeros((T_stackreg.shape[0], 4, 4))
    transforms_txy[:, 1:4, 1:4] = T_stackreg
    transforms_txy[:, 0, 0] = 1

    # to list of transforms
    transforms = [Transform(matrix=transform) for transform in transforms_txy]
    return transforms


def estimate_tczyx(
    fov: str,
    data_tczyx: da.Array,
    output_dirpath: Path,
    channel_index: int,
    center_crop_xy: list[int, int],
    z_focus_index_list: list[int],
    t_reference: str = "previous",
    verbose: bool = False,
) -> Transform:
    """
    Estimate the xy stabilization for a single position.

    Parameters
    ----------
    input_position_dirpath : Path
        Path to the input position directory.
    output_folder_path : Path
        Path to the output folder.
    df_z_focus_path : Path
        Path to the input focus CSV file.
    channel_index : int
        Index of the channel to process.
    center_crop_xy : list[int, int]
        Size of the crop in the XY plane.
    t_reference : str
        Reference timepoint.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    ArrayLike
        Transformation matrix.
    """

    T, C, Z, Y, X = data_tczyx.shape
    x_idx = slice(X // 2 - center_crop_xy[0] // 2, X // 2 + center_crop_xy[0] // 2)
    y_idx = slice(Y // 2 - center_crop_xy[1] // 2, Y // 2 + center_crop_xy[1] // 2)

    data_tzyx = np.stack(
            [
                data_tczyx[t, channel_index, z, y_idx, x_idx]
                for t, z in zip(range(T), z_focus_index_list)
            ]
        )
    data_tzyx = np.clip(data_tzyx, a_min=0, a_max=None)


    transforms = estimate_txy(data_tzyx, t_reference, verbose, output_dirpath)
    output_dirpath_fov = output_dirpath / f"{fov}"
    output_dirpath_fov.mkdir(parents=True, exist_ok=True)

    for t, transform in enumerate(transforms):
        np.save(output_dirpath_fov / f"{t}.npy", transform.matrix.astype(np.float32))

    return transforms

