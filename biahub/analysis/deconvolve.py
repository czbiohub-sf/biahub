import numpy as np
import torch

from iohub import open_ome_zarr
from waveorder.models.isotropic_fluorescent_thick_3d import apply_inverse_transfer_function


def compute_tranfser_function(
    psf_zyx_data: np.ndarray,
    output_zyx_shape: tuple,
) -> np.ndarray:
    zyx_padding = np.array(output_zyx_shape) - np.array(psf_zyx_data.shape)
    pad_width = [(x // 2, x // 2) if x % 2 == 0 else (x // 2, x // 2 + 1) for x in zyx_padding]
    padded_psf_data = np.pad(
        psf_zyx_data, pad_width=pad_width, mode="constant", constant_values=0
    )

    transfer_function = torch.abs(torch.fft.fftn(torch.tensor(padded_psf_data)))
    transfer_function /= torch.max(transfer_function)

    return transfer_function.numpy()


def deconvolve_data(
    czyx_raw_data: np.ndarray,
    transfer_function: torch.Tensor = None,
    transfer_function_store_path: str = None,
    regularization_strength: float = 1e-3,
) -> np.ndarray:
    if transfer_function is None:
        with open_ome_zarr(transfer_function_store_path, layout='fov', mode='r') as ds:
            transfer_function = torch.tensor(ds.data[0, 0])

    output = []
    for zyx_raw_data in czyx_raw_data:
        zyx_decon_data = apply_inverse_transfer_function(
            torch.tensor(zyx_raw_data),
            transfer_function,
            z_padding=0,
            regularization_strength=regularization_strength,
        )
        output.append(zyx_decon_data.numpy())

    return np.stack(output)
