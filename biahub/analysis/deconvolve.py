import numpy as np
import torch
from waveorder.models.isotropic_fluorescent_thick_3d import apply_inverse_transfer_function

def compute_tranfser_function(
        psf: np.ndarray
) -> torch.Tensor:
    transfer_function = torch.abs(torch.fft.fftn(torch.tensor(psf)))
    transfer_function /= torch.max(transfer_function)
    return transfer_function

def deconvolve_data(
        czyx_raw_data: np.ndarray,
        transfer_function: torch.Tensor,
        regularization_strength=1e-3,
) -> np.ndarray:
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