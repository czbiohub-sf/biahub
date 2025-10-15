from typing import Tuple

from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, PositiveInt


class ZFitRecord(BaseModel):
    z_bg: NonNegativeFloat
    z_amp: NonNegativeFloat
    z_mu: NonNegativeFloat
    z_sigma: NonNegativeFloat
    z_fwhm: NonNegativeFloat
    z_bg_sde: NonNegativeFloat
    z_amp_sde: NonNegativeFloat
    z_mu_sde: NonNegativeFloat
    z_sigma_sde: NonNegativeFloat


class YXFitRecord(BaseModel):
    yx_bg: NonNegativeFloat
    yx_amp: NonNegativeFloat
    y_mu: NonNegativeFloat
    x_mu: NonNegativeFloat
    yx_cyy: float
    yx_cyx: float
    yx_cxx: float
    y_fwhm: NonNegativeFloat
    x_fwhm: NonNegativeFloat
    yx_pc1_fwhm: NonNegativeFloat
    yx_pc2_fwhm: NonNegativeFloat
    yx_bg_sde: NonNegativeFloat
    yx_amp_sde: NonNegativeFloat
    y_mu_sde: NonNegativeFloat
    x_mu_sde: NonNegativeFloat
    yx_cyy_sde: NonNegativeFloat
    yx_cyx_sde: NonNegativeFloat
    yx_cxx_sde: NonNegativeFloat


class ZYXFitRecord(BaseModel):
    zyx_bg: NonNegativeFloat
    zyx_amp: NonNegativeFloat
    zyx_z_mu: NonNegativeFloat
    zyx_y_mu: NonNegativeFloat
    zyx_x_mu: NonNegativeFloat
    zyx_czz: float
    zyx_czy: float
    zyx_czx: float
    zyx_cyy: float
    zyx_cyx: float
    zyx_cxx: float
    zyx_z_fwhm: NonNegativeFloat
    zyx_y_fwhm: NonNegativeFloat
    zyx_x_fwhm: NonNegativeFloat
    zyx_pc1_fwhm: NonNegativeFloat
    zyx_pc2_fwhm: NonNegativeFloat
    zyx_pc3_fwhm: NonNegativeFloat
    zyx_bg_sde: NonNegativeFloat
    zyx_amp_sde: NonNegativeFloat
    zyx_z_mu_sde: NonNegativeFloat
    zyx_y_mu_sde: NonNegativeFloat
    zyx_x_mu_sde: NonNegativeFloat
    zyx_czz_sde: NonNegativeFloat
    zyx_czy_sde: NonNegativeFloat
    zyx_czx_sde: NonNegativeFloat
    zyx_cyy_sde: NonNegativeFloat
    zyx_cyx_sde: NonNegativeFloat
    zyx_cxx_sde: NonNegativeFloat


class PSFRecord(BaseModel):
    z_fit: ZFitRecord
    yx_fit: YXFitRecord
    zyx_fit: ZYXFitRecord


class PSFAnalysisInputs(BaseModel):
    date: Tuple[int, int, int]
    microscope: str
    magnification: PositiveInt
    objective_id: str
    na: PositiveFloat
    spacing: Tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    crop_size: Tuple[PositiveInt, PositiveInt, PositiveInt]
    image_name: str
    temperature: float
    airy_unit: PositiveFloat
    bead_size: PositiveFloat
    bead_supplier: str
    mounting_medium: str
    operator: str
    microscope_type: str
    excitation: PositiveFloat
    emission: PositiveFloat
    comment: str
    dpi: PositiveInt
