from biahub.vendor.napari_psf_analysis.image import Calibrated3DImage
from biahub.vendor.napari_psf_analysis.psf_analysis.fit.fitter import (
    YXFitter,
    ZFitter,
    ZYXFitter,
)
from biahub.vendor.napari_psf_analysis.psf_analysis.records import (
    PSFRecord,
    YXFitRecord,
    ZFitRecord,
    ZYXFitRecord,
)


class PSF:
    image: Calibrated3DImage = None
    psf_record: PSFRecord = None

    def __init__(self, image: Calibrated3DImage):
        self.image = image

    def analyze(self) -> None:
        z_fitter = ZFitter(image=self.image)
        yx_fitter = YXFitter(image=self.image)
        zyx_fitter = ZYXFitter(image=self.image)

        z_fit_record: ZFitRecord = z_fitter.fit()
        yx_fit_record: YXFitRecord = yx_fitter.fit()
        zyx_fit_record: ZYXFitRecord = zyx_fitter.fit()

        self.psf_record = PSFRecord(
            z_fit=z_fit_record,
            yx_fit=yx_fit_record,
            zyx_fit=zyx_fit_record,
        )

    def get_record(self) -> PSFRecord:
        return self.psf_record

    def get_summary_dict(self) -> dict:
        return {
            **self.psf_record.z_fit.model_dump(),
            **self.psf_record.yx_fit.model_dump(),
            **self.psf_record.zyx_fit.model_dump(),
        }
