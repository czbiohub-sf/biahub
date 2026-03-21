#!/bin/bash

module load anaconda
conda activate biahub

biahub estimate-psf \
  -i ../../../../0-convert/PSF.zarr/0/FOV0/0 \
  -c estimate_psf.yml \
  -o PSF.zarr
