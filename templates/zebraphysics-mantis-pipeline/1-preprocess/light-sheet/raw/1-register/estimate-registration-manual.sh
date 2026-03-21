#!/bin/bash

module load anaconda
conda activate biahub

# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi


biahub estimate-registration \
    -t ../../../label-free/0-reconstruct/${DATASET}.zarr/0/8/000000 \
    -s ../0-deskew/${DATASET}.zarr/0/8/000000 \
    -c estimate-registration-manual.yml \
    -o approx_transform_settings.yml \
    -rt "Phase3D" \
    -rs "mCherry EX561 EM600-37"\
    -rs "GFP EX488 EM525-45"
