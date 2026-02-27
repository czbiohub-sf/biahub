#!/bin/bash

module load anaconda
conda activate biahub

# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

# change to config file for manual or ants if needed

biahub estimate-registration \
    -t ../../../label-free/0-reconstruct/${DATASET}.zarr/0/8/000000 \
    -s ../0-deskew/${DATASET}.zarr/0/8/000000 \
    -o registration_settings.yml \
    -c estimate-registration-beads.yml\
    -rt "Phase3D" \
    -rs "mCherry EX561 EM600-37"\
    -rs "GFP EX488 EM525-45"



