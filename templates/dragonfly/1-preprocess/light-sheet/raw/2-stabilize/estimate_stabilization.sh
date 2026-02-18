#!/bin/bash


module load anaconda
conda activate biahub

# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub estimate-stabilization \
  -i ../0-deskew/${DATASET}.zarr/*/*/*  \
  -o . \
  -c estimate-stabilization-z.yml

