#!/bin/bash

module load anaconda/latest
conda activate biahub

# $DATASET is set as environmental variable

if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub stabilize \
    -i ../0-reconstruct/${DATASET}.zarr/*/*/* \
    -o phase/${DATASET}.zarr \
    -c xyz_stabilization_settings/*_*_*.yml

