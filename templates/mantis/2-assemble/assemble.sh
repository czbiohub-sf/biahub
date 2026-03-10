#!/bin/bash

module load anaconda
conda activate biahub

# $DATASET is set as env variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub concatenate \
  -c concatenate_cropped.yml \
  -o ${DATASET}.zarr

# Create a symbolic link for track data
ln -s ../1-preprocess/label-free/3-track/${DATASET}_cropped.zarr tracking.zarr
