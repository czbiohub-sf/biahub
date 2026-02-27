#!/bin/bash

module load anaconda/latest
conda activate biahub

# $DATASET is set as environmental variable

if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub stabilize \
    -i ../1-register/${DATASET}.zarr/*/*/* \
    -o ${DATASET}.zarr \
    -c combined_stabilization_settings/*_*_*.yml

