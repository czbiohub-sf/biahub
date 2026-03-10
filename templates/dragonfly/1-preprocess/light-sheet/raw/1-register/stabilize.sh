#!/bin/bash

module load anaconda/latest
conda activate biahub

# $DATASET is set as environmental variable

if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub stabilize \
    -i ../0-deskew/${DATASET}.zarr/*/*/* \
    -o ${DATASET}.zarr \
    -c registration_settings.yml

