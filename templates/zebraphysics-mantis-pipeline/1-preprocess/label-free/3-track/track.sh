#!/bin/bash

module load anaconda/latest
conda activate biahub

# $DATASET is set as environmental variable

if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub track \
    -o ${DATASET}.zarr \
    -c tracking_settings.yml

