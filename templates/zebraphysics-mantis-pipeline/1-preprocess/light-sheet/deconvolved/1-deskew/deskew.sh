#!/bin/bash

# $DATASET is set as env variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

module load anaconda/latest
conda activate biahub

biahub deskew \
    -i ../0-decon/${DATASET}.zarr/*/*/* \
    -c deskew_settings.yml \
    -o ${DATASET}.zarr \
    -sb sbatch_file.sh

