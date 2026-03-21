#!/bin/bash

module load anaconda
conda activate biahub

# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub deskew \
    -i ../../../../0-convert/${DATASET}_symlink/${DATASET}_lightsheet_1.zarr/*/*/* \
    -c deskew_settings.yml \
    -o ${DATASET}.zarr \
    -sb sbatch_file.sh

