#!/bin/bash

module load anaconda
conda activate biahub

# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub reconstruct \
    -i ../../../0-convert/${DATASET}_symlink/${DATASET}_labelfree_1.zarr/*/*/* \
    -c phase_config.yaml \
    -o ${DATASET}.zarr \
    -j 12
