#!/bin/bash

module load anaconda
conda activate biahub

# $DATASET is set as env variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub estimate-crop \
  -c concatenate.yml \
  -o concatenate_cropped.yml \
  --lf-mask-radius 0.95
