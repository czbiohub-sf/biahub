#!/bin/bash

module load anaconda
conda activate biahub

# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

biahub deconvolve \
  -i ../../../../0-convert/${DATASET}_symlink/${DATASET}_lightsheet_1.zarr/*/*/* \
  -c decon.yml \
  -o ${DATASET}.zarr \
  --psf-dirpath PSF.zarr \
  -sb sbatch_file.sh
