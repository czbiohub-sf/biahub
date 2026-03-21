#!/bin/bash

module load anaconda
conda activate iohub

# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

iohub rename-wells -i "${DATASET}_symlink/${DATASET}_labelfree_1.zarr" -c well_map.csv

iohub rename-wells -i "${DATASET}_symlink/${DATASET}_lightsheet_1.zarr" -c well_map.csv
