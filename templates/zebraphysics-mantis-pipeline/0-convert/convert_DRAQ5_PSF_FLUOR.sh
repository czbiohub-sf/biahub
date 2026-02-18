#!/bin/bash

module load anaconda
conda activate iohub

# $DATASET is set as environmental variable
if [[ -z "${DATASET}" ]]; then
  echo "DATASET environmental variable is not set"
  exit 1
fi

# Convert PSF
iohub convert \
  -i "/hpc/instruments/cm.mantis/${DATASET}/PSF_1/PSF_lightsheet_1" \
  -o "PSF.zarr"

# Convert FLUOR
for i in 1 2
do
  iohub convert \
    -i "/hpc/instruments/cm.mantis/${DATASET}/FLUOR_$i/FLUOR_lightsheet_1" \
    -o "FLUOR_$i.zarr"
done

# Convert DRAQ5
for mode in "lightsheet" "labelfree"
do
 iohub convert \
   -i "/hpc/instruments/cm.mantis/${DATASET}/DRAQ5_1/DRAQ5_${mode}_1" \
   -o "DRAQ5/DRAQ5_${mode}.zarr"
done
