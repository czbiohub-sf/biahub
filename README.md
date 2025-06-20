# biahub

Bio-image analysis hub supporting high-throughput data reconstruction on HPC clusters with [Slurm](https://slurm.schedmd.com/documentation.html) workload management.

![acquisition and reconstruction schematic](https://github.com/czbiohub-sf/shrimPy/blob/main/docs/figure_3a.png)

`biahub` was originally developed to reconstruct data acquired on the [mantis](https://doi.org/10.1093/pnasnexus/pgae323) microscope using the [shrimPy](https://github.com/czbiohub-sf/shrimPy) acquisition engine, and has since been extended to process diverse multimodal datasets. `biahub` reconstruction workflows rely on OME-ZARR datasets (for example, as created with [iohub](https://github.com/czbiohub-sf/iohub)) which enable efficient parallelization across compute nodes. Available reconstruction routines are listed below; more information can be obtained with `biahub --help`.

## Install

```
conda create -n biahub python==3.11
conda activate biahub

git clone https://github.com/czbiohub-sf/biahub.git
pip install -e ./biahub
```

## Data reconstruction

Data reconstruction uses a command line interface. All reconstruction calls take an input `-i` and an output `-o`, and most reconstruction calls use configuration files passed via a `-c` option. Reconstruction workflows launch multiple Slurm jobs and can also be run locally using the `-l` flag.

A typical set of CLI calls to go from raw data to registered volumes looks like:

```sh
# CONVERT TO ZARR
iohub convert \
    -i ./acq_name/acq_name_labelfree_1 \
    -o ./acq_name_labelfree.zarr \
iohub convert \
    -i ./acq_name/acq_name_lightsheet_1 \
    -o ./acq_name_lightsheet.zarr

# DECONVOLVE FLUORESCENCE
# (optional) characterize the PSF
biahub characterize-psf
    -i ./beads.zarr \
    -c ./characterize_params.yml \
    -o ./report/
# estimate PSF parameters
biahub estimate-psf \
    -i ./beads.zarr \
    -c ./psf_params.yml \
    -o ./psf.zarr
# deconvolve data
biahub deconvolve \
    -i ./acq_name_lightsheet.zarr \
    -c ./deconvolve_params.yml \
    --psf-dirpath ./psf.zarr
    -o ./acq_name_lightsheet_deconvolved.zarr

# DESKEW FLUORESCENCE
# estimate deskew parameters
biahub estimate-deskew \
    -i ./acq_name_lightsheet.zarr/0/0/0 \
    -o ./deskew.yml
# apply deskew parameters
biahub deskew \
    -i ./acq_name_lightsheet.zarr/*/*/* \
    -c ./deskew_params.yml \
    -o ./acq_name_lightsheet_deskewed.zarr

# RECONSTRUCT PHASE/BIREFRINGENCE
biahub reconstruct \
    -i ./acq_name_labelfree.zarr/*/*/* \
    -c ./recon.yml \
    -o ./acq_name_labelfree_reconstructed.zarr

# STABILIZE
# estimate stabilization parameters
biahub estimate-stabilization \
    -i ./acq_name_labelfree.zarr/*/*/* \
    -o ./stabilization.yml \
    --stabilize-xy \
    --stabilize-z
# stabilize data
biahub stabilize \
    -i ./acq_name_labelfree.zarr/*/*/* \
    -c ./stabilization.yml \
    -o ./acq_name_labelfree_stabilized.zarr/*/*/*

# REGISTER
# estimate registration parameters
biahub estimate-registration \
    -s ./acq_name_labelfree_reconstructed.zarr/0/0/0 \
    -t ./acq_name_lightsheet_deskewed.zarr/0/0/0 \
    -o ./register.yml
# optimize registration parameters
biahub optimize-registration \
    -s ./acq_name_labelfree_reconstructed.zarr/0/0/0 \
    -t ./acq_name_lightsheet_deskewed.zarr/0/0/0 \
    -c ./register.yml \
    -o ./register_optimized.yml
# register data
biahub register \
    -s ./acq_name_labelfree_reconstructed.zarr/*/*/* \
    -t ./acq_name_lightsheet_deskewed.zarr/*/*/* \
    -c ./register_optimized.yml \
    -o ./acq_name_registered.zarr

# CONCATENATE CHANNELS
biahub concatenate \
    -c ./concatenate.yml \
    -o ./acq_name_concatenated.zarr

# STITCH
# estimate stitching parameters
biahub estimate-stitch \
    -i ./acq_name.zarr/*/*/* \
    -o ./stitching.yml
# optimize stitching parameters
biahub optimize-stitch \
    -i ./stitching.yml \
    -o ./optimized-stitching.yml \
    --channel DAPI
# stitch fields of view
biahub stitch \
    -i ./acq_name.zarr/*/*/* \
    -c ./optimized-stitching.yml \
    -o ./acq_name_stitched.zarr/*/*/*
```

## Contributing
We would appreciate the bug reports and code contributions if you use this package. If you would like to contribute to this package, please read the [contributing guide](CONTRIBUTING.md).
