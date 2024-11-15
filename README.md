# biahub
Bio-image analysis hub.

## Install

```
conda create -n biahub python==3.10
conda activate biahub

git clone https://github.com/czbiohub-sf/biahub.git
pip install -e ./biahub
```

## Data reconstruction

Data reconstruction uses a command line interface. All reconstruction calls take an input `-i` and an output `-o`, and most reconstruction calls use configuration files passed via a `-c` option.

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
recorder reconstruct \
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
biahub estimate-stitching \
    -i ./acq_name.zarr/*/*/* \
    -o ./stitching.yml \
    --channel DAPI
    --percent-overlap 0.05
# stitch fields of view
biahub stitch \
    -i ./acq_name.zarr/*/*/* \
    -c ./stitching.yml \
    -o ./acq_name_stitched.zarr/*/*/*
```

## Contributing
We would appreciate the bug reports and code contributions if you use this package. If you would like to contribute to this package, please read the [contributing guide](CONTRIBUTING.md).
