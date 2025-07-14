
# Image Processing

## ome-tiff to zarr conversion
The data acquired by Micromanager is converted to ome zarr format using [iohub](https://github.com/czbiohub-sf/iohub) library.

## Quantitative label-free (phase) reconstruction
Brightfield volumes were reconstructed to get phase information using [waveorder](https://github.com/mehta-lab/waveorder). Phase image is more informative as it enhances the morphological features visible in the brightfield images and adds information on the relative density distribution in cells.

## Image registration
The three image channels were registered using [biahub](https://github.com/czbiohub-sf/biahub). BF and fluorescence channels were misregistered due to different views of the two different cameras, and the two fluorescence channels were misregistered due to differences in optics used for filtering the specific wavelengths of light.

## Image download and visualization
Download the complete high-resolution dataset with multiple field-of-views for all displayed conditions from the following [link](https://public.czbiohub.org/organelle_box/datasets/A549/organelle_box_v1.zarr). 800 x 800 pixel crop of one field-of-view is displayed.

You can visualize the zarr data using [napari](https://napari.org/).

Tools for working with zarr format are available on [napari-iohub](https://github.com/czbiohub-sf/napari-iohub).