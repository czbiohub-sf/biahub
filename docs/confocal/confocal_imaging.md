
# Correlative Confocal - QPI Imaging

This protocol describes correlative fluorescence (confocal) and label-free (quantitative phase) imaging of cells tagged with one or more fluorescent reporters. The following protocol was developed for imaging cell lines from the [Organelle Box resource](https://organellebox.sf.czbiohub.org/). 
## Sample preparation
* 150,000 of A549 cells were plated in 24 well glass bottom plate.
* Next day, nucleus was stain with Hoechst in 1:10,000 dilution in PBS for 10 mins.
* Cells were fixed with either 4% of PFA or cold MtOH for 20 mins. 
* Then, PBS was used during imaging.

## Our microscope

We use a confocal with a transmitted-light path for correlative fluorescence and label-free imaging.
### Microscope body
Leica DMi8 inverted microscope with an Okolab cage incubator.

### Objective
* Magnification - 63X
* Numerical aperture - 1.47
* Immersion media - Oil

### Imaging channels
* DAPI: (Hoechst), 358 nm excitation, 461 nm emission.
* FITC: (nuclear translocation sensor), 495 nm excitation, 519 nm emission.
* Label-free channels: BF (brightfield), 450 nm illumination.

### Imaging software
We used [Micro-Manager](https://micro-manager.org/) to automate the imaging. We acquired Multi-channel 3D (C, Z, Y, X ) images using [Multi-D acquisition dialog](https://micro-manager.org/Version_2.0_Users_Guide).

### Conditions imaged
* Live, 
* PFA fixed, 
* Methanol fixed

### Spatial resolution
* z-slices at 0.2 um spacing
* x-y pixel at 0.103 um resolution

### Imaging conditions
Live cells are imaged under incubated conditions at 37 degrees Celsius and 5% CO2, 90% humidity. Fixed cells were imaged at room temperature.

Laser intensity for various markers were optimized for best SNR. The intensity of the markers in live condition is ranked as follows , increasing through the list:

ATP1B3 < RPL36 < NCLN < SEC61B < LAMP1 < PNPLA2 < EDC4 < ACTB < EEA1 < TOMM20 < PEX3 < CLTA < SLC3A2 < ATG101 < VSP35 < G3BP1 < MAP1LC3B < RAB11A < H2BC21 < NPM1

The ranking of intensity of the markers can differ with fixation conditions.
