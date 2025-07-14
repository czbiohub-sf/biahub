
# Correlative Confocal - QPI Imaging

The microscope images of the markers were acquired using correlative confocal fluorescence microscopy and quantitative phase microscopy.

## Sample preparation
For fixed and live cell imaging 150,000 of A549 cells were plated in 24 well glass bottom plate.

### Fixed samples:
* The day after plating the cells, the nucleus was stained with Hoechst in 1:10,000 dilution in PBS for 10 mins.
* Cells were fixed with either 4% of PFA or cold MtOH for 20 mins.
* Add imaging media (water?)

### Live samples:
* Add imaging media

## Microscope specifications

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
Multimodal 4D (C,Z,X,Y) imaging using MicroManager.
Imaging was automated using MicroManager open-source software (version 2.0).

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
