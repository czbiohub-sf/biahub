## import required packages/functions
import os
import numpy as np
import matplotlib.pyplot as plt
from sympy import Point, Line, N
import napari
from scipy.ndimage import rotate
from skimage.transform import rescale
from iohub.ngff import open_ome_zarr

from aqr_mantis_transform_utils import *

##------------------------------------------------------------------------------

#### load curated registration bead coordinate files
#### |-->these positions have been extracted after rescaling of and -90deg  
####     rotation of light-sheet data to better match label-free data

## light-sheet beads
a = np.load('light-sheet_registration_curated_bead_coordinates.npy') 
## label-free beads
b = np.load('label-free_registration_curated_bead_coordinates.npy') 

##------------------------------------------------------------------------------

#### ordered rotation of ICP matched bead reference coordinate sets
## start with z rotation
z_rotation_angle = np.mean(measure_z_axis_rotation_angle(a,b))
z_rotation = rotate_around_z_axis(a, z_rotation_angle)
print('rotate around z-axis by ', z_rotation_angle, 'radians yields -->')
print(z_rotation)

## feed result of z rotation into y rotation
y_rotation_angle = np.mean(measure_y_axis_rotation_angle(a,b))
y_rotation = rotate_around_y_axis(z_rotation, y_rotation_angle)
print('rotate around y-axis by ', y_rotation_angle, 'radians yields -->')
print(y_rotation)

## feed compounded result of z,y rotation into x rotation
x_rotation_angle = np.mean(measure_x_axis_rotation_angle(a,b))
x_rotation = rotate_around_x_axis(y_rotation, x_rotation_angle)
print('rotate around x-axis by ', x_rotation_angle, 'radians yields -->')
print(x_rotation)

## find translation vector after rotation angles & matrices have been applied
translation_by_point = x_rotation-b
print('translate zyx-ordered rotation result by ', translation_by_point, 'yields-->')
print(x_rotation-translation_by_point)

## take mean of translation by point for downstream use
translation_vector = np.mean(translation_by_point, axis = 0)
print('mean translation for full fovs:', translation_vector)

##------------------------------------------------------------------------------

#### test bead rotation & translation results on neuromast dataset
## most of this section needs to be modified to run as a loop through timepoints
## or modified to split timepoints to run each in parallel

## grab resolution of each arm. ls==light-sheet. lf==label-free
## currently hardcoded because it was easier for running a bunch of tests
ls_resolution = (0.174, 0.116, 0.116)
lf_resolution = (0.174,0.1494,0.1494)

## load image data (currently loads a single timepoint)
## needs to be changed to load from zarr store
ls_img = np.load('neuromast_light-sheet_timepoint_start.npy')[1,...]
lf_img = np.squeeze(np.load('neuromast_label-free_timepoint_start.npy'))

## perform initial -90deg rotation of lightsheet data as done with beads
ls_rot_neg90 = rotate(ls_img, -90, axes=(1,2))
## rescale initial lightsheet rotated data to match label-free resolution
ls_rotated_rescaled = rescale(ls_rot_neg90, np.asarray(ls_resolution)/np.asarray(lf_resolution))

## flatten image to be registered into 2D position and value arrays
ls_coordinates = np.asarray(np.where(ls_rotated_rescaled)).T
ls_values = np.ndarray.flatten(ls_rotated_rescaled)

### make sure flattened arrays have same dimension_0 size
#print(ls_coordinates.shape[0]==ls_values.shape[0])

## perform z,y,x ordered rotations on neuromast data using bead determined angles
z_rotated_coordinates = rotate_around_z_axis(ls_coordinates, z_rotation_angle)
zy_rotated_coordinates = rotate_around_y_axis(z_rotated_coordinates, y_rotation_angle)
zyx_rotated_coordinates = rotate_around_x_axis(zy_rotated_coordinates, x_rotation_angle)

## translation after ordered rotation using mean translation vector determined from bead data
## determined from bead data
zyx_rotated_coordinates_translation = zyx_rotated_coordinates - translation_vector

##------------------------------------------------------------------------------

## create new array for registered voxels within label-free field of view
## this needs to be adapated to fill a new empty zarr store with the same shape
## as the label-free data
z_lf_shape_limit, y_lf_shape_limit, x_lf_shape_limit = lf_img.shape

ls_registered = np.zeros_like(lf_img)
counter=0
for i in range(ls_coordinates.shape[0]):
    iz, iy, ix = zyx_rotated_coordinates_translation[i,...]
    iz, iy, ix = (int(iz), int(iy), int(ix))
    #print(iz,iy,ix)
    if (iz>=0 and iz<z_lf_shape_limit) and (iy>=0 and iy<y_lf_shape_limit) and (ix>=0 and ix<x_lf_shape_limit):
        ls_registered[iz,iy,ix] = ls_values[i]
    else:
        continue

###------------------------------------------------------------------------------
#
### (if running in a notebook) plot max projections for quick visual inspection
#
#plt.imshow(ls_rotated_rescaled.max(axis=0))
#plt.show()
#
#plt.imshow(lf_img.max(axis=0))
#plt.show()
#
#plt.imshow(ls_registered.max(axis=0))
#plt.show()
#
### (if running in a notebook) plot corner vertices of transformed-translated data to make sure  
### no strange shear has been introduced each plot should look like an 'N'
#
#rotated_bbox_vertices_zy = np.asarray([(zyx_rotated_coordinates[:,0].min(),zyx_rotated_coordinates[:,1].min()),
#                                       (zyx_rotated_coordinates[:,0].min(),zyx_rotated_coordinates[:,1].max()),
#                                       (zyx_rotated_coordinates[:,0].max(),zyx_rotated_coordinates[:,1].min()),
#                                       (zyx_rotated_coordinates[:,0].max(),zyx_rotated_coordinates[:,1].max())])
#
#plt.figure(figsize=(2,2))
#plt.plot(rotated_bbox_vertices_zy[:,0], rotated_bbox_vertices_zy[:,1])
#plt.show()
#
#rotated_bbox_vertices_zx = np.asarray([(zyx_rotated_coordinates[:,0].min(),zyx_rotated_coordinates[:,2].min()),
#                                       (zyx_rotated_coordinates[:,0].min(),zyx_rotated_coordinates[:,2].max()),
#                                       (zyx_rotated_coordinates[:,0].max(),zyx_rotated_coordinates[:,2].min()),
#                                       (zyx_rotated_coordinates[:,0].max(),zyx_rotated_coordinates[:,2].max())])
#
#plt.figure(figsize=(2,2))
#plt.plot(rotated_bbox_vertices_zx[:,0], rotated_bbox_vertices_zx[:,1])
#plt.show()
#
#rotated_bbox_vertices_yx = np.asarray([(zyx_rotated_coordinates[:,1].min(),zyx_rotated_coordinates[:,2].min()),
#                                       (zyx_rotated_coordinates[:,1].min(),zyx_rotated_coordinates[:,2].max()),
#                                       (zyx_rotated_coordinates[:,1].max(),zyx_rotated_coordinates[:,2].min()),
#                                       (zyx_rotated_coordinates[:,1].max(),zyx_rotated_coordinates[:,2].max())])
#
#plt.figure(figsize=(2,2))
#plt.plot(rotated_bbox_vertices_yx[:,0], rotated_bbox_vertices_yx[:,1])
#plt.show()
#
##------------------------------------------------------------------------------

#### push to napari for further visual inspection

## initialize viewer
viewer = napari.Viewer()

## add images to viewer
#viewer.add_image(ls_rotated_rescaled, name='light-sheet_rotated_rescaled')
viewer.add_image(lf_img, name='label-free_img')
viewer.add_image(ls_registered, name='light-sheet_registered')

##------------------------------------------------------------------------------

#### save registered array
np.save('light-sheet_registered_timepoint', ls_registered)
print('timepoint _ complete')
