# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:01:12 2015

@author: sansomk
"""

import dicom
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


#dcmpath='/Users/sansomk/Downloads/0.4/102'
dcmpath='/home/ksansom/caseFiles/mri/images/0.4/102'
dcm_files = []
location = []
for dirname, subdirlist, filelist in os.walk(dcmpath):
    for filen in  filelist:
        #print(dirname, subdirlist, filelist)
        #break
        try:
            filePath = os.path.join(dirname,filen)
            #print(filePath)
            f = dicom.read_file(filePath, stop_before_pixels=True)
            location.append(f.)
            dcm_files.append(filePath)
        except:
            print("error: {0}".format(filen))
            
print('hello')


# Get ref file
RefDs = dicom.read_file(dcm_files[0])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(dcm_files))

#check ti see whether the image has been interpolated
if (hasattr(RefDs, 'SpacingBetweenSlices')):
    if (RefDs.SpacingBetweenSlices < RefDs.SliceThickness ):
        z_spacing = RefDs.SpacingBetweenSlices
else:
    z_spacing = float(RefDs.SliceThickness)

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]),
                     float(RefDs.PixelSpacing[1]), z_spacing)

x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
print(ArrayDicom.shape)
# loop through all the DICOM files
for filenameDCM in dcm_files:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, : , dcm_files.index(filenameDCM)] = ds.pixel_array

testindx = np.where(ArrayDicom != 0)
#print(np.max(testindx[0]), np.max(testindx[1]))
#print(np.min(testindx[0]), np.min(testindx[1]))
minx = np.min(testindx[0])
maxx = np.max(testindx[0])
miny = np.min(testindx[1])
maxy = np.max(testindx[1])
minz = np.min(testindx[2])
maxz = np.max(testindx[2])

#ConstPixelDims = (int(maxx-minx), int(maxy-miny), len(dcm_files))
#new = RefDs.pixel_array
#new[testindx] = -9999999

fig1 = plt.figure(dpi=300)
ax1 = plt.axes()
ax1.set_aspect('equal', 'datalim')
ax1.pcolormesh(x[miny:maxy], y[minx:maxx],
               np.flipud(ArrayDicom[minx:maxx, miny:maxy, 490]), cmap='gray')

'''
for filen in dcm_files:
    #print(filen)
    f = dicom.read_file(filen, stop_before_pixels=True)
    #print(f.PixelSpacing, f.PixelSpacing, f.SpacingBetweenSlices)
    print(f.SliceLocation)
'''