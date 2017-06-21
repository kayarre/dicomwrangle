# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:10:15 2016

@author: ksansom
"""

"""
  Tool converts matfile from dan's ultrasound images to
  a format that can be read by vtk, or ITK-SNAP

"""

import scipy.io as io
import numpy as np
from evtk.hl import imageToVTK
from evtk.hl import gridToVTK


mat = io.loadmat('/home/ksansom/caseFiles/ultrasound/grayscale_carotid/iCPC3D04a_v3d_x1_uint8.mat')

voxsize = np.float32(mat['voxsize'][0][0])
vox3d = mat['vox3d']

#n, m, s = vox3d.shape

#x = np.linspace(0., n*voxsize, n + 1, dtype=np.float32)
#y = np.linspace(0., m*voxsize, m + 1, dtype=np.float32)
#z = np.linspace(0, s*voxsize, s+1, dtype=np.float32)

dd = np.ravel(vox3d, order='F')

#gridToVTK("/home/ksansom/caseFiles/ultrasound/grayscale_carotid/test_grid", x, y, z, cellData = {'intensity': dd})

#volume interpolation doesn't work because there won't be any point data.
#imageToVTK("/home/ksansom/caseFiles/ultrasound/grayscale_carotid/test_image", origin = (0.0,0.0,0.0), spacing = (voxsize,voxsize,voxsize), cellData = {'intensity': np.ascontiguousarray(vox3d)})


#np.ascontiguousarray(vox3d).dump("ultrasound.npy")

from tvtk.api import tvtk, write_data

grid = tvtk.ImageData(spacing=(voxsize, voxsize, voxsize), origin=(0.0, 0.0, 0.0), 
                  dimensions=vox3d.shape, format='binary')
grid.point_data.scalars = dd
grid.point_data.scalars.name = 'intensity'

# Writes legacy ".vtk" format if filename ends with "vtk", otherwise
# this will write data using the newer xml-based format.
write_data(grid, '/home/ksansom/caseFiles/ultrasound/grayscale_carotid/test_mayavi.vtk')
