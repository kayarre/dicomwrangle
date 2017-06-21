from scipy import io
import numpy as np
from evtk.hl import imageToVTK
from evtk.hl import gridToVTK

test = {}
io.loadmat('iCPC3D04a_v3d_x1_uint8.mat',mdict=test)

vox3dnp = np.array(test['vox3d'])
imageToVTK('./test', origin=(0.0,0.0,0.0), spacing = (sp,sp,sp), cellData = {'grayscale': np.ascontiguousarray(vox3dnp)})

sp = test['voxsize'][0][0]
w, h, d = vox3dnp.shape
x = np.linspace(0.0, sp*(w), w+1, dtype=np.float32)
y = np.linspace(0.0, sp*(h), h+1, dtype=np.float32)
z = np.linspace(0.0, sp*(d), d+1, dtype=np.float32)


from tvtk.api import tvtk, write_data

grid = tvtk.ImageData(spacing=(sp, sp, sp), origin=(0.0, 0.0, 0.0), dimensions=vox3dnp.shape)
grid.point_data.scalars = np.ravel(vox3dnp, order='F')
grid.point_data.scalars.name = 'grayscale'

# Writes legacy ".vtk" format if filename ends with "vtk", otherwise
# this will write data using the newer xml-based format.
write_data(grid, 'test3.vtk')
