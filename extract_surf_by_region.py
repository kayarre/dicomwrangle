import vtk
#import numpy as np
import os
#import matplotlib.pyplot as plt


# define the mesh to slice
mesh = vtk.vtkXMLPolyDataReader()
file_path_mesh = "/home/ksansom/caseFiles/mesh_test"
file_name = "DSI020CALb_vmtk_decimate_trim_ext2_remesh_test.vtp"
file_path = os.path.join(file_path_mesh, file_name)
split_ext = os.path.splitext(file_path)[0]

mesh.SetFileName(file_path)
mesh.Update()
n_arrays = mesh.GetNumberOfCellArrays()

cell_array_id = 0
for i in range(n_arrays):
    array_name = mesh.GetCellArrayName(i)

    if ( array_name == 'RegionId'):
        print("found the thing: {0}".format(array_name))
        cell_array_id = i
        break

array_name_test = mesh.GetOutput().GetCellData().GetArrayName(cell_array_id)
print(cell_array_id, array_name_test)
regions = mesh.GetOutput().GetCellData().GetArray(cell_array_id)
region_range = regions.GetRange()
lower = int(region_range[0])
upper = int(region_range[1]+1)

for i in range(lower, upper):
    print(i)
    thresh = vtk.vtkThreshold()
    thresh.SetInputConnection(mesh.GetOutputPort())
    thresh.SetInputArrayToProcess(cell_array_id, 0, 0,
                                  vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                                  "RegionId")
    thresh.ThresholdBetween(i, i)
    thresh.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(thresh.GetOutputPort())
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(surfer.GetOutputPort())
    cleaner.Update()

    n_cells = cleaner.GetOutput().GetNumberOfCells()
    print("surface cells", n_cells)

    # Setup the colors array
    colors = vtk.vtkIntArray()
    colors.SetNumberOfComponents(1)
    colors.SetNumberOfValues(n_cells)
    colors.SetName("RegionId")
    for k in range(n_cells):
        colors.SetValue(k, i)

    cleaner.GetOutput().GetCellData().SetScalars(colors)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputConnection(cleaner.GetOutputPort())
    file_name_list = os.path.join(split_ext,
                          "{0}_{1}.vtp".format(split_ext, i))
    writer.SetFileName(file_name_list)
    writer.Write()

    # now extract feature edges
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputConnection(surfer.GetOutputPort())
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()

    boundaryClean = vtk.vtkCleanPolyData()
    boundaryClean.SetInputConnection(boundaryEdges.GetOutputPort())

    boundaryStrips = vtk.vtkStripper()
    boundaryStrips.SetInputConnection(boundaryClean.GetOutputPort())
    boundaryStrips.Update()

    boundaryPoly = vtk.vtkPolyData()
    boundaryPoly.SetPoints(boundaryStrips.GetOutput().GetPoints())
    boundaryPoly.SetPolys(boundaryStrips.GetOutput().GetLines())

    n_cells = boundaryPoly.GetNumberOfCells()
    print("n_cells", n_cells)

    # Setup the colors array
    colors = vtk.vtkIntArray()
    colors.SetNumberOfComponents(1)
    colors.SetName("RegionId")
    for j in range(n_cells):
        # Add the colors we created to the colors array
        if (n_cells > 1):
            colors.InsertNextValue(j+2)
        else:
            colors.InsertNextValue(i)

    boundaryPoly.GetCellData().SetScalars(colors)

    writer2 = vtk.vtkXMLPolyDataWriter()
    writer2.SetInputData(boundaryPoly)
    file_name_list2 = os.path.join(split_ext,
                          "{0}_cnt_{1}.vtp".format(split_ext, i))
    writer2.SetFileName(file_name_list2)
    writer2.Write()


"""

#read the image volume
#image_dtype = np.uint16
im_vol = np.zeros((f_image.Rows, f_image.Columns))

im_vol = f_image.pixel_array

plt.imshow(im_vol)
plt.show()


t = vtk.vtkImageData()
t.SetDimensions(int(f_image.Rows), int(f_image.Columns), 1)
t.SetOrigin( 0.0 , 0.0 , 0.0 )
t.SetSpacing( 1.0 , 1.0 , 1.0 )
t.AllocateScalars(vtk.VTK_UNSIGNED_INT, 0)

max_I = np.max(im_vol)
print(max_I)
dims = t.GetDimensions()
print(dims)
for i in range(dims[0]):
    for j in range(dims[1]):
        #xCoord = 1 - np.fabs((i - dims[0]/2.0) / (dims[0]/2.0) )
        #yCoord = 1 - np.fabs((j - dims[1]/2.0) / (dims[0]/2.0) )
        #print(xCoord, yCoord)
        t.SetScalarComponentFromDouble(i, j, 0, 0, float(im_vol[j,i]))

flip = vtk.vtkImageFlip()
flip.SetInputData(t)
flip.SetFilteredAxis(1)
flip.Update()
#The image viewers and writers are only happy with unsigned char
# images.  This will convert the floats into that format.
resultScale = vtk.vtkImageShiftScale()
resultScale.SetOutputScalarTypeToUnsignedInt()
resultScale.SetShift(0)
resultScale.SetScale(1.0)
resultScale.SetInputConnection(flip.GetOutputPort())
#resultScale.SetInputData(t)
resultScale.Update()

lanczos = vtk.vtkImageSincInterpolator()
lanczos.SetWindowFunctionToLanczos()

reslice = vtk.vtkImageReslice()
reslice.SetInputConnection(resultScale.GetOutputPort())
reslice.SetInterpolator(lanczos)
reslice.SetOutputSpacing(0.2, 0.2, 1.0)
reslice.SetOutputOrigin(0.0,0.0,0.0)
#reslice.SetOutputExtent(t.GetExtent())


#out_image = vtk.vtkImageData()
#out_image = reslice.GetOutput()
resultScale2 = vtk.vtkImageShiftScale()
resultScale2.SetOutputScalarTypeToUnsignedShort()
resultScale2.SetShift(0)
resultScale2.SetScale(32767.0 / max_I)
resultScale2.SetInputConnection(reslice.GetOutputPort())
#resultScale.SetInputData(t)
resultScale2.Update()

pngwriter = vtk.vtkPNGWriter()
pngwriter.SetFileName("test.png")
pngwriter.SetInputConnection(resultScale2.GetOutputPort())
pngwriter.Write()

#rows, cols, _ = out_image.GetDimensions()
#sc = out_image.GetCellData().GetScalars()
#print(sc)
#np_image = a.reshape(rows, cols, -1)

#plt.imshow(np_image)
#plt.show()

#Visualize
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(poly_filt.GetOutputPort())
actor = vtk.vtkActor()
actor.SetMapper(mapper)

viewer = vtk.vtkImageViewer2()
viewer.SetInputConnection(reslice.GetOutputPort())
viewer.SetColorWindow(max_I)
viewer.SetColorLevel(max_I/2)

iren = vtk.vtkRenderWindowInteractor()
viewer.SetupInteractor(iren)

viewer.Render()
viewer.GetRenderer().ResetCamera()
iren.Initialize()
viewer.Render()
iren.Start()

"""
