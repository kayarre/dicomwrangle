import vtk
import numpy as np
import pickle
import pydicom
import os
import matplotlib.pyplot as plt

# dicom part
"""

iop = [float(v) for v in f.ImageOrientationPatient]
ipp = [float(v) for v in f.ImagePositionPatient]
v1 = np.array(iop[0:3])
v2 = np.array(iop[3:])
n = np.cross(v2,v1)


p = np.array(ipp)
"""

# read in the pickle file with the cine image data
image_dir = "/home/ksansom/caseFiles/mri/VWI_proj/case2/cine/term_ica"
pkl_path = os.path.join(image_dir, "term_ica.pkl")

with open(pkl_path, 'rb') as handle:
    cine_dict = pickle.load(handle)

file_ = cine_dict["other"][0] # read the first file
f_image = pydicom.read_file(file_, stop_before_pixels=False)

#direction cosines
iop = [float(v) for v in f_image.ImageOrientationPatient]
dir_cos = iop

ipp = [float(v) for v in f_image.ImagePositionPatient]
v1 = np.array(iop[0:3])
v2 = np.array(iop[3:])
#normal vector
n = np.cross(v2,v1)
#center of first pixel point
p = np.array(ipp)

# going from LPS to RAS
n_cp = np.copy(n)
n_cp[-1] = -n_cp[-1]
p_cp = np.copy(p)
p_cp[0] = -p_cp[0]
p_cp[1] = -p_cp[1]
#image spacing
spacing = np.array([float(x) for x in f_image.PixelSpacing])

# define the mesh to slice
mesh = vtk.vtkPLYReader()
file_path_mesh = "/home/ksansom/caseFiles/mri/VWI_proj/case2/vmtk"
file_name = "case2_vmtk_decimate_350k.ply"
f_name, ext_ = os.path.splitext(file_name)
mesh.SetFileName(os.path.join(file_path_mesh, file_name))
mesh.Update()

cutter = vtk.vtkCutter()
cutter.SetInputConnection(mesh.GetOutputPort())

cut_plane = vtk.vtkPlane()
cut_plane.SetOrigin(p_cp)
#cut_plane.GetOrigin()
cut_plane.SetNormal(n_cp)
#cut_plane.GetNormal()

cutter.SetCutFunction(cut_plane)

stripper = vtk.vtkStripper()
stripper.SetInputConnection(cutter.GetOutputPort())
stripper.Update()

extract = vtk.vtkPolyDataConnectivityFilter()
extract.SetInputConnection(stripper.GetOutputPort())
extract.SetExtractionModeToAllRegions()
extract.GetExtractionMode()
extract.ColorRegionsOn()
extract.Update()

n_regions = extract.GetNumberOfExtractedRegions()
extract.GetOutput().GetPointData().SetActiveScalars("RegionId")

file_name_list = []
for i in range(n_regions):
    thresh = vtk.vtkThreshold()
    thresh.SetInputConnection(extract.GetOutputPort())
    thresh.SetInputArrayToProcess(0, 0, 0,
                                  vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                  "RegionId")
    thresh.ThresholdBetween(i, i)
    thresh.Update()
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(thresh.GetOutputPort())
    surfer.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputConnection(surfer.GetOutputPort())
    file_name_list.append(os.path.join(file_path_mesh,
                          "{0}_split_{1}.vtp".format(f_name, i)))
    writer.SetFileName(file_name_list[-1])
    writer.Write()


#convert polydata to image plane
reader = vtk.vtkXMLPolyDataReader()
#assume this is the right one
reader.SetFileName(file_name_list[0]) # first one is the one desired
reader.Update()

reader2 = vtk.vtkPLYReader()
reader2.SetFileName(os.path.join(file_path_mesh, file_name))
reader2.Update()

trans = vtk.vtkMatrix4x4()
trans.DeepCopy((dir_cos[3]*spacing[0], dir_cos[0]*spacing[1], n[0], p[0],
                 dir_cos[4]*spacing[0], dir_cos[1]*spacing[1], n[1], p[1],
                 dir_cos[5]*spacing[0], dir_cos[2]*spacing[1], n[2], p[2],
                 0, 0, 0, 1.0))

#print(trans)
# convert from itk format to vtk format
lps2ras = vtk.vtkMatrix4x4()
lps2ras.SetElement(0,0,-1)
lps2ras.SetElement(1,1,-1)
ras2lps = vtk.vtkMatrix4x4()
ras2lps.DeepCopy(lps2ras) # lps2ras is diagonal therefore the inverse is identical
vtkmat = vtk.vtkMatrix4x4()

# https://www.slicer.org/wiki/Documentation/Nightly/Modules/Transforms
vtk.vtkMatrix4x4.Multiply4x4(lps2ras, trans, vtkmat)
vtk.vtkMatrix4x4.Multiply4x4(vtkmat, ras2lps, vtkmat)
print(vtkmat.Determinant())
# Convert the sense of the transform (from ITK resampling to Slicer modeling transform)
invert = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(vtkmat, invert)
print(invert)
# linear transform matrix
invert_lt = vtk.vtkMatrixToLinearTransform()
invert_lt.SetInput(invert)
invert_lt.Update()

post = vtk.vtkTransform()
post.RotateZ(270)
post.RotateX(180)
post.Concatenate(invert_lt)
post.Update()

poly_filt = vtk.vtkTransformPolyDataFilter()
poly_filt.SetInputConnection(reader.GetOutputPort())
poly_filt.SetTransform(post)

poly_filt2 = vtk.vtkTransformPolyDataFilter()
poly_filt2.SetInputConnection(reader2.GetOutputPort())
poly_filt2.SetTransform(post)

writer2 = vtk.vtkXMLPolyDataWriter()
writer2.SetInputConnection(poly_filt.GetOutputPort())
writer2.SetFileName(os.path.join(file_path_mesh,
                      "test_trans.vtp"))
writer2.Write()

writer3 = vtk.vtkXMLPolyDataWriter()
writer3.SetInputConnection(poly_filt2.GetOutputPort())
writer3.SetFileName(os.path.join(file_path_mesh,
                      "test_trans_test.vtp"))
writer3.Write()

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
