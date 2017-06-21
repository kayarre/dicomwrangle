import vtk
import numpy as np
import pickle
import pydicom
import os
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import matplotlib.patches as patches
from scipy import signal
#import scipy.misc
import scipy.ndimage
from scipy import interpolate, integrate

resize = True
scalefactor = 8.0
#interp_type = "nearest"
interp_type = "lanczos"
read_pickle = True # read the images from pickle file

case_dir = "/home/ksansom/caseFiles/mri/VWI_proj/case2"

# read in the pickle file with the cine image data
image_dir = os.path.join(case_dir,"cine","term_ica")
pkl_path = os.path.join(image_dir, "term_ica.pkl")

with open(pkl_path, 'rb') as handle:
    cine_dict = pickle.load(handle)

file_ = cine_dict["other"][0] # read the first file
f_image = pydicom.read_file(file_, stop_before_pixels=False)
spacing = np.array([float(x) for x in f_image.PixelSpacing])

im_vol_1 = np.zeros((f_image.Rows, f_image.Columns))

im_vol_1 = f_image.pixel_array

reader = vtk.vtkXMLPolyDataReader()
file_path_mesh = os.path.join(case_dir, "vmtk")
reader.SetFileName(os.path.join(file_path_mesh,
                      "test_trans.vtp"))
reader.Update()


pd = reader.GetOutput()
bounds = np.empty((6))
pd.GetBounds(bounds)
points = pd.GetPoints()
npts = pd.GetNumberOfPoints()

x = []
y = []
for i in range(npts):
    pt = pd.GetPoint(i)
    x.append(pt[0])
    y.append(pt[1])

def getMask(currentImage, path_):
    ny, nx = np.shape(currentImage)
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    grid = path_.contains_points(points).reshape((ny,nx))
    return grid

if (resize == True):
    scale = scalefactor
else:
    scale = 1.0

#im_vol_resize = scipy.misc.imresize(im_vol_1, scale, interp=interp_type, mode='F')
im_vol_resize = scipy.ndimage.zoom(im_vol_1, scale, output=None,
                                   order=3, mode='constant', cval=0.0,
                                   prefilter=True)

x_pts = [(x_+0.5)*scale for x_ in x]
y_pts = [(y_+0.5)*scale for y_ in y]
verts = np.column_stack((x_pts, y_pts))
contour = mplpath.Path(verts, closed=True)

fig = plt.figure()
ax = fig.add_subplot(111)
patch = patches.PathPatch(contour, facecolor=(1.0,165./255.0,0,0.25), lw=1)
ax.add_patch(patch)

#plt.plot(x_pts, y_pts)
ax.set_xlim([round((bounds[0]-3)*scale), round((bounds[1]+3)*scale)])
ax.set_ylim([round((bounds[2]-3)*scale), round((bounds[3]+3)*scale)])
ax.imshow(im_vol_resize)


# show ROI masks
grid = getMask(im_vol_resize, contour)
print(grid.shape)
ax.imshow(grid, interpolation='None', cmap="gray",  alpha=0.5)
#pl.title('ROI masks of the two ROIs')
plt.show()

# extract the data
sub_dirs = ["mag", "other", "x", "y", "z"]

xyz_keys = ["x", "y", "z"]
cubes = {}
dict_time = {}
image_dtype = np.float32

pkl_path_image = os.path.join(image_dir, "term_ica_image.pkl")

if read_pickle == True:
    with open(pkl_path_image, 'rb') as handle:
        dict_data_save = pickle.load(handle)

    cubes = dict_data_save["img"]
    nx,ny,N = cubes["x"].shape
    dict_time = dict_data_save["time"]
else:
    for k in cine_dict.keys():
        N = len(cine_dict[k])
        #print(N)
        # assuming image size is constant
        if k in xyz_keys:
            count = 0
            nx = int(f_image.Rows)*int(scale)
            ny = int(f_image.Columns)*int(scale)
            im_vol = np.empty((nx, ny, N), dtype=image_dtype)
            time_list = []
            for f_ in cine_dict[k]:
                f_image = pydicom.read_file(f_, stop_before_pixels=False)
                #print(f_image.TriggerTime)
                img = f_image.pixel_array
                img = (img.astype(image_dtype)*np.float32(f_image.RescaleSlope) +
                       np.float32(f_image.RescaleIntercept))
                if resize == True:
                    #re_img = scipy.misc.imresize(img, scale, interp=interp_type, mode='F')
                    re_img = scipy.ndimage.zoom(img, scale, output=None,
                                                order=3, mode='constant', cval=0.0,
                                                prefilter=True)
                else:
                    re_img = img
                im_vol[:,:,count] = re_img
                time_list.append(float(f_image.TriggerTime))
                if (count == 0):
                    print(f_image.SeriesDescription, f_image.SeriesNumber)
                    print(f_image.NominalInterval)
                    print(re_img.dtype, img.dtype)
                count += 1
            cubes[k] = im_vol
            dict_time[k] = time_list

    dict_data_save = {"img": cubes, "time": dict_time}
    with open(pkl_path_image, 'wb') as handle:
        pickle.dump(dict_data_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
from vtk.util import numpy_support
#x_data_shape = cubes["x"].shape # keep this to convert back to numpy
#x_data = cubes["x"].reshape(x_data_shape)
x_data = numpy_support.numpy_to_vtk(num_array=cubes["x"].transpose(2, 1, 0).ravel(), deep=True, array_type=vtk.VTK_DOUBLE)

x_data_im = vtk.vtkImageData()
x_data_im.SetExtent(0, nx-1, 0, ny-1, 0, N-1)
x_data_im.SetOrigin( 0.0, 0.0, 0.0)
x_data_im.SetSpacing( 1.0, 1.0, 1.0)
x_data_im.AllocateScalars(vtk.VTK_DOUBLE, 1)
x_data_im.GetPointData().SetScalars(x_data)

lanczos = vtk.vtkImageSincInterpolator()
lanczos.SetWindowFunctionToLanczos()
reslice1 = vtk.vtkImageReslice()
reslice1.SetInputData(x_data_im)
#reslice1.SetResliceTransform(transform)
reslice1.SetInterpolator(lanczos)
reslice1.SetOutputSpacing(0.125, 0.125, 0.5)
reslice1.SetOutputOrigin( 0.0, 0.0, 0.0)
#reslice1.SetOutputExtent(0,127,0,127,0,0)

writer=vtk.vtkXMLImageDataWriter()
writer.SetInputConnection(reslice1.GetOutputPort())
writer.SetFileName('test.vti')
writer.Update()
writer.Write()
"""

test_q = np.sqrt( np.square(cubes["x"]) +
                  np.square(cubes["y"]) +
                  np.square(cubes["z"]))
#test_x = np.sum(np.sqrt( np.square(cubes["x"]))*grid.astype(np.float32))
#test_y = np.sum(np.sqrt( np.square(cubes["y"]))*grid.astype(np.float32))
#test_z = np.sum(np.sqrt( np.square(cubes["z"]))*grid.astype(np.float32))

q_sum = []
q_sum_test = []
for i in range(N): #assume size doesn't change
    q_mask_sum = np.sum(test_q[:,:,i]*grid.astype(np.float32))
    q_sum.append(q_mask_sum)
    q_sum_test.append(q_mask_sum*np.prod(spacing[0:2])/(100.0*scale**2)) # cm^3

mean_hr  = []
mean_diff = []
for k in xyz_keys:
    t = np.mean(np.diff(dict_time[k]))
    print(t)
    mean_diff.append(t)
    mean_hr.append(60000.0/dict_time[k][-1])
print( "mean heart rate {0} BPM".format(np.mean(mean_hr)))
print( "mean dt {0} milliseconds".format(np.mean(mean_diff)))

print("total flow {0} cm^3/beat".format(np.trapz( q_sum_test, dx = np.mean(mean_diff)/1000)))

mean_time = []
for i in range(N):
    mt = (dict_time["x"][i] + dict_time["y"][i] + dict_time["z"][i]) / 3.0
    mean_time.append(mt)

new_time = np.linspace(0, mean_time[-1], 40)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(new_time, q_sum_test, c='b', label='flowRate')
#ax2.scatter(dict_time["y"], q_sum_test, label='flowRatey')
#ax2.scatter(dict_time["z"], q_sum_test, label='flowRatez')
#ax2.plot(dict_time["x"], q_sum_test)
ax2.set_xlabel(r'$t$', fontsize=20)
ax2.set_ylabel(r'Q, $Q(t)$ $cm^3/min$', fontsize=20)
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.9))
plt.show()

q_sum_roll = np.roll(q_sum_test, -16)

#q_interp = interpolate.CubicSpline(new_time, q_sum_roll, bc_type='periodic')
q_interp = interpolate.InterpolatedUnivariateSpline(new_time, q_sum_roll, k=3, ext=0)
t = np.linspace(0.0, mean_time[-1], 1024)
q_test = q_interp(t)

gs2 = plt.GridSpec(1, 1, wspace=0.2, hspace=0.2)
# Create a figure

fig3 = plt.figure(figsize=(11, 9))
ax3 = fig3.add_subplot(gs2[0, :])
ax3.set_title('terminal ICA Waveform', fontsize=20)
ax3.set_xlabel(r'time $t$ $milliseconds$', fontsize=20)
ax3.set_ylabel(r'Flowrate $cm^3/min$', fontsize=20)

ax3.plot(t, q_test, c='b', linestyle='-', label='interp')

ax3.legend(loc='center left', bbox_to_anchor=(1, 0.9))
plt.show()

flow_data = {"flow":q_test, "time" :t }

# check for waveform directory

waveform_dir = os.path.join(case_dir, "waveform")
if( not os.path.exists(waveform_dir)):
    os.makedirs(waveform_dir)

pkl_waveform_data = os.path.join(waveform_dir, "waveform_{0}.pkl".format("term_ica"))
with open(pkl_waveform_data, 'wb') as handle:
    pickle.dump(flow_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
