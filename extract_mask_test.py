import vtk
import numpy as np
import pickle
import pydicom
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.path as mplpath
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
from scipy import signal
#import scipy.misc
import scipy.ndimage
from scipy import interpolate, integrate

resize = False
scalefactor = 8.0
shift = [0.,0.]
#interp_type = "nearest" #interp_type = "lanczos"
read_pickle = False # read the images from pickle file

case_dir = "/home/ksansom/caseFiles/mri/VWI_proj/case2"
sub_dirs = ["magx", "otherx", "x", "y", "z"]
xyz_keys = ["x", "y", "z"]
image_dtype = np.float32

def create_dir(dir_path, subdir):
    directory = os.path.join(dir_path, subdir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def getMask(currentImage, path_):
    ny, nx = np.shape(currentImage)
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    grid = path_.contains_points(points).reshape((ny,nx))
    return grid

def press(event):

    print('press', event.key)
    sys.stdout.flush()
    if (event.key == 'left'):
        shift[0] -= 1.0
    elif (event.key == 'right'):
        shift[0] += 1.0
    elif (event.key == 'up'):
        shift[1] += 1
    elif (event.key == "down"):
        shift[1] -= 1.0

    print(dir(patch))
    path_ = patch.get_path()# = ax.add_patch(patchip)
    print(path_._readonly)
    print(help(path_.vertices))
    verts_test = path_.vertices
    verts_test[:,0] += shift[0]
    verts_test[:,1] += shift[1]

    new_contour = mplpath.Path(verts_test, closed=True)
    print(dir(patch.set))
    patch_handle.set_path(new_contour)

    #patch_handle2.get_path().vertices(verts_test)
    #ax5 = fig.add_subplot(gs[0:3,0:3])
    #patch2 = patches.PathPatch(contour, facecolor='none', linewidth=0.8, #edgecolor=(1.0,165./255.0,0,0.50))#, alpha=0.5)
    #ax5.imshow(im_vol_resize, interpolation='bilinear', cmap="gray",  alpha=1.0)
    #patch_handle2 = ax5.add_patch(patch2)
    #t_start = ax.transData
    #t_start2 = ax5.transData
    #t = mpl.transforms.Affine2D().translate(shift[0], shift[1])
    #t_end = t_start + t
    #t_end2 = t_start2 + t

    #print(t_start)
    #print(t)
    #print(t_end)
    #patch_handle2.set_transform(t_end2)
    #print(t.get_matrix())
    #print(t_end2.get_matrix())
    #patch_handle.set_transform(t_end)
    #visible = xl.get_visible()
    #    xl.set_visible(not visible)
    fig.canvas.draw()

class Recalculate(object):

    def update(self, event):
        x_update = [x_ for x_ in x_pts]
        y_update = [y_ for y_ in y_pts]
        verts_test = np.column_stack((x_update, y_update))
        new_contour = mplpath.Path(verts_test, closed=True)
        # create mask
        new_grid = getMask(im_vol_resize, new_contour)
        # calculate the flow
        q_sum = []
        for i in range(N): #assume size doesn't change
            q_mask_sum = np.sum(test_q[:,:,i]*new_grid.astype(image_dtype))
            q_sum.append(q_mask_sum*np.prod(spacing[0:2])/(100.0*scale**2)) # cm^3

        print("total flow {0} cm^3/beat".format(np.trapz( q_sum, dx = np.mean(mean_diff)/1000)))
        q_sum_roll = np.roll(q_sum, -16)
        #q_interp = interpolate.CubicSpline(new_time, q_sum_roll, bc_type='periodic')
        q_interp = interpolate.InterpolatedUnivariateSpline(new_time, q_sum_roll, k=3, ext=0)
        q_test = q_interp(t_interp)

        #plotting update
        grid_handle.set_data(new_grid)
        #print(new_time)
        scat_data = np.stack((new_time, q_sum), axis=1)
        #ax2.relim()      # make sure all the data fits
        #ax2.autoscale()
        #print(scat_data.shape)
        scatter_handle.set_offsets(scat_data)
        interp_handle[-1].set_ydata(q_test)
        ax3.relim()      # make sure all the data fits
        ax3.autoscale()
        ax2.set_xlim(ax3.get_xlim())
        ax2.set_ylim(ax3.get_ylim())
        fig.canvas.draw()

def update_cine(val):
    ind = int(slider.val)
    s = [slice(ind, ind + 1) if i == 2 else slice(None)
             for i in range(3)]
    for sd in xyz_keys:
        im = cubes[sd][s].squeeze()
        cine_ax_dict[sd].set_data(im)
    ax6.set_title("time: {0}".format(new_time[ind]))
    ax2.scatter(new_time, q_sum, c='b', label='flowRate')
    ax2.scatter(new_time[ind], q_sum[ind], c='r', label='time={0}'.format(new_time[ind]))
    fig.canvas.draw()

# read in the pickle file with the cine image data paths
image_dir = os.path.join(case_dir, "cine")
pkl_path_images = os.path.join(image_dir, "sorted_cine_dict.pkl")
with open(pkl_path_images, 'rb') as handle:
    case_dict = pickle.load(handle)

contour_path = os.path.join(case_dir, "contours")
pkl_contour_path = os.path.join(contour_path, "contour_paths.pkl")
with open(pkl_contour_path, 'rb') as handle:
    contour_dict = pickle.load(handle)

figures_path = create_dir(case_dir, "contour_figures")

# check for waveform directory
waveform_dir = create_dir(case_dir, "waveform")

# create a dictionary for the image_volumes to be stored
image_dict = {k:{} for k in case_dict.keys()}
flow_dict = {k:{} for k in case_dict.keys()}

pkl_path_image_vol = os.path.join(image_dir, "numpy_image_vol.pkl")
if read_pickle == True:
    with open(pkl_path_image_vol, 'rb') as handle:
        image_dict = pickle.load(handle)

for case in case_dict.keys():
    if( case not in contour_dict.keys()):
        print("there is a key error between the images and the contours")
    else:

        file_ = case_dict[case]["images"]["otherx"][0] # read the first file
        f_image = pydicom.read_file(file_, stop_before_pixels=False)
        spacing = np.array([float(x) for x in f_image.PixelSpacing])

        im_vol_1 = np.zeros((f_image.Rows, f_image.Columns))
        im_vol_1 = f_image.pixel_array

        if (resize == True):
            scale = scalefactor
        else:
            scale = 1.0

        #im_vol_resize = scipy.misc.imresize(im_vol_1, scale, interp=interp_type, mode='F')
        im_vol_resize = scipy.ndimage.zoom(im_vol_1, scale, output=None,
                                           order=3, mode='constant', cval=0.0,
                                           prefilter=True)
        # extract the data
        cubes = {}
        dict_time = {}
        if read_pickle == True:
            cubes = image_dict[case]["img"]
            nx,ny,N = cubes["x"].shape
            dict_time = image_dict[case]["time"]
        else:
            for k in case_dict[case]["images"].keys():
                N = len(case_dict[case]["images"][k])
                #print(N)
                # assuming image size is constant
                if k in xyz_keys:
                    count = 0
                    nx = int(f_image.Rows)*int(scale)
                    ny = int(f_image.Columns)*int(scale)
                    im_vol = np.empty((nx, ny, N), dtype=image_dtype)
                    time_list = []
                    for f_ in case_dict[case]["images"][k]:
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

        image_dict[case] = {"img": cubes, "time": dict_time}

        norm_vec = contour_dict[case]["normal"]
        #print(cubes["x"].shape)
        #map of q values
        #test_q = np.sqrt( np.square(cubes["x"]) +
        #                  np.square(cubes["y"]) +
        #                  np.square(cubes["z"]))
        #negative values just mean need to multiply by -1, invert the normal
        test_q = (cubes["x"]*norm_vec[0] +
                  cubes["y"]*norm_vec[1] +
                  cubes["z"]*norm_vec[2])

        for loc in contour_dict[case]["trans"].keys():
            reader = vtk.vtkXMLPolyDataReader()
            file_path_contour = contour_dict[case]["trans"][loc]
            reader.SetFileName(file_path_contour)
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

            x_pts = [(x_+0.0)*scale for x_ in x]
            y_pts = [(y_+0.0)*scale for y_ in y]
            verts = np.column_stack((x_pts, y_pts))
            contour = mplpath.Path(verts, closed=True)

            # time information is independent of the mask information
            mean_hr  = []
            mean_diff = []
            for k in xyz_keys:
                t = np.mean(np.diff(dict_time[k]))
                print(t)
                mean_diff.append(t)
                mean_hr.append(60000.0/dict_time[k][-1])
            print( "mean heart rate {0} BPM".format(np.mean(mean_hr)))
            print( "mean dt {0} milliseconds".format(np.mean(mean_diff)))

            mean_time = []
            for i in range(N):
                mt = (dict_time["x"][i] + dict_time["y"][i] + dict_time["z"][i]) / 3.0
                mean_time.append(mt)

            new_time = np.linspace(0, mean_time[-1], 40)
            t_interp = np.linspace(0.0, mean_time[-1], 1024)


            # create mask
            grid = getMask(im_vol_resize, contour)
            # calculate the flow
            q_sum = []
            for i in range(N): #assume size doesn't change
                q_mask_sum = np.sum(test_q[:,:,i]*grid.astype(image_dtype))
                q_sum.append(q_mask_sum*np.prod(spacing[0:2])/(100.0*scale**2)) # cm^3

            print("total flow {0} cm^3/beat".format(np.trapz( q_sum, dx = np.mean(mean_diff)/1000)))

            q_sum_roll = np.roll(q_sum, -16)

            #q_interp = interpolate.CubicSpline(new_time, q_sum_roll, bc_type='periodic')
            q_interp = interpolate.InterpolatedUnivariateSpline(new_time, q_sum_roll, k=3, ext=0)

            q_test = q_interp(t_interp)

            #figures
            gs = plt.GridSpec(6, 6, wspace=0.2, hspace=0.2)
            fig = plt.figure(figsize=(17, 9))
            fig.canvas.mpl_connect('key_press_event', press)
            ax = fig.add_subplot(gs[3:,0:3])
            #xl = ax.set_xlabel('easy come, easy go')
            #ax.set_title('Press a key')
            patch = patches.PathPatch(contour, facecolor=(1.0,165./255.0,0,0.25), lw=1 )#, alpha=0.5)
            patch_handle = ax.add_patch(patch)

            #plt.plot(x_pts, y_pts)
            x_bounds = [round((bounds[0]-4)*scale), round((bounds[1]+4)*scale)]
            y_bounds = [round((bounds[2]-4)*scale), round((bounds[3]+4)*scale)]
            ax.set_xlim(x_bounds)
            ax.set_ylim(y_bounds)
            ax.imshow(im_vol_resize, interpolation='bilinear', cmap="gray",  alpha=1.0)

            ax5 = fig.add_subplot(gs[0:3,0:3])
            patch2 = patches.PathPatch(contour, facecolor='none', linewidth=0.8, edgecolor=(1.0,165./255.0,0,0.50))#, alpha=0.5)
            ax5.imshow(im_vol_resize, interpolation='bilinear', cmap="gray",  alpha=1.0)
            patch_handle2 = ax5.add_patch(patch2)
            #new_contour = patch_.get_path()
            ax5.set_xlim(x_bounds)
            ax5.set_ylim(y_bounds)

            # show ROI mask
            grid_handle = ax.imshow(grid, interpolation='None', cmap="gray",  alpha=0.1)
            #print(dir(grid_handle))
            ax2 = fig.add_subplot(gs[0:3, 4:])
            scatter_handle = ax2.scatter(new_time, q_sum, c='b', label='flowRate')
            ax2.scatter(new_time[0], q_sum[0], c='r', label='cine time')
            #print(dir(scatter_handle))
            #ax2.scatter(dict_time["y"], q_sum, label='flowRatey')
            #ax2.scatter(dict_time["z"], q_sum, label='flowRatez')
            #ax2.plot(dict_time["x"], q_sum)
            #ax2.set_xlabel(r'$t$', fontsize=20)
            ax2.set_title('terminal ICA Waveform', fontsize=20)
            ax2.set_ylabel(r'Q, $Q(t)$ $cm^3/min$', fontsize=20)
            ax2.xaxis.set_ticks_position('none')
            ax2.xaxis.set_ticklabels([])
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.9))

            ax3 = fig.add_subplot(gs[3:, 4:])
            ax3.set_xlabel(r'time $t$ $milliseconds$', fontsize=20)
            ax3.set_ylabel(r'Flowrate $cm^3/min$', fontsize=20)
            ax3.set_xlim(ax2.get_xlim())
            ax3.set_ylim(ax2.get_ylim())

            interp_handle = ax3.plot(t_interp, q_test, c='b', linestyle='-', label='interp')
            #print(dir(interp_handle[0]))
            ax3.legend(loc='center left', bbox_to_anchor=(1, 0.9))

            cine_ax_dict = {}
            ax6 = fig.add_subplot(gs[0:2,2:4])
            ax6.set_title("time: {0}".format(new_time[0]))
            cine_ax_dict["x"] = ax6.imshow(cubes['x'][:,:,0], interpolation='bilinear', cmap="viridis",  alpha=1.0)
            patch6 = patches.PathPatch(contour, facecolor='none', linewidth=0.8, edgecolor=(1.0,165./255.0,0,0.50))
            ax6.add_patch(patch6)
            ax6.set_xlim(x_bounds)
            ax6.set_ylim(y_bounds)

            ax7 = fig.add_subplot(gs[2:4,2:4])
            cine_ax_dict["y"] = ax7.imshow(cubes['y'][:,:,0], interpolation='bilinear', cmap="viridis",  alpha=1.0)
            patch7 = patches.PathPatch(contour, facecolor='none', linewidth=0.8, edgecolor=(1.0,165./255.0,0,0.50))
            ax7.add_patch(patch7)
            ax7.set_xlim(x_bounds)
            ax7.set_ylim(y_bounds)

            ax8 = fig.add_subplot(gs[4:,2:4])
            cine_ax_dict["z"] = ax8.imshow(cubes['z'][:,:,0], interpolation='bilinear', cmap="viridis",  alpha=1.0)
            patch8 = patches.PathPatch(contour, facecolor='none', linewidth=0.8, edgecolor=(1.0,165./255.0,0,0.50))
            ax8.add_patch(patch8)
            ax8.set_xlim(x_bounds)
            ax8.set_ylim(y_bounds)

            # define slider
            axcolor = 'lightslategray'
            ax_c = fig.add_axes([0.25, 0.05, 0.60, 0.03], aspect="auto", facecolor=axcolor)

            slider = Slider(ax_c, 'Axis %i index' % 2, 0, cubes["x"].shape[2] - 1,
                            valinit=0, valfmt='%i')



            slider.on_changed(update_cine)

            callback = Recalculate()
            axRecalc = plt.axes([0.9, 0.04, 0.08, 0.05])
            bnext = Button(axRecalc, 'Recalculate')
            bnext.on_clicked(callback.update)

            fig.savefig(os.path.join(figures_path,
                                      "flowrate_plots_{0}_{1}.png".format(case, loc)))
            plt.show()
            plt.close(fig)
            #plt.show()
            flow_dict[case][loc] = {"flow":q_test, "time" : t_interp,
                                    "flow_raw": q_sum, "time_raw": dict_time }


pkl_waveform_data = os.path.join(waveform_dir, "waveform_data.pkl")
with open(pkl_waveform_data, 'wb') as handle:
    pickle.dump(flow_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if (not read_pickle):
    with open(pkl_path_image_vol, 'wb') as handle:
        pickle.dump(image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
