# -*- coding: utf-8 -*-
"""
Reads the pickle file of sorted cine data
and creates image volume viewer for each location
@author: sansomk
"""

import pydicom
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import pickle


def cube_show_slider(cubes, sub_dirs, axis=2, **kwargs):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    """
    if 'title' in kwargs.keys():
        title = kwargs.pop('title')
    else:
        title = "test"

    # check dim
    for k in cubes.keys():
        if not cubes[k].ndim == 3:
            raise ValueError("cube should be an ndarray with ndim == 3")

    image_rows = 3
    image_col  = 8
    # generate figure
    gs = plt.GridSpec(image_rows, image_col, wspace=0.25, hspace=0.2)

    # Create a figure
    fig = plt.figure(figsize=(11, 17))
    fig.patch.set_facecolor('black')
    fig.suptitle(title, fontsize=20, color='w')
    #fig.suptitle(title, fontsize=20)
    ax = {}

    # Mag image
    #["mag", "other", "x", "y", "z"]
    ax["magx"] = fig.add_subplot(gs[:, 0:3])
    ax["magx"].set_title('Magnitude', {'color': 'w', 'fontsize': 15})
    #ax["mag"].yaxis.label.set_color('white')
    ax["otherx"] = fig.add_subplot(gs[:, 3:6])
    ax["otherx"].set_title('other', {'color': 'w', 'fontsize': 15})
    ax["x"] = fig.add_subplot(gs[0, 6:])
    ax["x"].set_ylabel('x', {'color': 'w', 'fontsize': 15})
    ax["x"].yaxis.set_label_position("right")
    ax["y"] = fig.add_subplot(gs[1, 6:])
    ax["y"].set_ylabel('y', {'color': 'w', 'fontsize': 15})
    ax["y"].yaxis.set_label_position("right")
    ax["z"] = fig.add_subplot(gs[2, 6:])
    ax["z"].set_ylabel('z', {'color': 'w', 'fontsize': 15})
    ax["z"].yaxis.set_label_position("right")

    # select first image
    s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
    l = {}
    for sd in sub_dirs:
        #print(s, cubes[sd].shape)
        im = cubes[sd][s].squeeze()
        l[sd] = ax[sd].imshow(im, **kwargs)

    # define slider
    axcolor = 'lightslategray'
    ax_c = fig.add_axes([0.25, 0.05, 0.65, 0.03], aspect="auto", facecolor=axcolor)

    slider = Slider(ax_c, 'Axis %i index' % axis, 0, cubes["magx"].shape[axis] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None)
                 for i in range(3)]
        for sd in sub_dirs:
            im = cubes[sd][s].squeeze()
            l[sd].set_data(im, **kwargs)
        fig.canvas.draw()

    slider.on_changed(update)

    plt.show()

sub_dirs = ["magx", "otherx", "x", "y", "z"]
xyz_keys = ["x", "y", "z"]

dir_path = "/home/ksansom/caseFiles/mri/VWI_proj/case2/cine"
pkl_path_dcm = os.path.join(dir_path, "sorted_cine_dict.pkl")

if( os.path.isfile(pkl_path_dcm)):
    try:
        with open(pkl_path_dcm, 'rb') as handle:
            case_dict = pickle.load(handle)
    #except pickle.UnpicklingError as e:
        # normal, somewhat expected
    except (AttributeError,  EOFError, ImportError, IndexError) as e:
        # secondary errors
        print(traceback.format_exc(e))
    except Exception as e:
        # everything else, possibly fatal
        print(traceback.format_exc(e))

else:
    #testing if don't have pickle file of already sorted stuff
    test_path = "/home/ksansom/caseFiles/mri/VWI_proj/case2/cine/term_ica"
    head, tail = os.path.split(test_path)
    #path to each set of files
    #first_dirs = [os.path.join(test_path, sd) for sd in sub_dirs])
    cine_dict = {}
    for sd in sub_dirs:
        sd_path = os.path.join(test_path, sd)
        get_files = [os.path.join(sd_path,fn) for fn in os.listdir(sd_path)]

        info_list = []
        for f in get_files:
            try:
                f_image_1 = pydicom.read_file(f, stop_before_pixels=True)
            except:
                print("error: {0}".format(f))
                continue
            info_list.append((f, float(f_image_1.TriggerTime)))

        sorted_list = sorted(info_list, key=lambda x: x[1])
        cine_dict[sd] = [s[0] for s in sorted_list]
        #print(cine_dict[sd])

    pkl_path = os.path.join(test_path, "{0}.pkl".format(tail))
    with open(pkl_path, 'wb') as handle:
        pickle.dump(cine_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    case_dict = { tail : { "images" : cine_dict } }

case_cubes = {loc:{} for loc in case_dict.keys()} # dictionary of image volumes
print(case_cubes.keys())
for loc in case_dict.keys():
    print_description = False
    for k in case_dict[loc]["images"].keys():
        count = 0
        N = len(case_dict[loc]["images"][k])
        #print(N)
        # assuming image size is constant
        image_dtype = np.uint16
        if k in xyz_keys:
            image_dtype = np.float64
        #print(k, case_dict[loc]["images"][k][0])
        f_image_1 = pydicom.read_file(case_dict[loc]["images"][k][0], stop_before_pixels=True)
        im_vol = np.zeros((f_image_1.Rows, f_image_1.Columns,N),dtype=image_dtype)
        for f_ in case_dict[loc]["images"][k]:
            #print(f_)
            f_image = pydicom.read_file(f_, stop_before_pixels=False)
            if(print_description == False):
                print(f_image_1.SeriesDescription, f_image_1.SeriesNumber)
                print_description = True
            img = f_image.pixel_array
            if k in xyz_keys:
                img = img.astype(image_dtype)*np.float64(f_image.RescaleSlope) + np.float64(f_image.RescaleIntercept)
            im_vol[:,:,count] = img
            count += 1
        case_cubes[loc][k] = im_vol
    #print(time_list)

    #determine if its phase contrast image
    print(case_cubes[loc].keys())
    cube_show_slider(case_cubes[loc], sub_dirs, axis=2, title=loc)

#pkl_path_vol = os.path.join(dir_path, "image_volumes.pkl")
#with open(pkl_path_vol, 'wb') as handle:
#    pickle.dump(case_cubes, handle, protocol=pickle.HIGHEST_PROTOCOL)
