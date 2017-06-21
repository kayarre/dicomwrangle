# -*- coding: utf-8 -*-
"""
Need to use roipoly to getmask and get velocity information
@author: sansomk
"""

import pydicom
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import pickle

#import shutil

#from roipoly import roipoly

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
    fig.suptitle("title", fontsize=20)
    #fig.suptitle(title, fontsize=20)
    ax = {}

    # Mag image
    ["mag", "other", "x", "y", "z"]
    ax["mag"] = fig.add_subplot(gs[:, 0:3])
    ax["mag"].set_title('Magnitude', {'color': 'w', 'fontsize': 15})
    #ax["mag"].yaxis.label.set_color('white')
    ax["other"] = fig.add_subplot(gs[:, 3:6])
    ax["other"].set_title('other', {'color': 'w', 'fontsize': 15})
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
        im = cubes[sd][s].squeeze()
        l[sd] = ax[sd].imshow(im, **kwargs)

    # define slider
    axcolor = 'lightslategray'
    ax_c = fig.add_axes([0.25, 0.05, 0.65, 0.03], aspect="auto", facecolor=axcolor)

    slider = Slider(ax_c, 'Axis %i index' % axis, 0, cubes["mag"].shape[axis] - 1,
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



dir_path = "/home/ksansom/caseFiles/mri/VWI_proj/case2/cine"

dirs = ["left2_aca", "left_aca", "left_mca", "oblique2_ica", "oblique_ica", "term_ica"]

sub_dirs = ["mag", "other", "x", "y", "z"]

xyz_keys = ["x", "y", "z"]

test_path = "/home/ksansom/caseFiles/mri/VWI_proj/case2/cine/term_ica"

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

pkl_path = os.path.join(test_path, "term_ica.pkl")
with open(pkl_path, 'wb') as handle:
    pickle.dump(cine_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

cubes = {} # dictionary of image volumes
dict_time = {}
test = False
for k in cine_dict.keys():
    count = 0
    N = len(cine_dict[k])
    #print(N)
    # assuming image size is constant
    image_dtype = np.uint16
    if k in xyz_keys:
        image_dtype = np.float64
    im_vol = np.zeros((f_image_1.Rows, f_image_1.Columns,N),dtype=image_dtype)
    time_list = []
    for f_ in cine_dict[k]:
        f_image = pydicom.read_file(f_, stop_before_pixels=False)
        if(test == False):
            print(f_image_1.SeriesDescription, f_image_1.SeriesNumber)
            test = True
        img = f_image.pixel_array
        if k in xyz_keys:
            img = img.astype(image_dtype)*np.float64(f_image.RescaleSlope) + np.float64(f_image.RescaleIntercept)
        im_vol[:,:,count] = img
        count += 1
        time_list.append(float(f_image.TriggerTime))
    cubes[k] = im_vol
    dict_time[k] = time_list
    #print(time_list)

#determine if its phase contrast image
cube_show_slider(cubes, sub_dirs, axis=2, title="test")
