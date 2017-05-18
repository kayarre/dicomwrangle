# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:45:03 2015

@author: sansomk
"""

import pydicom
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np

#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button, RadioButtons
#import fnmatch

import sqlite3 as lite


def cube_show_slider(cube, axis=2, **kwargs):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    """

    # check dim
    if not cube.ndim == 3:
        raise ValueError("cube should be an ndarray with ndim == 3")

    # generate figure
    fig = plt.figure()
    ax = plt.subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # select first image
    s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
    im = cube[s].squeeze()

    # display image
    l = ax.imshow(im, **kwargs)

    # define slider
    axcolor = 'lightgoldenrodyellow'
    ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

    slider = Slider(ax, 'Axis %i index' % axis, 0, cube.shape[axis] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None)
                 for i in range(3)]
        im = cube[s].squeeze()
        l.set_data(im, **kwargs)
        fig.canvas.draw()

    slider.on_changed(update)

    plt.show()

database_path = '/home/ksansom/Documents/SlicerDICOMDatabase/ctkDICOM.sql'
con = lite.connect(database_path)
cur = con.cursor()

#dcmpath = "/home/ksansom/caseFiles/mri/VWI_proj/dicom/case2/MRI-Phase_Contrast"
patient_id = "H4015577"

patient_query = 'SELECT UID FROM Patients WHERE PatientID="{0}"'.format(patient_id)
print("print query", patient_query)
query_ = cur.execute(patient_query)
patient_uid = query_.fetchall()
#print(patient_uid[0][0])

study_query = 'SELECT StudyInstanceUID FROM Studies WHERE PatientsUID IN ({0})'.format(patient_query)
print("print study query", study_query)
query_ = cur.execute(study_query)
study_uid = query_.fetchall()
study_list = [a[0] for a in study_uid]
print("study number {0}".format(len(study_list)))

case_dict = {}
for study in study_list:
    series_query  = 'SELECT SeriesInstanceUID, SeriesNumber FROM Series WHERE StudyInstanceUID="{0}"'.format(study)
    query_ = cur.execute(series_query)
    series_uid = query_.fetchall()
    series_list = [a[0] for a in series_uid]
    print("series number {0}".format(len(series_list)))

    for series, series_n in series_uid:
        image_query  = 'SELECT FileName FROM Images WHERE SeriesInstanceUID="{0}"'.format(series)
        query_ = cur.execute(image_query)
        images = query_.fetchall()
        image_list = [a[0] for a in images]
        #check first image
        try:
            f_image_1 = pydicom.read_file(image_list[0], stop_before_pixels=True)
        except:
            print("error: {0}".format(filen))
            continue


        if (not hasattr(f_image_1, "TriggerTime")):
            #determine if its phase contrast image
            print("probably not phase contrast")
            continue
        
        print("series Description: {0}".format(f_image_1.SeriesDescription))
        #print(f_image)
        # assume the same for all proceeding images
        X = f_image_1.Rows
        Y = f_image_1.Columns

        info_list = []
        for f_ in image_list:
            try:
                f_image = pydicom.read_file(f_, stop_before_pixels=True)
            except:
                print("error: {0}".format(filen))
                continue

            #check for image size
            if( X != f_image.Rows and Y != f_image.Columns):
                print("found repeated image that doesn't match size")
                print(X, f_image.Rows, Y, f_image.Columns)
                print(study, series, series_n, f_)
                print(f_image_1.StudyInstanceUID,
                      f_image_1.SeriesInstanceUID,
                      f_image_1.SeriesNumber, image_list[0])
                print("")
            else:
                info_list.append((f_, float(f_image.SliceLocation),
                                  float(f_image.TriggerTime)))
        #sorted_list = sorted(info_list, key = operator.itemgetter(1, 2))
        sorted_list = sorted(info_list, key=lambda x: (x[1], x[2]))
        #for item in sorted_list:
        #    print(item[1], item[2])

        count = 0
        N = len(sorted_list)
        im_vol = np.zeros((X,Y,N),dtype=np.uint16)
        for f_ in sorted_list:
            f_image = pydicom.read_file(f_[0], stop_before_pixels=False)
            im_vol[:,:,count] = f_image.pixel_array
            count += 1
        #determine if its phase contrast image

        cube_show_slider(im_vol)



#image_query = 'SELECT FileName FROM Images WHERE SeriesInstanceUID IN (SELECT SeriesInstanceUID FROM Series WHERE StudyInstanceUID IN (SELECT StudyInstanceUID FROM Studies WHERE PatientsUID IN (SELECT UID FROM Patients WHERE PatientID="{0}")))'.format(patient_id)
#query_ = cur.execute(image_query)
