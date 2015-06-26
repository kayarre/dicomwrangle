# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:45:03 2015

@author: sansomk
"""

import dicom
import os
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button, RadioButtons
import fnmatch


#dcmpath='/Users/sansomk/Downloads/E431791260_FlowVol_01/' # mac
dcmpath = "/home/sansomk/caseFiles/mri/images/E431791260_FlowVol_01/mag"
dcm_files = []
count = 0
dict_test = {}
# tags 
# TriggerTime = time of the image
# SliceLocation = spatial location of slice.
# SliceThickness = the thickness of the image
# need to figure our how to convert images to axial ones
slice_location = []
trigger_time = []
image_dict = {}
count = 0
fn_dict = {"X":"FlowX_*.dcm", "Y":"Flowy_*.dcm", "Z":"FlowZ_*.dcm", "MAG":"Mag_*.dcm"}
new_dict = {}
for dirname, subdirlist, filelist in os.walk(dcmpath):
    for filen in  filelist:
        try:
            filePath = os.path.join(dirname,filen)
            #print(filePath)
            f = dicom.read_file(filePath, stop_before_pixels=True)
            #print(dirname, subdirlist, filelist)
            #print(filePath)
            #print(f.SliceLocation)
        except:
            print("error: {0}".format(filen))
            continue
        
        #dictionary of images
        if (f.TriggerTime not in image_dict.keys()):
            image_dict[f.TriggerTime] = {}
        if (f.SliceLocation not in image_dict[f.TriggerTime].keys()):
            image_dict[f.TriggerTime][f.SliceLocation] = {}
        
        for fn_key in fn_dict.keys():
            if( fn_key not in image_dict[f.TriggerTime][f.SliceLocation].keys()):
                    image_dict[f.TriggerTime][f.SliceLocation][fn_key] = {}
            #print(fn_key, filen, fn_dict[fn_key])
            if (fnmatch.fnmatch(filen, fn_dict[fn_key])):
                #print('did i get here')
                if (f.SOPInstanceUID not in image_dict[f.TriggerTime][f.SliceLocation][fn_key].keys()):
                    image_dict[f.TriggerTime][f.SliceLocation][fn_key][f.SOPInstanceUID] = [filePath]
                    #print(image_dict[fn_key])
        if (f.TriggerTime not in trigger_time):
            trigger_time.append(f.TriggerTime)
        if (f.SliceLocation not in slice_location):
            slice_location.append(f.SliceLocation)

#print(slice_location, trigger_time)
print(sorted(image_dict[image_dict.keys()[0]].keys()))
"""
for time in sorted(trigger_time):
    new_dict[time] = {}
    for loc in sorted(slice_location):
        new_dict[time][loc] = {}
        for var in fn_dict.keys():
            new_dict[time][loc][var] = {}
            for image in image_dict.keys():
                new_dict[time][loc][var][image] = {}
                if (image_dict[image][0] == var and
                    image_dict[image][1] == time and 
                    image_dict[image][2] == loc):
                    new_dict[time][loc][var][image] = image_dict[image] 

print(new_dict)

for key in dict_test.keys():
    print(key, dict_test[key])
print(count)
"""