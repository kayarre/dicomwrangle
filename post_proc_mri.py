# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:54:15 2015

@author: sansomk
"""
import numpy as np
#import matplotlib.pyplot as plt 
#from mpl_toolkits.mplot3d import Axes3D
#import scipy.special as special #jn, jn_zeros
#from matplotlib import cm
import os
#from scipy import interpolate
import glob
import re
import dicom
#from matplotlib.widgets import Slider, Button, RadioButtons
import fnmatch
import pickle
import hashlib

def build_dcm_dict(dcmpath, fn_dict, image_dict_pkl="image_dict.pkl"):
    slice_location = []
    trigger_time = []
    image_dict = {}
    hash_value = hashlib.sha1(dcmpath).hexdigest()
    #count = 0
    for dirname, subdirlist, filelist in os.walk(dcmpath):
        for filen in  filelist:
            filePath = os.path.join(dirname,filen)    
            #print(filePath)
            #print(dirname, subdirlist, filelist)
            #print(filePath)
            #print(f.SliceLocation)
            try:
                f = dicom.read_file(filePath, stop_before_pixels=True)
            except Exception as e: 
               print(str(e))
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
    print("writing {0} to current working directory".format(image_dict_pkl))
    with open(os.path.join(os.getcwd(), image_dict_pkl), "wb") as pkl_f:
        pickle.dump(hash_value, pkl_f, -1)
        pickle.dump(image_dict, pkl_f, -1)
    return image_dict

def load_dcm_dict(dcmpath, fn_dict, image_dict_pkl="image_dict.pkl"):    
    try:
        with open(os.path.join(os.getcwd(), image_dict_pkl), "rb") as pkl_f:
            hash_value = pickle.load(pkl_f)
            image_dict = pickle.load(pkl_f)
        if (hash_value != hashlib.sha1(dcmpath).hexdigest()):
            raise Exception("pickle file doesn't match, rebuilding")
    except Exception as e: 
        print(str(e))
        print("no image dictionary pickle file, building one")
        # create the dictionary
        #dcm_files = []
        #count = 0
        #dict_test = {}
        # tags 
        # TriggerTime = time of the image
        # SliceLocation = spatial location of slice.
        # SliceThickness = the thickness of the image
        # need to figure our how to convert images to axial ones
        image_dict = build_dcm_dict(dcmpath, fn_dict, image_dict_pkl="image_dict.pkl")
    else:
        print('Read the pickle file from the current directory')
    trigger_time = image_dict.keys()
    slice_location = image_dict[trigger_time[0]].keys()
        

    return image_dict, sorted(slice_location), sorted(trigger_time)
    
def create_image_volume(image_dict, mri_2_cfd_map, image_type, return_coord=True):
    trigger_t = mri_2_cfd_map[0]
    slice_location = image_dict[trigger_t].keys()
    dcm_files = []    
    for loc in slice_location:
      for image_id in image_dict[trigger_t][loc][image_type].keys():
        dcm_files.append(image_dict[trigger_t][loc][image_type][image_id][0])

    path_loc = zip(dcm_files, slice_location)
    path_loc.sort(key=lambda x: x[1])
    dcm_files, slice_location = zip(*path_loc)
    #print(slice_location)
    
    # get reference image
    #print(len(dcm_files), dcm_files)
    ref_image = dicom.read_file(dcm_files[0])
    # load dimensions based on the number of rows columns and slices
    const_pixel_dims = (int(ref_image.Rows), int(ref_image.Columns), len(dcm_files))
    
    #check it whether image has been interpolated
    if (hasattr(ref_image, 'SpacingBetweenSlices')):
      if(ref_image.SpacingBetweenSlices < ref_image.SliceThickness):
        z_spacing = float(ref_image.SpacingBetweenSlices)
      else:
        z_spacing = float(ref_image.SliceThickness)
    else:
      z_spacing = float(ref_image.SliceThickness)


    # the array is sized based on 'const_pixel_dims
    
    array_dicom = np.zeros(const_pixel_dims, dtype=np.float64) #ref_image.pixel_array.dtype)
    print(array_dicom.shape)
    #loop through all the DICOM FILES
    for filenamedcm in dcm_files:
      #read the file
      ds = dicom.read_file(filenamedcm)
      #store the raw image data
      array_dicom[:, :, dcm_files.index(filenamedcm)] = (
          np.asarray(ds.pixel_array, dtype=np.float64) * (
          np.float64(ds.RescaleSlope)) + np.float64(ds.RescaleIntercept))
    
    #testindx = np.where(array_dicom !=0)
    #minx = np.min(testindx[0])
    #miny = np.min(testindx[1])
    #minz = np.min(testindx[2])
    #maxx = np.max(testindx[0])
    #maxy = np.max(testindx[1])
    #maxz = np.max(testindx[2])
    
    if (return_coord == True):
      # load spacing values in mm
      const_pixel_spacing = (float(ref_image.PixelSpacing[0]), 
                             float(ref_image.PixelSpacing[1]), z_spacing)
      x = np.arange(const_pixel_spacing[0]/2.0, (const_pixel_dims[0]+1)*const_pixel_spacing[0]-1.0,
                    const_pixel_spacing[0])
      y = np.arange(const_pixel_spacing[1]/2.0, (const_pixel_dims[1]+1)*const_pixel_spacing[1]-1.0,
                    const_pixel_spacing[1])
      z = np.arange(const_pixel_spacing[2]/2.0, (const_pixel_dims[2]+1)*const_pixel_spacing[2]-1.0,
                    const_pixel_spacing[2])
      return array_dicom, x,y,z
    else:
      return array_dicom

def tryint(s):
    try:
        return int(s)
    except:
        return s
     
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

from bisect import bisect_left
def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0], pos
    if pos == len(myList):
        return myList[-1], pos - 1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after, pos
    else:
       return before, pos - 1


def sort_cfd_sol(dir_path, search_name, t_orig, trigger_time_list):
  mri_time_list = [float(t)/1000.0 + t_orig for t in trigger_time_list]
  sol_files = glob.glob(os.path.join(dir_path, search_name))
  sort_nicely(sol_files)
  split_paths = [ os.path.split(sol_file) for sol_file in sol_files]
  #print(split_paths)
  
  split_names = [file_name.split('-') for file_name in zip(*split_paths)[-1]]
  #print(split_names)
  
  cfd_file_t = [float(split_name[-1]) for split_name in split_names]
  #print(cfd_file_t)
  
  cfd_file_list =[]
  #print(len(mri_time_list), len(cfd_file_t))
  for t in mri_time_list:
    val, pos = takeClosest(cfd_file_t, t)
    #print(pos)    
    cfd_file_list.append(sol_files[pos])
    #print(takeClosest(cfd_file_t, t), t)
  return zip(trigger_time_list, cfd_file_list)
  
def read_cfd_sol_file(mapped_tuple, scale, return_coord=True):
  x_coords = []
  y_coords = []
  z_coords = []    
  vx = []
  vy = []
  vz = []
  mri_time = mapped_tuple[0]
  cfd_file_path = mapped_tuple[1]

  print(mri_time, cfd_file_path)


  with open(cfd_file_path, 'rU') as cfd_file:
    for idx, line in enumerate(cfd_file):
      if (idx == 0):
        f_vars = line.split()
        print(f_vars)
      else:
        nums = line.split()
        #print(nums)
        if (return_coord == True):
          #nodes.append(int(nums[0]))
          x_coords.append(float(nums[1]))
          y_coords.append(float(nums[2]))
          z_coords.append(float(nums[3]))

        vx.append(float(nums[5]))
        vy.append(float(nums[6]))
        vz.append(float(nums[7]))
  if(scale == "m2mm"):
      rescale = 1000.00
  return np.asarray((x_coords, y_coords, z_coords,
                    vx, vy, vz))*rescale

    
  ''' 
  for path_n in sol_files:
    path, folder = os.path.split(path_n) 
    split_name = folder.split('-')
    print(split_name, '-'.join(split_name))
  '''  

  
'''
# mri data part
  #dcm_path = "/home/sansomk/caseFiles/mri/images/E431791260_Merge/0.4/102"
    dcm_path = "/home/sansomk/caseFiles/mri/images/E431791260_FlowVol_01"
    dcm_files = []
    slice_location = []
    acq_N = []
    for dirname, subdirlist, filelist in os.walk(dcm_path):
      for filen in filelist:
        filePath = os.path.join(dirname, filen)
        try:          
          #print(filePath)
          f = dicom.read_file(filePath, stop_before_pixels=True)
        except Exception as e: 
          print(str(e))
          print("error: {0}".format(filen))
        if (f.TriggerTime < 1.0 and filePath not in dcm_files):
          dcm_files.append(filePath)
          slice_location.append(f.SliceLocation)
          acq_N.append(f.AcquisitionNumber)
        #else:
        #  dcm_files.append(filePath)
    path_loc = zip(dcm_files, slice_location)
    path_loc.sort(key=lambda x: x[1])
    dcm_file, slice_location = zip(*path_loc)
    print(slice_location, acq_N)
    # get reference image
    print(len(dcm_files))
    ref_image = dicom.read_file(dcm_files[0])
    # load dimensions based on the number of rows columns and slices
    const_pixel_dims = (int(ref_image.Rows), int(ref_image.Columns), len(dcm_files))
    
    #check it whether image has been interpolated
    if (hasattr(ref_image, 'SpacingBetweenSlices')):
      if(ref_image.SpacingBetweenSlices < ref_image.SliceThickness):
        z_spacing = float(ref_image.SpacingBetweenSlices)
    else:
      z_spacing = float(ref_image.SliceThickness)

    # load spacing values in mm
    const_pixel_spacing = (float(ref_image.PixelSpacing[0]), 
                           float(ref_image.PixelSpacing[1]), z_spacing)
    #x = np.arange(0.0, (const_pixel_dims[0]+1)*const_pixel_spacing[0],
    #              const_pixel_spacing[0])
    #y = np.arange(0.0, (const_pixel_dims[1]+1)*const_pixel_spacing[1],
    #              const_pixel_spacing[1])
    #z = np.arange(0.0, (const_pixel_dims[2]+1)*const_pixel_spacing[2],
    #              const_pixel_spacing[2])
    # the array is sized based on 'const_pixel_dims
    
    array_dicom = np.zeros(const_pixel_dims, dtype=ref_image.pixel_array.dtype)
    print(array_dicom.shape)
    #loop through all the DICOM FILES
    for filenamedcm in dcm_files:
      #read the file
      ds = dicom.read_file(filenamedcm)
      #store the raw image data
      array_dicom[:, :, dcm_files.index(filenamedcm)] = ds.pixel_array
    
    testindx = np.where(array_dicom !=0)
    minx = np.min(testindx[0])
    miny = np.min(testindx[1])
    minz = np.min(testindx[2])
    maxx = np.max(testindx[0])
    maxy = np.max(testindx[1])
    maxz = np.max(testindx[2])
    
    
    # Create some data
    #x, y, z = np.ogrid[-5:5:100j, -5:5:100j, -5:5:100j]
    #data = np.sin(3*x)/x + 0.05*z**2 + np.cos(3*y)

    m = VolumeSlicer(data=array_dicom[minx:maxx, miny:maxy,:])
    m.configure_traits()
'''
if __name__ == '__main__':
  #dcmpath='/Users/sansomk/caseFiles/mri/E431791260_FlowVol_01/' # mac
  #dcmpath = "/home/sansomk/caseFiles/mri/images/E431791260_FlowVol_01/"
  dcmpath = "/home/ksansom/caseFiles/mri//images/E431791260_FlowVol_01/"
  image_dict_pkl = "image_dict.pkl"
  fn_dict = {"X":"FlowX_*.dcm", "Y":"Flowy_*.dcm", "Z":"FlowZ_*.dcm", "MAG":"Mag_*.dcm"}
  image_dict, slice_location, trigger_time = load_dcm_dict(dcmpath, fn_dict, image_dict_pkl)
  print("trigger time", trigger_time)

  #cfd_ascii_path = "/home/sansomk/caseFiles/mri/cfd"
  #cfd_ascii_path = "/Users/sansomk/caseFiles/mri/solution_ascii"
  cfd_ascii_path = "/raid/home/ksansom/caseFiles/mri/healthy/output_2/fluent_2/ascii_dir/"
  search_name = "mri_carotid-*"
  t_init = 2.8440  
  mri_2_cfd_map = sort_cfd_sol(cfd_ascii_path, search_name, t_init, trigger_time)
  print("map", mri_2_cfd_map)
  
  array_mag, x, y, z = create_image_volume(image_dict, mri_2_cfd_map[0], "MAG", True)
  X, Y, Z = np.meshgrid(x,y,z)
  print("mri array shapes", array_mag.shape, X.shape, Y.shape, Z.shape)  
  print("max mri value", np.max(np.max(np.max(array_mag))))
  
  from volume_slicer_adv import VolumeSlicer
  m = VolumeSlicer(data=array_mag)
  m.configure_traits()
  '''
  field = read_cfd_sol_file(mapped_tuple=mri_2_cfd_map[0], scale="m2mm")
  print(field.shape)
  vmag_cfd = np.sqrt(np.power(field[3],2) + np.power(field[4],2)  + np.power(field[5],2))
  print("max cfd value", np.max(np.max(np.max(vmag_cfd))))
  print("mri mag shape", vmag_cfd.shape)
  
  """
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')  
  ax.scatter(field[0], field[1], field[2], c=vmag_cfd)
  """
  
  #from numpy import array
  from scipy.interpolate import RegularGridInterpolator as rgi
  from scipy import interpolate 
  interp_function = rgi((x,y,z), array_mag, bounds_error=False, fill_value=0.0)
  get_points_test = interp_function((field[0], field[1], field[2]))
  #print(x, y, z)
  cfd_coord = field[0:3].reshape(field[0:3].shape[0], -1).T
  print(cfd_coord.shape)
  cfd_interp = interpolate.LinearNDInterpolator(cfd_coord, vmag_cfd, 0.0, False)
  
  #np.reshape()
  mri_coord = np.asarray((X,Y,Z)).reshape(field[0:3].shape[0], -1).T
  print(mri_coord.shape)
  mag_test = cfd_interp(mri_coord)
  mag_test = np.reshape(mag_test,)
  
  print(mag_test.shape)
  m = VolumeSlicer(data=mag_test)#[minx:maxx, miny:maxy,:])
  m.configure_traits()
  '''
  