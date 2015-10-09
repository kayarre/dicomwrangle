# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:54:15 2015

@author: sansomk

based on the following
http://cmic.cs.ucl.ac.uk/fileadmin/cmic/Documents/DavidAtkinson/DICOM_6up.pdf
"""
import numpy as np
import os
import dicom

class StandardOrientationException(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

def bipedCheck(iminfo):
  """
  If Anatomical Orientation Type (0010,2210) is absent or has a value
  of BIPED, the x-axis is increasing to the left hand side of the patient.
  The y-axis is increasing to the posterior side of the patient.
  The z-axis is increasing toward the head of the patient.
  @param dicom data iminfo contains image info
  
  """
    #check coordinate system
  biped = False
  if((0x0010, 0x2210) in iminfo.keys()):
    axis_str = iminfo[(0x0010, 0x2210)]
    if(axis_str == "BIPED"):
      biped = True
    else:
      print("something weird with coordinate sys")
  else:
    biped = True
    
  if( biped == False):
    raise StandardOrientationException("not using standard axes")
  # add the standard axes here
  return


def transformMatrix(iminfo):
  """
  This function calculates the 4x4 transform matrix from the image
  coordinates to patient coordinates.
  """
  dt = np.float64
  

    
    
  # converts strings to doubles
  # image position patient  
  ipp = np.array(iminfo.ImagePositionPatient, dtype=dt)
  #image orientation patient  
  iop = np.array(iminfo.ImageOrientationPatient, dtype=dt)
  print("ipp", ipp)
  print("iop", iop)
  # pixel spacing
  ps = np.array(iminfo.PixelSpacing, dtype=dt)
  
  #check it whether image has been interpolated
  if (hasattr(iminfo, 'SpacingBetweenSlices')):
      if(iminfo.SpacingBetweenSlices < iminfo.SliceThickness):
          z_spac = iminfo.SpacingBetweenSlices
      else:
          z_spac = iminfo.SliceThickness
  else:
      z_spac= iminfo.SliceThickness
      
  print("z_spac", z_spac)
  z_spacing = dt(z_spac)
  #Translate to put top left pixel at ImagePositionPatient
  Tipp = np.array([[1.0, 0.0, 0.0, ipp[0]], 
                   [0.0, 1.0, 0.0, ipp[1]],
                   [0.0, 0.0, 1.0, ipp[2]],
                   [0.0, 0.0, 0.0, 1.0]], dtype=dt)
  # r and c make up direction cosines
  r = iop[0:3]
  print("r=iop[0:3]", r)
  c = iop[3:6]
  print("c=iop[3:6]", c)
  s = np.cross(r, c) # take the cross product
  print("s=rxc", s)
  R = np.array([[r[0], c[0], s[0], 0],
                [r[1], c[1], s[1], 0],
                [r[2], c[2], s[2], 0],
                [0.0, 0.0, 0.0, 1.0]], dtype=dt)
  
  # not sure about this, both images are 3D but have different values
  
  if(iminfo.MRAcquisitionType=='3D'): # 3D turboflash
    # info.SliceThickness
    S = np.array([[ps[1], 0.0, 0.0, 0.0],
                  [0.0, ps[0], 0.0, 0.0],
                  [0.0, 0.0, z_spacing, 0.0],
                  [0.0, 0.0, 0.0, 1.0]], dtype=dt)
                  
  else: # 2D epi dti
    # info.SpacingBetweenSlices
    S = np.array([[ps[1], 0.0, 0.0, 0.0],
                [0.0, ps[0], 0.0, 0.0],
                [0.0, 0.0, z_spacing, 0.0],
                [0.0, 0.0, 0.0, 1.0]], dtype=dt)
  print("S", S)
  T0 = np.eye(4, k=0, dtype=dt)
  '''
  T0 = np.array([[1., 0.0, 0.0, 0.0], 
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0]],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
  '''
  
  M = np.dot(Tipp, np.dot(R, np.dot(S, T0)))

  return M, R

def getTransformMatrix(iminfo1, iminfo2):
  """
  This function calculates the 4x4 transform and 3x3 rotation matrix 
  between two image coordinate systems. 
  M=Tipp*R*S*T0;
  Tipp:translation
  R:rotation
  S:pixel spacing
  T0:translate to center(0,0,0) if necessary
  info1: dicominfo of 1st coordinate system
  info2: dicominfo of 2nd coordinate system
  Rot: rotation matrix between coordinate system
  
  Kurt Sansom 2015
  based on matlab code by Alper Yaman
  """
  Mtf, Rtf = transformMatrix(iminfo1)
  Mdti, Rdti = transformMatrix(iminfo2)
  M = np.dot(np.linalg.inv(Mdti),Mtf)
  Rot = np.dot(np.linalg.inv(Rdti), Rtf)

  return M, Rot

def transform2(iminfo1):
  cos = iminfo1.ImageOrientationPatient
  cosines = [np.float64(val) for val in cos]
  ipp = iminfo1.ImagePositionPatient
  
  normal = [0., 0., 0.]
  normal[0] = cosines[1]*cosines[5] - cosines[2]*cosines[4]
  normal[1] = cosines[2]*cosines[3] - cosines[0]*cosines[5]
  normal[2] = cosines[0]*cosines[4] - cosines[1]*cosines[3]
  #print(normal)
  dist = 0.0
  for i in range(3):
    dist += normal[i]*np.float64(ipp[i])
  
  #print(dist)
  return dist
  
if __name__ == '__main__':
  #dcm_path = "/home/ksansom/caseFiles/mri/images/0.4/102/"
  dcm_path = "/Users/sansomk/caseFiles/mri/E431791260_merge/0.4/102"
  ds1 = dicom.read_file(os.path.join(dcm_path, 'E431791260S201I001.dcm'))
  ds2 = dicom.read_file(os.path.join(dcm_path, 'E431791260S201I002.dcm'))
  
  M, Rot = getTransformMatrix(ds1, ds2)
  print(M)
  print(Rot)
  dist1 = transform2(ds1)
  dist2 = transform2(ds2)
  print(dist2-dist1)