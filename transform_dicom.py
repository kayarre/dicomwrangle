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

def transformMatrix(iminfo):
  """
  This function calculates the 4x4 transform matrix from the image
  coordinates to patient coordinates.
  """
  # converts strings to doubles
  # image position patient  
  ipp = np.array(iminfo.ImagePositionPatient, dtype=np.float64)
  #image orientation patient  
  iop = np.array(iminfo.ImageOrientationPatient, dtype=np.float64)
  # pixel spacing
  ps = np.array(iminfo.PixelSpacing, dtype=np.float64)
  
  #check it whether image has been interpolated
  if (hasattr(iminfo, 'SpacingBetweenSlices')):
      if(iminfo.SpacingBetweenSlices < iminfo.SliceThickness):
          z_spacing = np.float64(iminfo.SpacingBetweenSlices)
      else:
          z_spacing = np.float64(iminfo.SliceThickness)
  else:
      z_spacing = np.float64(iminfo.SliceThickness)
      
  #Translate to put top left pixel at ImagePositionPatient
  Tipp = np.array([[1., 0.0, 0.0, ipp[0]], 
                   [0.0, 1.0, 0.0, ipp[1]],
                   [0.0, 0.0, 1.0, ipp[2]],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
  # r and c make up direction cosines
  r = iop[0:3]
  c = iop[3:6]
  s = np.cross(r, c) # take the cross product
  R = np.array([[r[0], c[0], s[0], 0],
                [r[1], c[1], s[1], 0],
                [r[2], c[2], s[2], 0],
                [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
  
  # not sure about this, both images are 3D but have different values
  
  if(iminfo.MRAcquisitionType=='3D'): # 3D turboflash
    # info.SliceThickness
    S = np.array([[ps[1], 0.0, 0.0, 0.0],
                  [0.0, ps[0], 0.0, 0.0],
                  [0.0, 0.0, z_spacing, 0.0],
                  [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
                  
  else: # 2D epi dti
    # info.SpacingBetweenSlices
    S = np.array([[ps[1], 0.0, 0.0, 0.0],
                [0.0, ps[0], 0.0, 0.0],
                [0.0, 0.0, z_spacing, 0.0],
                [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

  T0 = np.eye(4, k=0, dtype=np.float64)
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

if __name__ == '__main__':
  ds1 = dicom.read_file("/home/ksansom/caseFiles/mri/images/0.4/102/E431791260S201I214.dcm")
  ds2 = dicom.read_file("/home/ksansom/caseFiles/mri/images/E431791260_FlowVol_01/mag/Mag_397.dcm")
  

  
  M, Rot = getTransformMatrix(ds1, ds2)
  print(M)
  print(Rot)