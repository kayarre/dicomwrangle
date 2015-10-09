# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:45:55 2015

@author: sansomk
"""

# D. Mason, Aug 2009 
from dicom.datadict import DicomDictionary, NameDict, CleanName 
import dicom 

new_dict_items = { 
0x10011001: ('UL', '1', "Test One", ''), 
0x10011002: ('OB', '1', "Test Two", ''), 
0x10011003: ('UI', '1', "Test Three", ''), 
} 
DicomDictionary.update(new_dict_items) 

new_names_dict = dict([(CleanName(tag), tag) for tag in 
new_dict_items]) 
NameDict.update(new_names_dict) 

filename = r"c:\co\pydicom\source\dicom\testfiles\CT_small.dcm" 
ds = dicom.read_file(filename) 

ds.TestOne = 42 
ds.TestTwo = '12345' 
ds.TestThree = '1.2.3.4.5' 

print ds.top() 