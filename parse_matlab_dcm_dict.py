# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:25:17 2015

@author: sansomk
"""

def write_dict(f, dict_name, attributes, tagIsString):
    if tagIsString:
        entry_format = """'{Tag}': ('{VR}', '{VM}', "{Name}", '{Retired}', '{Keyword}')"""
    else:
        entry_format = """{Tag}: ('{VR}', '{VM}', "{Name}", '{Retired}', '{Keyword}')"""
    f.write("\n%s = {\n    " % dict_name)
    f.write(",\n    ".join(entry_format.format(**attr) for attr in attributes))
    f.write("\n}\n")


import os
import dicom
import binascii

path_dcm_dict = '/Users/sansomk/Downloads'
f = open(os.path.join(path_dcm_dict, "dicom-dict-philips.txt"), "rU")
lines = f.readlines()
dcm_dict = {}
private_dict = {}
assume_good = True
for line in lines:
    if (line[0] == "#"):
        #print(line)
        continue
    else:
        #print( line)
        line_list = line.split()
        #print('this lines',line_list)
        tag = line_list[0].translate(None, "()")
        tagpair = tag.split(",")
        print(tagpair[0], tagpair[1])
        assume_good = True
        try:
            tag1 = binascii.unhexlify("0x"+tagpair[0])
            tag2 = binascii.unhexlify("0x"+tagpair[1])
        except Exception as e:
            assume_good = False
            print(str(e))
        print((tag1,tag2))
        break
        #tag2 = r"0x{0}".format(tagpair[1])
        if (assume_good):
            dcm_dict[(tag1, tag2)] = (line_list[1], line_list[3].rstrip("\r\n"), line_list[2], 
                   '', line_list[2])
        else:
            private_dict[(tagpair[0], tagpair[1])] = (line_list[1], line_list[3].rstrip("\r\n"), line_list[2], 
                   '', line_list[2])
        #print("0x{0}{1}: (('{2}', '{3}', ""{4}"", '{5}', '{6}'))".format(
        #    tagpair[0], tagpair[1], line_list[1], line_list[3].rstrip("\r\n"),
        #    line_list[2], '', line_list[2]))
        #0x00000000: ('UL', '1', "Command Group Length", '', 'CommandGroupLength'),
        #0x00000001: ('UL', '1', "Command Length to End", 'Retired', 'CommandLengthToEnd'),

test_file = '/Users/sansomk/caseFiles/mri/E431791260_FlowVol_01/flow/FlowX_337.dcm'

ds = dicom.read_file(test_file)
print(dcm_dict.keys()[0])
print(ds.keys()[0])
for key_v in ds.keys():
    print(key_v)
    if(key_v in dcm_dict.keys()):
        print("in", key_v, ds[key_v])
    else:
        print("out", key_v, ds[key_v])