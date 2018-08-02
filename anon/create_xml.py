
import xml.etree.ElementTree as ET

tag_list = "taglist.txt"

lines = [line.rstrip('\n') for line in open(tag_list)]

#create the file structure
data = ET.Element('data') 

no_edit_list = ["0018,1000", "0010,0020", "0008,0080", "0008,0081", "0008,1010",
                "0008,1030", "0018,1000", "0020,000d", "0032,1060", "0040,0254", 
                "0040,0253"]

for line in lines:
    #print(line)
    tag, tag_name = line.split("\t")
    item = ET.Element('item')
    name = ET.SubElement(item, "name")
    name.text = tag
    des = ET.SubElement(item, "description")
    des.text = tag_name
    edit = ET.SubElement(item, "editable")
    if tag in no_edit_list:
        edit.text = "no"
    else:
        edit.text = "yes"
    
    data.append(item)
    
    
#ET.dump(data)
# create a new XML file with the results
#mydata = ET.tostring(data)  
#myfile = open("items2.xml", "w")  
#myfile.write(mydata)

from xml.dom import minidom

xmlstr = minidom.parseString(ET.tostring(data)).toprettyxml(indent="   ")
with open("items2.xml", "w") as f:
    f.write(xmlstr)
