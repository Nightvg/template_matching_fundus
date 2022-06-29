import xml.etree.ElementTree as ET
import numpy as np

root = ET.parse('F:/Uni/Bachelorarbeit/Programs/annotations.xml').getroot()

points = np.array([[float(b.attrib['points'].split(',')[0]), float(b.attrib['points'].split(',')[1])] for a in root.findall('image') for b in a.findall('points')])
x_std = points[:,0].std()
y_std = points[:,1].std()

print(x_std)
print(y_std)
 