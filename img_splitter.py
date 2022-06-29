
import cv2
import os

p_from = 'E:/Uni/Bachelorarbeit/TMP'
p_to = 'E:/Uni/Bachelorarbeit/TMP_RGB'
names = ['blue.png', 'green.png', 'red.png']
files = [filenames for (dirpath, dirname, filenames) in os.walk(p_from)]
for f in files[0]:
    img = cv2.imread(p_from + '/' + f)
    img = cv2.split(img)
    i = 0
    for k in img:
        cv2.imwrite(p_to + '/' + f + names[i % 3], k)
        i += 1
