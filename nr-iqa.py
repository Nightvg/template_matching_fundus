#Need check
from brisque import BRISQUE
import os
import cv2
import numpy as np

b = BRISQUE()
path = 'E:/Uni/Bachelorarbeit/Programs/brisque_test'
files = [filenames for (dirpath, dirname, filenames) in os.walk(path)]
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
print(files)
for f in files[0]:
    print(f + ' : ' + str(b.get_score(path + '/' + f)))
