import numpy as np
import cv2
import os
from tqdm import tqdm

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

files = [filenames for (dirpath, dirname, filenames) in os.walk('F:/Uni/Bachelorarbeit/Programs/gettransforms/images')]
imgs = [cv2.imread('F:/Uni/Bachelorarbeit/Programs/gettransforms/images/' + file) for file in files[0]]
mask = cv2.imread('F:/Uni/Bachelorarbeit/Programs/mask2.png', 0)


file = open('brightness.txt','w')
k = 0
for i in tqdm(imgs):
    b, g, r = cv2.split(i)
    b = b[mask > 0].astype(float)
    g = g[mask > 0].astype(float)
    r = r[mask > 0].astype(float)

    img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)  
    file.write(files[0][k] + '\t' + "{:.2f}".format(np.median(b)) + '\t' + "{:.2f}".format(np.mean(b)) + '\t' + "{:.2f}".format(np.std(b)))
    file.write('\t' + "{:.2f}".format(np.median(g)) + '\t' + "{:.2f}".format(np.mean(g)) + '\t' + "{:.2f}".format(np.std(g)))
    file.write('\t' + "{:.2f}".format(np.median(r)) + '\t' + "{:.2f}".format(np.mean(r)) + '\t' + "{:.2f}".format(np.std(r)) + '\t' + "{:.2f}".format(cv2.Laplacian(img, cv2.CV_64F).var()) + '\t' + "{:.2f}".format(signaltonoise(i[mask > 0], None)) + '\n')
    k += 1
file.close()