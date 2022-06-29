import cv2
import numpy as np
import os
from tqdm import tqdm
import json

imgs = [[file, cv2.imread('F:/Uni/Bachelorarbeit/Programs/gettransforms/images/' + file, 0)] for (dirpath, dirname, filenames) in os.walk('F:/Uni/Bachelorarbeit/Programs/gettransforms/images') for file in filenames]

config = json.load(open('F:/Uni/Bachelorarbeit/Trans_imgs/config.json'))
orb = cv2.ORB_create(nfeatures=config['orb']['nf'], firstLevel=config['orb']['fl'], edgeThreshold=config['orb']['ps'], patchSize=config['orb']['ps'], fastThreshold=config['orb']['ft'])
sift = cv2.SIFT_create(nfeatures=config['sift']['nf'], contrastThreshold=config['sift']['ct'], edgeThreshold=config['sift']['et'], sigma=config['sift']['si'])
mask = cv2.imread('F:/Uni/Bachelorarbeit/Programs/mask2.png', 0)
ins = {}

for i in tqdm(imgs):
    tmp = {}
    tmps, _ = sift.detectAndCompute(i[1], mask)
    tmpo, _ = orb.detectAndCompute(i[1], mask)
    ins[i[0]] = {'sift' : np.unique(cv2.KeyPoint.convert(tmps), axis=0).tolist(), 'orb' : np.unique(cv2.KeyPoint.convert(tmpo), axis=0).tolist()} 

file = open('fpcoords.json','w')
json.dump(ins, file)
file.close()