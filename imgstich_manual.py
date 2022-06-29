
import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np
import math
import random

def getVectors(points1: list) -> np.ndarray:
    return np.array([p1 - p2 for i, p1 in enumerate(np.array(points1)) for p2 in np.array(points1[:i] + points1[i+1:])])


def arrayMatches(vec1: np.ndarray, vec2: np.ndarray, THRESHOLD) -> list:
    good = []
    m = np.array([[np.linalg.norm(x - y) for x in vec2] for y in vec1])
    bool_m = (m <= THRESHOLD).astype(int)

    sums = np.array([sum(x) == 1 for x in bool_m])
    if not sums.all():
        sums = np.array([sum(x) == 1 for x in bool_m.T])
        if not sums.all():
            print("threshold to tight")
            return


def randListMatches(vec1: np.ndarray, vec2: np.ndarray, THRESHOLD, MAX_TRIES=100) -> list:
    good = []
    b, tmp = vec1, vec2 if vec1.shape[0] > vec2.shape[0] else vec2, vec1
    count = 0
    s = np.zeros(b.shape)
    s[:tmp.shape[0]] = tmp
    sub = [np.linalg.norm(x) for x in (b - s)]
    prob, bo = 1.0, True

    while bo and len(sub) > 0 and count < MAX_TRIES:
        i, ma, mi = random.randint(len(sub)), sub.index(max(sub)), sub.index(min(sub))
        if min(sub) < THRESHOLD and s[mi] != np.array([0,0]):
            good.append((b[mi], s[mi]))
            np.delete(b, mi, axis=0)
            np.delete(s, mi, axis=0)
            del sub[mi]
            count = 0
        else:    
            s[i], s[ma] = s[ma], s[i]
            sub[i], sub[ma] = np.linalg.norm(b[i] - s[i]), np.linalg.norm(b[ma] - s[ma])
            count += 1

        if(len(good) >= 4):
            prob = 1 - len(good) / (len(sub) + len(good))
            bo = random.random() < prob
        
    return good

def warp(good_matches, warp_img, size):
    good_matches = np.array(good_matches)
    try:
        if len(good_matches) < 3:
            return np.zeros(size)
        src_points = np.float32(good_matches[:, 0, :])
        dst_points = np.float32(good_matches[:, 1, :])

        M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        return cv2.warpPerspective(warp_img, M, size)

    except AssertionError as e:
        print(e)


# Loading resources
root = ET.parse('E:/Uni/Bachelorarbeit/Programs/testtask/annotations.xml').getroot()
files = [filenames for (dirpath, dirname, filenames) in os.walk('E:/Uni/Bachelorarbeit/Programs/testtask/images')]
imgs = [cv2.imread('E:/Uni/Bachelorarbeit/Programs/testtask/images/' + file, 0) for file in files[0]]
kp = {}
for image in root.findall('image'):
    kp_tmp = []
    for s in image.findall('points'):
        x, y = s.attrib['points'].split(',')
        kp_tmp.append([float(x), float(y)])
    kp[image.attrib['name']] = kp_tmp

print('Loading done')
PIC1, PIC2, PIC3 = '25_001 - normal.PNGblue.png', '25_001 - normal.PNGgreen.png', '25_001 - normal.PNGred.png'

# Get Matches
good_matches1 = arrayMatches(kp[PIC2], kp[PIC1], 20) #TODO
good_matches2 = arrayMatches(kp[PIC2], kp[PIC3], 20) #TODO -> anpassen an die Vektoridee

try:
    h, w = imgs[files[0].index(PIC1)].shape
    b = warp(good_matches1, imgs[files[0].index(PIC1)], (w, h))
    g = imgs[files[0].index(PIC2)]
    r = warp(good_matches2, imgs[files[0].index(PIC3)], (w, h))
    cv2.imwrite('test.png', cv2.merge([b, g, r]))
except cv2.error as e:
    print(e)
