from fileinput import filename
import json
import numpy as np
import cv2
import os
from tqdm import tqdm

def get_src_dst(pts1, pts2):
    src = []
    dst = []
    for i in pts1.items():
        if i[0] in pts2:
            src.append(i[1])
            dst.append(pts2[i[0]])
    return np.array(src), np.array(dst)

def get_absolute_differences(matrix : np.ndarray, src : np.ndarray, dst :np.ndarray) -> float:
    if matrix.size == 0:
        return -1
    matrix = matrix.astype(float)
    if matrix.shape != (3, 3):
        matrix = np.vstack((matrix, np.array([0.0,0.0,1.0])))
    src = [[t[0] / t[2], t[1] / t[2]] for t in np.dot(np.column_stack((src, np.ones((src.shape[0],)))), matrix.T)]
    return np.mean([np.linalg.norm(x - y) for x,y in zip(src, dst)])


annotations = ['23','24','25','26','27','28','29']
transformations = []
t_list = {}

for i in tqdm(annotations):
    a = json.load(open('F:/Uni/Bachelorarbeit/Programs/annotations/' + i + '.json'))
    patient = {}
    for j in a:
        tmp = {t['value']['keypointlabels'][0] : [t['value']['x'] / 100 * t['original_width'], t['value']['y'] / 100 * t['original_height']] for t in j['annotations'][0]['result']}
        patient[j['file_upload'][9:].replace('_',' ')] = tmp
        if not os.path.isfile('F:/Uni/Bachelorarbeit/Programs/kp_coords_v2/' + j['file_upload'][9:].replace('_',' ')  + '.json'):
            json.dump(tmp, open('F:/Uni/Bachelorarbeit/Programs/kp_coords_v2/' + j['file_upload'][9:].replace('_',' ')  + '.json', 'w'))
    for im1 in tqdm(patient.items(), leave=False):
        to_file = {}

        tmplist = []
        for im2 in patient.items():
            if im1 == im2:
                continue
            src, dst = get_src_dst(im1[1],im2[1])
            if len(src) < 7:
                continue
            affin = None
            perspective = None
            tmplist.append(im2[0])
            if not os.path.isfile('F:/Uni/Bachelorarbeit/Programs/imdata/' + im1[0] +'.json'):
                tmp = {}
                affin, _ = cv2.estimateAffine2D(src, dst, None, cv2.RANSAC, 5.0)
                perspective, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                tmp['affin'] = {'matrix' : affin.tolist(), 'difference' : get_absolute_differences(affin, src, dst)}
                tmp['perspective'] = {'matrix' : perspective.tolist(), 'difference' : get_absolute_differences(perspective, src, dst)}
                to_file[im2[0]] = tmp            
        if len(to_file) > 0:
            json.dump(to_file, open('F:/Uni/Bachelorarbeit/Programs/imdata/' + im1[0] + '.json', 'w'))
        if(len(tmplist) > 0):
            t_list[im1[0]] = tmplist
if not os.path.isfile('F:/Uni/Bachelorarbeit/Programs/t_list.json'):
    json.dump(t_list, open('F:/Uni/Bachelorarbeit/Programs/t_list.json', 'w'))