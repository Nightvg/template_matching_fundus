import cv2
import os
import numpy as np
from tqdm import tqdm
import json
from manualtransform import get_absolute_differences
from manualtransform import get_src_dst

def sift_calc(im_f, im_t, pf, pt, args, affin=True):
    sift = cv2.SIFT_create(nfeatures=args['nf'], contrastThreshold=args['ct'], edgeThreshold=args['et'], sigma=args['si'])
    mask = cv2.imread('F:/Uni/Bachelorarbeit/Programs/mask2.png', 0)
    kp1, des1 = sift.detectAndCompute(im_f, mask)
    kp2, des2 = sift.detectAndCompute(im_t, mask)
    bf = cv2.BFMatcher()
    if len(kp1) != 0 and len(kp2) != 0:
        try:
            matches = bf.knnMatch(des1, des2, k=2) 
            good = []      
            if len(matches[0]) < 2:
                for m in matches:
                    good.append(m[0])
            else:
                for m, n in matches:
                    if m.distance < 0.75*n.distance:
                        good.append(m)
            MIN_MATCH_COUNT = 10
            if len(good) >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                sift_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, None, cv2.RANSAC, 5.0) if affin else cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if sift_matrix is not None:
                    #im_f = cv2.warpAffine(im_f, sift_matrix, (im_t.shape[1], im_t.shape[0])) if affin else cv2.warpPerspective(im_f, sift_matrix, (im_t.shape[1], im_t.shape[0]))
                    diff = get_absolute_differences(sift_matrix, pf, pt)
                    return str(sift_matrix.tolist()) + '\t' + str(diff)    
                else:
                    return str((np.identity(3)*3).tolist()) + '\t' + '-1'
        except:
            return str((np.identity(3)*2).tolist()) + '\t' + '-1'
    return str(np.identity(3).tolist()) + '\t' + '-1'

def orb_calc(im_f, im_t, pf, pt, args, affin=True):
    orb = cv2.ORB_create(nfeatures=args['nf'], firstLevel=args['fl'], edgeThreshold=args['ps'], patchSize=args['ps'], fastThreshold=args['ft'])
    mask = cv2.imread('F:/Uni/Bachelorarbeit/Programs/mask2.png', 0)
    kp1, des1 = orb.detectAndCompute(im_f, mask)
    kp2, des2 = orb.detectAndCompute(im_t, mask)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if len(kp1) != 0 and len(kp2) != 0:
        try: 
            matches = bf.match(des1, des2) 
            MIN_MATCH_COUNT = 10
            if len(matches) >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                orb_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, None, cv2.RANSAC, 5.0) if affin else cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if orb_matrix is not None:
                    #im_f = cv2.warpAffine(im_f, orb_matrix, (im_t.shape[1], im_t.shape[0])) if affin else cv2.warpPerspective(im_f, orb_matrix, (im_t.shape[1], im_t.shape[0]))
                    diff = get_absolute_differences(orb_matrix, pf, pt)
                    return str(orb_matrix.tolist()) + '\t' + str(diff)    
                else:
                    return str((np.identity(3)*3).tolist()) + '\t' + '-1'
        except:
            return str((np.identity(3)*2).tolist()) + '\t' + '-1'
    return str(np.identity(3).tolist()) + '\t' + '-1'

def load_kps(kppath='F:/Uni/Bachelorarbeit/Programs/kp_coords_v2'):
    return {file[:-5] : json.load(open(root + '/' + file, 'r')) for root, dir, filename in os.walk(kppath) for file in filename}

def load_all_images(impath = 'F:/Uni/Bachelorarbeit/Programs/gettransforms/images'):
    return {file.replace('_', ' ') : cv2.imread(impath + '/' + file) for (dirpath, dirname, filename) in os.walk(impath) for file in filename}

imgs = load_all_images()
kps = load_kps()
config = json.load(open('F:/Uni/Bachelorarbeit/Trans_imgs/config.json'))
t_list = json.load(open('F:/Uni/Bachelorarbeit/Programs/t_list.json'))
file = open('F:/Uni/Bachelorarbeit/Programs/erg.txt','w')

for im1 in tqdm(t_list.items()):
    for im2 in tqdm(im1[1], leave=False):
        src, dst = get_src_dst(kps[im1[0]], kps[im2])
        file.write(im1[0] + '\t' + im2 + '\t' + sift_calc(imgs[im1[0]], imgs[im2], src, dst, config['sift']) + '\t')
        file.write(sift_calc(imgs[im1[0]], imgs[im2], src, dst, config['sift'], affin=False) + '\t')
        file.write(orb_calc(imgs[im1[0]], imgs[im2], src, dst, config['orb']) + '\t')
        file.write(orb_calc(imgs[im1[0]], imgs[im2], src, dst, config['orb'], affin=False) + '\n')

file.close()