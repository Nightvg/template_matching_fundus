import numpy as np
import cv2
from tqdm import tqdm
import json
import xml.etree.ElementTree as ET

def legende(im):
  ef = np.zeros((im.shape[0], im.shape[1] + 300)).astype(float)
  ef[:,:im.shape[1]] = im
  for i in range(im.shape[0] - 1):
    color = int(i / im.shape[0] * 255)
    ef[ef.shape[0] - i - 1, (ef.shape[1] - 100):] = color
    if color % 50 == 0:
      cv2.putText(ef, str(color), (im.shape[0] + 100, ef.shape[1] - i), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 255, 255))
  return ef.astype(np.uint8)

def abs_diff(pf : np.ndarray, pt : np.ndarray, matrix: np.ndarray):
    tmp_pf = np.ones((pf.shape[0], 3))
    tmp_pf[:,:2] = pf
    tmp_pf = np.dot(matrix, tmp_pf.T)
    pf = tmp_pf.T[:,:2]
    result = 0
    count = 0
    for x,y in zip(pf,pt):
        if np.linalg.norm(x) == np.linalg.norm(matrix[:,2]) or np.linalg.norm(y) == 0:
            continue
        result += np.linalg.norm(x - y)
        count += 1
    return result / count

def manual_calc(pf, pt, M):
    matrix = np.load(open('F:/Uni/Bachelorarbeit/Programs/numpy_matrices/' + str(M) + '.npy','rb'), allow_pickle=True)
    diff = abs_diff(pf, pt, matrix)
    return {'matrix': matrix, 'diff': diff}

def sift_calc(im_f, im_t, pf, pt, args):
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
                #sift_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, None, cv2.RANSAC, 5.0)
                sift_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if sift_matrix is not None:
                    #im_f = cv2.warpAffine(im_f, sift_matrix, (im_t.shape[1], im_t.shape[0]))
                    im_f = cv2.warpPerspective(im_f, sift_matrix, (im_t.shape[1], im_t.shape[0]))
                    diff = abs_diff(pf, pt, sift_matrix)
                    return {'matrix' : sift_matrix, 'diff' : diff}
                else:
                    return {'matrix' : np.array([1]), 'diff': []} 
        except:
            return {'matrix' : np.array([0]), 'diff': []} 
    return {'matrix' : np.zeros((3, 3)), 'diff' : []}

def orb_calc(im_f, im_t, pf, pt, args):
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
                #orb_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, None, cv2.RANSAC, 5.0)
                orb_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if orb_matrix is not None:
                    #im_f = cv2.warpAffine(im_f, orb_matrix, (im_t.shape[1], im_t.shape[0]))
                    im_f = cv2.warpPerspective(im_f, orb_matrix, (im_t.shape[1], im_t.shape[0]))
                    diff = abs_diff(pf, pt, orb_matrix)
                else:
                    return {'matrix' : np.array([1]), 'diff': []} 
                return {'matrix' : orb_matrix, 'diff' : diff}    
        except:
            return {'matrix' : np.array([0]), 'diff': []}    
    return {'matrix' : np.zeros((3, 3)), 'diff' : []}

def everything(im_f, im_t, M, pf, pt, fr, to, args):
    #Image preparation for algorithmic matching
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im_f_gray = im_f[:,:,1]
    im_t_gray = im_t[:,:,1]
    args['sift']['dname'] = fr + '#' + to
    args['orb']['dname'] = fr + '#' + to
    try:
        #calc = {'manual': manual_calc(pf, pt, M), 'sift': sift_calc(im_f_gray, im_t_gray, pf, pt, args['sift']), 'orb': orb_calc(im_f_gray, im_t_gray, pf, pt, args['orb']), 'from': fr, 'to': to}
        calc = {'sift': sift_calc(im_f_gray, im_t_gray, pf, pt, args['sift']), 'orb': orb_calc(im_f_gray, im_t_gray, pf, pt, args['orb']), 'from': fr, 'to': to}
    except:
        print('something strange happend')
    return calc

results = []
s = open('matrices.txt','r').read().split('\n')
kps = open('keypoints.txt', 'r').read().split('\n')
config = json.load(open('F:/Uni/Bachelorarbeit/Trans_imgs/config.json'))

kps.pop(-1)
s.pop(-1)
mps = dict([[x.split('#')[0], np.load(open('F:/Uni/Bachelorarbeit/Programs/kp_coords/' + str(x.split('#')[1]) + '.npy','rb'), allow_pickle=True)] for x in kps])
# with open('keypoints.json','w') as tmp:
#     json.dump(mps, tmp)

for elem in tqdm(s):      
    f, t, M = elem.split('#')
    im_f = cv2.imread('F:/Uni/Bachelorarbeit/Programs/gettransforms/images/' + f)
    im_t = cv2.imread('F:/Uni/Bachelorarbeit/Programs/gettransforms/images/' + t)
    results.append(everything(im_f, im_t, M, mps[f], mps[t], f, t, config))

file = open('infos_perspective.txt','w')
json.dump(config, file)
file.write('\n')
for r in results:
    file.write(r['from'] + '\t' + r['to'] + '\t')
    #file.write('manual#' + str(r['manual']['matrix'].tolist()) + '#' + str(r['manual']['diff']) + '\n')
    file.write(str(r['sift']['matrix'].tolist()) + '\t' + str(r['sift']['diff']) + '\t')
    file.write(str(r['orb']['matrix'].tolist()) + '\t' + str(r['orb']['diff']) + '\n')
file.close()