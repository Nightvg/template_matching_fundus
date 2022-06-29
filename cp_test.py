import numpy as np
import cv2
from tqdm import tqdm

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

def sift_calc(im_f, im_t, pf, pt, args):
    sift = cv2.SIFT_create(nfeatures=args['nf'], contrastThreshold=args['ct'], edgeThreshold=args['et'], sigma=args['si'])
    mask = cv2.imread('E:/Uni/Bachelorarbeit/Programs/mask2.png', 0)
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
                sift_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, None, cv2.RANSAC, 5.0)
                if sift_matrix is not None:
                    im_f = cv2.warpAffine(im_f, sift_matrix, (im_t.shape[1], im_t.shape[0]))
                    diff = abs_diff(pf, pt, sift_matrix)
                    return {'matrix' : sift_matrix, 'diff' : diff}
                else:
                    return {'matrix' : np.array([1]), 'diff': []} 
        except:
            return {'matrix' : np.array([0]), 'diff': []} 
    return {'matrix' : np.zeros((3, 3)), 'diff' : []}

def orb_calc(im_f, im_t, pf, pt, args):
    orb = cv2.ORB_create(nfeatures=args['nf'], firstLevel=args['fl'], edgeThreshold=args['ps'], patchSize=args['ps'], fastThreshold=args['ft'])
    mask = cv2.imread('E:/Uni/Bachelorarbeit/Programs/mask2.png', 0)
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
                orb_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, None, cv2.RANSAC, 5.0)
                if orb_matrix is not None:
                    im_f = cv2.warpAffine(im_f, orb_matrix, (im_t.shape[1], im_t.shape[0]))
                    diff = abs_diff(pf, pt, orb_matrix)
                else:
                    return {'matrix' : np.array([1]), 'diff': []} 
                return {'matrix' : orb_matrix, 'diff' : diff}    
        except:
            return {'matrix' : np.array([0]), 'diff': []}    
    return {'matrix' : np.zeros((3, 3)), 'diff' : []}

def calc(im_f, im_t, pf, pt, fr, to, args):
    return {'sift': sift_calc(im_f[:,:,1], im_t[:,:,1], pf, pt, args['sift']), 'orb': orb_calc(im_f[:,:,1], im_t[:,:,1], pf, pt, args['orb']), 'from': fr, 'to': to, 'args': args}

kps = open('keypoints.txt', 'r').read().split('\n')
kps.pop(-1)
print('Loading done\n')
rgb = "25_003 - normal.PNG"
f3 = "25_007 - Filter 3.PNG"
f5 = "25_010 - Filter 5.PNG"
im_rgb = cv2.imread('E:/Uni/Bachelorarbeit/Programs/gettransforms/images/' + rgb)
im_f3 = cv2.imread('E:/Uni/Bachelorarbeit/Programs/gettransforms/images/' + f3)
im_f5 = cv2.imread('E:/Uni/Bachelorarbeit/Programs/gettransforms/images/' + f5)


nf_l = [10000]
ct_l = [0.001, 0.005, 0.01]
et_l = [40]
si_l = [1.6, 1.8, 2.0, 2.2, 2.5]

fl_l = [0]
ps_l = [25,31,35]
ft_l = [0,2,4,6,8]

t_config = []

for nf in nf_l:
    for ct in ct_l:
        for et in et_l:
            for si in si_l:
                t_config.append({'sift':{'nf':nf,'ct':ct,'et':et,'si':si}})

i = 0
for nf in nf_l:
    for fl in fl_l:
        for ps in ps_l:
            for ft in ft_l:
                t_config[i]['orb'] = {'nf':nf,'fl':fl,'ps':ps,'ft':ft}
                i += 1

print('Config done\n')
mps = dict([[x.split('#')[0], np.load(open('E:/Uni/Bachelorarbeit/Programs/kp_coords/' + str(x.split('#')[1]) + '.npy','rb'), allow_pickle=True)] for x in kps])

f5_results = []
for elem in tqdm(t_config):
    f5_results.appen(calc(im_f5, im_rgb, mps[f5], mps[rgb], f5, rgb, elem))

f3_results = []
for elem in tqdm(t_config):
    f3_results.append(calc(im_f3, im_rgb, mps[f3], mps[rgb], f3, rgb, elem))

file = open('infos_example.txt','w')
for r in f5_results:
    file.write(r['from'] + '#' + r['to'] + '#' + str(r['args']) + '\n')
    file.write('sift#' + str(r['sift']['matrix'].tolist()) + '#' + str(r['sift']['diff']) + '\n')
    file.write('orb#' + str(r['orb']['matrix'].tolist()) + '#' + str(r['orb']['diff']) + '\n')
for r in f3_results:
    file.write(r['from'] + '#' + r['to'] + '#' + str(r['args']) + '\n')
    file.write('sift#' + str(r['sift']['matrix'].tolist()) + '#' + str(r['sift']['diff']) + '\n')
    file.write('orb#' + str(r['orb']['matrix'].tolist()) + '#' + str(r['orb']['diff']) + '\n')
file.close()