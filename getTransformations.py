
import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np
import uuid
import random
from tqdm import tqdm
import multiprocessing as mp

class transform:

    def __init__(self, f: str, t: str, m: str) -> None:
        self.f = f
        self.t = t
        self.m = m

    def __str__(self) -> str:
        return self.f + '#' + self.t + '#' + self.m + '\n'

    def __eq__(self, o: object) -> bool:
        return type(o) == type(self) and ((o.f == self.f and o.t == self.t) or (o.t == self.f and o.f == self.t))

class image:

    def __init__(self, im: np.ndarray, name: str) -> None:
        self.im = im
        self.kp = np.zeros((10,2))
        self.name = name

    def __eq__(self, o: object) -> bool:
        return type(o) == type(self) and self.name.split('_')[0] == o.getName().split('_')[0]

    def addKp(self, kp: np.ndarray, key: int) -> bool:
        if kp.shape == (2,):
            self.kp[key] = kp
            return True
        else:
            return False

    def getName(self) -> str:
        return self.name

    def getKp(self) -> np.ndarray:
        return self.kp

    def getIm(self) -> np.ndarray:
        return self.im

    def getNonzeros(self) -> int:
        return self.nonzeros

class patient:

    def __init__(self, im: image) -> None:
        self.imgs = []
        self.addIm(im)

    def __eq__(self, o: object) -> bool:
        return o == self.name or type(o) == type(self) and o.getName() == self.name

    def __getitem__(self, key) -> image:
        return self.imgs[key]

    def addImgs(self, imgs: list) -> None:
        self.imgs = imgs

    def addIm(self, im: image) -> None:
        self.imgs.append(im)

    def getImgs(self) -> list:
        return self.imgs


def calc_patient(p: patient) -> list:
    matrix = []
    for i in p.getImgs():
        for j in p.getImgs():  
            if i.getName() == j.getName():
                continue

            #use of manual matches and labeled keypoints, hence no further calculation 

            source = i.getKp()
            destin = j.getKp()
            t = 0

            for i2,j2 in zip(source, destin):
                if np.linalg.norm(i2) == 0 or np.linalg.norm(j2) == 0:
                    source = np.delete(source, t, axis=0)
                    destin = np.delete(destin, t, axis=0)
                    t -= 1
                t += 1

            im_f = i.getIm()
            im_t = j.getIm()
            keypoints = np.zeros((im_f.shape[0], im_f.shape[1] + im_t.shape[1])).astype(np.uint8)
            keypoints[:,:im_f.shape[1]] = im_f
            keypoints[:,im_f.shape[1]:] = im_t
            keypoints = cv2.cvtColor(keypoints, cv2.COLOR_GRAY2BGR)
            for i2, j2 in zip(source, destin):
                randcolor = (random.randint(50,200),random.randint(50,200),random.randint(50,200))
                cv2.circle(keypoints, (int(i2[0]), int(i2[1])), 10, randcolor, 2)
                cv2.circle(keypoints, (int(j2[0]) + im_f.shape[1], int(j2[1])), 10, randcolor, 2)
                cv2.line(keypoints, (int(i2[0]), int(i2[1])), (int(j2[0]) + im_f.shape[1], int(j2[1])), randcolor, 2)
            #keypoints = cv2.resize(keypoints, (1800,900))
            cv2.imwrite('F:/Uni/Bachelorarbeit/Programs/matched/' + i.getName() + '#' + j.getName() + '.png', keypoints)

            M, _ = cv2.estimateAffine2D(source, destin)
            
            a = uuid.uuid4()
            matrix.append(str(transform(i.getName(), j.getName(), str(a))))
            np.save(open('F:/Uni/Bachelorarbeit/Programs/numpy_matrices/' + str(a) + '.npy','wb'), M)
            #print('Finished Image calc ' + i.getName() + ' / ' + j.getName())
    return matrix

def get_results(result):
    results.append(result)
    pbar.update()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # Loading resources
    results = []
    root = ET.parse('F:/Uni/Bachelorarbeit/Programs/gettransforms/annotations.xml').getroot()
    files = [filenames for (dirpath, dirname, filenames) in os.walk('F:/Uni/Bachelorarbeit/Programs/gettransforms/images')]
    imgs = {file : cv2.imread('F:/Uni/Bachelorarbeit/Programs/gettransforms/images/' + file, 0) for file in files[0]}
    patients = {}
    w_points = open('keypoints.txt','w')
    w_points.close()

    for img in root.findall('image'):
        tmp_im = image(imgs[img.attrib['name']], img.attrib['name'])
        for s in img.findall('points'):
            x, y = s.attrib['points'].split(',')
            tmp_im.addKp(np.array((float(x), float(y))), int(s.attrib['label']) - 1)
        n = uuid.uuid4()
        w_points = open('keypoints.txt','a')
        w_points.write(img.attrib['name'] + '#' + str(n) + '\n')
        w_points.close()
        np.save(open('F:/Uni/Bachelorarbeit/Programs/kp_coords/' + str(n) + '.npy','wb'), tmp_im.kp)
        #kps = cv2.cvtColor(tmp_im.im, cv2.COLOR_GRAY2BGR)
        #for i in tmp_im.kp:
        #    cv2.circle(kps, (int(i[0]), int(i[1])), 10, (0,0,255), 2)
        #    cv2.putText(kps, str(i[0]) + ',' + str(i[1]), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0,0,255))
        #    cv2.imwrite('E:/Uni/Bachelorarbeit/Programs/kp_coords/' + tmp_im.getName() + '_kp.png', kps)
        if np.count_nonzero(tmp_im.kp) < 10:
            continue
        if len(patients) != 0 and img.attrib['name'].split('_')[0] in patients:
            patients[img.attrib['name'].split('_')[0]].addIm(tmp_im)
        else:
            patients[img.attrib['name'].split('_')[0]] = patient(tmp_im)
            
    print('Loading done')
    #
    pbar = tqdm(len(list(patients.values())))
    with mp.get_context().Pool(len(patients.values()) if len(patients.values()) < mp.cpu_count() else mp.cpu_count()) as pool:
        for p in list(patients.values()):
           pool.apply_async(calc_patient, (p,), callback=get_results)
        pool.close()
        pool.join()
    file = open('matrices.txt','w')
    for i in results:
        for t in i:
            file.write(t)
    file.close()
