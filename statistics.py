import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm
import os
from matplotlib.ticker import FormatStrFormatter

def affTransform(pts, M):
    src = np.ones((len(pts), 1, 2))
    src[:, 0] = pts
    dst = cv2.transform(src, M)
    return np.array(dst[:, 0, :], dtype='float32') 

def plots_for_points(points):

    source = None
    res = None
    if os.path.isfile('statistics'+str(v)+'.json'):
        f = json.load(open('statistics'+str(v)+'.json'))
        source = np.array(f['points'])
        res = np.array(f['values'])
    else:
        source = np.array([[np.random.uniform(0.0, 3816.0), np.random.uniform(0.0, 3744.0)] for i in range(points)])

        x_std = 4.321847598732146
        y_std = 4.225069361692351

        ref_matrix = np.array([[np.cos(np.pi/12.0), -np.sin(np.pi/12.0), 150.0],[np.sin(np.pi/12.0), np.cos(np.pi/12.0), 100.0]]) #Referenzmatrix -> Ration um 15° und Translation um den Vektor (150,100)

        destin = affTransform(source, ref_matrix)
        res = np.array([])

        for i in tqdm.tqdm(range(20000)):
            tmp_destin = []
            for a in destin:
                rx, ry = np.clip(np.random.normal(a[0], x_std), 0.0, 3816.0), np.clip(np.random.normal(a[1], y_std), 0.0, 3744.0)
                tmp_destin += [[rx, ry]]
            t_matrix, _ = cv2.estimateAffine2D(np.array(source), np.array(tmp_destin), None, cv2.RANSAC, 5.0)
            res = np.array([abs(t_matrix - ref_matrix)]).reshape(6) if len(res) == 0 else np.vstack((res, np.array([abs(t_matrix - ref_matrix)]).reshape(6)))

        with open('statistics'+str(points)+'.json', 'w') as f:
            json.dump({'values' : res.tolist(), 'points' : source.tolist()}, f)

    resL = np.hstack((res[:,:2], res[:,3:5]))
    resR = np.column_stack((res[:,2], res[:,5]))
    return resL, resR

i = 1
for v,color in zip([7,10,20],['#ffffff','#eeeeee', '#aaaaaa']):
    resL, resR = plots_for_points(v)
    ax1 = plt.subplot(3, 2, i)
    box1 = ax1.boxplot(resL, showfliers=False, patch_artist=True, boxprops=dict(facecolor=color))
    #plt.setp(box1['boxes'],color=color)
    ax1.set_xticks([1,2,3,4],['a','b','d','e'])
    ax1.yaxis.tick_right()
    ax1.set_ylabel('Absoluter Wert')
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    i += 1
    ax2 = plt.subplot(3, 2, i)
    box2 = ax2.boxplot(resR, showfliers=False, patch_artist=True, boxprops=dict(facecolor=color))
    #plt.setp(box2['boxes'],color=color)
    ax2.set_xticks([1,2],['c','f'])
    ax2.set_ylabel('Absoluter Wert [Pixel]')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.tick_right()
    i += 1

plt.show()
        

