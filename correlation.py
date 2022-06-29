import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

img_infos = pd.read_csv('imagedata.csv', delimiter='\t', decimal='.')
trans_infos = pd.read_csv('transformations.csv', delimiter='\t', decimal=',')


trans_infos.drop(['Matrix1','Matrix2'], axis=1, inplace=True)
unison = pd.merge(trans_infos, img_infos, left_on='From', right_on='Name')
unison = pd.merge(unison, img_infos, left_on='To', right_on='Name')
unison.drop(columns=['Name_x','Name_y'], axis=1, inplace=True)
data = unison.drop(['From','To'], axis=1)

corr = data.corr()
mask = cv2.imread('corrmask.png', 0)
corr[mask == 0] = 0
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='seismic', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()
    