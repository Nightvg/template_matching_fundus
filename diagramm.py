import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import cv2

###Affine Errorbars###
# elements = open('statistics.txt','r').read().split('\n')
# elements.pop(-1)
# elements.pop(0)
# min = 4.8
# max = 5.2
# stats = []
# for a in elements:
#     tmp = a.split('\t')
#     tmp.pop(-1)
#     stats.append([float(tmp[0]), float(tmp[1])])

# classes = []
# for i in range(12):
#     classes += [[]]
#     for j in stats:
#         if j[1] >= min + 0.4 * i and j[1] < max + 0.4 * i:
#             classes[i] += [j[0]]

# xpoints = [5 + i * 0.4 for i in range(12)]
# ypoints = [np.mean(a) for a in classes]
# size = [len(a) for a in classes]
# errorbars = [np.std(a) for a in classes]
# f, (ax1, ax2) = plt.subplots(2)

# ax1.errorbar(xpoints, ypoints, yerr=errorbars, fmt='s')
# #ax1.set_xlim([4,21])
# #ax1.set_ylim([4,21])
# ax2.scatter(xpoints, size)
# plt.show()


###Cat Transform###
# im = cv2.imread('F:/Uni/Bachelorarbeit/PImages/cat.png')
# pts1 = np.float32([[50, 50],
#                    [200, 50], 
#                    [50, 200]])
  
# pts2 = np.float32([[10, 100],
#                    [200, 50], 
#                    [100, 250]])
  
# M = cv2.getAffineTransform(pts1, pts2)
# dst = cv2.warpAffine(im, np.dot(M,[[.62,0,-170],[0,.62,490],[0,0,1]]), (im.shape[1], im.shape[0]))
# cv2.imwrite('F:/Uni/Bachelorarbeit/PImages/cat_transform.png', dst)

###Abs Correlation###
ti = pd.read_csv('transformations.csv', delimiter='\t', decimal=',')
i = pd.read_csv('imagedata.csv', delimiter='\t', decimal='.')

ti.drop(['Matrix1','Matrix2'], axis=1, inplace=True)

ti['diff Med R'] = [abs(i.loc[i['Name'] == x].iloc[0]['MED_R'] - i.loc[i['Name'] == y].iloc[0]['MED_R']) for x,y in zip(ti['From'], ti['To'])]
ti['diff Med G'] = [abs(i.loc[i['Name'] == x].iloc[0]['MED_G'] - i.loc[i['Name'] == y].iloc[0]['MED_G']) for x,y in zip(ti['From'], ti['To'])]
ti['diff Med B'] = [abs(i.loc[i['Name'] == x].iloc[0]['MED_B'] - i.loc[i['Name'] == y].iloc[0]['MED_B']) for x,y in zip(ti['From'], ti['To'])]
ti['diff Std R'] = [abs(i.loc[i['Name'] == x].iloc[0]['STD_R'] - i.loc[i['Name'] == y].iloc[0]['STD_R']) for x,y in zip(ti['From'], ti['To'])]
ti['diff Std G'] = [abs(i.loc[i['Name'] == x].iloc[0]['STD_G'] - i.loc[i['Name'] == y].iloc[0]['MED_G']) for x,y in zip(ti['From'], ti['To'])]
ti['diff Std B'] = [abs(i.loc[i['Name'] == x].iloc[0]['STD_B'] - i.loc[i['Name'] == y].iloc[0]['MED_G']) for x,y in zip(ti['From'], ti['To'])]
ti['diff Mean R'] = [abs(i.loc[i['Name'] == x].iloc[0]['MW_R'] - i.loc[i['Name'] == y].iloc[0]['MW_R']) for x,y in zip(ti['From'], ti['To'])]
ti['diff Mean G'] = [abs(i.loc[i['Name'] == x].iloc[0]['MW_G'] - i.loc[i['Name'] == y].iloc[0]['MW_G']) for x,y in zip(ti['From'], ti['To'])]
ti['diff Mean B'] = [abs(i.loc[i['Name'] == x].iloc[0]['MW_B'] - i.loc[i['Name'] == y].iloc[0]['MW_B']) for x,y in zip(ti['From'], ti['To'])]
ti['diff Sharpness'] = [abs(i.loc[i['Name'] == x].iloc[0]['Schärfe'] - i.loc[i['Name'] == y].iloc[0]['Schärfe']) for x,y in zip(ti['From'], ti['To'])]
ti['diff SNR'] = [abs(i.loc[i['Name'] == x].iloc[0]['SNR'] - i.loc[i['Name'] == y].iloc[0]['SNR']) for x,y in zip(ti['From'], ti['To'])]

ti.drop(['From', 'To'], axis=1, inplace=True)
corr = ti.corr().head(2)


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='seismic', vmin=-1, vmax=1)
fig.colorbar(cax)
ax.set_xticks(np.arange(0,len(ti.columns)))
plt.xticks(rotation=90)
ax.set_yticks(np.arange(2))
ax.set_xticklabels(['SIFT MDP','ORB MDP',
                    'Differenz Median Rot', 'Differenz Median Grün', 'Differenz Median Blau',
                    'Differenz STD Rot', 'Differenz STD Grün', 'Differenz STD Blau',
                    'Differenz Mittelwert Rot', 'Differenz Mittelwert Grün', 'Differenz Mittelwert Blau',
                    'Differenz Schärfe', 'Differenz SNR'])
ax.set_yticklabels(['SIFT MDP','ORB MDP'])
plt.show()

