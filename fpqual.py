import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle
import numpy as np
import csv
import json
from tqdm import tqdm

def toHomogeneous(points: np.ndarray):
    return np.column_stack((points, np.ones((points.shape[0],)).astype(float)))

def transform(points: np.ndarray, matrix: np.ndarray):
    return np.dot(matrix, points.T).T 
    
file = list(csv.reader(open('transformations.csv', newline=''), delimiter='\t'))
fps = json.load(open('fpcoords.json'))

file.pop(0)
for i in tqdm(file):
    f,t,smdp,omdp = i[0],i[1],i[3],i[5]
    sift_matrix, orb_matrix = np.array(json.loads(i[2])), np.array(json.loads(i[4]))
    fsift, tsift, forb, torb = transform(toHomogeneous(np.array(fps[f]['sift'])), sift_matrix.astype(float)), fps[t]['sift'], transform(toHomogeneous(np.array(fps[f]['orb'])), orb_matrix.astype(float)), fps[t]['orb']
    fig, (axs, axo) = plt.subplots(1, 2)
    #figmng = plt.get_current_fig_manager()
    #figmng.window.state('zoomed')
    b1 = Circle((3816//2,3744//2),3816//2+110, fill=False, color='black')
    b2 = Circle((3816//2,3744//2),3816//2+110, fill=False, color='black')
    axs.set_aspect('equal')
    axo.set_aspect('equal')
    axs.title.set_text('SIFT')
    axo.title.set_text('ORB')
    axs.axis('off')
    axo.axis('off')
    axs.scatter([x[0] for x in fsift], [y[1] for y in fsift], color='red', marker='.', edgecolors='none', alpha=0.5)
    axo.scatter([x[0] for x in forb], [y[1] for y in forb], color='red', marker='.', edgecolors='none', alpha=0.5)
    axs.add_patch(b1)
    axo.add_patch(b2)
    #fig.canvas.draw()
    #w, h = fig.canvas.get_width_height()
    #reds = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, -1).copy()
    #axs.clear()
    #axo.clear()
    #axs.axis('off')
    #axo.axis('off')
    #axs.add_patch(b1)
    #axo.add_patch(b2)
    axs.scatter([x[0] for x in tsift], [y[1] for y in tsift], color='blue', marker='.', edgecolors='none', alpha=0.5)
    axo.scatter([x[0] for x in torb], [y[1] for y in torb], color='blue', marker='.', edgecolors='none', alpha=0.5)
    #fig.canvas.draw()
    #blues = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, -1).copy()
    #img = np.zeros((reds.shape[0], reds.shape[1], 3))
    #reds[reds[:, :, -1] == 0] = 0
    #blues[blues[:, :, -1] == 0] = 0
    #img = np.maximum(reds, blues)
    plt.savefig('F:/Uni/Bachelorarbeit/Programs/fps/' + f + '#' + t + '.png')
    plt.close()
    #Image.fromarray(img).save('F:/Uni/Bachelorarbeit/Programs/fps/' + f + '#' + t + '.png')
