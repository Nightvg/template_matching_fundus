import matplotlib
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')

def legende(im):
  ef = np.zeros((im.shape[0], im.shape[1] + 300)).astype(float)
  ef[:,:im.shape[1]] = im
  for i in range(im.shape[0] - 1):
    color = int(i / im.shape[0] * 255)
    ef[ef.shape[0] - i - 1, (ef.shape[1] - 100):] = color
    if color % 50 == 0:
      cv2.putText(ef, str(color), (im.shape[0] + 100, ef.shape[1] - i), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 255, 255))
  return ef.astype(np.uint8)

def calc_img(im_f, im_t, M, f, t, METHOD=1):

  f1 = im_f[:,:,1]
  im_f = cv2.warpAffine(im_f, M, (im_t.shape[1], im_t.shape[0]))

  f2 = im_f[:,:,1]
  t2 = im_t[:,:,1]
  mask_t = cv2.imread('F:/Uni/Bachelorarbeit/Programs/mask2.png', 0)
  mask_f = cv2.warpAffine(mask_t, M, (im_t.shape[1], im_t.shape[0]))
  mask_combine = mask_t & mask_f
  cv2.imwrite('E:/test.png', mask_combine)
  f2[mask_f != 255] = 255
  t2[mask_t != 255] = 255
  f1[mask_t != 255] = 255

  if METHOD != 1:
    f2 = cv2.equalizeHist(im_f[:,:,1]).astype(int)
    t2 = cv2.equalizeHist(im_t[:,:,1]).astype(int)
  if METHOD == 1:
    f2 = ((f2 - f2[mask_f == 255].min()) / f2[mask_f == 255].max()) * 255
    t2 = ((t2 - t2[mask_t == 255].min()) / t2[mask_t == 255].max()) * 255
    f1 = ((f1 - f1[mask_t == 255].min()) / f1[mask_t == 255].max()) * 255
  

  erg = abs(f2 - t2).astype(np.uint8)
  erg[mask_combine != 255] = 0
  # erg = legende(erg)
  
  #erg = cv2.resize(erg, (1200,1200))
  overlay = np.zeros(im_f.shape).astype(np.uint8)
  overlay[:,:,0] = 0
  overlay[:,:,1] = f2.astype(np.uint8)
  overlay[:,:,2] = t2.astype(np.uint8)
  overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
  #overlay[mask_combine != 255] = 0
  
  #f2 = cv2.resize(f2, (1200,1200))
  #t2 = cv2.resize(t2, (1200,1200))
  fig, ax = plt.subplots(1)
  tmp = ax.imshow(erg, vmin=0, vmax=255)
  tmp.set_cmap('jet')
  ax.axis('off')
  plt.colorbar(tmp)
  fig.savefig('F:/Uni/Bachelorarbeit/Programs/testblend_v3/' + f + '#' + t + '_HEAT.png', dpi=fig.dpi)
  plt.close('all')
  fig2, axes = plt.subplots(1, 3)
  #fig2.tight_layout()
  for axs,im,c in zip(axes, [f1,t2,overlay],['gray', 'gray', None]):
    axs.imshow(im, cmap=c)
    axs.axis('off')
  plt.subplots_adjust(wspace=0, hspace=0)
  fig2.savefig('F:/Uni/Bachelorarbeit/Programs/testblend_v3/' + f + '#' + t + '_OVERLAY.png', dpi=fig2.dpi)
  plt.figure().clear()
  plt.close()
  plt.cla()
  plt.clf()

  # cv2.imwrite('E:/Uni/Bachelorarbeit/Programs/testblend/' + f + '#' + t + '_HEAT.png', erg)
  # cv2.imwrite('E:/Uni/Bachelorarbeit/Programs/testblend/' + f + '#' + t + '_OVERLAY.png', overlay)

if __name__ == '__main__':
  s = open('matrices.txt','r').read().split('\n')
  s.pop(-1)
  for i in tqdm(s):
      
      f, t, M = i.split('#')
      im_f = cv2.imread('F:/Uni/Bachelorarbeit/Programs/gettransforms/images/' + f)
      im_t = cv2.imread('F:/Uni/Bachelorarbeit/Programs/gettransforms/images/' + t)
      matrix = np.load(open('F:/Uni/Bachelorarbeit/Programs/numpy_matrices/' + str(M) + '.npy','rb'), allow_pickle=True)
      
      calc_img(im_f, im_t, matrix, f, t)