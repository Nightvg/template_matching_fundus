import matplotlib.pyplot as plt
import math
import numpy as np
import cv2

#Machband
# img = np.ones((200, 1000))

# img[:100] = [int(x / 1000 * 255) for x in range(1000)]
# img[100:] = [32 * math.floor(int(x / 1000 * 255) / 32) + 16 for x in range(1000)]

# x3 = [-((x % 125) - 62.5)**3 / 10000 + 32 * math.floor(int(x / 1000 * 255) / 32) for x in range(1000)]
# xact = [32 * math.floor(int(x / 1000 * 255) / 32) for x in range(1000)]
# for i in range(62):
#     x3[i] = 0
#     if i > 0:
#         x3[-i] = 255 - 31
# #x3[-62:] = [255]

# f, (ax1, ax2) = plt.subplots(2, sharex=True)
# ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
# ax1.axis('off')
# ax2.plot(x3, color='r')
# ax2.plot(xact, color='black')
# ax2.set_yticklabels([])
# ax2.set_xticklabels([])
# ax2.set_xlabel('Pixel')
# ax2.set_ylabel('Intensität')

# plt.show()

#RANSAC



