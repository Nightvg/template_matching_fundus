
import sys
import cv2
import numpy as np


def combine(self, img1, img2, img3):
    return False


img1 = cv2.imread('E:/Uni/Bachelorarbeit/Programs/brisque_test/25_012 - normal.PNG')
img2 = cv2.imread('E:/Uni/Bachelorarbeit/Programs/brisque_test/25_010 - Filter 5.PNG')

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img1_gray = clahe.apply(img1_gray)
img2_gray = clahe.apply(img2_gray)

try:
    sift = cv2.SIFT_create(nfeatures=10000)
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    img1_build = cv2.drawKeypoints(img1_gray, kp1, None, color=(0, 0, 255), flags=0)
    img2_build = cv2.drawKeypoints(img2_gray, kp2, None, color=(0, 0, 255), flags=0)
    cv2.imwrite('features1.png', img1_build)
    cv2.imwrite('features2.png', img2_build)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1_gray.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        cv2.imwrite('features_both.png', img3)

    else:
        print("Not enough matches are found - %i/%i") % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    sys.exit()
except AssertionError as error:
    print(error)
    sys.exit()
