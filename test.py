import numpy as np
import cv2
from matplotlib import pyplot as plt
from service import Service


MIN_MATCH_COUNT = 10

source = cv2.imread('/Users/snake/Documents/images/result.jpg')
target = cv2.imread('/Users/snake/Documents/images/image1.jpg')

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(source, None)
kp2, des2 = sift.detectAndCompute(target, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k = 2)

good = []
for m, n in matches:
    if m.distance < n.distance * 0.5:
        good.append(m)

print len(good)

multiple = Service.get_distance(kp1[good[0].queryIdx].pt, kp1[good[1].queryIdx].pt) / \
           Service.get_distance(kp2[good[0].trainIdx].pt, kp2[good[1].trainIdx].pt)

print multiple

mat = cv2.imread('/Users/snake/Documents/images/result.jpg', cv2.IMREAD_GRAYSCALE)

def get_position(i, j, pt1, pt2, time):
    x = pt2[0] + (i - pt1[0]) * time
    y = pt2[1] + (j - pt1[1]) * time
    return int(x), int(y)

row = len(source)
col = len(source[0])

for i in xrange(row):
    for j in xrange(col):
        x, y = get_position(i, j, kp1[good[0].queryIdx].pt, kp1[good[1].queryIdx].pt, multiple)
        if x < len(target):
            if y < len(target[x]):
                target[x][y] = source[i][j]

dt = np.dtype('int8')
new = np.array(target, dtype=dt)
cv2.imwrite('/Users/snake/Documents/images/trans_image.jpg', new)
print len(mat)
print len(mat[0])
"""
# dst_img = ColorTransfer.transfer(dst_img, target)
cv2.imwrite('/Users/snake/Documents/images/middle_result.jpg',dst_img)
dst_img = cv2.resize(dst_img, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)

cv2.namedWindow('dst_image', cv2.WINDOW_NORMAL)
cv2.imshow('dst_image', dst_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""