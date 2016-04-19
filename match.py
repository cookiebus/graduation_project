import numpy as np
import cv2
from matplotlib import pyplot as plt

def match(img1, img2, MIN_MATCH_COUNT = 10):
    
    surf = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # bf  = cv2.BFMatcher()
    matches = flann.knnMatch(des1, des2, k=2)
    # matches = bf.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print "Good Matched Point:", len(good)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w, _ = img1.shape
        pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None
        return False, ''

    draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                       singlePointColor = None, 
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()
    dst_img = cv2.warpPerspective(img1, M, (h * 2, w))
    # dst_img = cv2.resize(dst_img, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('/Users/snake/Documents/images/result.jpg',dst_img)
    return True, dst_img


def query(query_image, train_image):
    print "querying %s %s" % (query_image, train_image)

    img1 = cv2.imread(query_image)  # queryImage
    img2 = cv2.imread(train_image, 0)  # trainImage
    is_matched, result = match(img1, img2, MIN_MATCH_COUNT = 0)
    if is_matched:
        is_matched, result = match(result, img2, MIN_MATCH_COUNT = 50)
        if is_matched:
            return True, result

    return False, ''


if __name__ == "__main__":

    for i in xrange(1, 2):
        query_image = '/Users/snake/Documents/images/test2.jpg'
        train_image = '/Users/snake/Documents/images/image5.jpg'
        is_matched, result = query(query_image, train_image)
        if is_matched:
            print "image %s is matched." % i
    
