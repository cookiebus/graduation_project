from images.models import Image
from images.constants import (
    RELATED,
    IMAGE_PATH_PREFIX,
    MIN_MATCH_COUNT
)
from matplotlib import pyplot as plt
from math import sqrt
from datetime import datetime
import cv2
import json
import numpy as np




class ImageService(object):

    @classmethod
    def get_max_match(cls, kp1, des1):
        images = Image.objects.all()
        aim = (0, None, None, None, None, '')

        for image in images:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            sift = cv2.xfeatures2d.SIFT_create()
            
            img2 = cv2.imread(IMAGE_PATH_PREFIX + image.image.url)
            kp2, des2 = sift.detectAndCompute(img2, None)

            matches2 = flann.knnMatch(des2, des1, k = 2)
            matches1 = flann.knnMatch(des1, des2, k = 2)
            d = {}
            for m, n in matches2:
                d[m.trainIdx] = m.queryIdx

            good = []
            for m, n in matches1:
                if m.queryIdx in d:
                    if d[m.queryIdx] == m.trainIdx:
                        good.append(m)
            good = Service.get_max_block(good, kp1, kp2)
            if len(good) > aim[0]:
                aim = (len(good), good[:], kp2[:], des2[:], img2, image.image.url)

        return aim

    @classmethod
    def get_target(cls, img_path):
        img1 = cv2.imread(img_path)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        
        _, good, kp2, des2, img2, image_url = cls.get_max_match(kp1, des1)

        print "Finish GET MAX MATCH"
        if _ == 0 or len(good) < 2:
            print "Not enough matches are found - %d/%d" % (_, MIN_MATCH_COUNT)
            return '', '', ''

        print "Good Matched Point:", len(good)
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w, _= img1.shape
        pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                           singlePointColor = None, 
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        dst_img = cv2.warpPerspective(img1, M, (h * 2, w))
        path = '/media/perspective/perspective_%s.jpg' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.imwrite(IMAGE_PATH_PREFIX + path, dst_img)
        # dst_img = cv2.resize(dst_img, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)

        source = dst_img
        target = img2
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
            if m.distance < n.distance * 0.7:
                good.append(m)
        print len(good)
        multiple = Service.get_distance(kp1[good[0].queryIdx].pt, kp1[good[1].queryIdx].pt) / \
                   Service.get_distance(kp2[good[0].trainIdx].pt, kp2[good[1].trainIdx].pt)

        print multiple


        row = len(source)
        col = len(source[0])

        for i in xrange(row):
            for j in xrange(col):
                x, y = get_position(i, j, kp1[good[0].queryIdx].pt, kp2[good[0].trainIdx].pt, multiple)
                if x < len(target):
                    if y < len(target[x]):
                        target[x][y] = source[i][j]

        # dt = np.dtype('int8')
        # new = np.array(target, dtype=dt)
        result_path = '/media/result/result_%s.jpg' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.imwrite(IMAGE_PATH_PREFIX + result_path, target)
        return image_url, path, result_path

    @classmethod
    def get_position(cls, i, j, pt1, pt2, time):
        x = pt2[0] + (i - pt1[0]) * time
        y = pt2[1] + (j - pt1[1]) * time
        return int(x), int(y)


class Service(object):

    edges = []
    node_num = 0
    color = []

    @classmethod
    def init(cls, n):
        for i in xrange(n):
            cls.edges.append([])
            cls.color.append(False)

        cls.node_num = n
        cls.edges_num = 0

    @classmethod
    def get_max_block(cls, matches, kp1, kp2):
        cls.init(len(matches))
        for u, match_a in enumerate(matches):
            for v, match_b in enumerate(matches):
                if v > u and cls.same_related(match_a, match_b, kp1, kp2):
                    cls.add_edge(u, v)

        max_blocks = []

        for i in xrange(cls.node_num):
            if not cls.color[i]:
                max_block_temp = []
                cls.dfs(i, max_block_temp)
                import copy
                if len(max_block_temp):
                    max_blocks.append(copy.copy(max_block_temp))

        needs = []
        print "number of blocks : ", len(max_blocks)
        for i in xrange(len(max_blocks)):
            needs.append(True)

        for i in xrange(len(max_blocks)):
            for j in xrange(i + 1, len(max_blocks)):
                if not cls.get_both(max_blocks[i], max_blocks[j], matches, kp1, kp2):
                    if len(max_blocks[i]) < len(max_blocks[j]):
                        needs[i] = False
                    else:
                        needs[j] = False

        print "Number of edges: ", cls.edges_num

        results = []
        total = 0
        for i, block in enumerate(max_blocks):
            if needs[i]:
                total += 1
                for index in block:
                    results.append(matches[index])
        print "Valid block : ", total
        print "Valid Matches : ", len(results)
        return results

    @classmethod
    def get_both(cls, block1, block2, matches, kp1, kp2):
        intersection_query = False
        for i in block1:
            for j in block2:
                if cls.get_related(matches[i].queryIdx, matches[j].queryIdx, kp1) != RELATED.DISJOINTNESS:
                    intersection_query = True
                    break
                if intersection_query:
                    break

        intersection_train = False
        for i in block1:
            for j in block2:
                if cls.get_related(matches[i].trainIdx, matches[j].trainIdx, kp2) != RELATED.DISJOINTNESS:
                    intersection_train = True
                    break
                if intersection_train:
                    break

        return intersection_train == intersection_query

    @classmethod
    def add_edge(cls, u, v):
        cls.edges_num += 1
        cls.edges[u].append(v)
        cls.edges[v].append(u)

    @classmethod
    def dfs(cls, x, max_block):
        cls.color[x] = True
        max_block.append(x)
        for y in cls.edges[x]:
            if not cls.color[y]:
                cls.dfs(y, max_block)

    @classmethod
    def same_related(cls, match_a, match_b, kp1, kp2):
        related_a = cls.get_related(match_a.queryIdx, match_b.queryIdx, kp1)
        related_b = cls.get_related(match_a.trainIdx, match_b.trainIdx, kp2)

        return related_a == related_b and related_a != RELATED.DISJOINTNESS


    @classmethod
    def get_related(cls, index1, index2, kp):
        o1, r1 = kp[index1].pt, kp[index1].size / 2 * 6
        o2, r2 = kp[index2].pt, kp[index2].size / 2 * 6
        distance = cls.get_distance(o1, o2)
        if distance > r1 + r2:
            return RELATED.DISJOINTNESS

        if r1 > r2 and r1 - r2 > distance:
            return RELATED.INCLUSION

        if r2 > r1 and r2 - r1 > distance:
            return RELATED.REVERSE_INCLUSION

        return RELATED.INTERSECTION

    @classmethod
    def get_distance(cls, o1, o2):
        return sqrt((o1[0] - o2[0]) ** 2 + (o1[1] - o2[1]) ** 2)
