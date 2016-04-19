import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import sqrt


class RELATED:
    DISJOINTNESS = 0
    INTERSECTION = 1
    INCLUSION = 2
    REVERSE_INCLUSION = 3


class Service:

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
                if len(max_block_temp) > 3:
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