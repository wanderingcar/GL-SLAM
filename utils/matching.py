# fastmatch from 2D LaserSLAM

import numpy as np
from math import cos, sin
from sklearn.neighbors import NearestNeighbors


def matching(curr_scan, prev_scan, initial, resolution):
    """
    Finding ideal transformation matrix by drifting small distances, angle
    :param curr_scan: current index scan data, Nxm (Nx2)
    :param prev_scan: previous index scan data, Nxm (Nx2)
    :param initial: initial transformation matrix by gps data 3x3
    :param resolution: drifting distance and angle array sized [dx, dy, d_theta]
    :return: ideal transformation matrix data 3x3
    """
    dx = resolution[0]  # drifting distance toward x axis
    dy = resolution[1]  # drifting distance toward y axis
    dt = resolution[2]  # drifting angle

    # dimension expansion
    m = 2
    src = np.ones((m + 1, curr_scan.shape[0]))
    dst = np.ones((m + 1, prev_scan.shape[0]))
    src[:m, :] = np.copy(curr_scan.T)  # 3xN
    dst[:m, :] = np.copy(prev_scan.T)  # 3xN

    # Go down the hill
    maxIter = 50
    maxDepth = 3
    iter = 0
    depth = 0

    while iter < maxIter and depth <= maxDepth:
        no_Change = True

        # Rotation
        for theta in [-dt, 0, dt]:
            ct = cos(theta)
            st = sin(theta)
            S = np.array([[ct, -st], [st, ct]]) @  # N*2

            # Translation
            for tx in [-dx, 0, dx]:
                Sx = np.around(S.T[0] + (tx - minX) * ipixel) + 1
                for ty in [-dy, 0, dy]:
                    Sy = np.around(S.T[1] + (ty - minY) * ipixel) + 1

                    i = np.where((1 < Sx) & (Sx < nCols) & (1 < Sy) & (Sy < nRows))
                    idx = np.around([Sy[i] - 1, Sx[i] - 1]).astype(np.int)
                    hits = (gridMap.metricMap[idx[0], idx[1]])
                    score = hits.sum()

                    # update
                    if score < bestScore:
                        no_Change = False
                        bestPose = [tx, ty, theta]
                        bestScore = score
                        bestHits = hits

        if no_Change:  # iteration 다 시도했는데 변화 없을 경우 -> resolution 높여서 다시 시도
            r = r / 2
            t = t / 2
            depth += 1

        # pose update, 반복
        pose = bestPose
        iter += 1

    return bestPose, bestHits


def evaluate():
