# fastmatch from 2D LaserSLAM

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM


def matching(curr_scan, prev_scan, initial, resolution, idx):
    """
    Finding ideal transformation matrix by drifting small distances, angle
    :param curr_scan: current index scan data, Nxm (Nx2)
    :param prev_scan: previous index scan data, Nxm (Nx2)
    :param initial: initial transformation matrix by gps data 3x3
    :param resolution: drifting distance and angle array sized [dx, dy, d_theta]
    :param idx: scan index
    :return: ideal transformation matrix data 3x3
    """

    # filtering low feature scans
    # if len(curr_scan) < 50 or len(prev_scan) < 50:
    #     return initial

    dx = resolution[0]  # drifting distance toward x axis
    dy = resolution[1]  # drifting distance toward y axis
    dt = resolution[2]  # drifting angle

    # dimension expansion
    m = 2
    src = np.zeros((m + 1, curr_scan.shape[0]))
    dst = np.zeros((m + 1, prev_scan.shape[0]))
    src[:m, :] = np.copy(curr_scan.T)  # 3xN
    dst[:m, :] = np.copy(prev_scan.T)  # 3xN

    # Go down the hill
    maxIter = 100
    maxDepth = 7
    iter = 0
    depth = 0
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst.T)

    initial_trans = initial @ src

    initial_error = evaluate(neigh, initial_trans.T)
    best_error = initial_error
    S = initial @ src
    best_S, best_T = S, initial

    while iter < maxIter and depth <= maxDepth:
        no_Change = True

        # Rotation
        for theta in [-dt, 0, dt]:
            T = np.eye(3)
            ct = np.cos(theta)
            st = np.sin(theta)
            T[:2, :2] = np.array([[ct, -st], [st, ct]])  # 3x3

            # Translation
            for tx in [-dx, 0, dx]:
                T[0, 2] = tx
                for ty in [-dy, 0, dy]:
                    T[1, 2] = ty

                    S = T @ src  # 3xN

                    error = evaluate(neigh, S.T)

                    # update
                    if error < best_error:
                        no_Change = False
                        best_T = T + np.array([[0, 0, tx], [0, 0, ty], [0, 0, 0]])
                        best_error = error
                        best_S = S

        if no_Change:  # iteration 다 시도했는데 변화 없을 경우 -> resolution 높여서 다시 시도
            dt /= 2
            dx /= 2
            dy /= 2
            depth += 1

        # iteration
        iter += 1

    print(best_error)

    if idx != "skip":
        plt.clf()
        plt.scatter(dst[0], dst[1], color='blue', label='dst')
        plt.scatter(best_S[0], best_S[1], color='red', label='est', marker="x")
        # plt.scatter(src[0], src[1], color='green', label='src', marker=".")
        plt.axis('equal')
        plt.legend()
        plt.title(str(best_error))
        loc = "./pm_result/" + str(idx) + ".png"
        plt.savefig(loc)

    return best_T


def evaluate(neigh, source):
    """
    Calculates mean error between source points and target points
    ** Error is a bit bigger if the number of target points is smaller
    :param neigh: scikit learn NearestNeighbor object
    :param source: source points, Nxm
    :return: mean error
    """
    distances, _ = neigh.kneighbors(source, return_distance=True)
    error = distances.ravel()
    return np.mean(error)
