import os
import numpy as np
import math


def read_symphony_scan_np(path):
    file_list = os.listdir(path)
    file_list.sort()

    file_path = path + "/" + file_list[0]
    lidar = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # len(file_list)
    for i in range(1, len(file_list)):
        file_path = path + "/" + file_list[i]
        new = np.loadtxt(file_path, delimiter=',', dtype=np.float64)
        lidar = np.vstack([lidar, new])
        print(file_list[i], "loaded")

    lidar = lidar[:, 1:]

    length = lidar.shape[0]
    ran = list(range(0, length, 5))
    lidar = lidar[ran, :]

    return lidar


def polar_to_xy(dat):
    """
    Switches polar coordinates lidar data to LOCAL cartesian coordinates data
    :param dat: np lidar data 1 X 541
    :return: mx2 np array, m: valid points
    """
    x = []
    y = []

    for j in range(541):
        distance = dat[j]
        if 0.5 < distance <= 20:  # filtering invalid(untrustworthy) lidar data
            angle = - 1 / 4 * math.pi + j / 360 * math.pi
            x.append(distance * math.cos(angle))
            y.append(distance * math.sin(angle))

    local = np.array([x, y])

    return local.T


def local_to_global(pose, local):
    """
    Switching LOCAL coordinates to GLOBAL coordinates
    :param pose: pose of vehicle in global coordinates
    :param local: LOCAL coordinates to be switched, mX2
    :return: GLOBAL coordinates of 'local', mX2
    """
    x, y, trans = pose[0], pose[1], pose[2]

    tr = np.array([x, y]).reshape(2, 1)
    local_t =local.T
    new = trans @ local + tr

    return new.T
