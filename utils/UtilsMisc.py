import os
import csv
import copy
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import minisam


def getConstDigitsNumber(val, num_digits):
    return "{:.{}f}".format(val, num_digits)


def getUnixTime():
    return int(time.time())


def load_gps():
    """
    loading gps data from image_auxiliary.csv
    :return:
    """
    full_table = np.loadtxt("./image_auxiliary.csv", delimiter=',', skiprows=1, dtype=np.float64)

    tf = np.array(range(len(full_table)))
    tf_tf = tf % 206 != 0
    gps = full_table[tf_tf, 2:5]

    print("gps loaded")

    return gps


def odom_gps(gps_0, gps_1):
    """
    Calculates odometry from gps data
    :param gps_0: gps data at t
    :param gps_1: gps data at t+1
    :return: SE2 odometry
    """
    dx = gps_1[0] - gps_0[0]
    dy = gps_1[1] - gps_0[1]

    if gps_1[2] > 0 > gps_0[2]:
        before = gps_0[2] + 2 * np.pi
        after = gps_1[2]
        dt = after - before
    elif gps_1[2] < 0 < gps_0[2]:
        before = gps_0[2]
        after = gps_1[2] + 2 * np.pi
        dt = after - before
    else:
        dt = gps_1[2] - gps_0[2]

    ct = np.cos(dt)
    st = np.sin(dt)

    se2 = np.array([[ct, -st, dx],
                    [st, ct, dt],
                    [0, 0, 1]])

    return se2


def theta_deg2so2(theta):
    theta_rad = np.deg2rad(theta)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    so2 = np.array([[c, -s], [s, c]])
    return so2


def theta_deg2se2(theta):
    se2 = np.eye(3)
    se2[:2, :2] = theta_deg2so2(theta)
    return se2


def getGraphNodePose(graph, idx):
    pose = graph.at(minisam.key('x', idx))
    pose_trans = pose.translation()
    pose_rot = pose.so2().matrix()
    return pose_trans, pose_rot


def saveOptimizedGraphPose(curr_node_idx, graph_optimized, filename):
    for opt_idx in range(curr_node_idx):
        pose_trans, pose_rot = getGraphNodePose(graph_optimized, opt_idx)
        pose_trans = np.reshape(pose_trans, (-1, 2)).squeeze()
        pose_rot = np.reshape(pose_rot, (-1, 4)).squeeze()
        optimized_pose_ith = np.array([pose_rot[0], pose_rot[1], pose_trans[0],
                                       pose_rot[2], pose_rot[3], pose_trans[1],
                                       0.0, 0.0, 0.1])
        if (opt_idx == 0):
            optimized_pose_list = optimized_pose_ith
        else:
            optimized_pose_list = np.vstack((optimized_pose_list, optimized_pose_ith))

    np.savetxt(filename, optimized_pose_list, delimiter=",")


class PoseGraphResultSaver:
    def __init__(self, init_pose, save_gap, num_frames, seq_idx, save_dir):
        self.pose_list = np.reshape(init_pose, (-1, 9))
        self.save_gap = save_gap
        self.num_frames = num_frames

        self.seq_idx = seq_idx
        self.save_dir = save_dir

    def saveUnoptimizedPoseGraphResult(self, cur_pose, cur_node_idx):
        # save 
        self.pose_list = np.vstack((self.pose_list, np.reshape(cur_pose, (-1, 9))))

        # write
        if cur_node_idx % self.save_gap == 0 or cur_node_idx == self.num_frames:
            # save odometry-only poses
            filename = "pose" + self.seq_idx + "unoptimized_" + str(getUnixTime()) + ".csv"
            filename = os.path.join(self.save_dir, filename)
            np.savetxt(filename, self.pose_list, delimiter=",")

    def saveOptimizedPoseGraphResult(self, cur_node_idx, graph_optimized):
        filename = "pose" + self.seq_idx + "optimized_" + str(getUnixTime()) + ".csv"
        filename = os.path.join(self.save_dir, filename)
        saveOptimizedGraphPose(cur_node_idx, graph_optimized, filename)

        optimized_pose_list = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
        self.pose_list = optimized_pose_list  # update with optimized pose

    def vizCurrentTrajectory(self, fig_idx):
        x = self.pose_list[:, 2]
        y = self.pose_list[:, 5]

        fig = plt.figure(fig_idx)
        plt.clf()
        plt.plot(x, y, color='blue')
        plt.axis('equal')
        plt.xlabel('x', labelpad=10)
        plt.ylabel('y', labelpad=10)
        plt.draw()
        plt.pause(0.01)  # is necessary for the plot to update for some reason
