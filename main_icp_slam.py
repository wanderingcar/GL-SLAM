import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from minisam import *
from utils.ScanContextManager import *
from utils.PoseGraphManager import *
from utils.UtilsMisc import *
import utils.UtilsPointcloud as Ptutils
import utils.matching as Match

np.set_printoptions(precision=4)

# params
parser = argparse.ArgumentParser(description='PyICP SLAM arguments')

parser.add_argument('--num_icp_points', type=int, default=500)  # 5000 is enough for real time
parser.add_argument('--num_rings', type=int, default=20)  # same as the original paper
parser.add_argument('--num_sectors', type=int, default=180)  # same as the original paper
parser.add_argument('--num_candidates', type=int, default=10)  # must be int
parser.add_argument('--try_gap_loop_detection', type=int, default=20)  # same as the original paper
parser.add_argument('--loop_threshold', type=float,
                    default=0.02)  # 0.11 is usually safe (for avoiding false loop closure)
parser.add_argument('--sequence_idx', type=str, default='00')
parser.add_argument('--save_gap', type=int, default=300)

args = parser.parse_args()

resolution = [0.01, 0.01, np.deg2rad(1)]

data_path = "./Laser_Data/140106/laser_data"
data = Ptutils.read_symphony_scan_np(data_path)
num_frames = data.shape[0]
gps = load_gps()

# Pose Graph Manager (for back-end optimization) initialization
PGM = PoseGraphManager()
PGM.addPriorFactor()

# Result saver
save_dir = "result/" + args.sequence_idx
if not os.path.exists(save_dir): os.makedirs(save_dir)
ResultSaver = PoseGraphResultSaver(init_pose=PGM.curr_se2,
                                   save_gap=args.save_gap,
                                   num_frames=num_frames,
                                   seq_idx=args.sequence_idx,
                                   save_dir=save_dir)

# Scan Context Manager (for loop detection) initialization
SCM = ScanContextManager(shape=[args.num_rings, args.num_sectors],
                         num_candidates=args.num_candidates,
                         threshold=args.loop_threshold)

# for saving the results as a video
fig_idx = 1
fig = plt.figure(fig_idx)
num_frames_to_skip_to_show = 5
num_frames_to_save = np.floor(num_frames / num_frames_to_skip_to_show)

icp_initial = None

# @@@ MAIN @@@: data stream
for for_idx in tqdm(range(num_frames)):

    # get current information
    raw_data = data[for_idx]  # raw data: lidar data with polar coordinates
    curr_scan_pts = Ptutils.polar_to_xy(raw_data)  # switch raw data into LOCAL cartesian data, filters invalid data

    # save current node
    PGM.curr_node_idx = for_idx  # make start with 0
    SCM.addNode(node_idx=PGM.curr_node_idx, ptcloud=curr_scan_pts)

    # Initialize
    if PGM.curr_node_idx == 0:
        PGM.prev_node_idx = PGM.curr_node_idx
        prev_scan_pts = copy.deepcopy(curr_scan_pts)
        icp_initial = np.eye(3)
        continue

    # calculate odometry

    # odom_transform, _, _, wtfcount = ICP.icp(mat_curr, mat_prev, wtfcount, init_pose=icp_initial, max_iterations=10)
    odom_transform = Match.matching(prev_scan_pts, curr_scan_pts, icp_initial, resolution, str(for_idx))

    # update the current (moved) pose
    PGM.curr_se2 = np.matmul(PGM.curr_se2, odom_transform)
    # Initial: gps data
    if for_idx != num_frames - 1:
        icp_initial = odom_gps(gps[for_idx], gps[for_idx + 1])

    # add the odometry factor to the graph
    PGM.addOdometryFactor(odom_transform)

    # renewal the prev information
    PGM.prev_node_idx = PGM.curr_node_idx
    prev_scan_pts = copy.deepcopy(curr_scan_pts)

    # loop detection and optimize the graph
    if PGM.curr_node_idx > 1 and PGM.curr_node_idx % args.try_gap_loop_detection == 0:
        # 1/ loop detection
        loop_idx, loop_dist, theta_deg = SCM.detectLoop()
        if loop_idx is None:  # NOT FOUND
            pass
        else:
            print("Loop event detected: ", PGM.curr_node_idx, loop_idx, loop_dist)
            # 2-1/ add the loop factor
            loop_scan_pts = SCM.getPtcloud(loop_idx)

            # loop_transform, _, _ = ICP.icp(mat_curr, mat_loop, noo,
            #                                init_pose=theta_deg2se2(theta_deg), max_iterations=20)
            init_pose = theta_deg2se2(theta_deg)
            loop_transform = Match.matching(loop_scan_pts, curr_scan_pts, init_pose, resolution, "skip")

            PGM.addLoopFactor(loop_transform, loop_idx)

            # 2-2/ graph optimization
            PGM.optimizePoseGraph()

            # 2-2/ save optimized poses
            ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)

    # save the ICP odometry pose result (no loop closure)
    ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se2, PGM.curr_node_idx)
    if for_idx % num_frames_to_skip_to_show == 0:
        ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)
