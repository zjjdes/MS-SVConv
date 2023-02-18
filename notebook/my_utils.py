from os.path import splitext
import math
import numpy as np
import open3d as o3d
import torch
from timeit import default_timer as timer

from torch_points3d.datasets.registration.pair import Pair

import teaserpp_python
from sklearn.neighbors import KDTree

"""
    Parameters
"""
# Calibration transformation matrix for KITTI lidar scans
# transforms a scan from the velodyne frame to the camera frame
T_CALIB_KITTI = np.array([[4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02],
       [-7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02],
       [9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01],
       [0, 0, 0, 1]]);

"""
    Point cloud I/O utils
"""
def remove_rover(pcd):
    """
    Remove the vehicle from the point cloud
    """
    coords = np.asarray(pcd.points)

    rover_width = 2 # x, 2
    rover_length = 5 # y, 5
    rover_height = 3.6 # z, 3.6

    # Assuming sensor is the origin and is on top of the vehicle
    coords = np.delete(coords, np.where((coords[:,0] > -rover_width / 2) & (coords[:,0] < rover_width / 2) & (coords[:,1] > -rover_length / 2) & (coords[:,1] < rover_length / 2) & (coords[:,2] < rover_height / 2) & (coords[:,2] > -rover_height / 2)), axis=0)

    new_pc = o3d.geometry.PointCloud()
    new_pc.points = o3d.utility.Vector3dVector(coords)

    return new_pc

def remove_ground(pcd):
    """
    Remove the ground points from the point cloud (obtained from LeGO-LOAM)
    Parameters set for Velodyne HDL-32E
    """
    n_scan = 32
    horizon_scan = 1800
    # ang_top = 10.67
    # ang_bottom = -30.67
    ang_top = 15
    ang_bottom = -25
    ang_res_x = 360/horizon_scan
    ang_res_y = (ang_top-ang_bottom) / (n_scan - 1)
    ground_scan_ind = 20
    sensor_mount_angle = 0
    sensor_minimum_range = 1

    coords = np.asarray(pcd.points)

    # Reorder points
    total_pts = coords.shape[0]
    reordered_coords = np.zeros(shape=[total_pts, 4])
    for [x, y, z] in coords:
        if np.linalg.norm([x, y, z]) < sensor_minimum_range:
            continue

        vertical_angle = np.arctan2(z, np.linalg.norm([x, y])) * 180.0 / np.pi

        if vertical_angle < ang_bottom or vertical_angle > ang_top:
            continue

        vert_id = np.round((vertical_angle - ang_bottom) / ang_res_y)

        horizon_angle = np.arctan2(y, x) * 180.0 / np.pi

        if horizon_angle < 0:
            horizon_angle += 360

        horizon_id = np.round(horizon_angle / ang_res_x)

        index = int(horizon_id + vert_id * horizon_scan)

        if index >= total_pts:
            continue

        reordered_coords[index, :3] = [x, y, z]

    # Ground removal
    if horizon_scan * ground_scan_ind >= total_pts:
        threshold = total_pts
    else:
        threshold = horizon_scan * ground_scan_ind

    for i in range(threshold):
        lower_id = i
        upper_id = i + horizon_scan

        if upper_id >= reordered_coords.shape[0]:
            continue

        diffX = reordered_coords[upper_id, 0] - reordered_coords[lower_id, 0]
        diffY = reordered_coords[upper_id, 1] - reordered_coords[lower_id, 1]
        diffZ = reordered_coords[upper_id, 2] - reordered_coords[lower_id, 2]

        angle = np.arctan2(diffZ, np.linalg.norm([diffX, diffY])) * 180.0 / np.pi

        if np.abs(angle - sensor_mount_angle) <= 10:
            reordered_coords[lower_id, 3] = 1
            reordered_coords[upper_id, 3] = 1

    reordered_coords = reordered_coords[np.logical_not(np.logical_and(reordered_coords[:, 0] == 0, reordered_coords[:, 1] == 0, reordered_coords[:, 2] == 0))]
    
    new_points = reordered_coords[:, :3][reordered_coords[:, -1] != 1]

    new_pc = o3d.geometry.PointCloud()
    new_pc.points = o3d.utility.Vector3dVector(new_points)

    return new_pc

def read_pcd(path, downsample=0, non_rover=False, non_ground=False):
    """
    Read point cloud
    """
    if splitext(path)[1] != '.bin':
        pcd = o3d.io.read_point_cloud(path)
        points = pcd.points
    else:
        points = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
        points = points[:, 0:3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    
    if downsample > 0:
        pcd = pcd.voxel_down_sample(downsample)
        points = pcd.points

    if non_rover:
        points = remove_rover(points)

    if non_ground:
        points = remove_ground(points)

    data = Pair(pos=torch.from_numpy(np.asarray(points)).float(),
                batch=torch.zeros(len(points)).long())
    return data


"""
    KITTI utils
"""
def load_kitti_poses(path):
    """
    Load KITTI poses
    """
    poses = []
    with open(path) as f:
        for line in f:
            pose = np.array(line.split()).astype(np.float32).reshape(3, 4)
            poses.append(pose)
    return poses

def get_kitti_gt_transformation(pose1_id, pose2_id, poses):
    """
    Compute transformation between two KITTI lidar scans
    (transforms from pose1 to pose2)
    """
    pose1 = np.asarray(poses[pose1_id])
    pose1 = np.vstack((pose1, np.array([0, 0, 0, 1])))
    pose1 = np.dot(np.dot(np.linalg.inv(T_CALIB_KITTI), pose1), T_CALIB_KITTI)

    pose2 = np.asarray(poses[pose2_id])
    pose2 = np.vstack((pose2, np.array([0, 0, 0, 1])))
    pose2 = np.dot(np.dot(np.linalg.inv(T_CALIB_KITTI), pose2), T_CALIB_KITTI)

    transformation = np.dot(np.linalg.inv(pose2), pose1)

    return transformation


"""
    Evaluation utils (from TEASER++)
"""
def compose_mat4_from_teaserpp_solution(solution):
    """
    Compose a 4-by-4 matrix from teaserpp solution
    """
    s = solution.scale
    rotR = solution.rotation
    t = solution.translation
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = rotR
    M = T.dot(R)

    if s == 1:
        M = T.dot(R)
    else:
        S = np.eye(4)
        S[0:3, 0:3] = np.diag([s, s, s])
        M = T.dot(R).dot(S)

    return M

def get_angular_error(R_gt, R_est):
    """
    Get angular error
    """
    try:
        A = (np.trace(np.dot(R_gt.T, R_est))-1) / 2.0
        if A < -1:
            A = -1
        if A > 1:
            A = 1
        rotError = math.fabs(math.acos(A));
        return math.degrees(rotError)
    except ValueError:
        import pdb; pdb.set_trace()
        return 99999

def compute_transformation_diff(est_mat, gt_mat):
    """
    Compute difference between two 4-by-4 SE3 transformation matrix
    """
    R_gt = gt_mat[:3,:3]
    R_est = est_mat[:3,:3]
    rot_error = get_angular_error(R_gt, R_est)

    t_gt = gt_mat[:,-1]
    t_est = est_mat[:,-1]
    trans_error = np.linalg.norm(t_gt - t_est)

    return rot_error, trans_error


"""
    Implementations of registration methods
"""
def ransac(pos_s, pos_t, feat_s, feat_t, distance_threshold=0.1):
    """
    RANSAC-based registration
    """
    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(pos_s.cpu().numpy())
    pcd_t = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(pos_t.cpu().numpy())

    f_s = o3d.pipelines.registration.Feature()
    f_s.data = feat_s.cpu().T.numpy()
    f_t = o3d.pipelines.registration.Feature()
    f_t.data = feat_t.cpu().T.numpy()

    start = timer()
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=pcd_s, target=pcd_t, source_feature=f_s, target_feature=f_t, 
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    end = timer()
    
    est_mat = result.transformation
    fitness = result.fitness
    inlier_rmse = result.inlier_rmse
    correspondence_set = result.correspondence_set

    return est_mat, fitness, inlier_rmse, correspondence_set, end - start

def fgr(pos_s, pos_t, feat_s, feat_t, distance_threshold=0.1):
    """
    FGR-based registration
    """
    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(pos_s.numpy())
    pcd_t = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(pos_t.numpy())

    f_s = o3d.pipelines.registration.Feature()
    f_s.data = feat_s.cpu().T.numpy()
    f_t = o3d.pipelines.registration.Feature()
    f_t.data = feat_t.cpu().T.numpy()

    start = timer()
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source=pcd_s, target=pcd_t, source_feature=f_s, target_feature=f_t, 
        option=o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
    end = timer()
    
    est_mat = result.transformation
    fitness = result.fitness
    inlier_rmse = result.inlier_rmse
    correspondence_set = result.correspondence_set

    return est_mat, fitness, inlier_rmse, correspondence_set, end - start

def find_mutually_nn_keypoints(ref_key, test_key, ref, test):
    """
    Use kdtree to find mutually closest keypoints 

    ref_key: reference keypoints (source)
    test_key: test keypoints (target)
    ref: reference feature (source feature)
    test: test feature (target feature)
    """
    ref_features = ref.data.T
    test_features = test.data.T
    ref_keypoints = np.asarray(ref_key.points)
    test_keypoints = np.asarray(test_key.points)
    n_samples = test_features.shape[0]

    ref_tree = KDTree(ref_features)
    test_tree = KDTree(test.data.T)
    test_NN_idx = ref_tree.query(test_features, return_distance=False)
    ref_NN_idx = test_tree.query(ref_features, return_distance=False)

    # find mutually closest points
    ref_match_idx = np.nonzero(
        np.arange(n_samples) == np.squeeze(test_NN_idx[ref_NN_idx])
    )[0]
    ref_matched_keypoints = ref_keypoints[ref_match_idx]
    test_matched_keypoints = test_keypoints[ref_NN_idx[ref_match_idx]]

    return np.transpose(ref_matched_keypoints), np.transpose(test_matched_keypoints)

def teaserpp(pos_s, pos_t, feat_s, feat_t, noise_bound=0.1):
    """
    TEASER++ global registration
    """
    # Load point clouds and features
    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(pos_s.numpy())
    pcd_t = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(pos_t.numpy())

    f_s = o3d.pipelines.registration.Feature()
    f_s.data = feat_s.cpu().T.numpy()
    f_t = o3d.pipelines.registration.Feature()
    f_t.data = feat_t.cpu().T.numpy()

    # Find mutually closest keypoints
    s_matched_key, t_matched_key = find_mutually_nn_keypoints(
        pcd_s, pcd_t, f_s, f_t
    )
    s_matched_key = np.squeeze(s_matched_key)
    t_matched_key = np.squeeze(t_matched_key)
    
    # Prepare TEASER++ Solver
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    # print("TEASER++ Parameters are:", solver_params)

    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    # Solve with TEASER++
    start = timer()
    teaserpp_solver.solve(s_matched_key, t_matched_key)
    end = timer()
    est_solution = teaserpp_solver.getSolution()
    est_mat = compose_mat4_from_teaserpp_solution(est_solution)

    max_clique = teaserpp_solver.getTranslationInliersMap()
    # print("Max clique size:", len(max_clique))

    final_inliers = teaserpp_solver.getTranslationInliers()

    return est_mat, max_clique, final_inliers, end - start