import os
from os import listdir, remove
from os.path import isfile, join
import pathlib
import time
import csv
import numpy as np
import open3d as o3d
import torch
from torch_points3d.applications.pretrained_api import PretainedRegistry
from torch_points3d.core.data_transform import GridSampling3D, AddFeatByKey, AddOnes, Random3AxisRotation
from torch_points3d.datasets.registration.pair import Pair
from torch_points3d.utils.registration import get_matches, fast_global_registration
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torch_points3d.metrics.registration_metrics import compute_hit_ratio
from scipy.spatial.distance import cdist

torch.cuda.set_per_process_memory_fraction(0.8, 0)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

def remove_rover(pcd):
    """Remove rover vehicle from the point cloud

    Args:
        pcd ([type]): [description]

    Returns:
        [type]: [description]
    """
    coords = np.asarray(pcd.points)

    rover_width = 2 # x, 2
    rover_length = 5 # y, 5
    rover_height = 1.8 # z, 3.6

    # Assuming sensor is the origin and is on top of the vehicle
    coords = np.delete(coords, np.where((coords[:,0] > -rover_width / 2) & (coords[:,0] < rover_width / 2) & (coords[:,1] > -rover_length / 2) & (coords[:,1] < rover_length / 2) & (coords[:,2] < rover_height)), axis=0)

    new_pc = o3d.geometry.PointCloud()
    new_pc.points = o3d.utility.Vector3dVector(coords)

    return new_pc


def remove_ground(pcd):
    n_scan = 32
    horizon_scan = 1800
    # ang_top = 10.67 # HDL-32E
    # ang_bottom = -30.67 # HDL-32E
    ang_top = 15 # VLP-32C
    ang_bottom = -25 # VLP-32C
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


def read_pcd(path, downsample=0, non_rover=True, non_ground=False):
    """Read point cloud

    Args:
        path ([string]): path to the point cloud

    Returns:
        [type]: [description]
    """
    pcd = o3d.io.read_point_cloud(path)

    if downsample > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    
    if non_rover:
        pcd = remove_rover(pcd)

    if non_ground:
        pcd = remove_ground(pcd)

    points = pcd.points

    data = Pair(pos=torch.from_numpy(np.asarray(points)).float(),
                batch=torch.zeros(len(points)).long())
    return data


def ransac(pos_s, pos_t, feat_s, feat_t, distance_threshold=0.1):
    """Register two point clouds using RANSAC

    Args:
        pos_s ([type]): [description]
        pos_t ([type]): [description]
        feat_s ([type]): [description]
        feat_t ([type]): [description]
        distance_threshold (float, optional): [description]. Defaults to 0.1.

    Returns:
        [type]: [description]
    """
    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(pos_s.cpu().numpy())
    pcd_t = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(pos_t.cpu().numpy())

    f_s = o3d.pipelines.registration.Feature()
    f_s.data = feat_s.T.cpu().numpy()
    f_t = o3d.pipelines.registration.Feature()
    f_t.data = feat_t.T.cpu().numpy()
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=pcd_s, target=pcd_t, source_feature=f_s, target_feature=f_t,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False),
        ransac_n=4,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    # return torch.from_numpy(result.transformation).float()
    return result


def fast_registration(pos_s, pos_t, feat_s, feat_t, distance_threshold=0.1):
    """Register two point clouds based on Fast Global Registration
        Q.-Y. Zhou, J. Park, and V. Koltun, Fast Global Registration, ECCV, 2016.

    Args:
        pos_s ([type]): [description]
        pos_t ([type]): [description]
        feat_s ([type]): [description]
        feat_t ([type]): [description]
        distance_threshold (float, optional): [description]. Defaults to 0.04.

    Returns:
        [type]: [description]
    """
    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(pos_s.cpu().numpy())
    pcd_t = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(pos_t.cpu().numpy())

    f_s = o3d.pipelines.registration.Feature()
    f_s.data = feat_s.T.cpu().numpy()
    f_t = o3d.pipelines.registration.Feature()
    f_t.data = feat_t.T.cpu().numpy()
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source=pcd_s, target=pcd_t, source_feature=f_s, target_feature=f_t, 
        option=o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def is_well_distributed(points):
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    azimuth[azimuth < 0] += np.pi
    diff = np.max(azimuth) - np.min(azimuth)
    print(' - Azimuth across ' + str(np.rad2deg(diff)) + ' degrees')
    if diff <= np.pi / 2:
        return False
    else:
        return True


def pdop_test(points):
    norm = np.linalg.norm(points, axis=1).reshape((-1, 1))
    A = points / norm
    Q = np.linalg.inv(np.dot(A.T,A))
    pdop = np.sqrt(np.sum(np.diag(Q)))
    print(' - PDOP=' + str(pdop))
    return pdop <= 5


def load_lists(pos_file, map_file, ref_file, rov_file):
    # Crude/estimated positions of the vehicle
    pos_xyz = np.genfromtxt(pos_file, delimiter=',').reshape((-1, 3))
    # pos_xyz = pos_xyz[:,1:]

    # Vehicle positions of the reference scans
    map_xyz = np.genfromtxt(map_file, delimiter=',').reshape((-1, 3))

    # IDs/timestamps of the reference scans 
    ref_id = np.genfromtxt(ref_file, dtype=str, delimiter=',').reshape((-1, 1))

    # IDs/timestamps of the rover scans 
    rov_id = np.genfromtxt(rov_file, dtype=str, delimiter=',').reshape((-1, 1))

    return pos_xyz, map_xyz, ref_id, rov_id


def main(velo_dir, out_dir, model_file, pos_file, map_file, ref_file, rov_file, max_ref, max_trials, min_fitness, voxel_size, num_kpts, non_rover, non_ground, downsample, ransac_dist):
    # Load reference list, rover list, reference vehicle positions, rover vehicle positions
    pos_xyz, map_xyz, ref_id, rov_id = load_lists(pos_file, map_file, ref_file, rov_file)

    # Euclidean distances between rover scans and reference scans
    dists = cdist(pos_xyz, map_xyz)

    # Lists of processed and unprocessed scans
    processed = np.empty((0, 2))
    unprocessed = np.empty((0, 1))

    # Load model
    model = PretainedRegistry.from_file(model_file, mock_property={}).cuda()

    # Data augmentation settings
    # transform = Compose([Random3AxisRotation(rot_x=180, rot_y=180, rot_z=180), GridSampling3D(size=voxel_size, quantize_coords=True, mode='last'), AddOnes(), AddFeatByKey(add_to_x=True, feat_name="ones")])
    transform = Compose([GridSampling3D(size=voxel_size, quantize_coords=True, mode='last'), AddOnes(), AddFeatByKey(add_to_x=True, feat_name="ones")])

    # Collect features of reference scans
    data_map = {}

    for i in range(dists.shape[0]):
        t_init = time.time()

        j = 0
        found = False

        # Locate the rover scan file
        rov_filename = str(rov_id[i])[2:-2]
        rov_path = join(velo_dir, rov_filename + '.pcd')
        data_s = transform(read_pcd(rov_path, downsample, non_rover, non_ground)) # read rover scan

        # Indices of sorted distances to all reference cans for current rover scan
        sorted_idx = np.argsort(dists[i, :])

        # Create results folder if it does not exist
        result_dir = join(out_dir, rov_filename)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        while not found and j <= max_ref:
            # Locate the reference scan file
            ref_filename = str(ref_id[sorted_idx[j]])[2:-2]
            ref_path = join(velo_dir, ref_filename + '.pcd')

            # Reuse features of reference scans if they have been computed before
            if ref_filename in data_map:
                data_t = data_map[ref_filename]
            else:
                data_t = transform(read_pcd(ref_path, downsample, non_rover, non_ground)) # read reference scan
                data_map[ref_filename] = data_t

            print('Processing ROVER: ' + rov_filename + ' with REFERENCE: ' + ref_filename + ', d=' + str(dists[i, sorted_idx[j]]))

            # Compute the matches
            with torch.no_grad():
                model.set_input(data_s, "cuda")
                feat_s = model.forward()
                model.set_input(data_t, "cuda")
                feat_t = model.forward()

            rand_s = torch.randint(0, len(feat_s), (num_kpts, ))
            rand_t = torch.randint(0, len(feat_t), (num_kpts, ))

            # matches = get_matches(feat_s[rand_s], feat_t[rand_t], sym=True) # nearest neighbour
            # sel_s = matches[:,0]
            # sel_t = matches[:,1]

            # Register scans using RANSAC
            for k in range(max_trials):
                # reg_result = fast_registration(data_s.pos[rand_s], data_t.pos[rand_t], feat_s[rand_s], feat_t[rand_t], ransac_dist)
                reg_result = ransac(data_s.pos[rand_s], data_t.pos[rand_t], feat_s[rand_s], feat_t[rand_t], ransac_dist)
                # reg_result = ransac(data_s.pos[rand_s][sel_s], data_t.pos[rand_t][sel_t], feat_s[rand_s][sel_s], feat_t[rand_t][sel_t], ransac_dist)

                Tr = reg_result.transformation
                inlier_rmse = reg_result.inlier_rmse
                corres = np.asarray(reg_result.correspondence_set)
                kpts_s = np.asarray(data_s.pos[rand_s][corres[:, 0], :]) # TODO: apply georeferencing here?
                kpts_t = np.asarray(data_t.pos[rand_t][corres[:, 1], :])
                # kpts_s = np.asarray(data_s.pos[rand_s][sel_s][corres[:, 0], :]) # TODO: apply georeferencing here?
                # kpts_t = np.asarray(data_t.pos[rand_t][sel_t][corres[:, 1], :])

                if reg_result.fitness >= min_fitness:
                    break
                elif k == max_trials - 1:
                    inlier_rmse = 0.0

            if inlier_rmse == 0.0 and j >= max_ref:
                unprocessed = np.vstack((unprocessed, np.array([rov_filename])))
                j += 1
                print(' - Failed to match ' + rov_filename + ' with any reference scan, skip.')
            elif inlier_rmse == 0.0:
                j += 1
                print(' - Fitness=' + str(reg_result.fitness) + ', ' + rov_filename + ' to be reprocessed with the next nearest reference scan.')
            else:
                print(' - Registration successful after ' + str(k + 1) + ' trials with reference scan #' + str(j + 1) + '. RMSE=' + str(inlier_rmse) + ', Fitness=' + str(reg_result.fitness) + ', matched ' + str(corres.shape[0]) + ' keypoints.')

                # Collect results: [reference scan ID, rover scan ID, RMSE]
                results = [[ref_filename], [rov_filename], [inlier_rmse]]

                # Save results
                with open(join(result_dir, 'Ref_Rov_RMSE.csv'), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(results)
                    writer.writerows(Tr)
                np.savetxt(join(result_dir, rov_filename + '_keypts.csv'), kpts_s, fmt='%.5f', delimiter=',')
                np.savetxt(join(result_dir, ref_filename + '_keypts.csv'), kpts_t, fmt='%.5f', delimiter=',')

                processed = np.vstack((processed, np.array([rov_filename, ref_filename])))

                # Toggle the found flag to terminate the loop
                found = True

            print(' - Runtime=' + str(time.time() - t_init) + ' seconds.\n')

    np.savetxt(join(out_dir, 'corres_scans.csv'), processed, fmt='%s', delimiter=',')
    np.savetxt(join(out_dir, 'not_processed.csv'), unprocessed, fmt='%s', delimiter=',')


if __name__ == '__main__':
    # HK_20190428
    base_dir = '../../../UrbanNav/HK_20190428'
    velo_dir = join(base_dir, 'velodyne')
    out_dir = join(base_dir, 'keypoints_ETH_3000_0.01_0.2')
    pos_file = join(base_dir, 'ublox_test_xyz_gt.csv')
    map_file = join(base_dir, 'velodyne_map/hdMap_xyz_gt.csv')
    ref_file = join(base_dir, 'map_list.txt')
    rov_file = join(base_dir, 'test_list.txt')

    model_file = './models/MS_SVCONV_4cm_X2_3head_eth.pt' # ETH
    # model_file = './models/MS_SVCONV_B4cm_X2_3head.pt' # ETH
    # model_file = './models/MS_SVCONV_2cm_X2_3head_3dm.pt' # 3DMatch
    # model_file = './models/MS_SVCONV_B2cm_X2_3head.pt' # TUM
    max_ref = 5 # maximum number of reference scans to test
    max_trials = 5 # maximum number of trials for RANSAC
    min_fitness = 0.01 # minimum fitness to accept transformation
    voxel_size = 0.04
    num_kpts = 3000
    non_rover = False # remove rover vehicle before processing
    non_ground = False # remove ground points before processing
    downsample = 0 # voxel size for downsampling, 0 means no downsampling
    ransac_dist = 0.2 # distance threshold for RANSAC

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    main(velo_dir, out_dir, model_file, pos_file, map_file, ref_file, rov_file, max_ref, max_trials, min_fitness, voxel_size, num_kpts, non_rover, non_ground, downsample, ransac_dist)
