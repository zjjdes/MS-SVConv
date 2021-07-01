import os
from os import listdir
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

torch.cuda.set_per_process_memory_fraction(0.8)


def read_pcd(path):
    """Read point cloud

    Args:
        path ([string]): path to the point cloud

    Returns:
        [type]: [description]
    """
    pcd = o3d.io.read_point_cloud(path)
    data = Pair(pos=torch.from_numpy(np.asarray(pcd.points)).float(),
                batch=torch.zeros(len(pcd.points)).long())
    return data


def ransac(pos_s, pos_t, feat_s, feat_t, distance_threshold=0.3):
    """Register two point clouds using RANSAC

    Args:
        pos_s ([type]): [description]
        pos_t ([type]): [description]
        feat_s ([type]): [description]
        feat_t ([type]): [description]
        distance_threshold (float, optional): [description]. Defaults to 0.3.

    Returns:
        [type]: [description]
    """
    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(pos_s.numpy())
    pcd_t = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(pos_t.numpy())

    f_s = o3d.pipelines.registration.Feature()
    f_s.data = feat_s.T.numpy()
    f_t = o3d.pipelines.registration.Feature()
    f_t.data = feat_t.T.numpy()
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


def load_lists(pos_file, map_file, ref_file, rov_file):
    # Crude/estimated positions of the vehicle
    pos_xyz = np.genfromtxt(pos_file, delimiter=',').reshape((-1, 4))
    pos_xyz = pos_xyz[:,1:]

    # Vehicle positions of the reference scans
    map_xyz = np.genfromtxt(map_file, delimiter=',').reshape((-1, 3))

    # IDs/timestamps of the reference scans 
    ref_id = np.genfromtxt(ref_file, dtype=str, delimiter=',').reshape((-1, 1))

    # IDs/timestamps of the rover scans 
    rov_id = np.genfromtxt(rov_file, dtype=str, delimiter=',').reshape((-1, 1))

    return pos_xyz, map_xyz, ref_id, rov_id


def main(velo_dir, tr_dir, out_dir, model_file, pos_file, map_file, ref_file, rov_file, max_ref, voxel_size, num_kpts):
    # Load reference list, rover list, reference vehicle positions, rover vehicle positions
    pos_xyz, map_xyz, ref_id, rov_id = load_lists(pos_file, map_file, ref_file, rov_file)

    # Euclidean distances between rover scans and reference scans
    dists = cdist(pos_xyz, map_xyz)

    # Lists of processed and unprocessed scans
    processed = np.empty((0, 2))
    unprocessed = np.empty((0, 1))

    # Load model
    model = PretainedRegistry.from_file(model_file, mock_property={})

    # Data augmentation settings
    transfo = Compose([Random3AxisRotation(rot_x=180, rot_y=180, rot_z=180), GridSampling3D(size=voxel_size, quantize_coords=True, mode='last'), AddOnes(), AddFeatByKey(add_to_x=True, feat_name="ones")])

    for i in range(dists.shape[0]):
        t_init = time.time()

        j = 0
        found = False

        # Locate the rover scan file
        rov_filename = str(rov_id[i])[2:-2]
        rov_path = join(velo_dir, rov_filename + '.ply')
        data_t = transfo(read_pcd(rov_path)) # read rover scan

        # Indices of sorted distances to all reference cans for current rover scan
        sorted_idx = np.argsort(dists[i, :])

        # Create results folder if it does not exist
        result_dir = join(out_dir, rov_filename)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        while not found and j <= max_ref:
            # Locate the reference scan file
            ref_filename = str(ref_id[sorted_idx[j]])[2:-2]
            ref_path = join(velo_dir, ref_filename + '.ply')

            data_s = transfo(read_pcd(ref_path)) # read reference scan

            print('Processing ROVER: ' + rov_filename + ' with REFERENCE: ' + ref_filename)

            # Compute the matches
            with torch.no_grad():
                model.set_input(data_s, "cuda")
                feat_s = model.forward()
                model.set_input(data_t, "cuda")
                feat_t = model.forward()

            rand_s = torch.randint(0, len(feat_s), (num_kpts, ))
            rand_t = torch.randint(0, len(feat_t), (num_kpts, ))
            matches = get_matches(feat_s[rand_s], feat_t[rand_t], sym=True)

            # Register scans using RANSAC
            ransac_result = ransac(data_s.pos[rand_s], data_t.pos[rand_t], feat_s[rand_s], feat_t[rand_t], voxel_size)
            # ransac_result = fast_global_registration(data_s.pos[rand_s], data_t.pos[rand_t])
            transformation = ransac_result.transformation
            inlier_rmse = ransac_result.inlier_rmse

            if inlier_rmse == 0.0 and j >= max_ref:
                unprocessed = np.vstack((unprocessed, np.array([rov_filename])))
                j += 1
                print(' - Failed to match ' + rov_filename + ' with any reference scan, skip.')
            elif inlier_rmse == 0.0:
                j += 1
                print(' - RMSE=' + str(inlier_rmse) + ', ' + rov_filename + 'to be reprocessed with the next nearest reference scan.')
            else:
                corres = np.asarray(ransac_result.correspondence_set)
                kpts_s = np.asarray(data_s.pos[rand_s][corres[:, 0], :]) # TODO: apply georeferencing here?
                kpts_t = np.asarray(data_t.pos[rand_t][corres[:, 1], :])

                print(' - Registration successful! RMSE=' + str(inlier_rmse) + ', matched ' + str(corres.shape[0]) + ' keypoints.')

                # Collect results: [rover scan ID, reference scan ID, RMSE]
                results = [[rov_filename], [ref_filename], [inlier_rmse]]

                # Save results
                with open(join(result_dir, 'Ref_Rov_RMSE.csv'), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(results)
                    writer.writerows(transformation)
                np.savetxt(join(result_dir, ref_filename + '_keypts.csv'), kpts_s, fmt='%1.4f', delimiter=',')
                np.savetxt(join(result_dir, rov_filename + '_keypts.csv'), kpts_t, fmt='%1.4f', delimiter=',')

                processed = np.vstack((processed, np.array([rov_filename, ref_filename])))

                # Toggle the found flag to terminate the loop
                found = True

            print(' - Runtime=' + str(time.time() - t_init) + ' seconds.\n')

    np.savetxt(join(out_dir, 'corres_scans.csv'), processed, fmt='%s', delimiter=',')
    np.savetxt(join(out_dir, 'not_processed.csv'), unprocessed, fmt='%s', delimiter=',')


if __name__ == '__main__':
    base_dir = '../../../UrbanNav/HK-Data20200314'
    velo_dir = join(base_dir, 'velodyne')
    tr_dir = join(base_dir, 'Tr_loam_georeferencing')
    out_dir = join(base_dir, 'keypoint_matching_output')
    model_file = './models/MS_SVCONV_B4cm_X2_3head.pt'
    pos_file = join(base_dir, 'ublox_sd_test.csv')
    map_file = join(base_dir, 'velodyne_map/hdMap_xyz.csv')
    ref_file = join(base_dir, 'map_list.txt')
    rov_file = join(base_dir, 'test_list.txt')
    max_ref = 5
    voxel_size = 0.3
    num_kpts = 2000

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    main(velo_dir, tr_dir, out_dir, model_file, pos_file, map_file, ref_file, rov_file, max_ref, voxel_size, num_kpts)
