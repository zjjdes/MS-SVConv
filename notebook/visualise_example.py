import numpy as np
import open3d as o3d

# Visualise point cloud (KITTI)
# points = np.fromfile("G:/LiDAR datasets/KITTI_partial_3s/velodyne_ref/9179789.bin", dtype=np.float32).reshape((-1, 4))
# points = np.fromfile("G:/LiDAR datasets/torch-points3d/data/kitti/raw/dataset/sequences/00/velodyne/000003.bin", dtype=np.float32).reshape((-1, 4))
# print(points.shape)

# Nuscenes
points = np.fromfile("G:/Download/nuscenes-mini/sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin", dtype=np.float32).reshape((-1, 5))
print(points.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])

o3d.visualization.draw_geometries([pcd])
