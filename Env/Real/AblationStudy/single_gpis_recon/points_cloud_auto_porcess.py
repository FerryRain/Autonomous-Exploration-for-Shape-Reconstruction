import numpy as np
import open3d as o3d

# 加载点云
points = np.load("Diress_global.npy")  # 替换为你的路径

center = (0,0,0)
# center = points.mean(axis=0)
directions = points - center
norms = np.linalg.norm(directions, axis=1, keepdims=True)
normalized_directions = directions / (norms + 1e-8)
expanded_points = center + normalized_directions * (norms + 10)




pcd_orig = o3d.geometry.PointCloud()
pcd_orig.points = o3d.utility.Vector3dVector(points)
pcd_orig.paint_uniform_color([1, 0, 0])
pcd_exp = o3d.geometry.PointCloud()
pcd_exp.points = o3d.utility.Vector3dVector(expanded_points)
pcd_exp.paint_uniform_color([0, 1, 0])

o3d.visualization.draw_geometries([pcd_orig, pcd_exp], window_name="Original vs Expanded PointCloud")
np.save("Diress_global_expand.npy", expanded_points)

print("Save Successful：expanded_point_cloud.npy")