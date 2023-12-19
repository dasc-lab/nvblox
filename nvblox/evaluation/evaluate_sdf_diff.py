from nvblox_evaluation.evaluation_utils.voxel_grid import VoxelGrid
import open3d as o3d
import numpy as np
from PIL import Image
import os

# Subtracts file1 - file2 for determining if certified
# map is conservative.
tsdf_folder = "tsdfs"
file1_name = "ros2_certified_esdf.ply"
file2_name = "ros2_esdf.ply"
keep_window_open = False
bounds = (-1, 1)  # For keeping SDF color scale consistent across multuple files.
# Get full relative path to all files in folder
sdf_path1 = os.path.join(tsdf_folder, file1_name)
sdf_path2 = os.path.join(tsdf_folder, file2_name)

# Extract slice along xy plane
vis = o3d.visualization.Visualizer()
vis.create_window()
slice = o3d.geometry.TriangleMesh()
viewpoint = None
if viewpoint is not None:
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(viewpoint)
    vis.remove_geometry(slice, reset_bounding_box=False)

sdf_grid1 = VoxelGrid.createFromPly(sdf_path1)
print(f"Shape of sdf_grid1: {sdf_grid1.shape()}")
print(f"Min indices of sdf_grid1: {sdf_grid1.min_indices}")
print(f"0,0,0 voxel value of sdf_grid1: {sdf_grid1.voxels[0,0,0]}")
sdf_grid2 = VoxelGrid.createFromPly(sdf_path2)
print(f"Shape of sdf_grid2: {sdf_grid2.shape()}")
print(f"Min indices of sdf_grid2: {sdf_grid2.min_indices}")
print(f"0,0,0 voxel value of sdf_grid1: {sdf_grid1.voxels[0,0,0]}")
sdf_grid = sdf_grid1 - sdf_grid2
print(f"Shape of diff: {sdf_grid.shape()}")
print(f"Min indices of diff: {sdf_grid.min_indices}")
print(f"0,0,0 voxel value of diff: {sdf_grid.voxels[0,0,0]}")
slice = sdf_grid.get_slice_mesh_at_ratio(
    0.5, axis="z", cube_size=1.0, bounds=bounds, highlight_positive=True
)
if slice is not None:
    vis.add_geometry(slice)
vis.poll_events()
vis.update_renderer()
image_float = np.asarray(vis.capture_screen_float_buffer())
image_uint8 = (image_float * 255).astype(np.uint8)
sdf_name = os.path.splitext(sdf_path1)[0] + "_diff"
Image.fromarray(image_uint8).save(sdf_name + ".png")
# Block on input
if keep_window_open:
    any_key = input("Press any key to continue")
