from nvblox_evaluation.evaluation_utils.voxel_grid import VoxelGrid
import open3d as o3d
import numpy as np
from PIL import Image
import os

# Load TSDF from ply file
# tsdf_path = "test_def_tsdf_gpu.ply"
tsdf_folder = "tsdfs"
keep_window_open = False
bounds = (-1, 1)  # For keeping SDF color scale consistent across multuple files.
# Get full relative path to all files in folder
tsdf_paths = [os.path.join(tsdf_folder, f) for f in os.listdir(tsdf_folder) if os.path.isfile(os.path.join(tsdf_folder, f)) and f.endswith(".ply")]

# Extract slice along xy plane
vis = o3d.visualization.Visualizer()
vis.create_window()
slice = o3d.geometry.TriangleMesh()
viewpoint = None
if viewpoint is not None:
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(viewpoint)
for tsdf_path in tsdf_paths:
    vis.remove_geometry(slice, reset_bounding_box=False)
    tsdf_grid = VoxelGrid.createFromPly(tsdf_path) 
    slice = tsdf_grid.get_slice_mesh_at_ratio(0.5, axis='z', cube_size=1.0, bounds=bounds)
    if slice is not None:
        vis.add_geometry(slice)
    vis.poll_events()
    vis.update_renderer()
    image_float = np.asarray(vis.capture_screen_float_buffer())
    image_uint8 = (image_float * 255).astype(np.uint8)
    # Get file name from path without extension
    tsdf_name = os.path.splitext(tsdf_path)[0]
    Image.fromarray(image_uint8).save(tsdf_name+".png")
# Block on input
if keep_window_open:
    any_key = input("Press any key to continue")
