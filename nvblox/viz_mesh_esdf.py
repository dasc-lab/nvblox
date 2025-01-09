import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time


def load_esdf_with_intensity(file_path):
    """
    Load an ESDF .ply file and extract points (x, y, z) and intensity.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    header_ended = False
    data_start_idx = 0
    for i, line in enumerate(lines):
        if "end_header" in line:
            header_ended = True
            data_start_idx = i + 1
            break

    if not header_ended:
        raise ValueError("The input file does not appear to be a valid .ply file.")

    # Load the point data (x, y, z, intensity)
    data = np.loadtxt(lines[data_start_idx:], dtype=float)
    if data.shape[1] != 4:
        raise ValueError("The input .ply file must have 4 columns: x, y, z, intensity.")

    points = data[:, :3]
    intensity = data[:, 3]
    return points, intensity


def apply_voxblox_colormap(intensity, truncation_distance=3):
    """
    Apply Voxblox-like colormap for ESDF visualization.
    """
    # Clamp intensity values to [0, truncation_distance]
    intensity_clamped = np.clip(intensity, 0, truncation_distance)

    # Normalize distances to [0, 1] range
    normalized_dist = intensity_clamped / truncation_distance

    # Map normalized distances to viridis colormap
    colors = plt.cm.viridis(normalized_dist)[:, :3]  # Extract RGB (ignore alpha)
    return colors


def animate_slices_with_voxblox_colormap(points, intensity, mesh_file, truncation_distance=3.0, voxel_size=0.02, ax=2):
    """
    Animate point cloud slices from bottom to top in the z direction,
    with the ground truth mesh overlayed and Voxblox-like colormap.
    """
    # Apply Voxblox colormap
    colors = apply_voxblox_colormap(intensity, truncation_distance)


    # Get z bounds of the point cloud
    min_z = np.min(points[:, ax])
    max_z = np.max(points[:, ax])

    print(f"min_z: {min_z}, max_z: {max_z}")

    # Ensure slice height captures at least 2 points
    unique_z = np.sort(np.unique(points[:, ax]))
    if len(unique_z) > 1:
        slice_height = voxel_size * 2.5
    else:
        raise ValueError("Point cloud does not have enough unique z-levels for slicing.")

    # Create Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Slices Animation", width=800, height=600)
    # Load ground truth mesh
    if mesh_file:
      mesh = o3d.io.read_triangle_mesh(mesh_file)
      mesh.compute_vertex_normals()
      vis.add_geometry(mesh)

    # Initialize point cloud for animation
    slice_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(slice_cloud)

    def update_slice(slice_index):
        """
        Update function for each frame of the animation.
        """
        z_min = min_z + slice_index * slice_height
        z_max = z_min + slice_height

        # Extract points within the slice
        slice_mask = (points[:, ax] >= z_min) & (points[:, ax] < z_max)
        slice_points = points[slice_mask]
        slice_colors = colors[slice_mask]

        # Check if any points are in the slice
        if len(slice_points) == 0:
            print("len=0")
            return  # Skip empty slices

        # Update the slice point cloud
        slice_cloud.points = o3d.utility.Vector3dVector(slice_points)
        slice_cloud.colors = o3d.utility.Vector3dVector(slice_colors)

        vis.update_geometry(slice_cloud)
        vis.poll_events()
        vis.update_renderer()

    # Run animation
    num_slices = int((max_z - min_z) / slice_height)

    while True:
        for slice_index in range(num_slices):
            update_slice(slice_index)
            time.sleep(0.1)

    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Animate ESDF point cloud slices with Voxblox-like colormap.")
    parser.add_argument("input_file", help="Path to the input ESDF .ply file (x, y, z, intensity).")
    parser.add_argument("--mesh_file", help="Path to the ground truth mesh file.")
    parser.add_argument("--truncation_distance", type=float, default=3.0, help="Truncation distance for ESDF visualization.")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size at which ESDF was generated.")
    parser.add_argument("--ax", type=int, default=2, help="axis to slice in. 2 = z")
    
    args = parser.parse_args()

    # Load the input ESDF file
    points, intensity = load_esdf_with_intensity(args.input_file)

    # Animate slices with Voxblox-like colormap
    animate_slices_with_voxblox_colormap(points, intensity, args.mesh_file, truncation_distance=args.truncation_distance, voxel_size=args.voxel_size, ax=args.ax)


if __name__ == "__main__":
    main()

