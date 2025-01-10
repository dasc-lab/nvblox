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

    print(f"loaded {points.shape[0]} points")

    return points, intensity


def apply_voxblox_colormap(intensity, min_dist = 0.0, max_dist = 3.0 ):
    """
    Apply Voxblox-like colormap for ESDF visualization.
    """

    print(np.unique(intensity))
    # Clamp intensity values to [-truncation_distance, truncation_distance]
    intensity_clamped = np.clip(intensity, min_dist, max_dist)

    # Normalize distances to [0, 1] range
    normalized_dist = (intensity_clamped - min_dist) / (max_dist - min_dist)

    print(normalized_dist)

    # Map normalized distances to viridis colormap
    colors = plt.cm.viridis(normalized_dist)[:, :3]  # Extract RGB (ignore alpha)
    return colors


def draw_slice(points, intensity, slice_z, mesh_file, truncation_distance=3.0, thickness=0.02, ax=2):
    """
    Draw a single slice at the specified z
    Animate point cloud slices from bottom to top in the z direction,
    with the ground truth mesh overlayed and Voxblox-like colormap.
    """

    print("unique intensities: ", np.unique(intensity))

    # Apply Voxblox colormap
    colors = apply_voxblox_colormap(intensity, max_dist = truncation_distance)


    # Get z bounds of the point cloud
    min_z = np.min(points[:, ax])
    max_z = np.max(points[:, ax])

    print(f"min_z: {min_z}, max_z: {max_z}")

    # create slice
    z_min = slice_z - thickness
    z_max = slice_z + thickness 

    # Extract points within the slice
    slice_mask = (points[:, ax] >= z_min) & (points[:, ax] < z_max)
    slice_points = points[slice_mask]
    slice_colors = colors[slice_mask]

    slice_intensity = intensity[slice_mask]
    print("slice intensity range: ", min(slice_intensity), max(slice_intensity))

    slice_cloud = o3d.geometry.PointCloud()
    slice_cloud.points = o3d.utility.Vector3dVector(slice_points)
    slice_cloud.colors = o3d.utility.Vector3dVector(slice_colors)


   # Create Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Slice", width=800, height=600)
    # Load ground truth mesh
    if mesh_file:
      mesh = o3d.io.read_triangle_mesh(mesh_file)
      mesh.compute_vertex_normals()
      vis.add_geometry(mesh)

    # Initialize point cloud for animation
    vis.add_geometry(slice_cloud)

    vis.run()


def main():
    parser = argparse.ArgumentParser(description="Animate ESDF point cloud slices with Voxblox-like colormap.")
    parser.add_argument("input_file", help="Path to the input ESDF .ply file (x, y, z, intensity).")
    parser.add_argument("slice_z", type=float)
    parser.add_argument("--mesh_file", help="Path to the ground truth mesh file.")
    parser.add_argument("--truncation_distance", type=float, default=3.0, help="Truncation distance for ESDF visualization.")
    parser.add_argument("--thickness", type=float, default=0.05, help="thickness of slce.")
    parser.add_argument("--ax", type=int, default=2, help="axis to slice in. 2 = z")
    
    args = parser.parse_args()

    # Load the input ESDF file
    points, intensity = load_esdf_with_intensity(args.input_file)

    # Animate slices with Voxblox-like colormap
    draw_slice(points, intensity, args.slice_z, args.mesh_file, truncation_distance=args.truncation_distance, thickness=args.thickness, ax=args.ax)

    input()


if __name__ == "__main__":
    main()

