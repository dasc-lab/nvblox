import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


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
    return points, intensity, lines[:data_start_idx]  # Return header


def main():
    parser = argparse.ArgumentParser(description="Apply a 4x4 transformation to an ESDF .ply file.")
    parser.add_argument("input_file", help="Path to the input ESDF .ply file (x, y, z, intensity).")
    args = parser.parse_args()


    # Load the input ESDF file
    points, intensity, header = load_esdf_with_intensity(args.input_file)

    # Normalize distances for color mapping
    min_dist = 0 #np.min(intensity)
    max_dist = np.max(intensity)
    normalized_dist = (intensity - min_dist) / (max_dist - min_dist)

    # Map normalized distances to a colormap
    colors = plt.cm.viridis(normalized_dist)[:, : 3]

    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud with Distances")


if __name__ == "__main__":
    main()
