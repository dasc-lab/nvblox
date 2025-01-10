import argparse
import numpy as np
import open3d as o3d


def load_mesh(file_path):
    """
    Load a mesh file with its original colors.
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertex_colors():
        mesh.compute_vertex_normals()  # Ensure normals for shading
    return mesh


def load_esdf_with_intensity(file_path):
    """
    Custom loader for ESDF .ply files that extracts points and intensity.
    The file is expected to have columns for x, y, z, and intensity.
    """
    # Read the .ply file
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Skip the header to find the start of vertex data
    header_ended = False
    data_start_idx = 0
    for i, line in enumerate(lines):
        if "end_header" in line:
            header_ended = True
            data_start_idx = i + 1
            break

    if not header_ended:
        raise ValueError("The input file does not appear to be a valid .ply file.")

    # Load the point data
    data = np.loadtxt(lines[data_start_idx:], dtype=float)
    if data.shape[1] != 4:
        raise ValueError("The input .ply file must have 4 columns: x, y, z, intensity.")

    # Separate points and intensity
    points = data[:, :3]
    intensity = data[:, 3]
    return points, intensity


def apply_color_gradient(intensity, color_map):
    """
    Apply a custom RGB gradient to intensity values.
    `color_map` should be a list of 3 values: [r, g, b].
    """
    # Normalize intensity
    normalized_intensity = (intensity - intensity.min()) / (intensity.ptp() + 1e-9)

    # Map intensity to the RGB gradient
    colors = np.outer(normalized_intensity, color_map)
    return colors


def visualize(mesh, esdfs):
    """
    Visualize the mesh and up to 3 ESDF point clouds.
    """
    geometries = [mesh]

    for idx, (points, intensity, color_map) in enumerate(esdfs):
        # Apply color gradient to intensity
        colors = apply_color_gradient(intensity, color_map)

        # Create an Open3D PointCloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(point_cloud)

    # Visualize the mesh and ESDFs
    o3d.visualization.draw_geometries(geometries, window_name="Mesh and ESDF Viewer")


def main():
    parser = argparse.ArgumentParser(description="Visualize a mesh with up to 3 ESDF point clouds.")
    parser.add_argument("mesh_file", help="Path to the mesh file (e.g., .ply, .obj).")
    parser.add_argument("esdf_files", nargs="*", help="Paths to up to 3 ESDF .ply files (x, y, z, intensity).")
    args = parser.parse_args()

    if len(args.esdf_files) > 3:
        raise ValueError("A maximum of 3 ESDF files can be provided.")

    # Load the mesh
    mesh = load_mesh(args.mesh_file)

    # Define color gradients for up to 3 ESDFs
    color_maps = [
        [1.0, 0.0, 0.0],  # Red gradient
        [0.0, 1.0, 0.0],  # Green gradient
        [0.0, 0.0, 1.0],  # Blue gradient
    ]

    # Load the ESDF files and associate them with color gradients
    esdfs = []
    for idx, esdf_file in enumerate(args.esdf_files):
        points, intensity = load_esdf_with_intensity(esdf_file)
        esdfs.append((points, intensity, color_maps[idx]))

    # Visualize the data
    visualize(mesh, esdfs)


if __name__ == "__main__":
    main()

