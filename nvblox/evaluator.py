import argparse
import numpy as np
import open3d as o3d
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from linetimer import CodeTimer

from matplotlib import pyplot as plt



def convert_mesh_to_pcd(file_path):
    """
    Load a .mesh file and convert it to point cloud
    """
    with CodeTimer("convert_mesh_to_pcd"):
        with CodeTimer("convert_mesh_to_pcd/read triangles"):
            mesh = o3d.io.read_triangle_mesh(file_path)

        # Create a point cloud using the mesh's vertices
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices

        # If the mesh has vertex colors, add them to the point cloud
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors

        # If the mesh has vertex normals, add them to the point cloud
        if mesh.has_vertex_normals():
            pcd.normals = mesh.vertex_normals

        return pcd


def load_esdf_with_distance(file_path):
    """
    Load an ESDF .ply file and extract points (x, y, z) and intensity (distance).
    """
    with CodeTimer("load_esdf/read_lines"):
        with open(file_path, "r") as f:
            lines = f.readlines()

    with CodeTimer("load_esdf/parse_header"):
        header_ended = False
        data_start_idx = 0
        for i, line in enumerate(lines):
            if "end_header" in line:
                header_ended = True
                data_start_idx = i + 1
                break

        if not header_ended:
            raise ValueError("The input file does not appear to be a valid .ply file.")

    with CodeTimer("load_esdf/loadtxt"):
        # Load the point data (x, y, z, intensity)
        data = np.loadtxt(lines[data_start_idx:], dtype=float)
        if data.shape[1] != 4:
            raise ValueError("The input .ply file must have 4 columns: x, y, z, intensity.")

        points = data[:, :3]
        distances= data[:, 3]
        return points, distances 

def interpolate_distances_linear_nd(pcd, esdf_pcd, esdf_dists):
    """
    Interpolates distances for points in a point cloud using linear interpolation.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud whose points need interpolation.
        esdf_pcd (o3d.geometry.PointCloud): ESDF point cloud.
        esdf_dists (np.ndarray): Distances corresponding to ESDF points.

    Returns:
        np.ndarray: Interpolated distances for each point in `pcd`.
    """
    with CodeTimer("interpolate"):
        with CodeTimer("interpolate/asarray"):
            # Extract ESDF points and distances
            esdf_pts = np.asarray(esdf_pcd.points)
            # Extract target point cloud points
            pcd_pts = np.asarray(pcd.points)

        with CodeTimer("interpolate/construct"):
            # Create a LinearNDInterpolator
            # interpolator = LinearNDInterpolator(esdf_pts, esdf_dists)
            interpolator = NearestNDInterpolator(esdf_pts, esdf_dists)

        with CodeTimer("interpolate/run_interpolator"):
            # Interpolate distances for each point in the target cloud
            interpolated_dists = interpolator(pcd_pts)

        with CodeTimer("interpolate/nan_to_num"):
            # Handle any NaN values (points outside the convex hull of the ESDF)
            interpolated_dists = np.nan_to_num(interpolated_dists, nan=0.0)

        return interpolated_dists


def vis_pcd_esdf(gt_pcd, esdf_pcd):
    """
    Visualizes ground truth point cloud and ESDF
    """

#    o3d.visualization.draw_geometries([gt_pcd, esdf_pcd])
    o3d.visualization.draw_geometries([gt_pcd])



def main():
    parser = argparse.ArgumentParser(description="Evaulate violations with ESDF and GT Mesh")
    parser.add_argument("mesh_file", 
        help="Path to the ground truth mesh file .mesh file (x, y, z, nx, ny, nz, red, green blue)")
    parser.add_argument("esdf_file", 
        help="Path to transformed esdf file .ply file (x, y, z, intensity)")
    args = parser.parse_args()

    # Load the ground truth mesh file as point cloud (no. of points = no. of vertices)
    gt_mesh_pcd = convert_mesh_to_pcd(args.mesh_file) 

    # Load the ESDF file as point cloud
    esdf_pts, esdf_dists = load_esdf_with_distance(args.esdf_file)

    # Create ESDF point cloud
    with CodeTimer("main/create_esdf_pcd"):
        esdf_pcd = o3d.geometry.PointCloud()
        esdf_pcd.points = o3d.utility.Vector3dVector(esdf_pts)

        # Color ESDF based on distances
        esdf_colors = np.zeros((len(esdf_dists), 3))  # Initialize color array
        esdf_colors[esdf_dists >= 0] = [0, 1, 0]  # Green for distance >= 0
        esdf_colors[esdf_dists < 0] = [1, 0, 0]   # Red for distance < 0
        esdf_pcd.colors = o3d.utility.Vector3dVector(esdf_colors)

    # Interpolate distances for gound truth point cloud based on ESDF
    interpolated_dists = interpolate_distances_linear_nd(gt_mesh_pcd, esdf_pcd, esdf_dists)

    N_gt_mesh_pts = len(gt_mesh_pcd.points)
    N_esdf_pts = len(esdf_pcd.points)

    in_colors = np.zeros((len(interpolated_dists), 3))

    pos_interpolated_dists = interpolated_dists[interpolated_dists >= 0.02]
    plt.hist(pos_interpolated_dists)
    plt.show()

    threshold_dist = 0.02
    with CodeTimer("create violating inds"):
        violating_inds = [i for i in range(N_gt_mesh_pts) if interpolated_dists[i] > threshold_dist ] 
    with CodeTimer("create non violating inds"):
        not_violating_inds = [i for i in range(N_gt_mesh_pts) if interpolated_dists[i] <= threshold_dist ] 
        # not_violating_inds = [i for i in range(N_gt_mesh_pts) if i not in violating_inds]

    N_violating = len(violating_inds)
    print(N_violating, N_gt_mesh_pts)
    print(1.0 * N_violating / N_gt_mesh_pts)
    print(max(interpolated_dists))

    
    with CodeTimer("main/assign_colors"):
        in_colors[violating_inds] = [0, 1, 0]
        in_colors[not_violating_inds] = [1, 0.9, 0.9]
        
        gt_mesh_pcd.colors = o3d.utility.Vector3dVector(in_colors)
    
    vis_pcd_esdf(gt_mesh_pcd, esdf_pcd)

if  __name__ == "__main__":
    main()