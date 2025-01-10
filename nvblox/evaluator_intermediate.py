import json
import argparse
import numpy as np
import scipy as sp
import open3d as o3d
from scipy.spatial import KDTree 

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


def interpolate_distances(gt_mesh, esdf_pts, esdf_dists, voxel_size=0.02, fill_value=np.nan):

    # check that the size of the esdf_pts = size of esdf_dists
    assert( esdf_pts.shape[0] == esdf_dists.shape[0])
    

    # create the kd-tree of the esdf points
    with CodeTimer("interpolate/construct kdtree"):
        tree = KDTree(esdf_pts)

    # query the tree at the gt_mesh points
    with CodeTimer("interpolate/query"):
        gt_mesh_pts = np.asarray(gt_mesh.points)
        ds, inds = tree.query(gt_mesh_pts)

    # get the obstacle distances at the gt_mesh_pts
    gt_mesh_dists = esdf_dists[inds]

    # for any gt_mesh point that is not in a voxel, set its value to the fill_value
    maxd  = np.sqrt(3) * voxel_size / 2
    gt_mesh_dists[ds > maxd] = fill_value

    return gt_mesh_dists


def write_ply(filename, points, intensities):
    """
    Write points and their corresponding intensities to a PLY file.

    Parameters:
        filename (str): The name of the output PLY file.
        points (numpy.ndarray): A Nx3 numpy array of 3D points.
        intensities (numpy.ndarray): A N-length numpy array of intensities.
    """
    if points.shape[0] != intensities.shape[0]:
        raise ValueError("Number of points and intensities must match.")

    # Create header for the PLY file
    header = f"""ply
format ascii 1.0
element vertex {points.shape[0]}
property float x
property float y
property float z
property float intensity
end_header
"""

    with CodeTimer("write_ply/construct data lines"):
        # Convert data to the PLY format
        data_lines = []
        for point, intensity in zip(points, intensities):
            x, y, z = point
            data_lines.append(f"{x} {y} {z} {intensity}")

    # Write to the PLY file
    with CodeTimer("write_ply/write"):
        with open(filename, 'w') as ply_file:
            ply_file.write(header)
            ply_file.write("\n".join(data_lines))

    return True


def main():
    parser = argparse.ArgumentParser(description="Evaulate violations with ESDF and GT Mesh")
    parser.add_argument("mesh_file", 
        help="Path to the ground truth mesh file .mesh file (x, y, z, nx, ny, nz, red, green blue)")
    parser.add_argument("esdf_file", 
        help="Path to transformed esdf file .ply file (x, y, z, intensity)")
    parser.add_argument("violation_file", 
        help="Path where the violation mesh file will be saved to.")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="voxel size of the esdf file")
    args = parser.parse_args()

    # Load the ground truth mesh file as point cloud (no. of points = no. of vertices)
    gt_mesh_pcd = convert_mesh_to_pcd(args.mesh_file) 

    # Load the ESDF file as point cloud
    esdf_pts, esdf_dists = load_esdf_with_distance(args.esdf_file)

    gt_mesh_dists = interpolate_distances(gt_mesh_pcd, esdf_pts, esdf_dists, args.voxel_size)

    # write to file
    print("Writing violations file")
    write_ply(args.violation_file, np.asarray(gt_mesh_pcd.points), gt_mesh_dists)




if  __name__ == "__main__":
    main()