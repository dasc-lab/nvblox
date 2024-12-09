import open3d as o3d
import os

# Load the first mesh
mesh1 = o3d.io.read_triangle_mesh("./results_eval/final_result/mesh.ply")

# Load the second mesh
#mesh2 = o3d.io.read_triangle_mesh("./results_eval/final_result/transformed_certi_mesh.ply")
#mesh2 = o3d.io.read_triangle_mesh("/root/data/Replica/office0_mesh.ply")

# Visualize both meshes in the same window
o3d.visualization.draw_geometries(
    [mesh1],
    window_name="Two Mesh Visualization",
    width=800,
    height=600,
    mesh_show_back_face=True
)
