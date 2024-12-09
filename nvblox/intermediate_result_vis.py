import open3d as o3d
import os

# Set paths to the directories containing mesh files
dir1 = './results_eval/meshes/ground_truth'
dir2 = './results_eval/meshes/certified'

# Get sorted list of mesh files from both directories
files1 = sorted([f for f in os.listdir(dir1) if f.endswith('.obj') or f.endswith('.stl') or f.endswith('.ply')])
files2 = sorted([f for f in os.listdir(dir2) if f.endswith('.obj') or f.endswith('.stl') or f.endswith('.ply')])

# Ensure both directories have the same number of comparable files
if len(files1) != len(files2):
    print("Warning: The directories have different numbers of files!")
    min_length = min(len(files1), len(files2))
    files1 = files1[:min_length]
    files2 = files2[:min_length]

# Function to visualize two meshes side by side in separate canvases
def visualize_side_by_side(mesh1_path, mesh2_path, title1, title2):
    # Initialize the Open3D GUI application
    o3d.visualization.gui.Application.instance.initialize()

    # Load the meshes
    mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
    mesh2 = o3d.io.read_triangle_mesh(mesh2_path)

    # Ensure normals are computed for better visualization
    if not mesh1.has_vertex_normals():
        mesh1.compute_vertex_normals()
    if not mesh2.has_vertex_normals():
        mesh2.compute_vertex_normals()

    # Create two visualizers
    vis1 = o3d.visualization.O3DVisualizer(title1, 640, 480)
    vis2 = o3d.visualization.O3DVisualizer(title2, 640, 480)

    # Add meshes to the visualizers
    vis1.add_geometry(mesh1)
    vis2.add_geometry(mesh2)

    # Show visualizers side by side
    vis1.show()
    vis2.show()

    # Run the application
    o3d.visualization.gui.Application.instance.run()

# Manual iteration through file pairs
index = 0
while index < len(files1):
    print(f"Comparing {files1[index]} and {files2[index]}")

    # Full paths to the mesh files
    mesh1_path = os.path.join(dir1, files1[index])
    mesh2_path = os.path.join(dir2, files2[index])
    
    # Visualize the meshes side by side
    visualize_side_by_side(mesh1_path, mesh2_path, f"Mesh 1: {files1[index]}", f"Mesh 2: {files2[index]}")

    # Ask user for input to continue or exit
    user_input = input("Press Enter to view the next pair, or type 'exit' to quit: ").strip().lower()
    if user_input == 'exit':
        break
    index += 1

print("Finished visualizing all mesh pairs.")
