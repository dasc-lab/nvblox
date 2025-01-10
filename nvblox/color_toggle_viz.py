import open3d as o3d
import argparse

def create_visualizer(mesh_files):
    """
    Create and configure the Open3D O3DVisualizer for mesh visualization.
    """
    # Initialize the Open3D GUI application
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # Create a 3D visualizer
    visualizer = o3d.visualization.O3DVisualizer("Mesh Viewer", 1024, 768)
    
    # Add coordinate axes and ground plane
    visualizer.show_axes = True
    visualizer.show_ground = True

    # Predefined colors for the meshes
    colors = [
        [1.0, 0.0, 0.0, 1.0],  # Red
        [0.0, 1.0, 0.0, 1.0],  # Green
        [0.0, 0.0, 1.0, 1.0],  # Blue
        [1.0, 1.0, 0.0, 1.0],  # Yellow
    ]

    # Load and add each mesh
    for idx, mesh_file in enumerate(mesh_files):
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()  # Compute normals for proper shading

        # Create a material and set its base color
        material = o3d.visualization.rendering.MaterialRecord()
        material.base_color = colors[idx % len(colors)]  # Cycle through colors
        material.shader = "defaultLit"  # Use a lit shader for proper shading

        name = f"Mesh {idx + 1}: {mesh_file}"
        visualizer.add_geometry(name, mesh, material)

    # Customize rendering options
    visualizer.show_settings = True  # Allow GUI settings
    visualizer.scene.set_background([0.1, 0.1, 0.1, 1.0])  # Set a dark background
    
    return app, visualizer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Mesh Viewer with Open3D 0.16.0")
    parser.add_argument("files", nargs="+", help="Paths to .ply files containing meshes")
    args = parser.parse_args()

    # Create the visualizer
    app, visualizer = create_visualizer(args.files)

    # Add the visualizer to the application and run
    app.add_window(visualizer)
    app.run()

if __name__ == "__main__":
    main()
