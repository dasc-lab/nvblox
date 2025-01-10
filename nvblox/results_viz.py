import open3d as o3d
import os

import imgui

def main(meshes):
    # Initialize imgui
    imgui.create_context()
    window_name = "Open3D Mesh Viewer"

    # Visibility flags
    show_mesh1 = True
    show_mesh2 = True
    show_mesh3 = True

    show_meshes = [True for mesh in meshes]

    # Main loop
    while True:
        imgui.new_frame()

        imgui.begin(window_name)

        for i in range(len(meshes)):
            _, show_meshes[i] = imgui.checkbox(f"Show Mesh {i}", show_meshes[i])
        
        imgui.end()

        # Update visualization
        geometries = []
        for i in range(len(meshes)):
            if show_meshes[i]:
                geometries.append(meshes[i])

        o3d.visualization.draw_geometries(geometries, window_name=window_name)

        imgui.render()

# if __name__ == "__main__":
#     main()


# import open3d as o3d
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering
# 
# class MeshViewerApp:
#     def __init__(self, meshes):
#         self.meshes = meshes
#         self.app = gui.Application.instance
#         self.app.initialize()
#         #self.window = gui.Window("Mesh Viewer", 1024, 768)
#         self.window = self.app.create_window() 
# 
#         # Create a scene widget for rendering
#         self.scene = gui.SceneWidget()
#         self.scene.scene = rendering.Open3DScene(self.window.renderer)
#         self.window.add_child(self.scene)
# 
#         # # Create a panel for checkboxes
#         self.panel = gui.Vert(0, gui.Margins(10))
#         self.checkboxes = {}
# 
#         # # Add meshes to the scene and create checkboxes
#         for i, (name, mesh) in enumerate(self.meshes.items()):
#             # Add mesh to the scene
#             self.scene.scene.add_geometry(name, mesh, rendering.MaterialRecord())
# 
#             # Create a checkbox for each mesh
#             checkbox = gui.Checkbox(name)
#             checkbox.checked = True
#             checkbox.set_on_checked(self._on_checkbox_toggled(name))
#             self.checkboxes[name] = checkbox
#             self.panel.add_child(checkbox)
# 
#         # Add panel to the main window
#         self.window.add_child(self.panel)
# 
#         # Configure camera
#         bounds = self.scene.scene.bounding_box
#         self.scene.setup_camera(30, bounds, bounds.get_center())
# 
#     def _on_checkbox_toggled(self, name):
#         def callback(checked):
#             if checked:
#                 self.scene.scene.show_geometry(name, show=True)
#             else:
#                 self.scene.scene.show_geometry(name, show=False)
# 
#         return callback
# 
#     def run(self):
#         self.app.run()


# Example usage
if __name__ == "__main__":

    # Load the first mesh
    mesh_1 = o3d.io.read_triangle_mesh("./results_eval/final_result/transformed_mesh.ply")
    
    # Load the second mesh
    mesh_2 = o3d.io.read_triangle_mesh("./results_eval/final_result/transformed_certi_mesh.ply")
    
    # load the ground truth mesh
    mesh_gt = o3d.io.read_triangle_mesh("./office0_mesh_triangles.ply")

    meshes = [
        #"GT": mesh_gt,
        #"Uncertified": mesh_1,
        mesh_2
    ]

    # Start the app
    viewer = main(meshes)



# 
# 
# # Visualize both meshes in the same window
# o3d.visualization.draw_geometries(
#     [mesh3,  mesh2],
#     window_name="Mesh Visualization",
#     width=800,
#     height=600,
#     mesh_show_back_face=True
# )
