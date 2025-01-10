import trimesh
import sys


def load_and_triangulate_quadmesh(file_path):
    """
    Load a quadmesh PLY file using trimesh and convert it into a triangulated mesh.

    Args:
        file_path (str): Path to the input quadmesh PLY file.

    Returns:
        trimesh.Trimesh: Triangulated trimesh object.
    """
    # Load the mesh with trimesh
    mesh = trimesh.load(file_path, process=False)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The input file does not contain a valid Trimesh object.")

    # Check if faces are quads
    if mesh.faces.shape[1] == 4:
        print("Converting quadmesh to triangulated mesh...")
        mesh = mesh.subdivide_to_triangles()

    return mesh

# Example usage
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("invalid args")
        exit(0)

    input_ply_file =  sys.argv[1]
    output_ply_file = sys.argv[2]

    # Load and triangulate the quadmesh
    triangulated_mesh = load_and_triangulate_quadmesh(input_ply_file)

    # Save the triangulated mesh
    triangulated_mesh.export(output_ply_file)
    print(f"Triangulated mesh saved to: {output_ply_file}")

    # Optional: visualize the mesh (requires pyglet or other visualization backends)
    # triangulated_mesh.show()

