/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <nvblox/mesh/mesh.h>

namespace nvblox {

template <typename MeshBlockType>
Mesh Mesh::fromLayer(const BlockLayer<MeshBlockType>& layer) {
  Mesh mesh;

  // Keep track of the vertex index.
  int next_index = 0;

  // Iterate over every block in the layer.
  const std::vector<Index3D> indices = layer.getAllBlockIndices();

  for (const Index3D& index : indices) {
    typename MeshBlockType::ConstPtr block = layer.getBlockAtIndex(index);

    // Copy over.
    const std::vector<Vector3f> vertices = block->getVertexVectorOnCPU();
    mesh.vertices.resize(mesh.vertices.size() + vertices.size());
    std::copy(vertices.begin(), vertices.end(),
              mesh.vertices.begin() + next_index);

    const std::vector<Vector3f> normals = block->getNormalVectorOnCPU();
    mesh.normals.resize(mesh.normals.size() + normals.size());
    std::copy(normals.begin(), normals.end(),
              mesh.normals.begin() + next_index);

    const std::vector<Color> colors = block->getColorVectorOnCPU();
    mesh.colors.resize(mesh.colors.size() + colors.size());
    std::copy(colors.begin(), colors.end(), mesh.colors.begin() + next_index);

    // Our simple mesh implementation has:
    // - per vertex colors
    // - per vertex normals
    CHECK((vertices.size() == normals.size()) || (normals.size() == 0));
    CHECK((vertices.size() == vertices.size()) || (colors.size() == 0));

    // Copy over the triangles.
    const std::vector<int> triangles = block->getTriangleVectorOnCPU();
    std::vector<int> triangle_indices(triangles.size());
    // Increment all triangle indices.
    std::transform(triangles.begin(), triangles.end(), triangle_indices.begin(),
                   std::bind2nd(std::plus<int>(), next_index));

    mesh.triangles.insert(mesh.triangles.end(), triangle_indices.begin(),
                          triangle_indices.end());

    next_index += vertices.size();
  }

  return mesh;
}

// Explicit Instantiation
template Mesh Mesh::fromLayer(const BlockLayer<MeshBlock>& layer);
template Mesh Mesh::fromLayer(const BlockLayer<CertifiedMeshBlock>& layer);

// transforms a mesh represented in the old frame into the new frame
Mesh transform_mesh(const Mesh& mesh, const Transform& T_new_old) {
  // create the new mesh
  Mesh new_mesh;

  // handle vertices
  for (auto& vertex : mesh.vertices) {
    Eigen::Vector3f new_vertex = T_new_old * vertex;
    new_mesh.vertices.push_back(new_vertex);
  }

  // handle normals
  for (auto& normal : mesh.normals) {
    Eigen::Vector3f new_normal = T_new_old.linear() * normal;
    new_mesh.normals.push_back(new_normal);
  }

  // handle triangles
  for (auto& triangle : mesh.triangles) {
    new_mesh.triangles.push_back(triangle);
  }

  // handle colors
  for (auto& color : mesh.colors) {
    new_mesh.colors.push_back(color);
  }

  return new_mesh;
}

}  // namespace nvblox