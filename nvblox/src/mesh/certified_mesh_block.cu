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
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {

CertifiedMeshBlock::CertifiedMeshBlock(MemoryType memory_type)
    : vertices(memory_type),
      normals(memory_type),
      colors(memory_type),
      triangles(memory_type) {}

CertifiedMeshBlock::CertifiedMeshBlock(const CertifiedMeshBlock& mesh_block)
    : CertifiedMeshBlock(mesh_block, mesh_block.vertices.memory_type()) {}

CertifiedMeshBlock::CertifiedMeshBlock(const CertifiedMeshBlock& mesh_block,
                                       MemoryType memory_type)
    : vertices(mesh_block.vertices, memory_type),
      normals(mesh_block.normals, memory_type),
      colors(mesh_block.colors, memory_type),
      triangles(mesh_block.triangles, memory_type) {}

void CertifiedMeshBlock::clear() {
  vertices.resize(0);
  normals.resize(0);
  triangles.resize(0);
  colors.resize(0);
}

void CertifiedMeshBlock::resizeToNumberOfVertices(size_t new_size) {
  vertices.resize(new_size);
  normals.resize(new_size);
  triangles.resize(new_size);
}

void CertifiedMeshBlock::reserveNumberOfVertices(size_t new_capacity) {
  vertices.reserve(new_capacity);
  normals.reserve(new_capacity);
  triangles.reserve(new_capacity);
}

CertifiedMeshBlock::Ptr CertifiedMeshBlock::allocate(MemoryType memory_type) {
  return std::make_shared<CertifiedMeshBlock>(memory_type);
}

std::vector<Vector3f> CertifiedMeshBlock::getVertexVectorOnCPU() const {
  return vertices.toVector();
}

std::vector<Vector3f> CertifiedMeshBlock::getNormalVectorOnCPU() const {
  return normals.toVector();
}

std::vector<int> CertifiedMeshBlock::getTriangleVectorOnCPU() const {
  return triangles.toVector();
}

std::vector<Color> CertifiedMeshBlock::getColorVectorOnCPU() const {
  return colors.toVector();
}

size_t CertifiedMeshBlock::size() const { return vertices.size(); }

size_t CertifiedMeshBlock::capacity() const { return vertices.capacity(); }

void CertifiedMeshBlock::expandColorsToMatchVertices() {
  colors.reserve(vertices.capacity());
  colors.resize(vertices.size());
}

// Set the pointers to point to the mesh block.
CudaMeshBlock::CudaMeshBlock(CertifiedMeshBlock* block) {
  CHECK_NOTNULL(block);
  vertices = block->vertices.data();
  normals = block->normals.data();
  triangles = block->triangles.data();
  colors = block->colors.data();

  vertices_size = block->vertices.size();
  triangles_size = block->triangles.size();
}// 

}  // namespace nvblox