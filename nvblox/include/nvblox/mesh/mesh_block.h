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
#pragma once

#include <memory>

#include "nvblox/core/color.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/layer.h"

namespace nvblox {

/// A mesh block containing all of the triangles from this block.
/// Each block contains only the UPPER part of its neighbors: i.e., the max
/// x, y, and z axes. Its neighbors are responsible for the rest.
struct MeshBlock {
  typedef std::shared_ptr<MeshBlock> Ptr;
  typedef std::shared_ptr<const MeshBlock> ConstPtr;

  /// Create a mesh block of the specified memory type.
  MeshBlock(MemoryType memory_type = MemoryType::kDevice);

  /// "Clone" copy constructor, with the possibility to change device type.
  MeshBlock(const MeshBlock& mesh_block);
  /// "Clone" to a different memory type.
  MeshBlock(const MeshBlock& mesh_block, MemoryType memory_type);

  // Mesh Data
  // These unified vectors contain the mesh data for this block. Note that
  // Colors and/or intensities are optional. The "triangles" vector is a vector
  // of indices into the vertices vector. Triplets of consecutive elements form
  // triangles with the indexed vertices as their corners.
  unified_vector<Vector3f> vertices;
  unified_vector<Vector3f> normals;
  unified_vector<Color> colors;
  unified_vector<int> triangles;

  /// Clear all data within the mesh block.
  void clear();

  /// Resize vertices and normals to the correct number of vertices.
  void resizeToNumberOfVertices(size_t new_size);
  /// Reserve space in the vertices and normals vectors.
  void reserveNumberOfVertices(size_t new_capacity);

  /// Size of the vertices vector.
  size_t size() const;
  /// Capacity (allocated size) of the vertices vector.
  size_t capacity() const;

  /// Resize colors/intensities such that:
  /// `colors.size()/intensities.size() == vertices.size()`
  void expandColorsToMatchVertices();

  // Copy mesh data to the CPU.
  std::vector<Vector3f> getVertexVectorOnCPU() const;
  std::vector<Vector3f> getNormalVectorOnCPU() const;
  std::vector<int> getTriangleVectorOnCPU() const;
  std::vector<Color> getColorVectorOnCPU() const;

  /// Note(alexmillane): Memory type ignored, MeshBlocks live in CPU memory.
  static Ptr allocate(MemoryType memory_type);
};

// Clone to define a CertifiedMeshBlock. LayerCake must have unique types... :(
struct CertifiedMeshBlock {
  typedef std::shared_ptr<CertifiedMeshBlock> Ptr;
  typedef std::shared_ptr<const CertifiedMeshBlock> ConstPtr;

  /// Create a mesh block of the specified memory type.
  CertifiedMeshBlock(MemoryType memory_type = MemoryType::kDevice);

  /// "Clone" copy constructor, with the possibility to change device type.
  CertifiedMeshBlock(const CertifiedMeshBlock& mesh_block);
  /// "Clone" to a different memory type.
  CertifiedMeshBlock(const CertifiedMeshBlock& mesh_block,
                     MemoryType memory_type);

  // Mesh Data
  // These unified vectors contain the mesh data for this block. Note that
  // Colors and/or intensities are optional. The "triangles" vector is a vector
  // of indices into the vertices vector. Triplets of consecutive elements form
  // triangles with the indexed vertices as their corners.
  unified_vector<Vector3f> vertices;
  unified_vector<Vector3f> normals;
  unified_vector<Color> colors;
  unified_vector<int> triangles;

  /// Clear all data within the mesh block.
  void clear();

  /// Resize vertices and normals to the correct number of vertices.
  void resizeToNumberOfVertices(size_t new_size);
  /// Reserve space in the vertices and normals vectors.
  void reserveNumberOfVertices(size_t new_capacity);

  /// Size of the vertices vector.
  size_t size() const;
  /// Capacity (allocated size) of the vertices vector.
  size_t capacity() const;

  /// Resize colors/intensities such that:
  /// `colors.size()/intensities.size() == vertices.size()`
  void expandColorsToMatchVertices();

  // Copy mesh data to the CPU.
  std::vector<Vector3f> getVertexVectorOnCPU() const;
  std::vector<Vector3f> getNormalVectorOnCPU() const;
  std::vector<int> getTriangleVectorOnCPU() const;
  std::vector<Color> getColorVectorOnCPU() const;

  /// Note(alexmillane): Memory type ignored, MeshBlocks live in CPU memory.
  static Ptr allocate(MemoryType memory_type);
};

/// Helper struct for mesh blocks on CUDA.
/// NOTE: We need this because we can't pass MeshBlock to kernel functions
/// because of the presence of unified_vector members.
struct CudaMeshBlock {
  CudaMeshBlock() = default;
  CudaMeshBlock(MeshBlock* block);
  CudaMeshBlock(CertifiedMeshBlock* block);

  Vector3f* vertices;
  Vector3f* normals;
  int* triangles;
  Color* colors;
  int vertices_size = 0;
  int triangles_size = 0;
};

/// Specialization of BlockLayer clone just for MeshBlocks.
template <>
inline BlockLayer<MeshBlock>::BlockLayer(const BlockLayer& other,
                                         MemoryType memory_type)
    : BlockLayer(other.block_size_, memory_type) {
  LOG(INFO) << "Deep copy of Mesh BlockLayer containing "
            << other.numAllocatedBlocks() << " blocks.";
  // Re-create all the blocks.
  std::vector<Index3D> all_block_indices = other.getAllBlockIndices();

  // Iterate over all blocks, clonin'.
  for (const Index3D& block_index : all_block_indices) {
    typename MeshBlock::ConstPtr block = other.getBlockAtIndex(block_index);
    if (block == nullptr) {
      continue;
    }
    blocks_.emplace(block_index,
                    std::make_shared<MeshBlock>(*block, memory_type));
  }
}

/// Specialization of BlockLayer clone just for CertifiedMeshBlocks.
template <>
inline BlockLayer<CertifiedMeshBlock>::BlockLayer(const BlockLayer& other,
                                                  MemoryType memory_type)
    : BlockLayer(other.block_size_, memory_type) {
  LOG(INFO) << "Deep copy of Mesh BlockLayer containing "
            << other.numAllocatedBlocks() << " blocks.";
  // Re-create all the blocks.
  std::vector<Index3D> all_block_indices = other.getAllBlockIndices();

  // Iterate over all blocks, clonin'.
  for (const Index3D& block_index : all_block_indices) {
    typename CertifiedMeshBlock::ConstPtr block =
        other.getBlockAtIndex(block_index);
    if (block == nullptr) {
      continue;
    }
    blocks_.emplace(block_index,
                    std::make_shared<CertifiedMeshBlock>(*block, memory_type));
  }
}

}  // namespace nvblox