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

#include "nvblox/utils/logging.h"

#include "nvblox/map/blox.h"

namespace nvblox {

__host__ __device__ inline float voxelSizeToBlockSize(const float voxel_size) {
  return voxel_size * VoxelBlock<bool>::kVoxelsPerSide;
}

__host__ __device__ inline float blockSizeToVoxelSize(const float block_size) {
  constexpr float kVoxelsPerSideInv = 1.0f / VoxelBlock<bool>::kVoxelsPerSide;
  return block_size * kVoxelsPerSideInv;
}

// TODO: this is badly done now for non-Eigen types.
/// Input position: metric (world) units with respect to the layer origin.
/// Output index: voxel index relative to the block origin.
Index3D getVoxelIndexFromPositionInLayer(const float block_size,
                                         const Vector3f& position) {
  const float voxel_size = blockSizeToVoxelSize(block_size);
  Index3D global_voxel_index =
      (position / voxel_size).array().floor().cast<int>();
  Index3D voxel_index = global_voxel_index.unaryExpr([&](int x) {
    return static_cast<int>(x % VoxelBlock<bool>::kVoxelsPerSide);
  });

  // Double-check that we get reasonable indices out.
  DCHECK((voxel_index.array() >= 0).all() &&
         (voxel_index.array() < VoxelBlock<bool>::kVoxelsPerSide).all());

  return voxel_index;
}

Index3D getBlockIndexFromPositionInLayer(const float block_size,
                                         const Vector3f& position) {
  Eigen::Vector3i index = (position / block_size).array().floor().cast<int>();
  return Index3D(index.x(), index.y(), index.z());
}

void getBlockAndVoxelIndexFromPositionInLayer(const float block_size,
                                              const Vector3f& position,
                                              Index3D* block_idx,
                                              Index3D* voxel_idx) {
  constexpr int kVoxelsPerSideMinusOne = VoxelBlock<bool>::kVoxelsPerSide - 1;
  const float voxel_size_inv = 1.0 / blockSizeToVoxelSize(block_size);
  *block_idx = (position / block_size).array().floor().cast<int>();
  *voxel_idx =
      ((position - block_size * block_idx->cast<float>()) * voxel_size_inv)
          .array()
          .cast<int>()
          .min(kVoxelsPerSideMinusOne);
}

Vector3f getPositionFromBlockIndexAndVoxelIndex(const float block_size,
                                                const Index3D& block_index,
                                                const Index3D& voxel_index) {
  const float voxel_size = blockSizeToVoxelSize(block_size);
  return Vector3f(block_size * block_index.cast<float>() +
                  voxel_size * voxel_index.cast<float>());
}

Vector3f getPositionFromBlockIndex(const float block_size,
                                   const Index3D& block_index) {
  // This is pretty trivial, huh.
  return Vector3f(block_size * block_index.cast<float>());
}

Vector3f getCenterPostionFromBlockIndexAndVoxelIndex(
    const float block_size, const Index3D& block_index,
    const Index3D& voxel_index) {
  constexpr float kHalfVoxelsPerSideInv =
      0.5f / VoxelBlock<bool>::kVoxelsPerSide;
  const float half_voxel_size = block_size * kHalfVoxelsPerSideInv;

  return getPositionFromBlockIndexAndVoxelIndex(block_size, block_index,
                                                voxel_index) +
         Vector3f(half_voxel_size, half_voxel_size, half_voxel_size);
}

std::vector<Index3D> getNeighborIndices(const Index3D& block_index) {
  std::vector<Index3D> neighbors;
  neighbors.reserve(26);
  for (int dx : {-1, 0, 1}) {
    for (int dy : {-1, 0, 1}) {
      for (int dz : {-1, 0, 1}) {
        if (dx != 0 || dy != 0 || dz != 0) {
          Index3D delta(dx, dy, dz);
          Index3D neighbor = block_index + delta;
          neighbors.emplace_back(neighbor);
        }
      }
    }
  }

  return neighbors;
}

//  Returns true if index is not in the vector indices.
bool is_not_in(const std::vector<Index3D>& indices, const Index3D& index) {
  // https://www.geeksforgeeks.org/check-if-vector-contains-given-element-in-cpp/
  auto it = std::find(indices.begin(), indices.end(), index);
  return it == indices.end();
}

}  // namespace nvblox
