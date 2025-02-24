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

#include <string>

#include "nvblox/map/layer.h"
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {
namespace io {

// Combines mesh blocks in a layer into a single list of vertices, triangles,
// and normals.
template <typename MeshBlockType>
void combineMeshBlocks(const BlockLayer<MeshBlockType>& layer,
                       std::vector<Vector3f>* vertices_ptr,
                       std::vector<Vector3f>* normals_ptr,
                       std::vector<int>* triangles_ptr);

template <typename MeshBlockType>
bool outputMeshLayerToPly(const BlockLayer<MeshBlockType>& layer,
                          const std::string& filename);

template <typename MeshBlockType>
bool outputMeshLayerToPly(const BlockLayer<MeshBlockType>& layer,
                          const char* filename);

}  // namespace io
}  // namespace nvblox