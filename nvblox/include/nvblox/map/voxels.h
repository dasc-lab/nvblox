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

#include <Eigen/Core>

#include "nvblox/core/color.h"

namespace nvblox {

/// A voxel storing TSDF (truncated signed distance field) values.
struct TsdfVoxel {
  /// Signed projective distance of the voxel from a surface.
  float distance = 0.0f;
  /// How many observations/how confident we are in this observation.
  float weight = 0.0f;
};

// Currently used in an identical manner to TSDF Voxel, but may change in the future.
// Need a separate type since LayerCake can only store one of each layer type currently.
struct CertifiedTsdfVoxel {
  /// Signed projective distance of the voxel from a surface.
  float distance = 0.0f;
  /// How many observations/how confident we are in this observation.
  float weight = 0.0f;
  // Certification Correction
  float correction = 0.0f;  // the correction applied by the deflation
                            // integrator due to imperfect state estimate
  // best_estimate_of_distance - correction = distance
};

/// Voxels that stores the distance and full direction to the nearest surface.
struct EsdfVoxel {
  // TODO(helen): optimize the memory layout here.
  /// Cached squared distance towards the parent.
  float squared_distance_vox = 0.0f;
  /// Direction towards the parent, *in units of voxels*.
  Eigen::Vector3i parent_direction = Eigen::Vector3i::Zero();
  /// Whether this voxel is inside the surface or not.
  bool is_inside = false;
  /// Whether this voxel has been observed.
  bool observed = false;
  /// Whether this voxel is a "site": i.e., near the zero-crossing and is
  /// eligible to be considered a parent.
  bool is_site = false;
};

/// Voxels that stores the distance and full direction to the nearest surface.
// Similar to CertifiedTsdfVoxel.
struct CertifiedEsdfVoxel {
  // TODO(helen): optimize the memory layout here.
  /// Cached squared distance towards the parent.
  float squared_distance_vox = 0.0f;
  /// Direction towards the parent, *in units of voxels*.
  Eigen::Vector3i parent_direction = Eigen::Vector3i::Zero();
  /// Whether this voxel is inside the surface or not.
  bool is_inside = false;
  /// Whether this voxel has been observed.
  bool observed = false;
  /// Whether this voxel is a "site": i.e., near the zero-crossing and is
  /// eligible to be considered a parent.
  bool is_site = false;
};

/// Voxel that stores the color near the surface.
struct ColorVoxel {
  /// The color!
  Color color = Color::Gray();
  /// How many observations/how confident we are in this observation.
  float weight = 0.0f;
};

struct OccupancyVoxel {
  float log_odds = 0.0f;
};

}  // namespace nvblox
