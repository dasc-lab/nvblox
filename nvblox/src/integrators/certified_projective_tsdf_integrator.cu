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
#include <nvblox/integrators/certified_projective_tsdf_integrator.h>
#include <nvblox/integrators/projective_tsdf_integrator.h>

#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/integrators/internal/cuda/impl/projective_integrator_impl.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/weighting_function.h"

namespace nvblox {

struct CertifiedUpdateTsdfVoxelFunctor {
  CertifiedUpdateTsdfVoxelFunctor() {}

  // Vector3f p_voxel_C, float depth, TsdfVoxel* voxel_ptr
  __device__ bool operator()(const float surface_depth_measured_,
                             const float voxel_depth_m, TsdfVoxel* voxel_ptr) {

    // Filter out invalid returns
    float surface_depth_measured = surface_depth_measured_;
    if (surface_depth_measured_ <= min_distance_m_) {
      surface_depth_measured = 7.0; // TODO(rgg): make this a parameter
    }
    // Get the distance between the voxel we're updating the surface.
    // Note that the distance is the projective distance, i.e. the distance
    // along the ray.
    const float voxel_to_surface_distance =
        surface_depth_measured - voxel_depth_m;
    // If we're behind the negative truncation distance, just continue.
    if (voxel_to_surface_distance < -truncation_distance_m_) {
      return false;
    }

    // Read CURRENT voxel values (from global GPU memory)
    const float voxel_distance_current = voxel_ptr->distance;
    // Fuse without using the weights to do en exponential moving average.

    // TODO(rgg): examine whether there is a more efficient way to mark observed
    // voxels? Weight update is needed because it also marks the voxel as
    // observed.

    // Get the weight of this observation from the sensor model.
    const float measurement_weight = weighting_function_(
        surface_depth_measured, voxel_depth_m, truncation_distance_m_);
    // TODO(rgg): remove magic number here
    const float voxel_weight_current = voxel_ptr->weight;
    const float weight =
        measurement_weight +
        0.09;  // 0.001 is the min threshold for observability, 0.1 for "softly"
              //  observed. Adding ~0.06 here seems like a reasonable
              //  compromise, but it'd be better to design a weighting function
              //  that does what we want it to.
    // const float weight =
        // fmin(measurement_weight + voxel_weight_current, max_weight_);
    // Fuse
    float fused_distance = voxel_to_surface_distance;
    // float fused_distance = (voxel_to_surface_distance * measurement_weight +
    //                         voxel_distance_current * voxel_weight_current) /
    //                        (measurement_weight + voxel_weight_current);

    // Clip
    if (fused_distance > 0.0f) {
      fused_distance = fmin(truncation_distance_m_, fused_distance);
    } else {
      fused_distance = fmax(-truncation_distance_m_, fused_distance);
    }
    voxel_ptr->weight = weight;
    // Write NEW voxel values (to global GPU memory)
    voxel_ptr->distance = fused_distance;
    return true;
  }

  float truncation_distance_m_ = 0.2f;
  float min_distance_m_ = 0.10f;  // Minimum distance to consider a return to be valid
  float max_weight_ = 100.0f;

  // TODO(rgg): update this with a new weighting function type
  WeightingFunction weighting_function_ =
      WeightingFunction(WeightingFunctionType::kConstantWeight);
};

CertifiedProjectiveTsdfIntegrator::CertifiedProjectiveTsdfIntegrator()
    : ProjectiveIntegrator<TsdfVoxel>() {
  update_functor_host_ptr_ =
      make_unified<CertifiedUpdateTsdfVoxelFunctor>(MemoryType::kHost);
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

CertifiedProjectiveTsdfIntegrator::~CertifiedProjectiveTsdfIntegrator() {
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

unified_ptr<CertifiedUpdateTsdfVoxelFunctor>
CertifiedProjectiveTsdfIntegrator::getTsdfUpdateFunctorOnDevice(
    float voxel_size) {
  // Set the update function params
  // NOTE(alex.millane): We do this with every frame integration to avoid
  // bug-prone logic for detecting when params have changed etc.
  update_functor_host_ptr_->max_weight_ = max_weight();
  update_functor_host_ptr_->truncation_distance_m_ =
      get_truncation_distance_m(voxel_size);
  update_functor_host_ptr_->weighting_function_ =
      WeightingFunction(weighting_function_type_);
  // Transfer to the device
  return update_functor_host_ptr_.clone(MemoryType::kDevice);
}

void CertifiedProjectiveTsdfIntegrator::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    TsdfLayer* layer, std::vector<Index3D>* updated_blocks) {
  // Get the update functor on the device
  unified_ptr<CertifiedUpdateTsdfVoxelFunctor> update_functor_device_ptr =
      getTsdfUpdateFunctorOnDevice(layer->voxel_size());
  // Integrate
  ProjectiveIntegrator<TsdfVoxel>::integrateFrame(
      depth_frame, T_L_C, camera,
      update_functor_host_ptr_.clone(MemoryType::kDevice).get(), layer,
      updated_blocks);
}

void CertifiedProjectiveTsdfIntegrator::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Lidar& lidar,
    TsdfLayer* layer, std::vector<Index3D>* updated_blocks) {
  // Get the update functor on the device
  unified_ptr<CertifiedUpdateTsdfVoxelFunctor> update_functor_device_ptr =
      getTsdfUpdateFunctorOnDevice(layer->voxel_size());
  // Integrate
  ProjectiveIntegrator<TsdfVoxel>::integrateFrame(
      depth_frame, T_L_C, lidar, update_functor_device_ptr.get(), layer,
      updated_blocks);
}

float CertifiedProjectiveTsdfIntegrator::max_weight() const {
  return max_weight_;
}

void CertifiedProjectiveTsdfIntegrator::max_weight(float max_weight) {
  CHECK_GT(max_weight, 0.0f);
  max_weight_ = max_weight;
}

WeightingFunctionType
CertifiedProjectiveTsdfIntegrator::weighting_function_type() const {
  return weighting_function_type_;
}

void CertifiedProjectiveTsdfIntegrator::weighting_function_type(
    WeightingFunctionType weighting_function_type) {
  weighting_function_type_ = weighting_function_type;
}

float CertifiedProjectiveTsdfIntegrator::marked_unobserved_voxels_distance_m()
    const {
  return marked_unobserved_voxels_distance_m_;
}

void CertifiedProjectiveTsdfIntegrator::marked_unobserved_voxels_distance_m(
    float marked_unobserved_voxels_distance_m) {
  marked_unobserved_voxels_distance_m_ = marked_unobserved_voxels_distance_m;
}

float CertifiedProjectiveTsdfIntegrator::marked_unobserved_voxels_weight()
    const {
  return marked_unobserved_voxels_weight_;
}

void CertifiedProjectiveTsdfIntegrator::marked_unobserved_voxels_weight(
    float marked_unobserved_voxels_weight) {
  marked_unobserved_voxels_weight_ = marked_unobserved_voxels_weight;
}

std::string CertifiedProjectiveTsdfIntegrator::getIntegratorName() const {
  return "certified_tsdf";
}

// Call with:
// - One threadBlock per VoxelBlock
// - 8x8x8 threads per threadBlock
__global__ void setUnobservedVoxelsCertKernel(const TsdfVoxel voxel_value,
                                              TsdfBlock** tsdf_block_ptrs) {
  // Get the voxel addressed by this thread.
  TsdfBlock* tsdf_block = tsdf_block_ptrs[blockIdx.x];
  TsdfVoxel* tsdf_voxel =
      &tsdf_block->voxels[threadIdx.z][threadIdx.y][threadIdx.x];
  // If voxel not observed set it to the constant value input to the kernel.
  // TODO(rgg): examine whether there is a more efficient way to mark observed
  // voxels?
  constexpr float kMinObservedWeight = 0.001;
  if (tsdf_voxel->weight < kMinObservedWeight) {
    *tsdf_voxel = voxel_value;
  }
}

void CertifiedProjectiveTsdfIntegrator::markUnobservedFreeInsideRadius(
    const Vector3f& center, float radius, TsdfLayer* layer,
    std::vector<Index3D>* updated_blocks_ptr) {
  CHECK_NOTNULL(layer);
  CHECK_GT(radius, 0.0f);
  // First get blocks in AABB
  const Vector3f min = center.array() - radius;
  const Vector3f max = center.array() + radius;
  const AxisAlignedBoundingBox aabb(min, max);
  const std::vector<Index3D> blocks_touched_by_aabb =
      getBlockIndicesTouchedByBoundingBox(layer->block_size(), aabb);
  // Narrow to radius
  const std::vector<Index3D> blocks_inside_radius = getBlocksWithinRadius(
      blocks_touched_by_aabb, layer->block_size(), center, radius);
  // Allocate (if they're not already);
  std::for_each(
      blocks_inside_radius.begin(), blocks_inside_radius.end(),
      [layer](const Index3D& idx) { layer->allocateBlockAtIndex(idx); });

  // TsdfBlock pointers to GPU
  const std::vector<TsdfBlock*> block_ptrs_host =
      getBlockPtrsFromIndices(blocks_inside_radius, layer);
  device_vector<TsdfBlock*> block_ptrs_device(block_ptrs_host);

  // The value given to "observed" voxels
  constexpr float kSlightlyObservedVoxelWeight = 0.1;
  const TsdfVoxel slightly_observed_tsdf_voxel{
      .distance = get_truncation_distance_m(layer->voxel_size()),
      .weight = kSlightlyObservedVoxelWeight};

  // Kernel launch
  const int num_thread_blocks = block_ptrs_device.size();
  constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;
  const dim3 num_threads_per_block(kVoxelsPerSide, kVoxelsPerSide,
                                   kVoxelsPerSide);
  setUnobservedVoxelsCertKernel<<<num_thread_blocks, num_threads_per_block, 0,
                                  integration_stream_>>>(
      slightly_observed_tsdf_voxel, block_ptrs_device.data());
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaPeekAtLastError());

  // Return blocks affected
  if (updated_blocks_ptr != nullptr) {
    *updated_blocks_ptr = blocks_inside_radius;
  }
}

}  // namespace nvblox
