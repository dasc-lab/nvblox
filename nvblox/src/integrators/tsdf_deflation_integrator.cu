#include <nvblox/integrators/tsdf_deflation_integrator.h>

#include "nvblox/integrators/internal/integrators_common.h"

namespace nvblox {

// @param T_L_C: Transform from local frame to camera frame
// @param eps_R: Uncertainty in rotation Frobenius norm
// @param eps_t: Uncertainty in translation norm
// @param voxel_size: Size of a voxel [m]
// @param block_size: Size of a block [m]
// @param t_delta: incremental translation from frame k to k+1
__global__ void deflateDistanceKernel(
    CertifiedTsdfBlock** block_ptrs, Index3D* block_indices,
    const Transform* T_L_C, const float eps_R, const float eps_t,
    const float voxel_size, const float block_size, const Vector3f* t_delta,
    const float min_distance, bool* is_block_fully_deflated) {
  // A single thread in each block initializes the output to true
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    is_block_fully_deflated[blockIdx.x] = true;
  }
  __syncthreads();

  CertifiedTsdfVoxel* voxel_ptr =
      &(block_ptrs[blockIdx.x]->voxels[threadIdx.x][threadIdx.y][threadIdx.z]);

  // Check if voxel is already deflated and skip if so
  if (voxel_ptr->distance < min_distance) {
    voxel_ptr->weight = 0.0f;  // Ensure weight is set to 0 even if there was a race condition earlier
    return;
  } else {
    // If one voxel in a block is not deflated beyond the limit,
    // the block is not fully deflated.
    // NOTE: There could be more than one thread writing this value, but because
    // all of them write false it is no issue.
    is_block_fully_deflated[blockIdx.x] = false;
  }

  // Check that the certified tsdf voxel has been observed and updated before
  // actually deflating the voxel's value
  // This is just for kernel performance, as we don't do anything with the information
  // of whether or not the voxel has been updated for now.
  // TODO(rgg): don't perform ESDF update on voxels that have not been observed / have been fully deflated.
  constexpr float kSlightlyObservedVoxelWeight = 0.01;
  if (voxel_ptr->weight < kSlightlyObservedVoxelWeight) {
    return;
  }

  // Theorem 1
  // Get xyz coordinates of voxel in global frame
  const Index3D block_index = block_indices[blockIdx.x];
  const Index3D voxel_index(threadIdx.x, threadIdx.y, threadIdx.z);
  Vector3f p = getCenterPostionFromBlockIndexAndVoxelIndex(
      block_size, block_index, voxel_index);

  // get the decrement
  // d_new = d - eps_R*norm(R_M^Bk+1*p + t_m^Bk+1 - t_Bk^Bk+1) - eps_t
  float decrement = eps_R * (*T_L_C * p - *t_delta).norm() + eps_t;

  // apply the decrement, but only to the correction, so that the estimated
  // distance isnt affected
  voxel_ptr->distance -= decrement;
  voxel_ptr->correction += decrement;

  // If the decrement has completely deflated the voxel,
  // reset the weight so that we can re-observe it and not
  // treat it as "known obstacle" when it is really just unknown.
  if (voxel_ptr->distance <= min_distance) {
    voxel_ptr->weight = 0.0f;
  }
}

__global__ void deflateDistanceKernel(CertifiedTsdfBlock** block_ptrs,
                                      const float decrement,
                                      const float min_distance,
                                      bool* is_block_fully_deflated) {
  // A single thread in each block initializes the output to true
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    is_block_fully_deflated[blockIdx.x] = true;
  }
  __syncthreads();

  /// !!!!!
  /// WARNING DEV:: NOT BEING USED!!!!!!!!!!!
  /// !!!!!

  CertifiedTsdfVoxel* voxel_ptr =
      &(block_ptrs[blockIdx.x]->voxels[threadIdx.x][threadIdx.y][threadIdx.z]);

  // Check if voxel is already deflated and skip if so
  if (voxel_ptr->distance - decrement < min_distance) {
    return;
  } else {
    // If one voxel in a block is not deflated beyond the limit,
    // the block is not fully deflated.
    // NOTE: There could be more than one thread writing this value, but because
    // all of them write false it is no issue.
    is_block_fully_deflated[blockIdx.x] = false;
  }
  voxel_ptr->distance -= decrement;
  // Dev: increase correction, decrement the value so the "estimated distance"
  // is still the same.
  voxel_ptr->correction += decrement;

  /// !!!!!
  /// WARNING DEV:: NOT BEING USED!!!!!!!!!!!
  /// !!!!!

  // Dev: try weird things with the weights. kinda works. but not really.
  // constexpr float kWeightDecrementFactor = 0.999;
  // voxel_ptr->weight = fmax(0, (1 - kWeightDecrementFactor) *
  // voxel_ptr->weight);
}

TsdfDeflationIntegrator::TsdfDeflationIntegrator() {
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

TsdfDeflationIntegrator::~TsdfDeflationIntegrator() {
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void TsdfDeflationIntegrator::deflate(
    VoxelBlockLayer<CertifiedTsdfVoxel>* layer_ptr, float decrement) {
  CHECK_NOTNULL(layer_ptr);
  if (layer_ptr->numAllocatedBlocks() == 0) {
    // Empty layer, nothing to do here.
    return;
  }
  deflateDistance(layer_ptr, decrement);
  if (deallocate_fully_deflated_blocks) {
    deallocateFullyDeflatedBlocks(layer_ptr);
  }
}

void TsdfDeflationIntegrator::deflate(
    VoxelBlockLayer<CertifiedTsdfVoxel>* layer_ptr, const Transform& T_L_C,
    float eps_R, float eps_t, float voxel_size, const Vector3f& t_delta) {
  CHECK_NOTNULL(layer_ptr);
  if (layer_ptr->numAllocatedBlocks() == 0) {
    // Empty layer, nothing to do here.
    return;
  }
  deflateDistance(layer_ptr, T_L_C, eps_R, eps_t, voxel_size, t_delta);
  if (deallocate_fully_deflated_blocks) {
    deallocateFullyDeflatedBlocks(layer_ptr);
  }
}

// Calls deflation kernel that depends on R, t, point location, and norm ball
// errors.

void TsdfDeflationIntegrator::deflateDistance(CertifiedTsdfLayer* layer_ptr,
                                              const Transform& T_L_C,
                                              float eps_R, float eps_t,
                                              float voxel_size,
                                              const Vector3f& t_delta) {
  CHECK_NOTNULL(layer_ptr);
  const int num_allocated_blocks = layer_ptr->numAllocatedBlocks();

  // Expand the buffers when needed
  if (num_allocated_blocks > allocated_block_ptrs_host_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * num_allocated_blocks);
    allocated_block_ptrs_host_.reserve(new_size);
    allocated_block_ptrs_device_.reserve(new_size);
    allocated_block_indices_host_.reserve(new_size);
    allocated_block_indices_device_.reserve(new_size);
    block_fully_deflated_device_.reserve(new_size);
    block_fully_deflated_host_.reserve(new_size);
  }

  // Get the block pointers on host and copy them to device
  allocated_block_ptrs_host_ = layer_ptr->getAllBlockPointers();
  allocated_block_ptrs_device_ = allocated_block_ptrs_host_;

  // Get the block indices on host and copy them to device
  allocated_block_indices_host_ = layer_ptr->getAllBlockIndices();
  allocated_block_indices_device_ = allocated_block_indices_host_;

  // get the block size
  const float block_size = voxelSizeToBlockSize(voxel_size);

  // Kernel call - One ThreadBlock launched per VoxelBlock
  block_fully_deflated_device_.resize(num_allocated_blocks);
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_allocated_blocks;
  // Allocate memory on device
  Transform* d_T_L_C;
  Vector3f* d_t_delta;
  cudaMalloc(&d_T_L_C, sizeof(Transform));
  cudaMalloc(&d_t_delta, sizeof(Vector3f));
  cudaMemcpy(d_T_L_C, &T_L_C, sizeof(Transform), cudaMemcpyHostToDevice);
  cudaMemcpy(d_t_delta, &t_delta, sizeof(Vector3f), cudaMemcpyHostToDevice);
  deflateDistanceKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                          integration_stream_>>>(
      allocated_block_ptrs_device_.data(),     // NOLINT
      allocated_block_indices_device_.data(),  // NOLINT
      d_T_L_C, eps_R, eps_t, voxel_size, block_size, d_t_delta, min_distance,
      block_fully_deflated_device_.data());  // NOLINT
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back to host
  block_fully_deflated_host_ = block_fully_deflated_device_;

  // Check if nothing is lost on the way
  CHECK(allocated_block_ptrs_host_.size() == num_allocated_blocks);
  CHECK(allocated_block_ptrs_device_.size() == num_allocated_blocks);
  CHECK(block_fully_deflated_device_.size() == num_allocated_blocks);
  CHECK(block_fully_deflated_host_.size() == num_allocated_blocks);
}

void TsdfDeflationIntegrator::deflateDistance(CertifiedTsdfLayer* layer_ptr,
                                              float decrement) {
  CHECK_NOTNULL(layer_ptr);
  const int num_allocated_blocks = layer_ptr->numAllocatedBlocks();

  // Expand the buffers when needed
  if (num_allocated_blocks > allocated_block_ptrs_host_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * num_allocated_blocks);
    allocated_block_ptrs_host_.reserve(new_size);
    allocated_block_ptrs_device_.reserve(new_size);
    allocated_block_indices_host_.reserve(new_size);
    allocated_block_indices_device_.reserve(new_size);
    block_fully_deflated_device_.reserve(new_size);
    block_fully_deflated_host_.reserve(new_size);
  }

  // Get the block pointers on host and copy them to device
  allocated_block_ptrs_host_ = layer_ptr->getAllBlockPointers();
  allocated_block_ptrs_device_ = allocated_block_ptrs_host_;

  // Kernel call - One ThreadBlock launched per VoxelBlock
  block_fully_deflated_device_.resize(num_allocated_blocks);
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_allocated_blocks;
  deflateDistanceKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                          integration_stream_>>>(
      allocated_block_ptrs_device_.data(),  // NOLINT
      decrement, min_distance,
      block_fully_deflated_device_.data());  // NOLINT
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back to host
  block_fully_deflated_host_ = block_fully_deflated_device_;

  // Check if nothing is lost on the way
  CHECK(allocated_block_ptrs_host_.size() == num_allocated_blocks);
  CHECK(allocated_block_ptrs_device_.size() == num_allocated_blocks);
  CHECK(block_fully_deflated_device_.size() == num_allocated_blocks);
  CHECK(block_fully_deflated_host_.size() == num_allocated_blocks);
}

void TsdfDeflationIntegrator::deallocateFullyDeflatedBlocks(
    CertifiedTsdfLayer* layer_ptr) {
  const int num_allocated_blocks = layer_ptr->numAllocatedBlocks();

  // Get the block indices on host
  std::vector<Index3D> allocated_block_indices_host =
      layer_ptr->getAllBlockIndices();

  // Find blocks that are fully decayed
  CHECK(num_allocated_blocks == allocated_block_indices_host.size());
  CHECK(num_allocated_blocks == block_fully_deflated_host_.size());
  for (size_t i = 0; i < num_allocated_blocks; i++) {
    if (block_fully_deflated_host_[i]) {
      layer_ptr->clearBlock(allocated_block_indices_host[i]);
    }
  }
}

}  // namespace nvblox
