#include <nvblox/integrators/tsdf_deflation_integrator.h>

#include "nvblox/integrators/internal/integrators_common.h"

namespace nvblox {

__global__ void deflateDistanceKernel(TsdfBlock** block_ptrs,
                                      const float decrement,
                                      const float min_distance,
                                      bool* is_block_fully_deflated) {
  // A single thread in each block initializes the output to true
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    is_block_fully_deflated[blockIdx.x] = true;
  }
  __syncthreads();

  TsdfVoxel* voxel_ptr =
      &(block_ptrs[blockIdx.x]->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

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
}

TsdfDeflationIntegrator::TsdfDeflationIntegrator() {
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

TsdfDeflationIntegrator::~TsdfDeflationIntegrator() {
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void TsdfDeflationIntegrator::deflate(VoxelBlockLayer<TsdfVoxel>* layer_ptr,
                                     float decrement) {
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

void TsdfDeflationIntegrator::deflateDistance(TsdfLayer* layer_ptr,
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
      decrement,
      min_distance,
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
    TsdfLayer* layer_ptr) {
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