#pragma once

#include "nvblox/core/log_odds.h"
#include "nvblox/map/common_names.h"

namespace nvblox {

/// The SdfDeflationIntegrator class can be used to deflate (decrease the
/// distance to obstacles) an ESDF or TSDF layer.
/// This is useful for maintaining safety in the presence of visual odometry error. 
class TsdfDeflationIntegrator {
 public:
  TsdfDeflationIntegrator();
  ~TsdfDeflationIntegrator();

  /// Deflate the occupancy grid.
  /// Does not affect unallocated blocks.
  /// @param layer_ptr The occupancy layer to deflate
  /// @param amount The amount to deflate by (typically positive)
  void deflate(VoxelBlockLayer<TsdfVoxel>* layer_ptr, float decrement);

  // Minimum value to decay to. Smaller values allowed if already present in SDF.
  float min_distance = -0.5;

  // Whether to deallocate blocks that are fully deflated (could be well within an obstacle)
  bool deallocate_fully_deflated_blocks = true;

 private:
  void deflateDistance(TsdfLayer* layer_ptr, float decrement);
  void deallocateFullyDeflatedBlocks(TsdfLayer* layer_ptr);
  // Internal buffers
  host_vector<TsdfBlock*> allocated_block_ptrs_host_;
  device_vector<TsdfBlock*> allocated_block_ptrs_device_;
  device_vector<bool> block_fully_deflated_device_;
  host_vector<bool> block_fully_deflated_host_;

  // CUDA stream to process ingration on
  cudaStream_t integration_stream_;
};

}  // namespace nvblox