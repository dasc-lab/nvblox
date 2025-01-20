#pragma once

#include "nvblox/core/log_odds.h"
#include "nvblox/map/common_names.h"

namespace nvblox {

/// The SdfDeflationIntegrator class can be used to deflate (decrease the
/// distance to obstacles) an ESDF or TSDF layer.
/// This is useful for maintaining safety in the presence of visual odometry
/// error.
class TsdfDeflationIntegrator {
 public:
  TsdfDeflationIntegrator(bool deallocate_fully_deflated_blocks = true);
  ~TsdfDeflationIntegrator();

  void deflate(VoxelBlockLayer<CertifiedTsdfVoxel>* layer_ptr,
               const Transform& T_L_C, 
               const Transform& T_Ck_Ckm1,
               const TransformCovariance& Sigma, 
               const float n_std=1.0);

  // Minimum value to decay to. Smaller values allowed if already present in
  // SDF.
  float min_distance = -0.10;

  // Setter
  void set_deallocate_fully_deflated_blocks(bool dealloc)
  {
    deallocate_fully_deflated_blocks_ = dealloc;
  }

  // Getter
  bool deallocate_fully_deflated_blocks() {
    return deallocate_fully_deflated_blocks_;
  }


 private:
  
  void deflateDistance(CertifiedTsdfLayer* layer_ptr,
                                              const Transform& T_L_C,
                                              const Transform& T_Ck_Ckm1,
                                              const TransformCovariance & Sigma, 
                                              const float n_std=1.0
                                              );
  void deallocateFullyDeflatedBlocks(CertifiedTsdfLayer* layer_ptr);


  // Internal buffers
  host_vector<CertifiedTsdfBlock*> allocated_block_ptrs_host_;
  device_vector<CertifiedTsdfBlock*> allocated_block_ptrs_device_;
  host_vector<Index3D> allocated_block_indices_host_;
  device_vector<Index3D> allocated_block_indices_device_;
  device_vector<bool> block_fully_deflated_device_;
  host_vector<bool> block_fully_deflated_host_;
  
  host_vector<float> decrement_range_host_;
  device_vector<float> decrement_range_device_;

  // CUDA stream to process ingration on
  cudaStream_t integration_stream_;
  
  // Whether to deallocate blocks that are fully deflated (could be well within
  // an obstacle)
  bool deallocate_fully_deflated_blocks_ = false;
};

}  // namespace nvblox
