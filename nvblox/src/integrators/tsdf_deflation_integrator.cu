#include <nvblox/integrators/tsdf_deflation_integrator.h>

#include "nvblox/integrators/internal/integrators_common.h"

#include <Eigen/Eigenvalues>

namespace nvblox {

// get the largest eigenvalue of a symmetric positive definite matrix. returns negative of the solver status if error.
__device__ float eigmax_3x3(const Eigen::Matrix3f & M) {

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver;
  solver.computeDirect(M);

  if (solver.info() == Eigen::Success){
    return solver.eigenvalues().maxCoeff();
  }
  else {
    return -float(solver.info());
  }

}


__device__  Eigen::Matrix<float, 3, 6> getSE3ActionJacobian(
                                     const Transform& T, const Vector3f& p) {
  // get the rotation matrix
  Eigen::Matrix<float, 3, 3> R = T.linear();

  // get a skew symmetric matrix from p
  Eigen::Matrix3f S;
  S << 0, -p.z(), p.y(), p.z(), 0, -p.x(), -p.y(), p.x(), 0;

  Eigen::Matrix<float, 3, 6> J;
  J.block(0, 0, 3, 3) = R;
  J.block(0, 3, 3, 3) = (-R * S);

  return J;
}

__device__ float getDecrement( const Transform & T_Ck_Ckm1, const Vector3f& p_camera, const TransformCovariance & Sigma, const float n_std){

  // construct the jacobian
  Eigen::Matrix<float, 3, 6> J = getSE3ActionJacobian(T_Ck_Ckm1, p_camera);

  // get the covariance of the point
  Eigen::Matrix<float, 3,3> Sigma_p = J * (Sigma) * J.transpose();

  // get the largest eigenvalue
  float max_eigval = eigmax_3x3(Sigma_p);

  if (max_eigval > 0)
  {
    float decrement = n_std * std::sqrt(max_eigval);
    return decrement;
  }
  else
  {
    // SOMETHING WENT WRONG!!
    float decrement = 100.0; // delete it all....
    return decrement;
  }
}


// @param T_Ck_Ckm1: Transform from previous camera frame to current camera
// frame
// @param Sigma: covariance of transform
__global__ void deflateDistanceKernel(
    CertifiedTsdfBlock** block_ptrs, Index3D* block_indices,
    const Transform* T_Ck_L, const Transform* T_Ck_Ckm1,
    const TransformCovariance* Sigma, const float n_std, const float block_size,
    const float min_distance, bool* is_block_fully_deflated, 
    float* decrement_range) {
  // A single thread in each block initializes the output to true
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    is_block_fully_deflated[blockIdx.x] = true;
    decrement_range[0] = 100.0;
    decrement_range[1] = -100.0;
  }
  __syncthreads();

  CertifiedTsdfVoxel* voxel_ptr =
      &(block_ptrs[blockIdx.x]->voxels[threadIdx.x][threadIdx.y][threadIdx.z]);

  // Check if voxel is already deflated and skip if so
  constexpr float kMinWeight = 1.0e-4;
  if ((voxel_ptr->distance < min_distance) || (voxel_ptr->weight <= kMinWeight)) {
    voxel_ptr->weight = 0.0f;  // Ensure weight is set to 0 even if there was a
                               // race condition earlier
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
  // This is just for kernel performance, as we don't do anything with the
  // information of whether or not the voxel has been updated for now.
  constexpr float kSlightlyObservedVoxelWeight = 0.01;
  if (voxel_ptr->weight < kSlightlyObservedVoxelWeight) {
    return;
  }

  // Get xyz coordinates of voxel in global frame
  const Index3D block_index = block_indices[blockIdx.x];
  const Index3D voxel_index(threadIdx.x, threadIdx.y, threadIdx.z);
  Vector3f p = getCenterPostionFromBlockIndexAndVoxelIndex(
      block_size, block_index, voxel_index);

  // convert the position vector to the camera frame
  Vector3f p_camera = (*T_Ck_L) * p;

  float decrement = getDecrement(*T_Ck_Ckm1, p_camera, *Sigma, n_std);


  // keep track of the decrement range
  decrement_range[0] = std::min(decrement_range[0], decrement);
  decrement_range[1] = std::max(decrement_range[1], decrement);

  // apply the decrement, but also to the correction, so that the estimated
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


TsdfDeflationIntegrator::TsdfDeflationIntegrator(bool deallocate_fully_deflated_blocks)
: deallocate_fully_deflated_blocks_(deallocate_fully_deflated_blocks)
 {
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

TsdfDeflationIntegrator::~TsdfDeflationIntegrator() {
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void TsdfDeflationIntegrator::deflate(
    VoxelBlockLayer<CertifiedTsdfVoxel>* layer_ptr, const Transform& T_L_C,
    const Transform& T_Ck_Ckm1, const TransformCovariance& Sigma,
    const float n_std) {
  CHECK_NOTNULL(layer_ptr);
  if (layer_ptr->numAllocatedBlocks() == 0) {
    // Empty layer, nothing to do here.
    return;
  }
  deflateDistance(layer_ptr, T_L_C, T_Ck_Ckm1, Sigma, n_std);

  if (deallocate_fully_deflated_blocks_) {
    deallocateFullyDeflatedBlocks(layer_ptr);
  }
}


// Calls deflation kernel that depends on the incremental pose
// point location, and transform covariance.
void TsdfDeflationIntegrator::deflateDistance(CertifiedTsdfLayer* layer_ptr,
                                              const Transform& T_L_C,
                                              const Transform& T_Ck_Ckm1,
                                              const TransformCovariance& Sigma,
                                              const float n_std) {
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

  // check that the decremement vectors are created.
  decrement_range_device_.resize(2);
  decrement_range_host_.resize(2);

  // Get the block pointers on host and copy them to device
  allocated_block_ptrs_host_ = layer_ptr->getAllBlockPointers();
  allocated_block_ptrs_device_ = allocated_block_ptrs_host_;

  // Get the block indices on host and copy them to device
  allocated_block_indices_host_ = layer_ptr->getAllBlockIndices();
  allocated_block_indices_device_ = allocated_block_indices_host_;

  // get the block size
  const float voxel_size = layer_ptr->voxel_size();
  const float block_size = voxelSizeToBlockSize(voxel_size);

  // get the inverse transform
  Transform T_Ck_L = T_L_C.inverse();

  // Kernel call - One ThreadBlock launched per VoxelBlock
  block_fully_deflated_device_.resize(num_allocated_blocks);
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_allocated_blocks;
  // Allocate memory on device
  Transform* d_T_Ck_L;
  Transform* d_T_Ck_Ckm1;
  TransformCovariance* d_Sigma;
  cudaMalloc(&d_T_Ck_L, sizeof(Transform));
  cudaMalloc(&d_T_Ck_Ckm1, sizeof(Transform));
  cudaMalloc(&d_Sigma, sizeof(TransformCovariance));

  cudaMemcpy(d_T_Ck_L, &T_Ck_L, sizeof(Transform), cudaMemcpyHostToDevice);
  cudaMemcpy(d_T_Ck_Ckm1, &T_Ck_Ckm1, sizeof(Transform),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sigma, &Sigma, sizeof(TransformCovariance),
             cudaMemcpyHostToDevice);

  deflateDistanceKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                          integration_stream_>>>(
      allocated_block_ptrs_device_.data(),     // NOLINT
      allocated_block_indices_device_.data(),  // NOLINT
      d_T_Ck_L, d_T_Ck_Ckm1, d_Sigma,          // NOLINT
      n_std, block_size, min_distance,         // NOLINT
      block_fully_deflated_device_.data(), 
      decrement_range_device_.data());    // NOLINT

  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back to host
  block_fully_deflated_host_ = block_fully_deflated_device_;
  decrement_range_host_ = decrement_range_device_;


  // Check if nothing is lost on the way
  CHECK(allocated_block_ptrs_host_.size() == num_allocated_blocks);
  CHECK(allocated_block_ptrs_device_.size() == num_allocated_blocks);
  CHECK(block_fully_deflated_device_.size() == num_allocated_blocks);
  CHECK(block_fully_deflated_host_.size() == num_allocated_blocks);

  LOG(INFO) << "Decrement Range: " << decrement_range_host_[0] << " " << decrement_range_host_[1];
}

void TsdfDeflationIntegrator::deallocateFullyDeflatedBlocks(
    CertifiedTsdfLayer* layer_ptr) {
  const int num_allocated_blocks = layer_ptr->numAllocatedBlocks();

  // Get the block indices on host
  std::vector<Index3D> allocated_block_indices_host =
      layer_ptr->getAllBlockIndices();

  // Find blocks that are fully decayed
  size_t num_cleared = 0;
  CHECK(num_allocated_blocks == allocated_block_indices_host.size());
  CHECK(num_allocated_blocks == block_fully_deflated_host_.size());
  for (size_t i = 0; i < num_allocated_blocks; i++) {
    if (block_fully_deflated_host_[i]) {
      layer_ptr->clearBlock(allocated_block_indices_host[i]);
      num_cleared += 1;
    }
  }
  LOG(INFO) << "CertifiedTSDF deallocated " << num_cleared << " blocks";
}

}  // namespace nvblox
