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
#include <fstream>
#include <iostream>

#include "nvblox/core/internal/warmup_cuda.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/certified_projective_tsdf_integrator.h"
#include "nvblox/integrators/tsdf_deflation_integrator.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/mesh/mesh_block.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/utils/timing.h"

#include <liegroups/liegroups.hpp>

namespace nvblox {

class PlaneEval {
 public:
  PlaneEval();

  void runBenchmark(const std::string& csv_output_path = "");
  bool outputMesh(const std::string& ply_output_path);
  bool outputCertifiedMesh(const std::string& ply_output_path);

 private:  
  // Voxel size for TSDF
  static constexpr float kVoxelSize = 0.05;
  static constexpr float kBlockSize =
      VoxelBlock<TsdfVoxel>::kVoxelsPerSide * kVoxelSize;

  // Set number of point in trajectory 
  static constexpr int kNumTrajectoryPoints = 70;

  // Maximum dimension of the environment
  static constexpr float kMaxEnvironmentDimension = 7.0f;

  // Actual layers.
  TsdfLayer tsdf_layer_;
  CertifiedTsdfLayer certified_tsdf_layer_;
  EsdfLayer esdf_layer_;
  MeshLayer mesh_layer_;
  CertifiedMeshLayer certified_mesh_layer_;

  // Simulated camera.
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;

  // Deflation parameters.
  float n_std_ = 1.0;
  float sigma_ = 0.0; // 1e-12

  LieGroups::Matrix6f odometry_error_cov_ = sigma_ * LieGroups::Matrix6f::Identity();
  Transform prev_T_S_C_{};
};

PlaneEval::PlaneEval()
    : tsdf_layer_(kVoxelSize, MemoryType::kDevice),
      certified_tsdf_layer_(kVoxelSize, MemoryType::kUnified),
      esdf_layer_(kVoxelSize, MemoryType::kUnified),
      mesh_layer_(kBlockSize, MemoryType::kUnified),
      certified_mesh_layer_(kBlockSize, MemoryType::kUnified),
      camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

// C++ <17 requires declaring static constexpr variables
// In C++17 this is no longer required, as static constexpr also implies inline
constexpr float PlaneEval::kVoxelSize;
constexpr float PlaneEval::kBlockSize;
constexpr int PlaneEval::kNumTrajectoryPoints;
constexpr float PlaneEval::kMaxEnvironmentDimension;

void PlaneEval::runBenchmark(const std::string& csv_output_path) {
  
  // Create an integrator with default settings.
  ProjectiveTsdfIntegrator integrator;
  CertifiedProjectiveTsdfIntegrator certified_tsdf_integrator;
  MeshIntegrator mesh_integrator;
  CertifiedMeshIntegrator certified_mesh_integrator;
  TsdfDeflationIntegrator tsdf_deflation_integrator;
  EsdfIntegrator esdf_integrator;
  esdf_integrator.max_distance_m(4.0f);

  // Define scene
  primitives::Scene scene;
  
  // Create a scene with a ground plane.
  // scene.addGroundLevel(0.0f);

  scene.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(1.0f, 0.0f, 0.0f), Vector3f(1.0f, 0.0f, 0.0f)
  ));
    scene.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(0.0f, 0.0f, 5.0f), Vector3f(0.0f, 0.0f, 1.0f)
  ));

  // Create a depth frame. We share this memory buffer for the entire
  // trajectory.
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);

  // Iterating over each transform
  for (size_t i = 0; i < kNumTrajectoryPoints; i++) {
 
    // Construct a transform from camera to scene with this.
    // Transform T_S_C_true;
    Transform T_S_C = Transform::Identity();
    T_S_C.matrix()(0, 3) = -10.0 * i / kNumTrajectoryPoints; // move -x by 10 m


    LOG(INFO) << T_S_C.matrix(); 

    // Generate a depth image of the scene.
    scene.generateDepthImageFromScene(
        camera_, T_S_C, 2 * kMaxEnvironmentDimension, &depth_frame);

    std::vector<Index3D> updated_blocks;
    // Integrate this depth image.
    {
      timing::Timer integration_timer("benchmark/integrate_tsdf");
      integrator.integrateFrame(depth_frame, T_S_C, camera_, &tsdf_layer_,
                                &updated_blocks);
    }

    // integrate the certfied tsdf layer
    {
      std::vector<Index3D> certified_updated_blocks;
      certified_tsdf_integrator.integrateFrame(
          depth_frame, T_S_C, camera_, &certified_tsdf_layer_,
          &certified_updated_blocks);
      // certified_esdf_blocks_to_update_.insert(certified_updated_blocks.begin(),
      //                                        certified_updated_blocks.end());

      Transform T_Ck_Ckm1 = T_S_C.inverse() * prev_T_S_C_;

      // TODO: run the deflation step too.
      tsdf_deflation_integrator.deflate(
        &certified_tsdf_layer_, T_S_C, T_Ck_Ckm1, odometry_error_cov_, n_std_); 

      prev_T_S_C_ = T_S_C;

      // const std::vector<Index3D> all_blocks = 
      //     certified_tsdf_layer_->getAllBlockIndices();

      // certified_esdf_blocks_to_update_.insert(all_blocks.begin(), all_blocks.end());

    }

    // // Integrate the mesh.
    // {
    //   timing::Timer mesh_timer("benchmark/integrate_mesh");
    //   mesh_integrator.integrateBlocksGPU(tsdf_layer_, updated_blocks,
    //                                      &mesh_layer_);
    // }

    // // Integrate the ESDF.
    // {
    //   timing::Timer esdf_timer("benchmark/integrate_esdf");
    //   esdf_integrator.integrateBlocks(tsdf_layer_, updated_blocks,
    //                                   &esdf_layer_);
    // }
  }

  // generate mesh from TSDF layer
  mesh_integrator.integrateBlocksGPU(
      tsdf_layer_, tsdf_layer_.getAllBlockIndices(),
      &mesh_layer_);

      // generate mesh from the certified TSDF layer
  certified_mesh_integrator.integrateBlocksGPU(
      certified_tsdf_layer_, certified_tsdf_layer_.getAllBlockIndices(),
      &certified_mesh_layer_);

}

bool PlaneEval::outputMesh(const std::string& ply_output_path) {
  timing::Timer timer_write("mesh/write");
  return io::outputMeshLayerToPly(mesh_layer_, ply_output_path);
}

bool PlaneEval::outputCertifiedMesh(const std::string& ply_output_path) {
  timing::Timer timer_write("certified_mesh/write");
  return io::outputMeshLayerToPly(certified_mesh_layer_, ply_output_path);
}

}  // namespace nvblox

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  nvblox::warmupCuda();

  // Explicitly state path of mesh and certified mesh
  std::string output_mesh_path = "./mesh.ply";
  std::string output_certified_mesh_path = "./certified_mesh.ply";

  // Block to modify paths for storing meshes
  if (argc >= 3) {
    output_mesh_path = argv[1];
    output_certified_mesh_path = argv[2];
  }

  nvblox::PlaneEval benchmark;
  benchmark.runBenchmark("");

  if (!output_mesh_path.empty()) {
    benchmark.outputMesh(output_mesh_path);
    benchmark.outputCertifiedMesh(output_certified_mesh_path);
  }

  std::cout << nvblox::timing::Timing::Print();

  return 0;
}
