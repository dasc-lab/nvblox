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
#include <cstdlib>
#include <ctime>

#include "nvblox/core/internal/warmup_cuda.h"
#include "nvblox/integrators/certified_esdf_integrator.h"
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
    bool outputGtMesh(const std::string& ply_output_path);
    bool outputMesh(const std::string& ply_output_path);
    bool outputTransformedMesh(const std::string& ply_output_path);
    bool outputCertifiedMesh(const std::string& ply_output_path);
    bool outputTransformedCertifiedMesh(const std::string& ply_output_path);
    bool outputEsdf(const std::string& ply_output_path);
    bool outputCertifiedEsdf(const std::string& ply_output_path);
    bool outputTsdf(const std::string& ply_output_path);
    bool outputCertifiedTsdf(const std::string& ply_output_path);
    bool saveTrajectory(const std::string& file_path);

  private:  
    // Voxel size for TSDF
    static constexpr float kVoxelSize = 0.05;
    static constexpr float kBlockSize =
        VoxelBlock<TsdfVoxel>::kVoxelsPerSide * kVoxelSize;

    // Set number of point in trajectory
    static constexpr int kNumTrajectoryPoints = 30;
    static constexpr float kDistanceTravelled = kNumTrajectoryPoints * 0.1;

    // Maximum dimension of the environment
    static constexpr float kMaxEnvironmentDimension = 7.0f;

    // Actual layers.
    TsdfLayer gt_tsdf_layer_;
    TsdfLayer tsdf_layer_;
    CertifiedTsdfLayer certified_tsdf_layer_;
    EsdfLayer esdf_layer_;
    CertifiedEsdfLayer certified_esdf_layer_;
    MeshLayer gt_mesh_layer_;
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
    float sigma_ = 1e-4; // 1e-12
    TransformCovariance Sigma_ = sigma_ * LieGroups::Matrix6f::Identity();

    Transform T_S_Ck_est_ = Transform::Identity();
    Transform T_S_Ck_ = Transform::Identity();

    std::vector<Transform> true_trajectory_; // save true trajectory
    std::vector<Transform> perturbed_trajectory_; // save the perturbed trajectory

    Mesh transformed_mesh_;
    Mesh transformed_certi_mesh_;

    // ESDF blocks
    Index3DSet esdf_blocks_to_update_;
    Index3DSet certified_esdf_blocks_to_update_;
  };

  PlaneEval::PlaneEval()
      : gt_tsdf_layer_(kVoxelSize, MemoryType::kUnified),
        tsdf_layer_(kVoxelSize, MemoryType::kUnified),
        certified_tsdf_layer_(kVoxelSize, MemoryType::kUnified),
        esdf_layer_(kVoxelSize, MemoryType::kUnified),
        certified_esdf_layer_(kVoxelSize, MemoryType::kUnified),
        gt_mesh_layer_(kBlockSize, MemoryType::kUnified),
        mesh_layer_(kBlockSize, MemoryType::kUnified),
        certified_mesh_layer_(kBlockSize, MemoryType::kUnified),
        camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

  // C++ <17 requires declaring static constexpr variables
  // In C++17 this is no longer required, as static constexpr also implies inline
  constexpr float PlaneEval::kVoxelSize;
  constexpr float PlaneEval::kBlockSize;
  constexpr int PlaneEval::kNumTrajectoryPoints;
  constexpr float PlaneEval::kMaxEnvironmentDimension;
  constexpr float PlaneEval::kDistanceTravelled;

  void PlaneEval::runBenchmark(const std::string& csv_output_path) {
    
    // Create an integrator with default settings.
    ProjectiveTsdfIntegrator integrator;
    CertifiedProjectiveTsdfIntegrator certified_tsdf_integrator;
    MeshIntegrator gt_mesh_integrator;
    MeshIntegrator mesh_integrator;
    CertifiedMeshIntegrator certified_mesh_integrator;
    TsdfDeflationIntegrator tsdf_deflation_integrator;
    EsdfIntegrator esdf_integrator;
    CertifiedEsdfIntegrator certified_esdf_integrator;
    esdf_integrator.max_distance_m(4.0f);
    esdf_integrator.min_weight(2.0f);
    certified_esdf_integrator.max_distance_m(4.0f);
    certified_esdf_integrator.min_weight(2.0f);

    // Define scene
    primitives::Scene scene;

    scene.aabb() = AxisAlignedBoundingBox(Vector3f(-38.0f, -5.0f, 0.0f),
                                          Vector3f(10.0f, 5.0f, 10.0f));

    // scene.addPrimitive(std::make_unique<primitives::Plane>(
    //     Vector3f(1.0f, 0.0f, 0.0f), Vector3f(1.0f, 0.0f, 0.0f)
    // ));
      scene.addPrimitive(std::make_unique<primitives::Plane>(
        Vector3f(0.0f, 0.0f, 3.0f), Vector3f(0.0f, 0.0f, 1.0f)
    ));

    // scene.addPrimitive(std::make_unique<primitives::Cube>(
    //   Vector3f(0.0f, 0.0f, 3.0f), Vector3f(2.0f, 1.0f, 1.0f)
    //   ));

    // extract ground truth mesh from the scene
    scene.generateLayerFromScene(10.0f, &gt_tsdf_layer_);

    // Create a depth frame. We share this memory buffer for the entire
    // trajectory.
    DepthImage depth_frame(camera_.height(), camera_.width(),
                          MemoryType::kUnified);

    // Initial transform from camera to scene.
    // Transform T_S_C_True;

    // Save True trajectories
    for  (size_t i = 0; i < kNumTrajectoryPoints; i++) {
      
      Transform T_S_C = Transform::Identity();

      // T_S_C = T_S_C * Eigen::AngleAxisf((-i * M_PI/4) / kNumTrajectoryPoints, Eigen::Vector3f::UnitY());

      // Calculate trajectory
      T_S_C.matrix()(0, 3) = -kDistanceTravelled * i / kNumTrajectoryPoints;

      // Save true trajectory
      true_trajectory_.push_back(T_S_C);

    }

    // initialize perturbed trajectory
    perturbed_trajectory_.push_back(true_trajectory_[0]);

    // Save Perturbed trajectories
    for (size_t i = 1; i < kNumTrajectoryPoints; i++) {

      // grab the true incremental pose
      Transform T_S_Bk = true_trajectory_[i];
      Transform T_S_Bkm1 = true_trajectory_[i - 1];
      Transform T_Bk_Bkm1 = T_S_Bk.inverse() * T_S_Bkm1;

      LieGroups::Vector6f tau = LieGroups::randn<float,6>(Sigma_);
      Transform T_pert = Transform(LieGroups::SE3::Exp(tau));
      Transform T_Bk_Bkm1_est = T_Bk_Bkm1 * T_pert;

      // grab the perturbed at the last frame
      Transform T_S_Bkm1_est = perturbed_trajectory_[i - 1];

      // compute the new estimated transform
      Transform T_S_Bk_est = T_S_Bkm1_est * T_Bk_Bkm1_est.inverse();

      perturbed_trajectory_.push_back(T_S_Bk_est);

    }

    // Iterating over each transform
    for (size_t i = 1; i < kNumTrajectoryPoints; i++) {
  
      Transform T_S_Ck = true_trajectory_[i];
      Transform T_S_Ck_est = perturbed_trajectory_[i];
      
      LOG(INFO) << T_S_Ck.matrix(); 

      // Generate a depth image of the scene.
      scene.generateDepthImageFromScene(
          camera_, T_S_Ck, 2 * kMaxEnvironmentDimension, &depth_frame);


      std::vector<Index3D> updated_blocks;
      // Integrate this depth image.
      {
        timing::Timer integration_timer("benchmark/integrate_tsdf");
        integrator.integrateFrame(depth_frame, T_S_Ck_est, camera_, &tsdf_layer_,
                                  &updated_blocks);

        esdf_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());

      }

      // integrate the certified tsdf layer
      {

        Transform T_Ck_Ckm1_est = T_S_Ck_est.inverse() * perturbed_trajectory_[i-1];

         // deflation of certified tsdf layers
        tsdf_deflation_integrator.deflate(
          &certified_tsdf_layer_, T_S_Ck_est, T_Ck_Ckm1_est, Sigma_, n_std_); 


        // certified tsdf layer integration
        std::vector<Index3D> certified_updated_blocks;
        certified_tsdf_integrator.integrateFrame(
            depth_frame, T_S_Ck_est, camera_, &certified_tsdf_layer_,
            &certified_updated_blocks);
        certified_esdf_blocks_to_update_.insert(certified_updated_blocks.begin(),
                                               certified_updated_blocks.end());


      
        const std::vector<Index3D> all_blocks = 
            certified_tsdf_layer_.getAllBlockIndices();

        certified_esdf_blocks_to_update_.insert(all_blocks.begin(), all_blocks.end());

      }

      // Update ESDF 
      {
        // // Convert the set of EsdfBlocks needing an update to a vector
        // std::vector<Index3D> esdf_blocks_to_update_vector(
        //   esdf_blocks_to_update_.begin(), esdf_blocks_to_update_.end());

        // std::vector<Index3D> certified_esdf_blocks_to_update_vector(
        //   certified_esdf_blocks_to_update_.begin(),
        //   certified_esdf_blocks_to_update_.end());

        // esdf_integrator.integrateBlocks(
        //   tsdf_layer_, esdf_blocks_to_update_vector,
        //   &esdf_layer_);

        // esdf_blocks_to_update_.clear();

        // certified_esdf_integrator.integrateBlocks(
        //   certified_tsdf_layer_, certified_esdf_blocks_to_update_vector,
        //   &certified_esdf_layer_);

        // certified_esdf_blocks_to_update_.clear();
      }

      T_S_Ck_est_ = perturbed_trajectory_[i];
      T_S_Ck_ = true_trajectory_[i];

    }

    // regenerate the esdf
    esdf_integrator.integrateLayer(tsdf_layer_, &esdf_layer_);
    certified_esdf_integrator.integrateLayer(certified_tsdf_layer_,
                                             &certified_esdf_layer_);

    // generate ground truth mesh
    gt_mesh_integrator.integrateBlocksGPU(
        gt_tsdf_layer_, gt_tsdf_layer_.getAllBlockIndices(),
        &gt_mesh_layer_);

    // generate mesh from TSDF layer
    mesh_integrator.integrateBlocksGPU(
        tsdf_layer_, tsdf_layer_.getAllBlockIndices(),
        &mesh_layer_);

    // generate mesh from the certified TSDF layer
    certified_mesh_integrator.integrateBlocksGPU(
        certified_tsdf_layer_, certified_tsdf_layer_.getAllBlockIndices(),
        &certified_mesh_layer_);

    // Scene estimated transformation
    // Transform T_S_S_est = T_S_Ckm1 * T_S_Ckm1_true.inverse() ;
    Transform T_S_S_est = T_S_Ck_ * T_S_Ck_est_.inverse();

    // generate transformed mesh
    Mesh mesh = Mesh::fromLayer(mesh_layer_);
    transformed_mesh_ = transform_mesh(mesh, T_S_S_est);

    // generate transformed certified mesh
    Mesh certi_mesh = Mesh::fromLayer(certified_mesh_layer_);
    transformed_certi_mesh_ = transform_mesh(certi_mesh, T_S_S_est);
  }


  bool PlaneEval::outputGtMesh(const std::string& ply_output_path) {
    timing::Timer timer_write("gt_mesh/write");
    return io::outputMeshLayerToPly(gt_mesh_layer_, ply_output_path);
  }

  bool PlaneEval::outputMesh(const std::string& ply_output_path) {
    timing::Timer timer_write("mesh/write");
    return io::outputMeshLayerToPly(mesh_layer_, ply_output_path);
  }

  bool PlaneEval::outputTransformedMesh(const std::string& ply_output_path) {
    timing::Timer timer_write("mesh/write");
    return io::outputMeshToPly(transformed_mesh_, ply_output_path);
  }

  bool PlaneEval::outputCertifiedMesh(const std::string& ply_output_path) {
    timing::Timer timer_write("certified_mesh/write");
    return io::outputMeshLayerToPly(certified_mesh_layer_, ply_output_path);
  }

  bool PlaneEval::outputTransformedCertifiedMesh(const std::string& ply_output_path) {
    timing::Timer timer_write("transformed_certified_mesh/write");
    return io::outputMeshToPly(transformed_certi_mesh_, ply_output_path);
  }

  bool PlaneEval::outputEsdf(const std::string& ply_output_path) {
    LOG(INFO) << "Writing ESDF";
    timing::Timer timer_write("esdf/write");
    return io::outputVoxelLayerToPly(esdf_layer_, ply_output_path);
  }

  bool PlaneEval::outputCertifiedEsdf(const std::string& ply_output_path) {
    LOG(INFO) << "Writing Certified ESDF";
    timing::Timer timer_write("certified_esdf/write");
    return io::outputVoxelLayerToPly(certified_esdf_layer_, ply_output_path);
  }

  bool PlaneEval::outputTsdf(const std::string& ply_output_path) {
    LOG(INFO) << "Writing TSDF";
    timing::Timer timer_write("tsdf/write");
    return io::outputVoxelLayerToPly(tsdf_layer_, ply_output_path);
  }

  bool PlaneEval::outputCertifiedTsdf(const std::string& ply_output_path) {
    LOG(INFO) << "Writing Certified TSDF";
    timing::Timer timer_write("certified_tsdf/write");
    return io::outputVoxelLayerToPly(certified_tsdf_layer_, ply_output_path);
  }

  bool PlaneEval::saveTrajectory(const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
      std::cerr << "Unable to open file: " << file_path << std::endl;
      return false;
    }

    for (const auto& transform : perturbed_trajectory_) {
      const auto& matrix = transform.matrix();
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          file << matrix(i, j);
          if (i != 3 || j != 3) {
            file << ",";
          }
        }
      }
      file << "\n";
    }
    return true;
  }

}  // namespace nvblox

int main(int argc, char* argv[]) {
  // srand(time(0));
  srand(0);

  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  nvblox::warmupCuda();

  // Explicitly state path of mesh and certified mesh
  std::string output_gt_mesh_path = "./gt_mesh.ply";
  std::string output_mesh_path = "./mesh.ply";
  std::string output_transformed_mesh_path = "./transformed_mesh.ply";
  std::string output_certified_mesh_path = "./certified_mesh.ply";
  std::string output_transformed_certified_mesh_path = "./transformed_certified_mesh.ply";
  std::string output_tsdf_path = "./tsdf.ply";
  std::string output_certified_tsdf_path = "./certified_tsdf.ply";
  std::string output_esdf_path = "./esdf.ply";
  std::string output_certified_esdf_path = "./certified_esdf.ply";
  std::string output_trajectory_path = "./trajectory.csv";


  // Block to modify paths for storing meshes
  if (argc >= 9) {
    output_gt_mesh_path = argv[1];
    output_mesh_path = argv[2];
    output_transformed_mesh_path = argv[3];
    output_certified_mesh_path = argv[4];
    output_transformed_certified_mesh_path = argv[5];
    output_esdf_path = argv[6];
    output_certified_esdf_path = argv[7];
    output_trajectory_path = argv[8];
  }

  nvblox::PlaneEval benchmark;
  benchmark.runBenchmark("");

  if (!output_mesh_path.empty()) {
    //  benchmark.outputGtMesh(output_gt_mesh_path);
    //  benchmark.outputMesh(output_mesh_path);
    //  benchmark.outputTransformedMesh(output_transformed_mesh_path);
    benchmark.outputCertifiedMesh(output_certified_mesh_path);
    //  benchmark.outputTransformedCertifiedMesh(output_transformed_certified_mesh_path);
    // benchmark.outputEsdf(output_esdf_path);
    benchmark.outputCertifiedEsdf(output_certified_esdf_path);
    //  benchmark.outputTsdf(output_tsdf_path);
    //  benchmark.outputCertifiedTsdf(output_certified_tsdf_path);
    //  benchmark.saveTrajectory(output_trajectory_path);
  }

  std::cout << nvblox::timing::Timing::Print();

  return 0;
}
