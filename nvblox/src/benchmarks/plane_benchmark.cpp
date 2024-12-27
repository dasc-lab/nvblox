#include <iostream>

#include "nvblox/core/internal/warmup_cuda.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/mesh/mesh_block.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

class PlaneBenchmark {
 public:
  PlaneBenchmark();

  void runBenchmark(const std::string& csv_output_path = "");
  bool outputMesh(const std::string& ply_output_path);
  bool checkTSDF();

 private:
  // Settings. Do not modify or the benchmark isn't comparable.
  static constexpr float kVoxelSize = 0.05;
  static constexpr float kBlockSize =
      VoxelBlock<TsdfVoxel>::kVoxelsPerSide * kVoxelSize;
  static constexpr int kNumTrajectoryPoints = 100;
  static constexpr float kMaxEnvironmentDimension = 10.0f;
  static constexpr float kTruncationDistanceVox = 2;
  static constexpr float kTruncationDistanceMeters =
      kTruncationDistanceVox * kVoxelSize;
  static constexpr float kDistanceErrorTolerance = kTruncationDistanceMeters;
  static constexpr float kMinWeight = 1.0;

  // Actual layers.
  TsdfLayer tsdf_layer_;
  EsdfLayer esdf_layer_;
  MeshLayer mesh_layer_;

  // Ground Truth Layers
  TsdfLayer gt_tsdf_layer_;

  // Simulated camera.
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;

  // Scene is bounded to the dimensions above.
  primitives::Scene scene;
};

PlaneBenchmark::PlaneBenchmark()
    : tsdf_layer_(kVoxelSize, MemoryType::kUnified),
      esdf_layer_(kVoxelSize, MemoryType::kUnified),
      mesh_layer_(kBlockSize, MemoryType::kUnified),
      gt_tsdf_layer_(kVoxelSize, MemoryType::kUnified),
      camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

// C++ <17 requires declaring static constexpr variables
// In C++17 this is no longer required, as static constexpr also implies inline
constexpr float PlaneBenchmark::kVoxelSize;
constexpr float PlaneBenchmark::kBlockSize;
constexpr int PlaneBenchmark::kNumTrajectoryPoints;
constexpr float PlaneBenchmark::kMaxEnvironmentDimension;

void PlaneBenchmark::runBenchmark(const std::string& csv_output_path) {
  // Create an integrator with default settings.
  ProjectiveTsdfIntegrator integrator;
  MeshIntegrator mesh_integrator;
  EsdfIntegrator esdf_integrator;
  esdf_integrator.max_distance_m(4.0f);
  integrator.truncation_distance_vox(kTruncationDistanceVox);

  // create a bounded scene
  scene.aabb() = AxisAlignedBoundingBox(
      Vector3f(-kMaxEnvironmentDimension, -kMaxEnvironmentDimension,
               -kMaxEnvironmentDimension),
      Vector3f(kMaxEnvironmentDimension, kMaxEnvironmentDimension,
               kMaxEnvironmentDimension));

  // Add a plane in the +z aixs (since the camera sees in the +z axis)
  scene.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(0.0f, 0.0f, 5.0f), Vector3f(0.0f, 0.0f, 1.0f)));

  scene.addPrimitive(std::make_unique<primitives::Cube>(
      Vector3f(0.0f, 0.0f, 3.0f), Vector3f(1.0f, 1.0f, 1.0f)));

  // Create a depth frame. We share this memory buffer for the entire
  // trajectory.
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);

  for (size_t i = 0; i < kNumTrajectoryPoints; i++) {
    // Construct a transform from camera to scene with this.
    const float x = 5.0f * i / kNumTrajectoryPoints;
    Transform T_S_C = Transform::Identity();
    T_S_C.matrix()(0, 3) = x;

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

    // Integrate the mesh.
    if (false) {
      timing::Timer mesh_timer("benchmark/integrate_mesh");
      mesh_integrator.integrateBlocksGPU(tsdf_layer_, updated_blocks,
                                         &mesh_layer_);
    }

    // Integrate the ESDF.
    if (false) {
      timing::Timer esdf_timer("benchmark/integrate_esdf");
      esdf_integrator.integrateBlocks(tsdf_layer_, updated_blocks,
                                      &esdf_layer_);
    }
  }

  // generate the mesh
  {
    timing::Timer mesh_timer("benchmark/integrate_mesh");
    mesh_integrator.integrateBlocksCPU(
        tsdf_layer_, tsdf_layer_.getAllBlockIndices(), &mesh_layer_);
  }
}

bool PlaneBenchmark::checkTSDF() {
  // generate the ground-truth TSDF

  scene.generateLayerFromScene(kTruncationDistanceMeters, &gt_tsdf_layer_);
  LOG(INFO) << "Generated GT TSDF";

  // now loop through and check the diff between the tsdf and the gt tsdf

  // Now do some checks...
  // Check every voxel in the map.
  int total_num_voxels = 0;
  int num_voxel_big_error = 0;
  float max_error = -10000.0f;
  float min_error = 10000.0f;
  auto lambda = [&](const Index3D& block_index, const Index3D& voxel_index,
                    const TsdfVoxel* voxel) {
    if (voxel->weight >= kMinWeight) {
      // Get the corresponding point from the GT layer.
      const TsdfVoxel* gt_voxel = getVoxelAtBlockAndVoxelIndex<TsdfVoxel>(
          gt_tsdf_layer_, block_index, voxel_index);
      if (gt_voxel != nullptr) {
        float error = voxel->distance - gt_voxel->distance;

        if (std::fabs(error) > kDistanceErrorTolerance) {
          num_voxel_big_error++;
        }

        min_error = std::min(min_error, error);
        max_error = std::max(max_error, error);
        total_num_voxels++;
      }
    }
  };
  callFunctionOnAllVoxels<TsdfVoxel>(tsdf_layer_, lambda);
  float percent_large_error = static_cast<float>(num_voxel_big_error) /
                              static_cast<float>(total_num_voxels) * 100.0f;
  std::cout << " num_voxel_big_error: " << num_voxel_big_error << std::endl;
  std::cout << " total_num_voxels: " << total_num_voxels << std::endl;
  std::cout << " percent_large_error: " << percent_large_error << std::endl;
  std::cout << " min_error: " << min_error << std::endl;
  std::cout << " max_error: " << max_error << std::endl;

  return true;
}

bool PlaneBenchmark::outputMesh(const std::string& ply_output_path) {
  timing::Timer timer_write("mesh/write");
  return io::outputMeshLayerToPly(mesh_layer_, ply_output_path);
}

}  // namespace nvblox

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  nvblox::warmupCuda();

  std::string output_mesh_path = "";
  if (argc >= 2) {
    output_mesh_path = argv[1];
  }

  nvblox::PlaneBenchmark benchmark;
  benchmark.runBenchmark("");

  if (!output_mesh_path.empty()) {
    benchmark.outputMesh(output_mesh_path);
  }

  // check the tsdf
  benchmark.checkTSDF();

  // std::cout << nvblox::timing::Timing::Print();

  return 0;
}
