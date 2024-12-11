
#include <iostream>

#include "nvblox/core/internal/warmup_cuda.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator_cpu.h"
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

using namespace nvblox;

primitives::Scene getSphereInBox() {
  // Scene is bounded to -5, -5, 0 to 5, 5, 5.
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, 0.0f),
                                        Vector3f(5.0f, 5.0f, 5.0f));
  // Create a scene with a ground plane and a sphere.
  scene.addGroundLevel(0.0f);
  scene.addCeiling(5.0f);
  scene.addPrimitive(
      std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 2.0f), 2.0f));
  // Add bounding planes at 5 meters. Basically makes it sphere in a box.
  scene.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);
  return scene;
}

class PlaneBenchmark {
 public:
  PlaneBenchmark()
      : layer_(voxel_size_m_, MemoryType::kUnified),
        camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

  void run();

  // Test layer
  constexpr static float voxel_size_m_ = 0.02;
  TsdfLayer layer_;

  // How much error we expect on the surface
  constexpr static float surface_reconstruction_allowable_distance_error_vox_ =
      2.0f;
  constexpr static float surface_reconstruction_allowable_distance_error_m_ =
      surface_reconstruction_allowable_distance_error_vox_ * voxel_size_m_;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;
};

void PlaneBenchmark::run() {
  constexpr float kTrajectoryRadius = 4.0f;
  constexpr float kTrajectoryHeight = 2.0f;
  constexpr int kNumTrajectoryPoints = 80;
  constexpr float kTruncationDistanceVox = 2;
  constexpr float kTruncationDistanceMeters =
      kTruncationDistanceVox * voxel_size_m_;
  // Maximum distance to consider for scene generation.
  constexpr float kMaxDist = 10.0;
  constexpr float kMinWeight = 1.0;

  // Tolerance for error.
  constexpr float kDistanceErrorTolerance = kTruncationDistanceMeters;

  // Get the ground truth SDF of a sphere in a box.
  primitives::Scene scene = getSphereInBox();
  TsdfLayer gt_tsdf_layer(voxel_size_m_, MemoryType::kUnified);
  scene.generateLayerFromScene(kTruncationDistanceMeters, &gt_tsdf_layer);
  // scene.generateLayerFromScene(5.0f, &gt_tsdf_layer); // this leads to wrong
  // results!!

  // Create an integrator.
  ProjectiveTsdfIntegrator integrator_gpu;
  integrator_gpu.truncation_distance_vox(kTruncationDistanceVox);

  // Simulate a trajectory of the requisite amount of points, on the circle
  // around the sphere.
  const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints);

  // Create a depth frame. We share this memory buffer for the entire
  // trajectory.
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);

  // Two layers, one for CPU integration and one for GPU integration
  // TsdfLayer layer_cpu(layer_.voxel_size(), MemoryType::kUnified);
  TsdfLayer tsdf_layer(layer_.voxel_size(), MemoryType::kUnified);

  for (size_t i = 0; i < kNumTrajectoryPoints; i++) {
    const float theta = radians_increment * i;
    // Convert polar to cartesian coordinates.
    Vector3f cartesian_coordinates(kTrajectoryRadius * std::cos(theta),
                                   kTrajectoryRadius * std::sin(theta),
                                   kTrajectoryHeight);
    // The camera has its z axis pointing towards the origin.
    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaternionf rotation_theta(
        Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

    // Construct a transform from camera to scene with this.
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_theta * rotation_base);
    T_S_C.pretranslate(cartesian_coordinates);

    // Generate a depth image of the scene.
    scene.generateDepthImageFromScene(camera_, T_S_C, kMaxDist, &depth_frame);

    // Integrate this depth image.
    integrator_gpu.integrateFrame(depth_frame, T_S_C, camera_, &tsdf_layer);
  }

  // Now do some checks...
  // Check every voxel in the map.
  int total_num_voxels = 0;
  int num_voxel_big_error = 0;
  float min_error = 1000.0f;
  float max_error = -1000.0f;
  auto lambda = [&](const Index3D& block_index, const Index3D& voxel_index,
                    const TsdfVoxel* voxel) {
    if (voxel->weight >= kMinWeight) {
      // Get the corresponding point from the GT layer.
      const TsdfVoxel* gt_voxel = getVoxelAtBlockAndVoxelIndex<TsdfVoxel>(
          gt_tsdf_layer, block_index, voxel_index);
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

  num_voxel_big_error = 0;
  total_num_voxels = 0;
  callFunctionOnAllVoxels<TsdfVoxel>(tsdf_layer, lambda);
  float percent_large_error = static_cast<float>(num_voxel_big_error) /
                              static_cast<float>(total_num_voxels) * 100.0f;
  // EXPECT_LT(percent_large_error, kAcceptablePercentageOverThreshold);
  std::cout << "  - num_voxel_big_error: " << num_voxel_big_error << std::endl;
  std::cout << "  - total_num_voxels: " << total_num_voxels << std::endl;
  std::cout << "  - percent_large_error: " << percent_large_error << "%"
            << std::endl;
  std::cout << "  - min_error: " << min_error << std::endl;
  std::cout << "  - max_error " << max_error << std::endl;

  if (false) {
    io::outputVoxelLayerToPly(tsdf_layer, "test_tsdf_projective_gpu.ply");
    io::outputVoxelLayerToPly(gt_tsdf_layer, "test_tsdf_projective_gt.ply");
  }
}

int main(int arg, char** argv) {
  PlaneBenchmark benchmark;

  benchmark.run();
}