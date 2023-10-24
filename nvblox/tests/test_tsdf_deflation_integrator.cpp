#include <gtest/gtest.h>
#include <cmath>

#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/tsdf_deflation_integrator.h"
#include "nvblox/interpolation/interpolation_3d.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/primitives/scene.h"

#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

constexpr float kAcceptablePercentageOverThreshold = 2.0;  // 2.0 %

class TsdfDeflationIntegratorTest : public ::testing::Test {
 protected:
  TsdfDeflationIntegratorTest()
      : layer_(voxel_size_m_, MemoryType::kUnified),
        camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

  // Test layer
  constexpr static float voxel_size_m_ = 0.1;
  TsdfLayer layer_;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;
};

TEST_F(TsdfDeflationIntegratorTest, SphereSceneTest) {
  constexpr float kTrajectoryRadius = 4.0f;
  constexpr float kTrajectoryHeight = 2.0f;
  constexpr int kNumTrajectoryPoints = 80;
  constexpr float kTruncationDistanceVox = 2;
  constexpr float kTruncationDistanceMeters =
      kTruncationDistanceVox * voxel_size_m_;
  constexpr float kMaxDist = 10.0;
  constexpr float kMinWeight = 1.0;
  const float decrement{0.1};

  // Tolerance for error.
  constexpr float kDistanceErrorTolerance = kTruncationDistanceMeters;

  // Get the ground truth SDF of a sphere in a box.
  primitives::Scene scene = test_utils::getSphereInBox();
  TsdfLayer gt_layer(voxel_size_m_, MemoryType::kUnified);
  scene.generateLayerFromScene(kTruncationDistanceMeters, &gt_layer);

  // Create an integrator.
  ProjectiveTsdfIntegrator integrator;
  TsdfDeflationIntegrator deflation_integrator;
  integrator.truncation_distance_vox(kTruncationDistanceVox);

  // Simulate a trajectory of the requisite amount of points, on the circle
  // around the sphere.
  const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints);

  // Create a depth frame. We share this memory buffer for the entire
  // trajectory.
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);

  TsdfLayer layer_gpu(layer_.voxel_size(), MemoryType::kUnified);
  TsdfLayer layer_gpu_deflated(layer_.voxel_size(), MemoryType::kUnified);

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
    integrator.integrateFrame(depth_frame, T_S_C, camera_, &layer_gpu);

    // Update the deflated TSDF.
    integrator.integrateFrame(depth_frame, T_S_C, camera_, &layer_gpu_deflated);
    deflation_integrator.deflate(&layer_gpu_deflated, decrement);
  }

  // Now do some checks...
  // Check every voxel in the map.
  int total_num_voxels = 0;
  int num_voxel_big_error = 0;
  int num_larger_in_deflated = 0;
  auto lambda = [&](const Index3D& block_index, const Index3D& voxel_index,
                    const TsdfVoxel* voxel) {
    if (voxel->weight >= kMinWeight) {
      // Get the corresponding point from the GT layer.
      const TsdfVoxel* gt_voxel = getVoxelAtBlockAndVoxelIndex<TsdfVoxel>(
          gt_layer, block_index, voxel_index);
      if (gt_voxel != nullptr) {
        if (std::fabs(voxel->distance - gt_voxel->distance) >
            kDistanceErrorTolerance) {
          num_voxel_big_error++;
        }
        total_num_voxels++;
      }
    }
  };

  float total_deflation = 0.0;
  float min_deflation = 0.0;
  float max_deflation = 0.0;
  int num_deflated_voxels = 0;
  auto lambda_compare =
      [&](const Index3D& block_index, const Index3D& voxel_index,
          const TsdfVoxel* voxel_orig, const TsdfVoxel* voxel_deflated) {
        // Check the deflated TSDF has smaller distances than the original.
        float deflation = voxel_orig->distance - voxel_deflated->distance;
        if (deflation < -kDistanceErrorTolerance) {
          num_larger_in_deflated++;
        }
        if (deflation < min_deflation) {
          min_deflation = deflation;
        }
        if (deflation > max_deflation) {
          max_deflation = deflation;
        }
        total_deflation += deflation;
        num_deflated_voxels++;
      };
  callFunctionOnAllVoxels<TsdfVoxel>(layer_gpu, lambda);
  callFunctionOnAllVoxels<TsdfVoxel>(layer_gpu, layer_gpu_deflated, lambda_compare);
  float average_deflation = total_deflation / static_cast<float>(num_deflated_voxels);
  std::cout << "GPU: average deflaton: " << average_deflation << std::endl;
  std::cout << "GPU: min deflaton: " << min_deflation << std::endl;
  std::cout << "GPU: max deflaton: " << max_deflation << std::endl;

  float percent_large_error = static_cast<float>(num_voxel_big_error) /
                              static_cast<float>(total_num_voxels) * 100.0f;
  EXPECT_LT(percent_large_error, kAcceptablePercentageOverThreshold);
  EXPECT_EQ(num_larger_in_deflated, 0);
  std::cout << "GPU: num_voxel_big_error: " << num_voxel_big_error << std::endl;
  std::cout << "GPU: total_num_voxels: " << total_num_voxels << std::endl;
  std::cout << "GPU: percent_large_error: " << percent_large_error << std::endl;
  std::cout << "GPU: num_larger_in_deflated: " << num_larger_in_deflated << std::endl;

  if (FLAGS_nvblox_test_file_output) {
    io::outputVoxelLayerToPly(layer_gpu, "test_tsdf_projective_gpu.ply");
    io::outputVoxelLayerToPly(gt_layer, "test_tsdf_projective_gt.ply");
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}