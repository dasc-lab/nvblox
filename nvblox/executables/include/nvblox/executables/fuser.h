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
#pragma once

#include <glog/logging.h>

#include <memory>
#include <string>

#include "nvblox/datasets/data_loader.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/layer_cake.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/mesh/mesh_block.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/rays/sphere_tracer.h"
#include "nvblox/utils/logging.h"

#include <liegroups/liegroups.hpp>

namespace nvblox {

class Fuser {
 public:
  Fuser() = default;
  Fuser(std::unique_ptr<datasets::RgbdDataLoaderInterface>&& data_loader);

  // Loads parameters from command line flags
  void readCommandLineFlags();

  // Runs an experiment
  int run();

  // Set various settings.
  void setVoxelSize(float voxel_size);
  void setProjectiveFrameSubsampling(int subsample);
  void setColorFrameSubsampling(int subsample);
  void setMeshFrameSubsampling(int subsample);
  void setEsdfFrameSubsampling(int subsample);
  void setEsdfMode(Mapper::EsdfMode esdf_mode);

  // Integrate certain layers.
  bool integrateFrame(const int frame_number);
  bool integrateFrames();
  void updateEsdf();

  // Output a pointcloud tsdf as PLY file.
  bool outputTsdfPointcloudPly();
  // Output a pointcloud occupancy as PLY file.
  bool outputOccupancyPointcloudPly();
  // Output a pointcloud ESDF as PLY file.
  bool outputESDFPointcloudPly();
  // Output a pointcloud Certified ESDF as PLY file.
  bool outputCertifiedESDFPointcloudPly();
  // Output a file with the mesh.
  bool outputMeshPly();
  // Output a file with the certified mesh.
  bool outputCertifiedMeshPly();
  // Output timings to a file
  bool outputTimingsToFile();
  // Output the serialized map to a file
  bool outputMapToFile();
  // Output the perturbed trajectory to a file
  bool outputTrajectoryToFile();

  // Get the mapper (useful for experiments where we modify mapper settings)
  Mapper& mapper();

  // Get the covariance matrix of the pose error per frame
  LieGroups::Matrix6f getOdometryErrorCovariance(){ return odometry_error_cov_; }

  // Set the covariance matrix of the odometry error per frame assuming its sigma * I
  bool setOdometryErrorCovariance(float sigma);

  // Set the covariance matrix of the odometry error per frame
  bool setOdometryErrorCovariance(LieGroups::Matrix6f Sigma);

  // Dataset settings.
  int num_frames_to_integrate_ = std::numeric_limits<int>::max();
  std::unique_ptr<datasets::RgbdDataLoaderInterface> data_loader_;

  // Params
  float voxel_size_m_ = 0.05;
  ProjectiveLayerType projective_layer_type_ = ProjectiveLayerType::kTsdf;
  int projective_frame_subsampling_ = 1;
  int color_frame_subsampling_ = 1;
  // By default we just do the mesh and esdf once at the end
  // (if output paths exist)
  int mesh_frame_subsampling_ = -1;
  int esdf_frame_subsampling_ = -1;

  // ESDF slice params
  float z_min_ = 0.5f;
  float z_max_ = 1.0f;
  float z_slice_ = 0.75f;

  // ESDF mode
  Mapper::EsdfMode esdf_mode_ = Mapper::EsdfMode::k3D;

  // Mapper - Contains map layers and integrators
  std::unique_ptr<Mapper> mapper_;

  // Odometry Error Params
  LieGroups::Matrix6f odometry_error_cov_ = LieGroups::Matrix6f::Zero();
  Transform T_L_Ckm1 = Transform::Identity();
  Transform T_L_Ckm1_true = Transform::Identity();
  std::vector<Transform> trajectory_;  // save the perturbed trajectory

  // Output paths
  std::string timing_output_path_;
  std::string tsdf_output_path_;
  std::string esdf_output_path_;
  std::string certified_esdf_output_path_;
  std::string occupancy_output_path_;
  std::string mesh_output_path_;
  std::string certified_mesh_output_path_;
  std::string map_output_path_;
  std::string trajectory_output_path_;
};

}  //  namespace nvblox
