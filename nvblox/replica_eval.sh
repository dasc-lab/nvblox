#!/bin/bash

# Define the directories
PARENT_DIR="results_eval"

GT_MESH_DIR="$PARENT_DIR/meshes/ground_truth"
CERTI_MESH_DIR="$PARENT_DIR/meshes/certified"
TRAJECTORY_DIR="$PARENT_DIR/trajectory/estimated"
FINAL_RES_DIR="$PARENT_DIR/final_result"

# Function to create the directory structure
create_directory_structure() {
    echo "Creating directory structure..."
    mkdir -p "$GT_MESH_DIR"
    mkdir -p "$CERTI_MESH_DIR"
    mkdir -p "$TRAJECTORY_DIR"
    mkdir -p "$FINAL_RES_DIR"
    echo "Directory structure created successfully."
}

# Check if the result_eval already exists
if [ -d "$PARENT_DIR" ]; then
    # Get the current timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # Create a backup of the existing directory
    BACKUP_DIR="${PARENT_DIR}_backup_$TIMESTAMP"
    echo "Backing up existing $PARENT_DIR to $BACKUP_DIR..."
    mv "$PARENT_DIR" "$BACKUP_DIR"
    echo "Backup created successfully."
fi

# Create the new directory structure
create_directory_structure

# Run fuse_replica on office0 dataset
echo "Running fuse_replica on office0"

# Define directories and files
BUILD_DIR="./build/executables"
DATASET_DIR="$HOME/data/Replica/office0"

# Run the fuse_replica executable for evaluation of entire trajectory
# "${BUILD_DIR}/fuse_replica" "$DATASET_DIR" --mesh_output_path="${FINAL_RES_DIR}/mesh.ply" \
#     --num_frames=10 \
#     --certified_mesh_output_path="${FINAL_RES_DIR}/certi_mesh.ply" \
#     --voxel_size=0.02 \
#     --trajectory_output_path="${FINAL_RES_DIR}/trajectory.txt" \
#     --odometry_error_covariance=1e-6 \
#     --inter_mesh_output_path="${GT_MESH_DIR}" \
#     --inter_certified_mesh_output_path="${CERTI_MESH_DIR}" \
#     --transformed_certified_mesh_output_path="${FINAL_RES_DIR}/tranformed_certi_mesh.ply" \
#     --inter_trajectory_output_path="${TRAJECTORY_DIR}" > output.txt 2>&1


# Run the fuse_replica executable
"${BUILD_DIR}/fuse_replica" "$DATASET_DIR" --mesh_output_path="${FINAL_RES_DIR}/mesh.ply" \
    --certified_mesh_output_path="${FINAL_RES_DIR}/certi_mesh.ply" \
    --transformed_certified_mesh_output_path="${FINAL_RES_DIR}/transformed_certi_mesh.ply" \
    --voxel_size=0.02 \
    --trajectory_output_path="${FINAL_RES_DIR}/trajectory.txt" \
    --odometry_error_covariance=1e-4  > output.txt 2>&1
