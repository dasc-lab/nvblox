#!/bin/bash

# Define the directories
EVAL_RES_DIR="./eval_results"

create_eval_directory_structure() {
    echo "Creating eval directory strucutre..."
    mkdir -p "$EVAL_RES_DIR"
    echo "Result directory created successfully."
}

# Check if the result_eval already exists
if [ -d "$EVAL_RES_DIR" ]; then
    # Get the current timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # Create a backup of the existing directory
    BACKUP_DIR="${EVAL_RES_DIR}_backup_$TIMESTAMP"
    echo "Backing up existing $EVAL_RES_DIR to $BACKUP_DIR..."
    mv "$EVAL_RES_DIR" "$BACKUP_DIR"
    echo "Backup created successfully."
fi

# Create the new directory structure
create_eval_directory_structure

# Run fuse_replica on office0 dataset
echo "Running fuse_replica on office0"

# Define directories and files
BUILD_DIR="./build/executables"
DATASET_DIR="$HOME/data/Replica/office0"

# python3 color_toggle_viz.py ./office0_mesh.ply ./results_eval/final_result/transformed_certi_mesh.ply


"${BUILD_DIR}/fuse_replica" "$DATASET_DIR" \
    --output_dir_path="$EVAL_RES_DIR" \
    --working_mode="HEURISTIC"\
    --voxel_size=0.02 \
    --num_frames=2000 \
    --standard_deviation=1 \
    --clearing_radius=3 \
    --odometry_error_covariance=1e-5 > output.txt 2>&1

python3 color_toggle_viz.py ./office0_mesh.ply ./eval_results/transformed_mesh.ply ./eval_results/transformed_certified_mesh.ply ./eval_results/transformed_heuristic_mesh.ply