#!/bin/bash

EVAL_RES_DIR=./eavl_results

# Run fuse_replica on office0 dataset
echo "Running fuse_replica on office0"

# Define directories and files
BUILD_DIR="./build/executables"
DATASET_DIR="$HOME/data/Replica/office0"


"${BUILD_DIR}/fuse_replica" "$DATASET_DIR" \
    --output_dir_path="$EVAL_RES_DIR" \
    --working_mode="BASELINE"\
    --voxel_size=0.02 \
    --num_frames=2000 \
    --standard_deviation=1 \
    --clearing_radius=3 \
    --odometry_error_covariance=1e-5 > output.txt 2>&1

python3 color_toggle_viz.py ./office0_mesh.ply ./eval_results/transformed_mesh.ply ./eval_results/transformed_certified_mesh.ply ./eval_results/transformed_heuristic_mesh.ply