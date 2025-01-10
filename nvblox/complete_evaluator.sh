#!/bin/bash

GT_MESH="./ground_truth_meshes"

REPLICA_DATASET_NAMES=(
    "office0" 
    "office1" 
    "office2"
    "office3"
    "office4"
    "room0"
    "room1"
    "room2")

for j in {0..7}; do
    REPLICA_DATASET_NAME=${REPLICA_DATASET_NAMES[$j]} 
    echo "Running dataset name: ${REPLICA_DATASET_NAME}"

    # Define the directory name
    EVAL_RES_DIR="eval_results/${REPLICA_DATASET_NAME}"
    
    # Check if the directory exists
    if [ ! -d "$EVAL_RES_DIR" ]; then
        echo "Directory '$EVAL_RES_DIR' does not exist. Creating it..."
        mkdir -p "$EVAL_RES_DIR"
    else
        echo "Directory '$EVAL_RES_DIR' already exists. Using it..."
    fi
    
    # Proceed with further operations in the directory
    echo "Directory is ready for use: $EVAL_RES_DIR"
    
    # Define dataset source directories
    BUILD_DIR="./build/executables"
    DATASET_DIR="$HOME/data/Replica/${REPLICA_DATASET_NAME}"

    echo " "
    echo "ENVIRONMENT ${REPLICA_DATASET_NAME}"
    
    # Define working modes
    WORKING_MODES=("BASELINE" "CERTIFIED" "HEURISTIC")
    
    # Loop to run all three modes
    for i in {0..2}; do
        WORKING_MODE=${WORKING_MODES[$i]}
        OUTPUT_LOG_PATH="${EVAL_RES_DIR}/output_${WORKING_MODE}.txt"

        echo "Run #$((i + 1)) with working mode: $WORKING_MODE"
        "${BUILD_DIR}/fuse_replica" "$DATASET_DIR" \
            --output_dir_path="${EVAL_RES_DIR}" \
            --working_mode="$WORKING_MODE" \
            --voxel_size=0.02 \
            --num_frames=2000 \
            --standard_deviation=1 \
            --clearing_radius=3 \
            --odometry_error_covariance=1e-5 > ${OUTPUT_LOG_PATH} 2>&1
        echo "Run #$((i + 1)) with working mode: $WORKING_MODE completed. Output logged to ${OUTPUT_LOG_PATH}"
    done
    
    echo "Transforming BASELINE ESDF"
    python3 transform_esdf.py ./"$EVAL_RES_DIR"/esdf.ply ./"$EVAL_RES_DIR"/transformed_esdf.ply ./"$EVAL_RES_DIR"/gt_transform.txt
    
    echo "Transforming CERTIFIED ESDF"
    python3 transform_esdf.py ./"$EVAL_RES_DIR"/certified_esdf.ply ./"$EVAL_RES_DIR"/transformed_certified_esdf.ply ./"$EVAL_RES_DIR"/gt_transform.txt
    
    echo "Transfroming HEURISTIC ESDF"
    python3 transform_esdf.py ./"$EVAL_RES_DIR"/heuristic_esdf.ply ./"$EVAL_RES_DIR"/transformed_heuristic_esdf.ply ./"$EVAL_RES_DIR"/gt_transform.txt
    
    # do vis
    # python3 color_toggle_viz.py ./office4_mesh.ply ./eval_results/transformed_mesh.ply ./eval_results/transformed_certified_mesh.ply ./eval_results/transformed_heuristic_mesh.ply

    echo "./"$EVAL_RES_DIR"/"$REPLICA_DATASET_NAME"_mesh.ply"

    echo "Interpolating Transformed BASELINE ESDF"
    python3 evaluator.py ./"$GT_MESH"/"$REPLICA_DATASET_NAME"_mesh.ply  ./"$EVAL_RES_DIR"/transformed_esdf.ply ./"$EVAL_RES_DIR"/baseline_plot.png ./"$EVAL_RES_DIR"/baseline.json

    echo "Interpolating Transformed CERTIFIED ESDF"
    python3 evaluator.py ./"$GT_MESH"/"$REPLICA_DATASET_NAME"_mesh.ply  ./"$EVAL_RES_DIR"/transformed_certified_esdf.ply ./"$EVAL_RES_DIR"/certified_plot.png ./"$EVAL_RES_DIR"/certified.json

    echo "Interpolating Transformed HEURISTIC ESDF"
    python3 evaluator.py ./"$GT_MESH"/"$REPLICA_DATASET_NAME"_mesh.ply  ./"$EVAL_RES_DIR"/transformed_heuristic_esdf.ply ./"$EVAL_RES_DIR"/heuristic_plot.png ./"$EVAL_RES_DIR"/heuristic.json

    echo "Saving BASELINE ESDF Slicing Visualization"
    python3 save_slice.py ./"$GT_MESH"/"$REPLICA_DATASET_NAME"_mesh.ply ./"$EVAL_RES_DIR"/transformed_esdf.ply --truncation_distance 0.5 --output_folder ./"$EVAL_RES_DIR/baseline_frames" --output_video ./"$EVAL_RES_DIR"/baseline.mp4

    echo "Saving CERTIFIED ESDF Slicing Visualization"
    python3 save_slice.py ./"$GT_MESH"/"$REPLICA_DATASET_NAME"_mesh.ply ./"$EVAL_RES_DIR"/transformed_certified_esdf.ply --truncation_distance 0.5 --output_folder ./"$EVAL_RES_DIR/certified_frames" --output_video ./"$EVAL_RES_DIR"/certified.mp4

    echo "Saving HEURISTIC ESDF Slicing Visualization"
    python3 save_slice.py ./"$GT_MESH"/"$REPLICA_DATASET_NAME"_mesh.ply ./"$EVAL_RES_DIR"/transformed_heuristic_esdf.ply --truncation_distance 0.5 --output_folder ./"$EVAL_RES_DIR/heuristic_frames" --output_video ./"$EVAL_RES_DIR"/heuristic.mp4


done