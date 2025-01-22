#!/bin/bash

# Define the range and step size
START=1
END=2000
STEP=500

# Generate equally spaced time steps including the end
FRAMES=($(seq $START $STEP $END))

# Ensure the end value is included if not already in the sequence
if [ "${FRAMES[-1]}" -ne "$END" ]; then
    FRAMES+=("$END")
fi

# Select the environment
REPLICA_DATASET_NAME="office3"

# Select the odometry error covariance
ERROR_COV=1e-5

# Proceed with further operations in the directory
echo "Directory is ready to use: $VIS_RES_DIR"

# Define dataset source directories
BUILD_DIR="./build/executables"
DATASET_DIR="$HOME/data/Replica/${REPLICA_DATASET_NAME}"

# Define working modes
WORKING_MODES=("BASELINE" "CERTIFIED" "HEURISTIC")

# Iterate over the set of frames
for N_frames in "${FRAMES[@]}"; do

    # Define the directory name
    VIS_RES_DIR="vis_results/${N_frames}"

    # Check if the directory exists
    if [ ! -d "$VIS_RES_DIR" ]; then
        echo "Directory '$VIS_RES_DIR' does not exist. Creating it..."
        mkdir -p "$VIS_RES_DIR"
    else   
        echo "Directory '$VIS_RES_DIR' already exists. Using it..."
    fi

    echo "Time step: $N_frames"

    # Loop to run all three modes
    for mode in {0..2}; do

        WORKING_MODE=${WORKING_MODES[$mode]}
        OUTPUT_LOG_PATH="${VIS_RES_DIR}/output_${WORKING_MODE}.txt"

        echo "Run #$((mode + 1)) with working mode: $WORKING_MODE"

        "${BUILD_DIR}/fuse_replica" "$DATASET_DIR" \
            --output_dir_path="${VIS_RES_DIR}" \
            --working_mode="$WORKING_MODE" \
            --voxel_size=0.02 \
            --num_frames="$N_frames" \
            --standard_deviation=3 \
            --clearing_radius=3 \
            --odometry_error_covariance="$ERROR_COV" > ${OUTPUT_LOG_PATH} 2>&1
        echo "Run #$((mode + 1)) with working mode: $WORKING_MODE completed. Output logged to ${OUTPUT_LOG_PATH}"

    done

    echo "Transforming BASELINE ESDF"
    python3 transform_esdf.py ./"$VIS_RES_DIR"/esdf.ply ./"$VIS_RES_DIR"/transformed_esdf.ply ./"$VIS_RES_DIR"/gt_transform.txt
        
    echo "Transforming CERTIFIED ESDF"
    python3 transform_esdf.py ./"$VIS_RES_DIR"/certified_esdf.ply ./"$VIS_RES_DIR"/transformed_certified_esdf.ply ./"$VIS_RES_DIR"/gt_transform.txt
        
    echo "Transfroming HEURISTIC ESDF"
    python3 transform_esdf.py ./"$VIS_RES_DIR"/heuristic_esdf.ply ./"$VIS_RES_DIR"/transformed_heuristic_esdf.ply ./"$VIS_RES_DIR"/gt_transform.txt

done
