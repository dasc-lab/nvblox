#!/bin/bash

echo "Running fuse_replica on office0"

exec_dir="./build/executables/"
dataset_dir="~/data/Replica/office0"
output_dir="${exec_dir}/outputs/"

mesh_output_file="${output_dir}/mesh.ply"


./${exec_dir}/fuse_replica --voxel_size=0.02 ${dataset_dir} ${mesh_output_file}


