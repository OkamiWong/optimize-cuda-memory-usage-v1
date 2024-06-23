#!/bin/bash

SAFE_TEMPERATURE=55
COMPUTE_DEVICE_ID=1

function get_gpu_temperature() {
  local device_id=$1
  echo $(nvidia-smi -q -i $device_id -d TEMPERATURE | grep "GPU Current Temp" | grep -E "[0-9]+" -o)
}

function wait_for_safe_temperature() {
  current_temperature=$(get_gpu_temperature $COMPUTE_DEVICE_ID)
  while ((current_temperature > SAFE_TEMPERATURE)); do
    sleep 3
    current_temperature=$(get_gpu_temperature $COMPUTE_DEVICE_ID)
  done
}

BIN="../../build/userApplications/tiledCholesky"
MEMOPT_CONFIG_TEMPLATE="./memopt-config.json"
OUTPUT_FILE="./stdout.out"
OUTPUT_FOLDER_PREFIX="tiled_cholesky_out_"
TIME_OVERHEAD_LIMITS=(1.09 1.10 1.11 1.12 1.13 1.14 1.15 1.16)

starting_directory=$(pwd)

for time_overhead_limit in ${TIME_OVERHEAD_LIMITS[@]}; do
  wait_for_safe_temperature

  cd $starting_directory

  output_folder="$OUTPUT_FOLDER_PREFIX$time_overhead_limit"
  mkdir -p $output_folder
  cd $output_folder

  jq ".optimization.acceptableRunningTimeFactor = $time_overhead_limit" ../$MEMOPT_CONFIG_TEMPLATE > config.json

  ../$BIN &> $OUTPUT_FILE
done
