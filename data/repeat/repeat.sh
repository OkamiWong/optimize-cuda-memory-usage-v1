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
OUTPUT_FOLDER_PREFIX="out_"
NUMBER_TO_REPEAT=10

starting_directory=$(pwd)

for i in $(seq 1 10); do
  wait_for_safe_temperature

  cd $starting_directory

  output_folder="$OUTPUT_FOLDER_PREFIX$i"
  mkdir -p $output_folder
  cd $output_folder

  cp ../$MEMOPT_CONFIG_TEMPLATE ./config.json

  ../$BIN &> $OUTPUT_FILE
done
