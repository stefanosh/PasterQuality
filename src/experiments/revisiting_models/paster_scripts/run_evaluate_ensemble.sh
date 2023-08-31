#!/bin/bash
# Script.sh

# Scope: for all datasets in output folder, run tuning and then evaluate for all models


# Extract the script name without the .sh extension
script_name=$(basename "$0" .sh) 

# Get the directory of the running script
script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")

# Define here all the datasets (they have to be defined in data/ and also in the .toml files respectively)
# datasets="tabular_100_trials_32_batch_size"
datasets="tabular_split_0 tabular_split_1 tabular_split_2 tabular_split_3 tabular_split_4"

n_trials=100
batch_size=32
evaluation_seeds=15

# Set to one GPU only to avoid any errors
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES='' #use all GPUs
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


for dataset_name in $datasets; do   
    # Record the start time
    start_time=$(date '+%Y-%m-%d %H:%M:%S')

    # TODO: delete logs folder?
    mkdir -p $script_dir/logs/$dataset_name/evaluate
    mkdir -p $script_dir/logs/$dataset_name/ensemble

    echo -e "\n ======================================================="
    echo -e "Start Evaluate all models for dataset: $dataset_name"
    ./evaluate.sh $dataset_name $n_trials $batch_size $evaluation_seeds
    echo -e "\n Finish Evaluate all models for dataset: $dataset_name"
    echo -e "======================================================="

    echo -e "\n ======================================================="
    echo -e "Start Ensemble  all models for dataset: $dataset_name"
    ./ensemble.sh $dataset_name $n_trials $batch_size
    echo -e "\n Finish Ensemble all models for dataset: $dataset_name"
    echo -e "======================================================="

    # Record the end time and calculate the execution time
    end_time=$(date '+%Y-%m-%d %H:%M:%S')
    execution_time=$(( $(date -d "$end_time" '+%s') - $(date -d "$start_time" '+%s') ))
    execution_minutes=$(( $execution_time / 60 ))
    echo "Script --$script_name-- execution time: $execution_time seconds"

    file_name="${script_name}_start_${start_time}_end_${end_time}_${dataset_name}_${n_trials}_trials_${batch_size}_batch_size"
    file_path="$script_dir/logs/$file_name"

    #TODO: error count is not correct
    echo "Stats for script --$script_name-- : Total errors: $error_count, n_trials: $n_trials, batch_size: $batch_size, dataset: $dataset_name and execution time: $execution_time seconds OR $execution_minutes minutes" >> "$file_path"
done
