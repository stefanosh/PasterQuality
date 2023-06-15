#!/bin/bash

# Record the start time
start_time=$(date '+%Y-%m-%d %H:%M:%S')

# Extract the script name without the .sh extension
script_name=$(basename "$0" .sh) 

# Get the directory of the running script
script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")

# Dataset directory containing the models and the benchmarking config files and results
dataset_name=$1
n_trials=$2
batch_size=$3

output_folder="output_pasterquality"

# Full path to the project directory
folder_dir="$PROJECT_DIR/$output_folder/$dataset_name"

# Define the log file path for Python errors
# log_file="$script_dir/logs/error.log"

# Initialize the error count
error_count=0

echo "Running all models for dataset: $dataset_name"
echo "Folder dir: $folder_dir"
echo "Script directory: $script_dir"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ==================================================================================================
#  Main functionality of the script
# ==================================================================================================
for folder in $folder_dir/*; do
    # Get the folder/model name
    folder_name=$(basename "$folder")  
    echo -e "\n Ensembling model: $folder_name"

    model_time=$(date '+%Y-%m-%d_%H%M%S')
    file_path="$script_dir/logs/$dataset_name/$script_name/${folder_name}_${model_time}"
    cd $PROJECT_DIR;

    #TODO: maybe take out the node as well?
    # TabNet should be run with revisiting_models_tf environment
    if [ "$folder_name" != "tabnet" ] && [ "$folder_name" != "node" ]; then
        # Run the tunining of the model
        python bin/ensemble.py $folder_name $output_folder/$dataset_name/$folder_name/tuned | tee $file_path /dev/tty
    elif [ "$folder_name" == "tabnet" ]; then
        conda run -n revisiting_models_tf python bin/ensemble.py tabnet $output_folder/$dataset_name/tabnet/tuned | tee $file_path /dev/tty
    fi
done

# ==================================================================================================
# Record the end time and calculate the execution time
end_time=$(date '+%Y-%m-%d %H:%M:%S')
execution_time=$(( $(date -d "$end_time" '+%s') - $(date -d "$start_time" '+%s') ))
execution_minutes=$(( $execution_time / 60 ))
echo "Script --$script_name-- execution time: $execution_time seconds"

file_name="${script_name}_start_${start_time}_end_${end_time}_$dataset_name_${n_trials}_trials_${batch_size}_batch_size"
file_path="$script_dir/logs/$file_name"

#TODO: error count is not correct
echo "Stats for script --$script_name-- : Total errors: $error_count, n_trials: $n_trials, batch_size: $batch_size, dataset: $dataset_name and execution time: $execution_time seconds OR $execution_minutes minutes" >> "$file_path"