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
    echo "Tuning model: $folder_name"

    model_time=$(date '+%Y-%m-%d_%H%M%S')
    file_path="$script_dir/logs/$dataset_name/$script_name/${folder_name}_${model_time}_${n_trials}_trials_${batch_size}_batch_size}"
    cd $PROJECT_DIR;

    if [ "$folder_name" != "tabnet" ] && [ "$folder_name" != "node" ]; then

        # Alter the n_trials parameter in the config file
        cd $PROJECT_DIR; python -c "
from pathlib import Path
p = Path('$PROJECT_DIR/$output_folder/$dataset_name/$folder_name/tuning/0.toml')
p.write_text(p.read_text().replace('n_trials = 1', 'n_trials = $n_trials'))
" #2> >(tee -a "$log_file" >&2) > >(tee -a /dev/null)

        # Alter the batch_size parameter in the config file
        cd $PROJECT_DIR; python -c "
from pathlib import Path
p = Path('$PROJECT_DIR/$output_folder/$dataset_name/$folder_name/tuning/0.toml')
p.write_text(p.read_text().replace('batch_size = 256', 'batch_size = $batch_size'))
" #2> >(tee -a "$log_file" >&2) > >(tee -a /dev/null)


        # Run the tunining of the model
        python bin/tune.py $output_folder/$dataset_name/$folder_name/tuning/0.toml -f | tee $file_path /dev/tty

        if [ $? -ne 0 ]; then
            # Increment the error count if the return code is non-zero
            ((error_count++))
        fi

    elif [ "$folder_name" == "tabnet" ]; then
        
        # From original tabnet paper https://arxiv.org/pdf/1908.07442.pdf
        # Consider • A large batch size is beneficial – if the memory constraints permit, 
        # as large as 1-10 % of the total training dataset size can help performance. The virtual batch size is typically much smaller.
        divider=$((2048 / $batch_size))
        virtual_batch_size=$((256 / $divider))
        
                # Alter the n_trials parameter in the config file
        cd $PROJECT_DIR; python -c "
from pathlib import Path
p = Path('$PROJECT_DIR/$output_folder/$dataset_name/$folder_name/tuning/0.toml')
p.write_text(p.read_text().replace('n_trials = 1', 'n_trials = $n_trials'))
" #2> >(tee -a "$log_file" >&2) > >(tee -a /dev/null)

        # Alter the batch_size parameter in the config file
        cd $PROJECT_DIR; python -c "
from pathlib import Path
p = Path('$PROJECT_DIR/$output_folder/$dataset_name/$folder_name/tuning/0.toml')
p.write_text(p.read_text().replace('batch_size = 2048', 'batch_size = $batch_size'))
" #2> >(tee -a "$log_file" >&2) > >(tee -a /dev/null)

        # Alter the batch_size parameter in the config file
        cd $PROJECT_DIR; python -c "
from pathlib import Path
p = Path('$PROJECT_DIR/$output_folder/$dataset_name/$folder_name/tuning/0.toml')
p.write_text(p.read_text().replace('virtual_batch_size = 256', 'batch_size = $virtua_batch_size'))
" #2> >(tee -a "$log_file" >&2) > >(tee -a /dev/null)
        
        conda run -n rtdl_tf_copy python bin/tune.py $output_folder/$dataset_name/tabnet/tuning/0.toml -f | tee $file_path /dev/tty

        if [ $? -ne 0 ]; then
            # Increment the error count if the return code is non-zero
            ((error_count++))
        fi
    fi
done

# ==================================================================================================
# Record the end time and calculate the execution time
end_time=$(date '+%Y-%m-%d %H:%M:%S')
execution_time=$(( $(date -d "$end_time" '+%s') - $(date -d "$start_time" '+%s') ))
execution_minutes=$(( $execution_time / 60 ))
echo "Script --$script_name-- execution time: $execution_time seconds"

file_name="${script_name}_start_${start_time}_end_${end_time}_${dataset_name}_${n_trials}_trials_${batch_size}_batch_size"
file_path="$script_dir/logs/$file_name"

#TODO: error count is not correct
echo "Stats for script --$script_name-- : Total errors: $error_count, n_trials: $n_trials, batch_size: $batch_size, dataset: $dataset_name and execution time: $execution_time seconds OR $execution_minutes minutes" >> "$file_path"