import os
import time
import json
from IPython.display import FileLink
import zipfile
from pathlib import Path
import shutil
import pandas as pd
import numpy as np


# automatically update the data_path in the 0.toml file for tuning, with the dataset_name of interest
def update_tuning_config_files(folder_path, name, trials=10, batch_size=32):
    for model_folder in os.listdir(folder_path):
        for folder in os.listdir(f'{folder_path}/{model_folder}'):
            if folder == 'tuning':      
                # Replacing this way because toml cannot lad arrays with homogenous types from 0.toml..
                p = Path(f'{folder_path}/{model_folder}/{folder}/0.toml')
                data_to_replace = f"path = 'data/{name}'"
                trials_to_replace = f"n_trials = {trials}"
                batch_size_to_replace = f"batch_size = {batch_size}"
                p.write_text(p.read_text().replace("path = 'data/paster'", data_to_replace))
                p.write_text(p.read_text().replace("n_trials = 1", trials_to_replace))
                p.write_text(p.read_text().replace("batch_size = 256", batch_size_to_replace))


# automatically update the data_path in the 0.toml file for tuning, with the dataset_name of interest
def update_best_tuned_config_files(folder_path, name):
    for model_folder in os.listdir(folder_path):
        for folder in os.listdir(f'{folder_path}/{model_folder}'):
            if folder == 'tuning':      
                # Replacing this way because toml cannot lad arrays with homogenous types from 0.toml..
                p = Path(f'{folder_path}/{model_folder}/{folder}/0/best.toml')
                data_to_replace = f"path = 'data/{name}'"
                p.write_text(p.read_text().replace("path = 'data/new_recorder_whole_merged_batches'", data_to_replace))

                p = Path(f'{folder_path}/{model_folder}/{folder}/0.toml')
                p.write_text(p.read_text().replace("path = 'data/tabular_100_trials_32_batch_size'", data_to_replace))                 


# Create a copy of the template folder for our new dataset
# datasets = ["tabular_split_0","tabular_split_1","tabular_split_2","tabular_split_3","tabular_split_4"]
for i in range(10):
    dataset_name = f"tabular_split_{i}"
    dirpath = "/home/stefanos/PasterQuality/src/experiments/revisiting_models/output_pasterquality"
    template_path = "/home/stefanos/PasterQuality/src/experiments/revisiting_models/output_pasterquality/tuned_evaluation_template/"  # this has data/new_recorder_whole_merged_batches in best.toml
    new_dataset_path = os.path.join(dirpath, f"{dataset_name}")

    # Copy model/config folders from the template folder to the target folder
    if not os.path.exists(new_dataset_path):
        shutil.copytree(template_path, new_dataset_path)

    # Update the config files for the new dataset to tune hyperparameters from scratch
    # update_tuning_config_files(new_dataset_path, dataset_name, trials=2, batch_size=32)

    # Update the config files for the new dataset to tune hyperparameters from scratch
    update_best_tuned_config_files(new_dataset_path, dataset_name)