from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
import os
import time
import json
from IPython.display import FileLink
import zipfile
from pathlib import Path
import shutil
import pandas as pd
import numpy as np

"""Create CV splits for tabular dataset for RTDL operation"""

df = pd.read_csv("/home/stefanos/PasterQuality/data/tabular/complete_dataset.csv")
paster_ids = df['paster_id'].values
random_seed = 42
split_percentage = 0.2
dir_path = "/home/stefanos/PasterQuality/data/tabular"
os.makedirs(dir_path, exist_ok=True)

kf = KFold(n_splits=5, random_state=random_seed, shuffle=True)
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_seed)
for i, (train_index, test_index) in enumerate(rkf.split(paster_ids)):
    os.makedirs(f"{dir_path}/csv/tabular_split_{i}/", exist_ok=True)
    os.makedirs(f"{dir_path}/npy/tabular_split_{i}/", exist_ok=True)
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]
    
    df.to_csv(f"{dir_path}/complete_dataset.csv")
    
    # Save train, test df with fold indices
    train_set.to_csv(f"{dir_path}/csv/tabular_split_{i}/train_set.csv")
    test_set.to_csv(f"{dir_path}/csv/tabular_split_{i}/test_set.csv")
    
    split_idx = 2  # Because we have PU, paster_id
    X_train = train_set[train_set.columns[:-split_idx]].copy()
    y_train = train_set[train_set.columns[-split_idx]].copy()
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=split_percentage,random_state=random_seed)
    
    X_test = test_set[test_set.columns[:-split_idx]].copy()
    y_test = test_set[test_set.columns[-split_idx]].copy()


    # Save data to .csv files
    X_train.to_csv(f"{dir_path}/csv/tabular_split_{i}/X_train.csv", index=False)
    X_val.to_csv(f"{dir_path}/csv/tabular_split_{i}/X_val.csv", index=False)
    X_test.to_csv(f"{dir_path}/csv/tabular_split_{i}/X_test.csv", index=False)
    y_train.to_csv(f"{dir_path}/csv/tabular_split_{i}/y_train.csv", index=False)
    y_val.to_csv(f"{dir_path}/csv/tabular_split_{i}/y_val.csv", index=False)
    y_test.to_csv(f"{dir_path}/csv/tabular_split_{i}/y_test.csv", index=False)

    
    # Convert Paster DFs to numpy arrays
    y_train = y_train.to_numpy().astype(np.float32)
    y_val = y_val.to_numpy().astype(np.float32)
    y_test = y_test.to_numpy().astype(np.float32)

    X_train = X_train.to_numpy().astype(np.float32)
    X_val = X_val.to_numpy().astype(np.float32)
    X_test = X_test.to_numpy().astype(np.float32)

    # Save data to .npy files
    for dataset in ["train", "val", "test"]:
        np.save(
            f"{dir_path}/npy/tabular_split_{i}/N_{dataset}.npy",
            eval(f"X_{dataset}"),
        )
        np.save(
             f"{dir_path}/npy/tabular_split_{i}/y_{dataset}.npy",
            eval(f"y_{dataset}"),
        )

    # Save the required metadata as JSON
    dataset_name = f"tabular_split_{i}"
    json_data = {}
    file_path = f"{dir_path}/npy/tabular_split_{i}/info.json"
    json_data["name"] = f"{dataset_name}___0"
    json_data["basename"] = dataset_name
    json_data["split"] = 0
    json_data["task_type"] = "regression"
    json_data["n_num_features"] = X_train.shape[1]
    json_data["n_cat_features"] = 0
    json_data["train_size"] = len(X_train)
    json_data["val_size"] = len(X_val)
    json_data["test_size"] = len(X_test)

    with open(file_path, "w") as file:
        json.dump(json_data, file)