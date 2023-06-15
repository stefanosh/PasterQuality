import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import random
from src.utils.data import check_common_ids


###########################################################
# For tabular dataset
###########################################################

df = pd.read_csv("/home/stefanos/pasteurAIzer/data/tabular/complete_dataset.csv")

dataset_name = "tabular"
split_percentage = 0.2
random_state = 42
split_idx = 2  # Because we have PU, pastre_id
X = df[df.columns[:-split_idx]].copy()
y = df[df.columns[-split_idx]].copy()

# By creating bins and performing stratified sampling based on those bins, we can ensure a more balanced distribution of the target variable across the train and test sets, including outliers.
n_bins = 3
bin_labels = range(n_bins)
y_binned = pd.cut(y, bins=n_bins, labels=bin_labels)

# Create train_data and test_data from df
X_train, X_test, y_train, y_test = train_test_split(
    X.copy(),
    y.copy(),
    test_size=split_percentage,
    random_state=random_state,
    stratify=y_binned,
)

# Split train into training and validation data
y_binned = pd.cut(y_train, bins=n_bins, labels=bin_labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=split_percentage,
    random_state=random_state,
    stratify=y_binned,
)


# Save data to .csv files
X_train.to_csv(f"/home/stefanos/pasteurAIzer/data/tabular/csv/X_train.csv", index=False)
X_val.to_csv(f"/home/stefanos/pasteurAIzer/data/tabular/csv/X_val.csv", index=False)
X_test.to_csv(f"/home/stefanos/pasteurAIzer/data/tabular/csv/X_test.csv", index=False)

y_train.to_csv(f"/home/stefanos/pasteurAIzer/data/tabular/csv/y_train.csv", index=False)
y_val.to_csv(f"/home/stefanos/pasteurAIzer/data/tabular/csv/y_val.csv", index=False)
y_test.to_csv(f"/home/stefanos/pasteurAIzer/data/tabular/csv/y_test.csv", index=False)

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
        f"/home/stefanos/pasteurAIzer/data/tabular/numpy/X_{dataset}.npy",
        eval(f"X_{dataset}"),
    )
    np.save(
        f"/home/stefanos/pasteurAIzer/data/tabular/numpy/y_{dataset}.npy",
        eval(f"y_{dataset}"),
    )

# Save the required metadata as JSON
json_data = {}
file_path = "/home/stefanos/pasteurAIzer/data/tabular/info.json"
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


###########################################################
# For sequential datasets
###########################################################

# Split based on the number of records ()==paster_ids==groups) and not arbitrarily by indices (similar to what GroupShuffleSplit would do)

# Load the full dataset to split into train, val and test
df = pd.read_csv(f"/home/stefanos/pasteurAIzer/data/sequential/complete_dataset.csv")
df = df.drop(columns=["timestamp"])

unique_ids = df.paster_id.unique()
random_seed = 42
split_percentage = 0.2
labels = ["water_temp", "product_temp"]

# Get paster_ids for test set
test_ids_to_split = int(len(unique_ids) * split_percentage)
random.seed(random_seed)
test_ids = random.sample(list(unique_ids), test_ids_to_split)

# Get paster_ids for validation set
train_ids = [x for x in unique_ids if x not in test_ids]
val_ids_to_split = int(len(train_ids) * split_percentage)
val_ids = random.sample(train_ids, val_ids_to_split)

# Get paster_ids for training set
train_ids = [x for x in train_ids if x not in val_ids]

print(
    f"Train ids: {len(train_ids)}. Percentage: {len(train_ids)/len(unique_ids)*100} %"
)
print(f"Val ids: {len(val_ids)}. Percentage: {len(val_ids)/len(unique_ids)*100} %")
print(f"Test ids: {len(test_ids)}. Percentage: {len(test_ids)/len(unique_ids)*100} %")

# Check if train_ds, val_ids, test_ids are mutually exclusive without using set and intersection
check_common_ids(train_ids, val_ids, test_ids, unique_ids)



# Filter ids from df for each set
train_df = df[df.paster_id.isin(train_ids)].reset_index(drop=True)
val_df = df[df.paster_id.isin(val_ids)].reset_index(drop=True)
test_df = df[df.paster_id.isin(test_ids)].reset_index(drop=True)

train_df = train_df.drop(columns=["paster_id"])
val_df = val_df.drop(columns=["paster_id"])

for set in ["train", "val"]:
    df = eval(f"{set}_df")
    X = df.drop(columns=["water_temp", "product_temp"])
    y = df[["water_temp", "product_temp"]]
    X.to_csv(
        f"/home/stefanos/pasteurAIzer/data/sequential/csv/X_{set}.csv", index=False
    )
    y.to_csv(
        f"/home/stefanos/pasteurAIzer/data/sequential/csv/y_{set}.csv", index=False
    )
    np.save(f"/home/stefanos/pasteurAIzer/data/sequential/numpy/X_{set}.npy", X)
    np.save(f"/home/stefanos/pasteurAIzer/data/sequential/numpy/y_{set}.npy", y)

# Leaving test outside of the loop because we need to keep the paster_id to make the recursive prediction pipeline evaluation
test_df.to_csv(
    f"/home/stefanos/pasteurAIzer/data/sequential/csv/test_df.csv", index=False
)


###########################################################
# For time series dataset
###########################################################

# Keeping the same split strategy as the sequential dataset

df = pd.read_csv(f"/home/stefanos/pasteurAIzer/data/time_series/complete_dataset.csv")
df = df.drop(columns=["timestamp"])

# Filter ids from df for each set
train_df = df[df.paster_id.isin(train_ids)].reset_index(drop=True)
val_df = df[df.paster_id.isin(val_ids)].reset_index(drop=True)
test_df = df[df.paster_id.isin(test_ids)].reset_index(drop=True)

train_df.to_csv(
    f"/home/stefanos/pasteurAIzer/data/time_series/csv/train_df.csv", index=False
)
val_df.to_csv(
    f"/home/stefanos/pasteurAIzer/data/time_series/csv/val_df.csv", index=False
)
test_df.to_csv(
    f"/home/stefanos/pasteurAIzer/data/time_series/csv/test_df.csv", index=False
)

for set in ["train", "val", "test"]:
    df = eval(f"{set}_df")
    X = []

    for pasteur_id in df.paster_id.unique():
        X.append(df[df.paster_id == pasteur_id]["zone_temp"].to_numpy())

    X = np.array(X, dtype=object)
    y = df.drop_duplicates(subset=["paster_id"])["PU"].to_numpy()
    np.save(f"/home/stefanos/pasteurAIzer/data/time_series/numpy/X_{set}.npy", X)
    np.save(f"/home/stefanos/pasteurAIzer/data/time_series/numpy/y_{set}.npy", y)