import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.evaluation import *


project_path = '/home/stefanos/pasteurAIzer'
test_filepath = f'{project_path}/data/sequential/csv/test_df.csv'
(sequential_test_df, sequential_pred_df, metrics_df, feature_columns) = initialize_sequential_evaluation(test_filepath)

print("Shape of test: ", sequential_test_df.shape)

# Load tabular sets
merge_trainval = True
X_train = pd.read_csv("/home/stefanos/pasteurAIzer/data/sequential/csv/X_train.csv")
X_val = pd.read_csv("/home/stefanos/pasteurAIzer/data/sequential/csv/X_val.csv")
y_train = pd.read_csv("/home/stefanos/pasteurAIzer/data/sequential/csv/y_train.csv")
y_val = pd.read_csv("/home/stefanos/pasteurAIzer/data/sequential/csv/y_val.csv")

X_train = X_train[feature_columns]
X_val = X_val[feature_columns]

if merge_trainval:
    X_train = pd.concat([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)
    X_val = None
    y_val = None
    print("Shape of train:", X_train.shape)
else:
    print(X_train.shape)
    print(X_val.shape)

# Fit a model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=10, random_state=42))
model.fit(X_train.values, y_train)

# Make predictions on the test set, using the recursive/sequential/step_by_step approach and not the traditional one as below
# The purpose is to evaluate the model per pasteurization batch and not by individual rows, using the predictions (water, product temperatures)
# of the previous timesteps as lag inputs for the next timesteps
# y_pred = model.predict(X_test)

sequential_pred_df, duration = sequential_evaluation(model, sequential_pred_df, feature_columns, verbose=True)

metrics_df = calculate_detailed_sequential_metrics(sequential_pred_df, sequential_test_df, metrics_df)
total_metrics_df = calculate_total_sequential_metrics(metrics_df)

metrics_df.to_csv(f'{project_path}/src/experiments/examples/results/detailed_records_metrics.csv', index=False)
metrics_df.describe().to_csv(f'{project_path}/src/experiments/examples/results/statistics_records_metrics.csv', index=True)
total_metrics_df.to_csv(f'{project_path}/src/experiments/examples/results/total_sequential_metrics.csv', index=False)