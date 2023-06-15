import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.evaluation import mean_absolute_percentage_error

# Load tabular sets
merge_trainval = True
X_train = pd.read_csv("/home/stefanos/pasteurAIzer/data/tabular/csv/X_train.csv")
X_val = pd.read_csv("/home/stefanos/pasteurAIzer/data/tabular/csv/X_val.csv")
X_test = pd.read_csv("/home/stefanos/pasteurAIzer/data/tabular/csv/X_test.csv")
y_train = pd.read_csv("/home/stefanos/pasteurAIzer/data/tabular/csv/y_train.csv")
y_val = pd.read_csv("/home/stefanos/pasteurAIzer/data/tabular/csv/y_val.csv")
y_test = pd.read_csv("/home/stefanos/pasteurAIzer/data/tabular/csv/y_test.csv")

if merge_trainval:
    X_train = pd.concat([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)
    X_val = None
    y_val = None

    print(X_train.shape)
    print(X_test.shape)
else:
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)


# Fit a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.ravel())

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")