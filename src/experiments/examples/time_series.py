import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.evaluation import mean_absolute_percentage_error


merge_trainval = True
X_train = np.load("/home/stefanos/pasteurAIzer/data/time_series/numpy/X_train.npy", allow_pickle=True)
X_val = np.load("/home/stefanos/pasteurAIzer/data/time_series/numpy/X_val.npy", allow_pickle=True)
X_test = np.load("/home/stefanos/pasteurAIzer/data/time_series/numpy/X_test.npy", allow_pickle=True)
y_train = np.load("/home/stefanos/pasteurAIzer/data/time_series/numpy/y_train.npy", allow_pickle=True)
y_val = np.load("/home/stefanos/pasteurAIzer/data/time_series/numpy/y_val.npy", allow_pickle=True)
y_test = np.load("/home/stefanos/pasteurAIzer/data/time_series/numpy/y_test.npy",  allow_pickle=True)

if merge_trainval:
    X_train = np.concatenate([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)
    X_val = None
    y_val = None

    print(X_train.shape)
    print(X_test.shape)
else:
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)