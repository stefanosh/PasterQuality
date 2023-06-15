import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from tqdm import tqdm


# Same as sklearn, but convert MAPE to percentage
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Given the list of product temperatures, calculate the pus for a given pasteurization batch/recording
def calculate_pus(product_pred_list):
    pu_pred = 0
    for i in range(len(product_pred_list)):
        pu_pred = pu_pred + (0.166667 * pow(1.393, (product_pred_list[i] - 60)))
    return pu_pred


# Load the initial dataframes and set correctly the datasets to be evaluated
def initialize_sequential_evaluation(test_filepath):
    
    sequential_test_df = pd.read_csv(test_filepath)

    sequential_pred_df = sequential_test_df.copy()

    # Create metrics dataframe
    metrics_columns = ["paster_id", "time_steps", "real_pus", "pred_pus", "error_pu",
            "rmse_water", "r2_water", "mape_water", "mae_water", "rmse_product", "r2_product", "mape_product", "mae_product" ]
    metrics_df = pd.DataFrame(columns=metrics_columns)
    metrics_df['paster_id'] = sequential_test_df['paster_id'].unique()

    feature_columns = ['factory_env_temp', 'time_step', 'paster_program', 'zone_number', 'zone_temp', 'prev_zone_temp', 'next_zone_temp', 'lag_water_temp', 'lag_product_temp']

    # Remove values from lag features that will be filled step by step with the predictions
    sequential_pred_df[['lag_water_temp', 'lag_product_temp']] = np.nan

    # Then fill again only the first value of the lags, for each paster_id, (in order to enable the first prediction of the model)
    for paster_id in sequential_test_df.paster_id.unique():
        lag_water_temp_backup = sequential_test_df.loc[sequential_test_df.paster_id == paster_id, 'lag_water_temp'].iloc[0]
        lag_product_temp_backup = sequential_test_df.loc[sequential_test_df.paster_id == paster_id, 'lag_product_temp'].iloc[0]
        idx = sequential_pred_df.loc[sequential_pred_df.paster_id == paster_id, 'lag_water_temp'].index[0]
        sequential_pred_df.loc[idx, 'lag_water_temp'] = lag_water_temp_backup
        sequential_pred_df.loc[idx, 'lag_product_temp'] = lag_product_temp_backup

    return (sequential_test_df, sequential_pred_df, metrics_df, feature_columns)


# For each pasteurization batch (=paster_id), make the step-by-step predictions
def sequential_evaluation(model, sequential_pred_df, feature_columns, verbose):
    start_time = time.time()
    record_counter = 0
    for paster_id in tqdm(sequential_pred_df.paster_id.unique()):
        record_time = time.time()
        record_counter += 1
        record_df = sequential_pred_df.loc[sequential_pred_df.paster_id == paster_id]
        lag_water_temp = 0
        lag_product_temp = 0
        for idx in record_df.index:
            # Iterate through the records and make the step_by_step predictions
            if idx == record_df.index[0]:
                input = sequential_pred_df.loc[idx, feature_columns].to_list()
            else:
                sequential_pred_df.loc[idx, 'lag_water_temp'] = lag_water_temp
                sequential_pred_df.loc[idx, 'lag_product_temp'] = lag_product_temp
            
            input = sequential_pred_df.loc[idx, feature_columns].to_list()
            input = np.array(input).reshape(1, -1)
            prediction = model.predict(input)[0].round(2)
            sequential_pred_df.loc[idx, 'water_temp'] = prediction[0]
            sequential_pred_df.loc[idx, 'product_temp'] = prediction[1]
            lag_water_temp = prediction[0]
            lag_product_temp = prediction[1]

        time_left = np.round(((time.time() - record_time) * (len(sequential_pred_df.paster_id.unique()) - record_counter)), 3)
        formatted_time_left = "{:,.3f}".format(time_left)
        time_left = float(formatted_time_left.replace(",", ""))
        if verbose:
            print("\n====================================================================================================")
            print(f"Evaluated paster_id: {paster_id} | {record_counter} out of {len(sequential_pred_df.paster_id.unique())} | Record length: {len(record_df)} | Time elapsed: {np.round((time.time() - record_time), 3)} seconds. | Time left: {time_left} seconds or {np.round((time_left / 60), 3)} minutes")
            print("====================================================================================================\n")
            
    end_time = time.time()
    duration = end_time - start_time
    print("Step by step simulation duration for all test instances:", duration, "seconds OR", duration / 60, "minutes")
    return sequential_pred_df, duration


# Calculate for each record the regression metrics for prediction water, product temperatures and the PUs
def calculate_detailed_sequential_metrics(sequential_pred_df, sequential_test_df, metrics_df):

    for paster_id in sequential_pred_df.paster_id.unique():
        y_test = sequential_test_df.loc[sequential_test_df.paster_id == paster_id, 'water_temp'].to_list()
        y_pred = sequential_pred_df.loc[sequential_pred_df.paster_id == paster_id, 'water_temp'].to_list()

        # Print if y_test contains nan values
        if np.isnan(y_test).any():
            print("y_test contains nan values")
            print(y_test)

        # Print if y_pred contains nan values
        if np.isnan(y_pred).any():
            print("y_pred contains nan values")
            print(y_pred)

        # Metrics for water
        metrics_df.loc[metrics_df.paster_id == paster_id, 'time_steps'] = len(y_test)
        metrics_df.loc[metrics_df.paster_id == paster_id, 'rmse_water'] = mean_squared_error(y_test, y_pred, squared=False)
        metrics_df.loc[metrics_df.paster_id == paster_id, 'r2_water'] = r2_score(y_test, y_pred)
        metrics_df.loc[metrics_df.paster_id == paster_id, 'mape_water'] = mean_absolute_percentage_error(y_test, y_pred)
        metrics_df.loc[metrics_df.paster_id == paster_id, 'mae_water'] = mean_absolute_error(y_test, y_pred)

        # Metrics for product
        y_test = sequential_test_df.loc[sequential_test_df.paster_id == paster_id, 'product_temp'].to_list()
        y_pred = sequential_pred_df.loc[sequential_pred_df.paster_id == paster_id, 'product_temp'].to_list()

        # Print if y_test contains nan values
        if np.isnan(y_test).any():
            print("y_test contains nan values")
            print(y_test)

        # Print if y_pred contains nan values
        if np.isnan(y_pred).any():
            print("y_pred contains nan values")
            print(y_pred)

        metrics_df.loc[metrics_df.paster_id == paster_id, 'rmse_product'] = mean_squared_error(y_test, y_pred, squared=False)
        metrics_df.loc[metrics_df.paster_id == paster_id, 'r2_product'] = r2_score(y_test, y_pred)
        metrics_df.loc[metrics_df.paster_id == paster_id, 'mape_product'] = mean_absolute_percentage_error(y_test, y_pred)
        metrics_df.loc[metrics_df.paster_id == paster_id, 'mae_product'] = mean_absolute_error(y_test, y_pred)


        # Metrics for pus
        real_pus = calculate_pus(y_test)
        pred_pus = calculate_pus(y_pred)
        metrics_df.loc[metrics_df.paster_id == paster_id, 'real_pus'] = real_pus
        metrics_df.loc[metrics_df.paster_id == paster_id, 'pred_pus'] = pred_pus
        metrics_df.loc[metrics_df.paster_id == paster_id, 'error_pu'] = abs(real_pus - pred_pus)

    for column in metrics_df.columns:
        if column not in ['paster_id']:
            metrics_df[column] = metrics_df[column].astype(float).round(2)
    
    return metrics_df


# Total metrics for water, product and pu, aggregated as weighted average (based on water and product time_steps) from all records
# Calculating weighted average of metrics for water and product because the records have different lenghts (time_steps)
# The difference with the classic metrics_df.describe() are small, but it's just more accurate this way
def calculate_total_sequential_metrics(metrics_df):
    lengths = metrics_df['time_steps'].to_list()
    total_metrics = {}
    for target in ['water', 'product']:
        for metric in ['rmse', 'r2', 'mape', 'mae']:
            weighted_mean = np.average(metrics_df[f'{metric}_{target}'], weights=lengths)
            weighted_variance = np.average((metrics_df[f'{metric}_{target}'] - weighted_mean)**2, weights=lengths)
            weighted_std = np.sqrt(weighted_variance)
            total_metrics[f'mean_{metric}_{target}'] = weighted_mean
            total_metrics[f'std_{metric}_{target}'] = weighted_std


    # Some statistics on real and predicted pus
    real_pus = metrics_df['real_pus'].to_list()
    pred_pus = metrics_df['pred_pus'].to_list()
    total_metrics['mean_real_pu'] = np.average(real_pus)
    total_metrics['std_real_pu'] = np.sqrt(np.average((real_pus - total_metrics['mean_real_pu'])**2))
    total_metrics['min_real_pu'] = np.min(real_pus)
    total_metrics['max_real_pu'] = np.max(real_pus)
    total_metrics['min_pred_pu'] = np.min(pred_pus)
    total_metrics['max_pred_pu'] = np.max(pred_pus)
    total_metrics['mean_pred_pu'] = np.average(pred_pus)
    total_metrics['std_pred_pu'] = np.sqrt(np.average((pred_pus - total_metrics['mean_pred_pu'])**2))
    
    # Total metrics for pus
    total_metrics['mape_pu'] = mean_absolute_percentage_error(real_pus, pred_pus)
    total_metrics['mae_pu'] = mean_absolute_error(real_pus, pred_pus)
    total_metrics['rmse_pu'] = mean_squared_error(real_pus, pred_pus, squared=False)
    total_metrics['r2_pu'] = r2_score(real_pus, pred_pus)

    total_metrics_df = pd.DataFrame(total_metrics, index=[0])
    return total_metrics_df