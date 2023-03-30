#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from datetime import datetime
import string


# General settings
warnings.simplefilter('ignore', ConvergenceWarning)
TRAINING_REPS = 10
LOG_INTERVAL = 1
TRAIN = True
FIRST_TRAINING = True
MOD_ORD = 5
FILE_NAME = 'armax5_meanpar_err_force_delout_norm.pkl'

# Load dataset
dataframe = pd.read_pickle('./dataset_error_force_del_norm.pkl')
X = dataframe["x"]
Y = dataframe["y"]
train_len = int(0.8 * len(X.values[0].time))
test_len = int(0.2 * len(X.values[0].time))

if TRAIN:
    # Define ARMAX model
    if FIRST_TRAINING:
        model = ARIMA(endog=Y.values[0].data[:train_len], exog=X.values[0].data[:train_len], order=(MOD_ORD, 0, MOD_ORD))
    else:
        model = pickle.load(open(FILE_NAME, 'rb'))
    res = model.fit()
    prev_params = res.params
    param_list = []  # np.empty((1, len(prev_params)))
    mean_params = np.empty(len(prev_params))
    param_list.append(prev_params)
    # Fit ARMAX model for each experiment
    for row in range(1, X.values.shape[0]):
        X_data_ = X.values[row].data[:train_len]
        Y_data_ = Y.values[row].data[:train_len]
        for rep in range(0, TRAINING_REPS):
            res_update = model.fit(start_params=prev_params).apply(endog=Y_data_, exog=X_data_)
            model_params = res_update.params
            model.update(params=model_params)
            prev_params = model_params
        param_list.append(model_params)
        # prev_params = np.random.uniform(low=-1.0, high=1.0, size=len(prev_params))
        prev_params = ARIMA(endog=Y.values[np.random.randint(low=0, high=X.values.shape[0])].data[:train_len],
                            exog=X.values[np.random.randint(low=0, high=X.values.shape[0])].data[:train_len],
                            order=(MOD_ORD, 0, MOD_ORD)).fit().params
        if row % LOG_INTERVAL == 0:
            print(prev_params)
    param_array = np.array(param_list)
    for col in range(0, len(model_params)):
        mean_params[col] = np.mean(param_array[:, col])
    final_res = model.fit(start_params=mean_params)
    # Save model
    pickle.dump(final_res, open(FILE_NAME, 'wb'))
else:
    res = pickle.load(open(FILE_NAME, 'rb'))
    # res = model.fit()

# Forecast
rand_idx = np.random.randint(low=0, high=99)
time = X.values[rand_idx].time
gtruth = Y.values[rand_idx].data
test_time = X.values[rand_idx].time[-test_len:]
test_input = X.values[rand_idx].data[-test_len:]
gtruth_test = Y.values[rand_idx].data[-test_len:]
forecast_samples = len(test_input)

forecasting = res.forecast(steps=forecast_samples, exog=test_input[:forecast_samples])
observations = res.apply(endog=gtruth, exog=X.values[rand_idx].data)


# Plots
title_str = 'idx: ' + str(rand_idx)
plt.figure(1)
plt.plot(time[:train_len], res.fittedvalues)
plt.plot(time[:train_len], gtruth[:train_len])
plt.xlabel('time')
plt.ylabel('observation')
plt.legend(['observation', 'truth'])
plt.title(title_str)
plt.grid()

plt.figure(2)
plt.plot(test_time[:forecast_samples], forecasting)
plt.plot(test_time[:forecast_samples], gtruth_test[:forecast_samples])
plt.xlabel('time')
plt.ylabel('forecasting')
plt.legend(['forecasting', 'truth'])
plt.title(title_str)
plt.grid()

plt.show()
