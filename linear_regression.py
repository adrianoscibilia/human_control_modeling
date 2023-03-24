#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# General settings
warnings.simplefilter('ignore', ConvergenceWarning)
TRAINING_REPS = 10
LOG_INTERVAL = 10
TRAIN = True
FIRST_TRAINING = False
MOD_ORD = 5
FILE_NAME = 'armax5_err_force_raw.pkl'

# Load dataset
dataframe = pd.read_pickle('./dataset_error_force_timeseries_raw.pkl')
X = dataframe["x"]
Y = dataframe["y"]
train_len = int(0.8 * len(X.values[0].time))
test_len = int(0.2 * len(X.values[0].time))
# X_train = []
# Y_train = []

if TRAIN:
    # Define ARMAX model
    if FIRST_TRAINING:
        model = ARIMA(endog=Y.values[0].data[:train_len], exog=X.values[0].data[:train_len], order=(MOD_ORD, 0, MOD_ORD),
                      enforce_stationarity=False,
                      enforce_invertibility=False)
    else:
        model = pickle.load(open(FILE_NAME, 'rb'))
    prev_params = model.fit().params

    # Fit ARMAX model for each experiment
    for row in range(1, X.values.shape[0]):
        X_data_ = X.values[row].data[:train_len]
        Y_data_ = Y.values[row].data[:train_len]
        # X_train.append(X_data_)
        # Y_train.append(Y_data_)
        for rep in range(0, TRAINING_REPS):
            model = ARIMA(endog=Y_data_, exog=X_data_, order=(MOD_ORD, 0, MOD_ORD),
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            model_params = model.update(params=prev_params)
            res = model.fit(start_params=model_params)  # .apply(endog=Y_data_, exog=X_data_)
            prev_params = res.params
        if row % LOG_INTERVAL == 0:
            print(prev_params)
    # Save model
    pickle.dump(model, open(FILE_NAME, 'wb'))
else:
    model = pickle.load(open(FILE_NAME, 'rb'))

# Forecast test set
rand_idx = np.random.randint(low=0, high=99)
test_time = X.values[rand_idx].time[-test_len:]
test_input = X.values[rand_idx].data[-test_len:]
gtruth = Y.values[rand_idx].data[-test_len:]

prediction = res.forecast(steps=test_len, exog=test_input)

# Plots
plt.figure(1)
plt.plot(test_time, prediction)
plt.plot(test_time, gtruth)
plt.xlabel('time')
plt.ylabel('prediction')
plt.grid()

plt.show()
