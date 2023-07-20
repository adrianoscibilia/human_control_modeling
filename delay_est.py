import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
import control as cnt


def window_param_extraction(input_array, window_size, LOG):
    window_size_samples = int(window_size / 0.008)
    start_index = 0
    time = np.linspace(start=0, stop=60, num=len(input_array))
    periods = []
    eps = []
    omega = []

    while start_index < len(input_array) - window_size_samples:
        window = input_array[start_index:(start_index + window_size_samples)]
        time_window = time[start_index:(start_index + window_size_samples)]
        window_peaks, _ = signal.find_peaks(window, height=0.01, distance=1)
        T = window_size/len(window_peaks)
        periods.append(T)
        w_n = 6.28/T
        omega.append(w_n)
        eps.append(3.5/(w_n*window_size))

        if LOG:
            print("T: ", T)
            print("w_n: ", w_n)

            plt.figure(1)
            plt.plot(window)
            plt.plot(window_peaks, window[window_peaks], 'x')
            plt.show()

        start_index += window_size_samples

    return periods, eps, omega


def tf_from_param(eps, omega, T_d_0):
    k = 1
    F = pow(omega, 2) * k * cnt.tf(1, [1, 2*eps*omega, pow(omega, 2)])
    (num, den) = cnt.pade(T_d_0, 4)
    U = cnt.tf(1, [1, 0]) - cnt.tf(1, [1, 0]) * cnt.tf(num, den)
    return F, U


dataframe = pd.read_pickle('dataset100_norm.pkl')

X = dataframe["x"]
Y = dataframe["y"]

data_x = np.empty((len(X.values), len(X.values[0]['data'])))
data_y = np.empty((len(Y.values), len(Y.values[0]['data'])))
for i in range(0, len(X.values)):
    data_x[i, :] = X.values[i]['data']
    data_y[i, :] = Y.values[i]['data']

time = np.linspace(start=0, stop=60, num=data_x.shape[1])
xdata = np.transpose(data_x)
ydata = np.transpose(data_y)

random_pk_test = np.random.randint(low=0, high=ydata.shape[1])
periods, eps, omega = window_param_extraction(ydata[:, random_pk_test], 5, False)

period_mean = np.mean(periods)
eps_mean = np.mean(eps)
omega_mean = np.mean(omega)
print("mean T: ", period_mean, "mean epsilon: ", eps_mean, "\tmean w_n: ", omega_mean)

F, U = tf_from_param(eps_mean, omega_mean, 1)
print("F: ", F, "U: ", U)