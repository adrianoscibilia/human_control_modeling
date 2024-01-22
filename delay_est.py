import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn.metrics as skl
from scipy import signal
import control as cnt


SIMULATE = True
EVALUATE = False


def window_param_extraction(input_array, window_size, LOG):
    window_size_samples = int(window_size / 0.008)
    start_index = 0
    # time = np.linspace(start=0, stop=60, num=len(input_array))
    periods = []
    eps = []
    omega = []

    while start_index < len(input_array) - window_size_samples:
        window = np.absolute(input_array[start_index:(start_index + window_size_samples)])
        # time_window = time[start_index:(start_index + window_size_samples)]
        window_peaks, _ = signal.find_peaks(window, height=0.01, distance=1)
        T = window_size/len(window_peaks)
        periods.append(T)
        w_n = 6.28/T
        omega.append(w_n)
        eps.append(3.9/(w_n*window_size))

        if LOG:
            # print("T: ", T)
            # print("w_n: ", w_n)
            plt.figure(1)
            plt.plot(window)
            plt.plot(window_peaks, window[window_peaks], 'x')
            plt.show()
        start_index += window_size_samples
    return periods, eps, omega


def tf_from_param(eps, omega, gain, T_d_0):
    k = gain
    F = pow(omega, 2) * k * cnt.tf(1, [1, 2*eps*omega, pow(omega, 2)])
    (num, den) = cnt.pade(T_d_0, 4)
    U = cnt.tf(1, [1, 0]) - cnt.tf(1, [1, 0]) * cnt.tf(num, den)
    return F, U


def estimate_delay(window_size, input_array, eps_mean, omega_mean, gain_0, T_d_0, PLOT):
    time = np.linspace(start=0, stop=60, num=len(input_array))

    gain = gain_0
    RMSE_gain = []
    gain_list = []
    while gain > 1:
        gain_list.append(gain)
        F, U = tf_from_param(eps_mean, omega_mean, gain, 0.35)
        sys = F * U
        response = cnt.forced_response(sys=sys, T=time, U=input_array)
        RMSE_gain.append(skl.mean_squared_error(y_true=input_array, y_pred=response.outputs, squared=False))

        if PLOT and gain == gain_0:
            fig_title = 'Gain optimization'
            plt.figure(2)
            plt.plot(time, response.outputs)
            plt.plot(time, input_array)
            plt.legend(['simulated', 'truth'])
            plt.grid()
            plt.title(fig_title)
            plt.show()

        gain -= 0.1
    RMSE_gain_min = np.argmin(RMSE_gain)
    gain_opt = np.around (gain_list[RMSE_gain_min], 2)

    window_size_samples = int(window_size / 0.008)
    T_d = T_d_0
    RMSE_delay = []
    delay_list = []
    while T_d > 0.1:
        delay_list.append(T_d)
        F, U = tf_from_param(eps_mean, omega_mean, gain_opt, T_d)
        sys = F * U
        start_index = 0
        RMSE = []
        while start_index < len(input_array) - window_size_samples:
            window = input_array[start_index:(start_index + window_size_samples)]
            time_window = time[start_index:(start_index + window_size_samples)]
            response_window = cnt.forced_response(sys=sys, T=time_window, U=window)
            RMSE_window = skl.mean_squared_error(y_true=window, y_pred=response_window.outputs, squared=False)
            RMSE.append(RMSE_window)
            start_index += window_size_samples

            if PLOT and T_d == T_d_0:
                fig_title = 'Delay optimization'
                plt.figure(3)
                plt.plot(time_window, response_window.outputs)
                plt.plot(time_window, window)
                plt.legend(['simulated', 'truth'])
                plt.grid()
                plt.title(fig_title)
                plt.show()

        RMSE_delay.append(np.mean(np.array(RMSE)))
        T_d -= 0.001
    RMSE_del_min = np.argmin(RMSE_delay)
    delay_opt = np.around(delay_list[RMSE_del_min], 3)

    return gain_opt, delay_opt, RMSE_delay[RMSE_del_min]

# MODULE ONLY DATASET
dataframe = pd.read_pickle('dataset/dataset_ref_f_act_mod_raw.pkl')

X = dataframe["x"]
Y = dataframe["y"]
# Y2 = dataframe["y2"]

data_x = np.empty((len(X.values), len(X.values[0]['data'])))
data_y = np.empty((len(Y.values), len(Y.values[0]['data'])))
# data_y2 = np.empty((len(Y2.values), len(Y2.values[0]['data'])))
for i in range(0, len(X.values)):
    data_x[i, :] = X.values[i]['data']
    data_y[i, :] = Y.values[i]['data']
    # data_y2[i, :] = Y2.values[i]['data']

# # LISTED RAW DATASET
# dataframe_xy = pd.read_pickle('dataset_ref_f_norm.pkl')
#
# X1 = dataframe_xy["x1"]
# X2 = dataframe_xy["x2"]
# Y1 = dataframe_xy["y1"]
# Y2 = dataframe_xy["y2"]
#
# data_x = np.empty((2*len(X1.values), len(X1.values[0]['data'])))
# data_y = np.empty((2*len(Y1.values), len(Y1.values[0]['data'])))
# for i in range (0, len(X1.values)):
#     data_x[i, :] = X1.values[i]['data']
#     data_x[i+100, :] = X2.values[i]['data']
#     data_y[i, :] = Y1.values[i]['data']
#     data_y[i+100, :] = Y2.values[i]['data']

time = np.linspace(start=0, stop=60, num=data_x.shape[1])
xdata = np.transpose(data_x)
ydata = np.transpose(data_y)
# y2data = np.transpose(data_y2)

if SIMULATE:
    random_pk_test = np.random.randint(low=0, high=ydata.shape[1])
    periods, eps, omega = window_param_extraction(ydata[:, random_pk_test], 5, False)

    period_mean = np.mean(periods)
    eps_mean = np.mean(eps)
    omega_mean = np.mean(omega)
    print("mean T: ", period_mean, "mean epsilon: ", eps_mean, "\tmean w_n: ", omega_mean)

    K, Td, RMSE_Td_min = estimate_delay(5, ydata[:, random_pk_test], eps_mean, omega_mean, 10, 1, False)
    print("gain opt: ", K, "delay opt: ", Td)

    F, U = tf_from_param(eps_mean, omega_mean, K, Td)
    sys = F * U
    y_sim = cnt.forced_response(sys=sys, T=time, U=ydata[:, random_pk_test])

    plt.figure(4)
    plt.title("Simulated response of optimal system")
    plt.plot(time, y_sim.outputs)
    plt.plot(time, ydata[:, random_pk_test], alpha=0.6)
    plt.legend(['simulated', 'true'])
    plt.grid()
    plt.show()

if EVALUATE:
    period_mean_list = []
    eps_mean_list = []
    omega_mean_list = []
    gains = []
    delays = []
    RMSE = []
    for subj_idx in range(0, ydata.shape[1]):
        periods, eps, omega = window_param_extraction(ydata[:, subj_idx], 5, False)
        period_mean = np.mean(periods)
        eps_mean = np.mean(eps)
        omega_mean = np.mean(omega)
        K, Td, RMSE_Td_min = estimate_delay(5, ydata[:, subj_idx], eps_mean, omega_mean, 5, 0.8, False)

        period_mean_list.append(period_mean)
        eps_mean_list.append(eps_mean)
        omega_mean_list.append(omega_mean)
        gains.append(K)
        delays.append(Td)
        RMSE.append(RMSE_Td_min)

    scores_df = pd.DataFrame({
        'Period': period_mean_list,
        'Epsilon': eps_mean_list,
        'Omega': omega_mean_list,
        'Gain': gains,
        'Delay': delays,
        'RMSE': RMSE
    })

    writer = pd.ExcelWriter("tables/scores_delay_identification_raw_2308.xlsx", engine='xlsxwriter')
    scores_df.to_excel(writer, sheet_name='Sheet1', startrow=1, header=False, index=False)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    # Get the dimensions of the dataframe.
    (max_row, max_col) = scores_df.shape

    # Create a list of column headers, to use in add_table().
    column_settings = []
    for header in scores_df.columns:
        column_settings.append({'header': header})

    # Add the table.
    worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})

    # Make the columns wider for clarity.
    worksheet.set_column(0, max_col - 1, 12)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()