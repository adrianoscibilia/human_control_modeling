#!/usr/bin/env python3

import numpy as np
from bagpy import bagreader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat
import pickle
import matplotlib.pyplot as plt
import numpy as np
from bagpy import bagreader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat


def find_bag_lenght(file_path):
    b = bagreader(file_path)
    ACT_POS_MSG = b.message_by_topic('/actual_position')
    EXPERIMENT_WINDOW_MSG = b.message_by_topic("/experiment_window")
    act_pos_pd = pd.read_csv(ACT_POS_MSG)
    experiment_window_pd = pd.read_csv(EXPERIMENT_WINDOW_MSG)
    act_pos = act_pos_pd['pose.position.x']
    experiment_window = experiment_window_pd['data']
    lenghts = []
    lenghts.append(len(act_pos))
    lenghts.append(len(experiment_window))
    minimum_lenght = np.amin(lenghts)
    final_lenght = 0
    for i in range(0, minimum_lenght):
        if experiment_window[i]:
            final_lenght += 1
    return final_lenght


def bag_read(file_path):
    b = bagreader(file_path)

    # read messages for each topic
    H_ERR_MSG = b.message_by_topic('/human_error')
    ACT_POS_MSG = b.message_by_topic('/actual_position')
    REF_POS_MSG = b.message_by_topic('/reference_position')
    # POS_ERR_MSG = b.message_by_topic('/position_error')
    FORCE_MSG = b.message_by_topic('/human_force')
    EXPERIMENT_WINDOW_MSG = b.message_by_topic("/experiment_window")

    # import csv as panda dataframe
    h_err_pd = pd.read_csv(H_ERR_MSG)
    act_pos_pd = pd.read_csv(ACT_POS_MSG)
    ref_pos_pd = pd.read_csv(REF_POS_MSG)
    # pos_err_pd = pd.read_csv(POS_ERR_MSG)
    force_pd = pd.read_csv(FORCE_MSG)
    experiment_window_pd = pd.read_csv(EXPERIMENT_WINDOW_MSG)

    # create vectors
    # time = act_pos_pd['Time'] - act_pos_pd.loc[0, 'Time']
    # h_err_x = h_err_pd['pose.position.x']
    # h_err_y = h_err_pd['pose.position.y']
    # act_pos_x = act_pos_pd['pose.position.x']
    # act_pos_y = act_pos_pd['pose.position.y']
    ref_pos_x = ref_pos_pd['pose.position.x']
    ref_pos_y = ref_pos_pd['pose.position.y']
    # pos_err_x = pos_err_pd['pose.position.x']
    # pos_err_y = pos_err_pd['pose.position.y']
    h_force_x = force_pd['wrench.force.x']
    h_force_y = force_pd['wrench.force.y']
    experiment_window = experiment_window_pd['data']

    # SIZE ADJUSTMENTS

    # time_adj = []
    # h_err_adj_x = []
    # h_err_adj_y = []
    force_adj_x = []
    force_adj_y = []
    ref_pos_adj_x = []
    ref_pos_adj_y = []
    # act_pos_adj_x = []
    # act_pos_adj_y = []
    lenghts = []
    error_adj_x = []
    error_adj_y = []

    lenghts.append(len(h_force_x))
    lenghts.append(len(h_force_y))
    # lenghts.append(len(h_err_x))
    # lenghts.append(len(h_err_y))
    lenghts.append(len(ref_pos_x))
    lenghts.append(len(ref_pos_y))
    # lenghts.append(len(act_pos_x))
    # lenghts.append(len(act_pos_y))
    lenghts.append(len(experiment_window))
    minimum_lenght = np.amin(lenghts)

    window_vector = np.zeros(len(experiment_window))
    for i in range(0, minimum_lenght):
        if experiment_window[i]:
            window_vector[i] = 1

    for idx in range(0, minimum_lenght):
        if window_vector[idx] == 1:
            # time_adj.append(time[idx])
            force_adj_x.append(h_force_x[idx]-h_force_x[0])
            force_adj_y.append(h_force_y[idx]-h_force_y[0])
            # h_err_adj_x.append(h_err_x[idx]-h_err_x[0])
            # h_err_adj_y.append(h_err_y[idx]-h_err_y[0])
            ref_pos_adj_x.append(ref_pos_x[idx] - ref_pos_x[0])
            ref_pos_adj_y.append(ref_pos_y[idx] - ref_pos_y[0])
            # act_pos_adj_x.append(act_pos_x[idx] - act_pos_x[0])
            # act_pos_adj_y.append(act_pos_y[idx] - act_pos_y[0])
            # error_adj_x.append(ref_pos_x[idx] - act_pos_x[idx])
            # error_adj_y.append(ref_pos_y[idx] - act_pos_y[idx])

    # CONVERT TO NP ARRAYS
    input1_array = np.array(ref_pos_adj_x)
    input2_array = np.array(ref_pos_adj_y)
    output1_array = np.array(force_adj_x)
    output2_array = np.array(force_adj_y)

    # input1_norm = []
    # input2_norm = []
    # output1_norm = []
    # output2_norm = []
    # for idx in range(0, len(input1_array)):
    #     input1_norm.append((2*(input1_array[idx] - np.min(input1_array))/(np.max(input1_array) - np.min(input1_array)))-1)
    #     input2_norm.append((2*(input2_array[idx] - np.min(input2_array))/(np.max(input2_array) - np.min(input2_array)))-1)
    #     output1_norm.append((2*(output1_array[idx] - np.min(output1_array))/(np.max(output1_array) - np.min(output1_array)))-1)
    #     output2_norm.append((2*(output2_array[idx] - np.min(output2_array))/(np.max(output2_array) - np.min(output2_array)))-1)
    # input1_norm_array = np.array(input1_norm)
    # input2_norm_array = np.array(input2_norm)
    # output1_norm_array = np.array(output1_norm)
    # output2_norm_array = np.array(output2_norm)

    return input1_array, input2_array, output1_array, output2_array


def arrays_to_dataframe(input1, input2, output1, output2):

    input1_df = pd.DataFrame({'data': input1})
    input2_df = pd.DataFrame({'data': input2})
    output1_df = pd.DataFrame({'data': output1})
    output2_df = pd.DataFrame({'data': output2})

        # NORMALIZE
        # Fit scalers

        # input_scaler = StandardScaler().fit(input_df.data.values.reshape(-1, 1))
        # output_scaler = StandardScaler().fit(output_df.data.values.reshape(-1, 1))
        # input_scaler = MinMaxScaler().fit(input_df.data.values.reshape(-1, 1))
        # output_scaler = MinMaxScaler().fit(output_df.data.values.reshape(-1, 1))
    return input1_df, input2_df, output1_df, output2_df




subjects_complete = ['/Alessandro_Scano', '/Claudia_Pagano', '/Francesco_Airoldi', '/Matteo_Malosio', '/Michele',
                     '/Giorgio_Nicola', '/Paolo_Franceschi', '/Marco_Faroni', '/Stefano_Mutti', '/Trunal']
subjects = ['/Claudia_Pagano', '/Marco_Faroni']
bag_folder_base = '/home/adriano/projects/bag/controller_adriano/'
file_extension = '.bag'
experiment_type = 'step'
underscore = '_'

n_of_iterations = 10
subjects = subjects_complete
DEL = False


# lenghts = []
# for n in range(0, len(subjects)):
#     bag_folder = bag_folder_base + subjects[n] + '/step/'
#     for i in range(0, n_of_iterations):
#         experiment_number = str(i+1)
#         file_name = experiment_type + underscore + experiment_number
#         bag_input = bag_folder + file_name + file_extension
#         len_row_col = find_bag_lenght(bag_input)
#         lenghts.append(len_row_col)
#
# min_len = np.amin(lenghts)

min_len = 7500

x1 = []
x2 = []
y1 = []
y2 = []
# x_scalers = []
# y_scalers = []
for n in range(0, len(subjects)):
    bag_folder = bag_folder_base + subjects[n] + '/step/'
    for i in range(0, n_of_iterations):
        experiment_number = str(i+1)
        file_name = experiment_type + underscore + experiment_number
        bag_input = bag_folder + file_name + file_extension
        x_n_i_1, x_n_i_2, y_n_i_1, y_n_i_2 = bag_read(bag_input)
        if len(x_n_i_1) > min_len or len(x_n_i_2) > min_len or len(y_n_i_1) > min_len or len(y_n_i_1) > min_len:
            x_n_i_1 = x_n_i_1[:min_len]
            x_n_i_2 = x_n_i_2[:min_len]
            y_n_i_1 = y_n_i_1[:min_len]
            y_n_i_2 = y_n_i_2[:min_len]
        x_n_1, x_n_2, y_n_1, y_n_2 = arrays_to_dataframe(x_n_i_1, x_n_i_2, y_n_i_1, y_n_i_2)
        # x_norm = x_scale.transform(x_n.data.values.reshape(-1, 1))
        # y_norm = x_scale.transform(y_n.data.values.reshape(-1, 1))
        x1.append(x_n_1)
        x2.append(x_n_2)
        y1.append(y_n_1)
        y2.append(y_n_2)
        # x_scalers.append(x_scale)
        # y_scalers.append(y_scale)


dataset = pd.DataFrame({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
pickle.dump(dataset, open('dataset/dataset_ref_f_raw.pkl', 'wb'))