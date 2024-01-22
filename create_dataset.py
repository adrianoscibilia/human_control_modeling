#!/usr/bin/env python3
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
    # H_ERR_MSG = b.message_by_topic('/human_error')
    ACT_POS_MSG = b.message_by_topic('/actual_position')
    REF_POS_MSG = b.message_by_topic('/reference_position')
    # POS_ERR_MSG = b.message_by_topic('/position_error')
    FORCE_MSG = b.message_by_topic('/human_force')
    EXPERIMENT_WINDOW_MSG = b.message_by_topic("/experiment_window")

    # import csv as panda dataframe
    # h_err_pd = pd.read_csv(H_ERR_MSG)
    act_pos_pd = pd.read_csv(ACT_POS_MSG)
    ref_pos_pd = pd.read_csv(REF_POS_MSG)
    # pos_err_pd = pd.read_csv(POS_ERR_MSG)
    force_pd = pd.read_csv(FORCE_MSG)
    experiment_window_pd = pd.read_csv(EXPERIMENT_WINDOW_MSG)

    # create vectors
    # time = act_pos_pd['Time'] - act_pos_pd.loc[0, 'Time']
    # h_err_x = h_err_pd['pose.position.x']
    # h_err_y = h_err_pd['pose.position.y']
    act_pos_x = act_pos_pd['pose.position.x']
    act_pos_y = act_pos_pd['pose.position.y']
    ref_pos_x = ref_pos_pd['pose.position.x']
    ref_pos_y = ref_pos_pd['pose.position.y']
    # pos_err_x = pos_err_pd['pose.position.x']
    # pos_err_y = pos_err_pd['pose.position.y']
    h_force_x = force_pd['wrench.force.x']
    h_force_y = force_pd['wrench.force.y']
    experiment_window = experiment_window_pd['data']

    for i in range(len(h_force_x)):
        ref_pos = np.sqrt((np.power(ref_pos_x, 2) + np.power(ref_pos_y, 2)))
        act_pos = np.sqrt((np.power(act_pos_x, 2) + np.power(act_pos_y, 2)))
        h_force = np.sqrt((np.power(h_force_x, 2) + np.power(h_force_y, 2)))
        # pos_err = np.sqrt((np.power(pos_err_x, 2) + np.power(pos_err_y, 2)))
        # h_err = np.sqrt((np.power(h_err_x, 2) + np.power(h_err_y, 2)))

    # SIZE ADJUSTMENTS
    ref_pos_input = ref_pos
    act_pos_input = act_pos

    # time_adj = []
    # h_err_adj = []
    force_adj = []
    ref_pos_adj = []
    act_pos_adj = []
    lenghts = []
    # error_adj = []
    # error_adj_dot = []
    lenghts.append(len(h_force))
    # lenghts.append(len(h_err_adj))
    lenghts.append(len(ref_pos_input))
    lenghts.append(len(act_pos_input))
    lenghts.append(len(experiment_window))
    minimum_lenght = np.amin(lenghts)

    window_vector = np.zeros(len(experiment_window))
    for i in range(0, minimum_lenght):
        if experiment_window[i]:
            window_vector[i] = 1

    for idx in range(0, minimum_lenght):
        if window_vector[idx] == 1:
            # time_adj.append(time[idx])
            force_adj.append(h_force[idx])
            # h_err_adj.append(h_err_adj[idx])
            ref_pos_adj.append(ref_pos_input[idx])
            act_pos_adj.append(act_pos_input[idx])
            # error_adj.append(ref_pos_input[idx] - act_pos_input[idx])

    # CONVERT TO NP ARRAYS
    input_array = np.array(ref_pos_adj)
    output_array = np.array(force_adj)
    output2_array = np.array(act_pos_adj)

    input_norm = []
    output_norm = []
    output2_norm = []
    for idx in range(0, len(ref_pos_adj)):
        # input_norm.append((input_array[idx] - np.mean(input_array))/np.std(input_array))
        # output_norm.append((output_array[idx] - np.mean(output_array))/np.std(output_array))
        input_norm.append((input_array[idx] - np.min(input_array))/(np.max(input_array) - np.min(input_array)))
        output_norm.append((output_array[idx] - np.min(output_array))/(np.max(output_array) - np.min(output_array)))
        output2_norm.append((output2_array[idx] - np.min(output2_array)) / (np.max(output2_array) - np.min(output2_array)))

    input_norm_array = np.array(input_norm)
    output_norm_array = np.array(output_norm)
    output2_norm_array = np.array(output2_norm)

    return input_norm_array, output_norm_array, output2_norm_array


def arrays_to_dataframe(input, output, output2, DEL):

    input_df = pd.DataFrame({'data': input})
    output_df = pd.DataFrame({'data': output})
    output2_df = pd.DataFrame({'data': output2})

    if DEL:
        # Add delayed copies
        delays = [125, 625, 1250]
        output_del = [np.empty(len(output)), np.empty(len(output)), np.empty(len(output))]
        idx = 0

        for d in delays:
            output_del[idx][:d] = 0
            output_del[idx][d:len(output)] = output[d:len(output)]
            idx += 1

        output_df_d1 = pd.DataFrame({'data': output_del[0]})
        output_df_d2 = pd.DataFrame({'data': output_del[1]})
        output_df_d3 = pd.DataFrame({'data': output_del[2]})

        return input_df, output_df, output_df_d1, output_df_d2, output_df_d3
    else:
        # NORMALIZE
        # Fit scalers

        # input_scaler = StandardScaler().fit(input_df.data.values.reshape(-1, 1))
        # output_scaler = StandardScaler().fit(output_df.data.values.reshape(-1, 1))
        # input_scaler = MinMaxScaler().fit(input_df.data.values.reshape(-1, 1))
        # output_scaler = MinMaxScaler().fit(output_df.data.values.reshape(-1, 1))
        return input_df, output_df, output2_df  # input_scaler, output_scaler




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

x = []
y = []
y2 = []
x_scalers = []
y_scalers = []
for n in range(0, len(subjects)):
    bag_folder = bag_folder_base + subjects[n] + '/step/'
    for i in range(0, n_of_iterations):
        experiment_number = str(i+1)
        file_name = experiment_type + underscore + experiment_number
        bag_input = bag_folder + file_name + file_extension
        x_n_i, y_n_i, y2_n_i = bag_read(bag_input)
        if len(x_n_i) > min_len:
            x_n_i = x_n_i[:min_len]
            y_n_i = y_n_i[:min_len]
            y2_n_i = y2_n_i[:min_len]
        if DEL:
            x_n, y_n, y_n_d1, y_n_d2, y_n_d3 = arrays_to_dataframe(x_n_i, y_n_i, DEL)
            x.append(x_n)
            x.append(x_n)
            x.append(x_n)
            x.append(x_n)
            y.append(y_n)
            y.append(y_n_d1)
            y.append(y_n_d2)
            y.append(y_n_d3)
        else:
            x_n, y_n, y2_n = arrays_to_dataframe(x_n_i, y_n_i, y2_n_i, DEL)
            # x_norm = x_scale.transform(x_n.data.values.reshape(-1, 1))
            # y_norm = x_scale.transform(y_n.data.values.reshape(-1, 1))
            x.append(x_n)
            y.append(y_n)
            y2.append(y2_n)
            # x_scalers.append(x_scale)
            # y_scalers.append(y_scale)


dataset = pd.DataFrame({'x': x, 'y': y, 'y2': y2})
pickle.dump(dataset, open('dataset/dataset_ref_f_act_mod_norm.pkl', 'wb'))
# pickle.dump(x_scalers, open('x_minmax_scalers_100_0707.pkl', 'wb'))
# pickle.dump(y_scalers, open('y_minmax_scalers_100_0707.pkl', 'wb'))