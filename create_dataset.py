#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import numpy as np
import control as cnt
from bagpy import bagreader
import pandas as pd
from scipy import signal
import sklearn.metrics as skl


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
    time = act_pos_pd['Time'] - act_pos_pd.loc[0, 'Time']
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

    for i in range(len(act_pos_x)):
        ref_pos = np.sqrt((np.power(ref_pos_x, 2) + np.power(ref_pos_y, 2)))
        act_pos = np.sqrt((np.power(act_pos_x, 2) + np.power(act_pos_y, 2)))
        h_force = np.sqrt((np.power(h_force_x, 2) + np.power(h_force_y, 2)))
        # pos_err = np.sqrt((np.power(pos_err_x, 2) + np.power(pos_err_y, 2)))
        # h_err = np.sqrt((np.power(h_err_x, 2) + np.power(h_err_y, 2)))

    # SIZE ADJUSTMENTS
    ref_pos_input = ref_pos
    act_pos_input = act_pos

    time_adj = []
    h_err_adj = []
    force_adj = []
    ref_pos_adj = []
    act_pos_adj = []
    lenghts = []
    error_adj = []
    error_adj_dot = []
    lenghts.append(len(h_force))
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
            time_adj.append(time[idx])
            force_adj.append(h_force[idx])
            ref_pos_adj.append(ref_pos_input[idx])
            act_pos_adj.append(act_pos_input[idx])
            error_adj.append(ref_pos_input[idx] - act_pos_input[idx])

    for i in range(1, len(time_adj)):
        time_adj[i] = time_adj[i] - time_adj[0]
    time_adj[0] = 0

    # CONVERT TO NP ARRAYS
    error_adj_array = np.array(error_adj)
    force_array = np.array(force_adj)
    # ref_pos_adj_array = np.array(ref_pos_adj)
    # act_pos_adj_array = np.array(act_pos_adj)
    return error_adj_array, force_array


subjects_complete = ['/Alessandro_Scano', '/Claudia_Pagano', '/Francesco_Airoldi', '/Matteo_Malosio', '/Michele',
                     '/Giorgio_Nicola', '/Paolo_Franceschi', '/Marco_Faroni', '/Stefano_Mutti', '/Trunal']
subjects = ['/Claudia_Pagano', '/Marco_Faroni']
bag_folder_base = '/home/adriano/projects/ros_ws/src/controller_adriano/bag'
file_extension = '.bag'
experiment_type = 'step'
underscore = '_'
n_of_iterations = 11

lenghts = []
len_row = 0
for n in range(0, len(subjects_complete)):
    bag_folder = bag_folder_base + subjects_complete[n] + '/step/'
    for i in range(1, n_of_iterations):
        experiment_number = str(i)
        file_name = experiment_type + underscore + experiment_number
        bag_input = bag_folder + file_name + file_extension
        len_row_col = find_bag_lenght(bag_input)
        len_row = len_row + len_row_col
    lenghts.append(len_row)
    len_row = 0

max_len = np.amax(lenghts)

x_n = np.empty(0)
y_n = np.empty(0)
x = np.empty((len(subjects_complete), max_len))
y = np.empty((len(subjects_complete), max_len))
len_row = 0

for n in range(0, len(subjects_complete)):
    bag_folder = bag_folder_base + subjects_complete[n] + '/step/'
    for i in range(1, n_of_iterations):
        experiment_number = str(i)
        file_name = experiment_type + underscore + experiment_number
        bag_input = bag_folder + file_name + file_extension
        x_n_i, y_n_i = bag_read(bag_input)
        len_row = len_row + len(x_n_i)
        x_n = np.append(x_n, x_n_i)
        y_n = np.append(y_n, y_n_i)
        if i == (n_of_iterations - 1):
            if len_row < max_len:
                len_diff = max_len - len_row
                zero_diff = np.zeros(len_diff)
                x_n = np.append(x_n, zero_diff)
                y_n = np.append(y_n, zero_diff)
    x[n, :] = x_n
    y[n, :] = y_n
    x_n = np.empty(0)
    y_n = np.empty(0)
    len_row = 0

dataset = {'x': [x], 'y': [y]}
dataframe = pd.DataFrame(dataset)
print("dataframe shape: ", dataframe.shape)
dataframe.to_pickle('dataframe.pkl')
dataframe.to_csv('dataframe.csv')
