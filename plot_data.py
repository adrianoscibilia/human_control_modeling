#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
import numpy as np
from bagpy import bagreader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat


def bag_read(file_path, str1, str2):
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

    # act_pos_x = act_pos_pd['pose.position.x']
    # act_pos_y = act_pos_pd['pose.position.y']
    # ref_pos_x = ref_pos_pd['pose.position.x']
    # ref_pos_y = ref_pos_pd['pose.position.y']
    plot_array1 = act_pos_pd[str1]
    plot_array2 = act_pos_pd[str2]
    experiment_window = experiment_window_pd['data']

    # for i in range(len(act_pos_x)):
    #     ref_pos = np.sqrt((np.power(ref_pos_x, 2) + np.power(ref_pos_y, 2)))
    #     act_pos = np.sqrt((np.power(act_pos_x, 2) + np.power(act_pos_y, 2)))
    #     h_force = np.sqrt((np.power(h_force_x, 2) + np.power(h_force_y, 2)))
    #     # pos_err = np.sqrt((np.power(pos_err_x, 2) + np.power(pos_err_y, 2)))
    #     # h_err = np.sqrt((np.power(h_err_x, 2) + np.power(h_err_y, 2)))

    lenghts = []
    lenghts.append(len(plot_array2))
    lenghts.append(len(plot_array2))
    lenghts.append(len(experiment_window))
    minimum_lenght = np.amin(lenghts)

    window_vector = np.zeros(len(experiment_window))
    for i in range(0, minimum_lenght):
        if experiment_window[i]:
            window_vector[i] = 1

    plt_list1_adj = []
    plt_list2_adj = []
    for idx in range(0, minimum_lenght):
        if window_vector[idx] == 1:
            plt_list1_adj.append(plot_array2[idx])
            plt_list2_adj.append(plot_array2[idx])

    # CONVERT TO NP ARRAYS
    input_array = np.array(plot_array1)
    output_array = np.array(plot_array2)

    return input_array, output_array


subjects_complete = ['/Alessandro_Scano', '/Claudia_Pagano', '/Francesco_Airoldi', '/Matteo_Malosio', '/Michele',
                     '/Giorgio_Nicola', '/Paolo_Franceschi', '/Marco_Faroni', '/Stefano_Mutti', '/Trunal']

#INPUT PARAMETERS
subject = '/Stefano_Mutti'
exp_number = 1
what_to_plot1 = 'pose.position.x'
what_to_plot2 = 'pose.position.y'

bag_folder_base = '/home/adriano/projects/bag/controller_adriano/'
file_extension = '.bag'
experiment_type = 'step'
underscore = '_'

bag_folder = bag_folder_base + subject + '/step/'
experiment_number = str(exp_number)
file_name = experiment_type + underscore + experiment_number
bag_input = bag_folder + file_name + file_extension
array1, array2 = bag_read(bag_input, what_to_plot1, what_to_plot2)

plt_time = np.linspace(start=0, stop=60, num=len(array1))

plt.figure(1)
plt.plot(plt_time, array1)
plt.xlabel("time")
plt.ylabel("h err x")
plt.grid()

plt.figure(2)
plt.plot(plt_time, array2)
plt.xlabel("time")
plt.ylabel("h err y")
plt.grid()
plt.show()