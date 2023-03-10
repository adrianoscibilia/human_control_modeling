#!/usr/bin/env python3
import csv

import bagpy
import scipy.fft
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy import signal


base_dir_from_linux = '/home/adriano/projects/ros_ws/src/controller_adriano/bag'
experiment_path = '/Stefano_Mutti/step/step_5.bag'
file_name = base_dir_from_linux + experiment_path
b = bagreader(file_name)

# read messages for each topic
H_ERR_MSG = b.message_by_topic('/human_error')
ACT_POS_MSG = b.message_by_topic('/actual_position')
REF_POS_MSG = b.message_by_topic('/reference_position')
POS_ERR_MSG = b.message_by_topic('/position_error')
FORCE_MSG = b.message_by_topic('/human_force')
EXPERIMENT_WINDOW_MSG = b.message_by_topic("/experiment_window")

# import csv as panda dataframe
h_err_pd = pd.read_csv(H_ERR_MSG)
act_pos_pd = pd.read_csv(ACT_POS_MSG)
ref_pos_pd = pd.read_csv(REF_POS_MSG)
pos_err_pd = pd.read_csv(POS_ERR_MSG)
force_pd = pd.read_csv(FORCE_MSG)
experiment_window_pd = pd.read_csv(EXPERIMENT_WINDOW_MSG)

# create vectors
time = h_err_pd['Time'] - h_err_pd.loc[0, 'Time']
h_err_x = h_err_pd['pose.position.x']
h_err_y = h_err_pd['pose.position.y']
act_pos_x = act_pos_pd['pose.position.x']
act_pos_y = act_pos_pd['pose.position.y']
ref_pos_x = ref_pos_pd['pose.position.x']
ref_pos_y = ref_pos_pd['pose.position.y']
pos_err_x = pos_err_pd['pose.position.x']
pos_err_y = pos_err_pd['pose.position.y']
h_force_x = force_pd['wrench.force.x']
h_force_y = force_pd['wrench.force.y']
experiment_window = experiment_window_pd['data']

for i in range(len(act_pos_x)):
    ref_pos = np.sqrt((np.power(ref_pos_x, 2) + np.power(ref_pos_y, 2)))
    act_pos = np.sqrt((np.power(act_pos_x, 2) + np.power(act_pos_y, 2)))
    h_force = np.sqrt((np.power(h_force_x, 2) + np.power(h_force_y, 2)))
    pos_err = np.sqrt((np.power(pos_err_x, 2) + np.power(pos_err_y, 2)))
    h_err = np.sqrt((np.power(h_err_x, 2) + np.power(h_err_y, 2)))

# SIZE ADJUSTMENTS
ref_pos_input = ref_pos
act_pos_input = act_pos
h_err_input = h_err

time_adj = []
h_err_adj = []
force_adj = []
ref_pos_adj = []
act_pos_adj = []
lenghts = []
error_adj = []
error_adj_dot = []
lenghts.append(len(h_err_input))
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
        h_err_adj.append(h_err_input[idx])
        force_adj.append(h_force[idx])
        ref_pos_adj.append(ref_pos_input[idx])
        act_pos_adj.append(act_pos_input[idx])
        error_adj.append(ref_pos_input[idx] - act_pos_input[idx])

for i in range(1, len(time_adj)):
    time_adj[i] = time_adj[i] - time_adj[0]
time_adj[0] = 0

# DERIVATIVE
error_adj_dot = np.gradient(h_err_adj)
error_adj_dot_plt = np.array(error_adj_dot)
error_adj_dot_plt_inv = error_adj_dot_plt * -1

# FIND DELAY WITH MAX AND MINS
error_adj_plt = np.array(h_err_adj)
error_adj_plt_inv = error_adj_plt * -1
peaks, _ = signal.find_peaks(error_adj_dot_plt, height=0.02, distance=150)
# mins, _ = signal.find_peaks(error_adj_plt_inv, height=-0.02)

reactions = []

for i in range(0, len(peaks)):
    for j in range(0, len(error_adj_dot_plt)):
        if (j > peaks[i]):
            if (error_adj_dot_plt[j] < -0.0002):
                reactions.append(j)
                break

reactions_plt = np.array(reactions)

delays = []
delay_lenght = len(peaks)
if len(reactions_plt) < len(peaks):
    delay_lenght = len(reactions_plt)

for i in range(0, delay_lenght):
    delays.append((reactions_plt[i] - peaks[i]) * 0.008)

delay_mean = np.mean(delays)

print("delays: ", delays)
print("mean delay: ", delay_mean)
print("number of virtual ref: ", delay_lenght)


# PLOTS
ref_pos_plt = ref_pos_adj - ref_pos_adj[0]
act_pos_plt = act_pos_adj - act_pos_adj[0]
error_prova_plt = ref_pos_plt - act_pos_plt

plt.figure(1)
plt.plot(time_adj, ref_pos_plt)
plt.plot(time_adj, act_pos_plt)
#plt.plot(error_adj_dot_plt)
#plt.plot(peaks, error_adj_dot_plt[peaks], "x")
#plt.plot(reactions_plt, error_adj_dot_plt[reactions_plt], "o")
plt.xlabel("time")
plt.grid()

plt.figure(2)
plt.plot(time_adj, force_adj)
plt.xlabel("time")
plt.ylabel("force")
plt.grid()

plt.figure(3)
plt.plot(time_adj, h_err_adj)
plt.plot(time_adj, error_prova_plt)
plt.xlabel("time")
plt.ylabel("error")
plt.grid()

plt.show()


# # create data folder
# bag_folder = 'E:\home\\adriano\projects\\ros_ws\src\controller_adriano\\bag\Trunal\step\\'
# for i in range(1, 11):
#     experiment_type = 'step'
#     underscore = '_'
#     experiment_number = str(i)
#     file_name = experiment_type + underscore + experiment_number
#     file_extension = '.bag'
#     bag_input = bag_folder + file_name + file_extension
#
#     n_of_points = baganalyze(bag_input)
#     outfile_name = bag_folder + "number_of_references"
#     outfile_extension = ".csv"
#     outfile = open(outfile_name + outfile_extension, 'a')
#
#     writer = csv.writer(outfile)
#     row = [experiment_number, n_of_points]
#     writer.writerow(row)
#     outfile.close()
