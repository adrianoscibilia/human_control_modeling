#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import numpy as np
# import math
# from sippy import functionset as fset
# from sippy import *
import control as cnt
from bagpy import bagreader
import pandas as pd
from scipy import signal
import sklearn.metrics as skl


def mse_tf_compute(W_co, T_d, omega, Tf_id):
    G = W_co * cnt.tf(1, [1, 0])
    (num, den) = cnt.pade(T_d, 4)
    G_nom = cnt.tf(num, den) * G
    G_nom_mag, G_nom_phase, G_nom_w = cnt.freqresp(G_nom, omega)
    MSE = skl.mean_squared_error(Tf_id, G_nom_mag)
    return MSE


def optimize_param_freq(W_co_in, T_d_in, omega, Tf_id):
    mse_w_vec = []
    mse_td_vec = []
    W_co_vec = []
    T_d_vec = []
    W_co_0 = W_co_in
    T_d_0 = T_d_in
    while W_co_0 > 0:
        W_co_vec.append(W_co_0)
        MSE_W_act = mse_tf_compute(W_co_0, T_d_in, omega, Tf_id)
        mse_w_vec.append(MSE_W_act)
        #print('%.2f'%W_co_0, "\t", MSE_W_act)
        W_co_0 -= 0.01
    #print("\n")
    MSE_W_min = np.argmin(mse_w_vec)
    W_co_opt = np.around(W_co_vec[MSE_W_min], 2)
    while T_d_0 > 0.4:
        T_d_vec.append(T_d_0)
        MSE_Td_act = mse_tf_compute(W_co_opt, T_d_0, omega, Tf_id)
        mse_td_vec.append(MSE_Td_act)
        #print('%.2f'%T_d_0, "\t", MSE_Td_act)
        T_d_0 -= 0.01
    MSE_Td_min = np.argmin(mse_td_vec)
    T_d_opt = np.around(T_d_vec[MSE_Td_min], 3)
    return W_co_opt, T_d_opt


def optimize_param_time(W_co_in, T_d_in, ref_pos_in, act_pos_in):
    mse_w_vec = []
    mse_td_vec = []
    W_co_vec = []
    T_d_vec = []
    W_co_0 = W_co_in
    T_d_0 = T_d_in

    #SET SIMULATED SIGNALS DIMENSIONS
    t_sim_0 = np.arange(0, 60, 0.008)
    U_sim_0 = ref_pos_in[0:len(ref_pos_in)] - ref_pos_in[1]
    Y_meas_0 = act_pos_in[0:len(act_pos_in)] - act_pos_in[1]
    U_sim = U_sim_0
    t_sim = t_sim_0
    Y_meas = Y_meas_0
    if len(U_sim_0) < len(t_sim_0):
        t_sim = t_sim_0[0:len(U_sim_0)]
    if len(U_sim_0) > len(t_sim_0):
        U_sim = U_sim_0[0:len(t_sim)]
        Y_meas = Y_meas_0[0:len(t_sim)]

    while W_co_0 > 0:
        W_co_vec.append(W_co_0)
        G = W_co_0 * cnt.tf(1, [1, 0])
        (num, den) = cnt.pade(T_d_in, 4)
        G_nom = cnt.tf(num, den) * G
        G_feedback = cnt.feedback(sys1=G_nom, sys2=1, sign=-1)
        response = cnt.forced_response(sys=G_feedback, T=t_sim, U=U_sim)
        MSE_W_act = skl.mean_squared_error(y_true=Y_meas, y_pred=response.outputs)
        mse_w_vec.append(MSE_W_act)
        W_co_0 -= 0.01
    MSE_W_min = np.argmin(mse_w_vec)
    W_co_opt = np.around(W_co_vec[MSE_W_min], 2)
    while T_d_0 > 0:
        T_d_vec.append(T_d_0)
        G = W_co_opt * cnt.tf(1, [1, 0])
        (num, den) = cnt.pade(T_d_0, 4)
        G_nom = cnt.tf(num, den) * G
        G_feedback = cnt.feedback(sys1=G_nom, sys2=1, sign=-1)
        response = cnt.forced_response(sys=G_feedback, T=t_sim, U=U_sim)
        MSE_Td_act = skl.mean_squared_error(y_true=Y_meas, y_pred=response.outputs)
        mse_td_vec.append(MSE_Td_act)
        T_d_0 -= 0.01
    MSE_Td_min = np.argmin(mse_td_vec)
    T_d_opt = np.around(T_d_vec[MSE_Td_min], 3)
    return W_co_opt, T_d_opt


def bag_read(file_path):
    b = bagreader(file_path)

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

    # DERIVATIVES
    error_adj_array = np.array(h_err_adj)
    ref_pos_adj_array = np.array(ref_pos_adj)
    act_pos_adj_array = np.array(act_pos_adj)
    return time_adj, ref_pos_adj_array, act_pos_adj_array, error_adj_array


def identify(mode, ref_pos_input, act_pos_input, pos_err_input):
    # mode=0 for PSD identification; mode=1 for time domain
    if mode == 0:
        # CSD AD PSD TF IDENTIFICATION
        f_uu, P_uu = signal.welch(pos_err_input, fs=125)
        f_yu, P_yu = signal.csd(act_pos_input, pos_err_input, fs=125)
        w_yu = 2 * np.pi * f_yu
        w_yu[0] = 1
        # omega_prova = np.linspace(1, 360)
        G_yu_frd = np.abs(P_yu / P_uu)
        W_co_opt, T_d_opt = optimize_param_freq(1, 0.69, w_yu, G_yu_frd)
        print("\n", "W_co_opt:", W_co_opt, "\tT_d_opt:", T_d_opt)
        return W_co_opt, T_d_opt
    if mode == 1:
        W_co_opt, T_d_opt = optimize_param_time(W_co_in=10, T_d_in=1, ref_pos_in=ref_pos_input, act_pos_in=act_pos_input)
        print("\n", "W_co_opt:", W_co_opt, "\tT_d_opt:", T_d_opt)
        return W_co_opt, T_d_opt
    else:
        print("\nWrong modality selected!")
        return -1


subjects = ['/Alessandro_Scano', '/Claudia_Pagano', '/Francesco_Airoldi', '/Matteo_Malosio', '/Michele', '/Giorgio_Nicola', '/Paolo_Franceschi', '/Marco_Faroni', '/Stefano_Mutti', '/Trunal']
bag_folder_base = '/home/adriano/projects/ros_ws/src/controller_adriano/bag'
#base_dir_from_linux = '/home/adriano/projects/ros_ws/src/controller_adriano/bag'
#experiment_path = '/Stefano_Mutti/step/step_5.bag'

# #Single file modality
# subject = '/Giorgio_Nicola'
# experiment_number = str(5)
# bag_folder = bag_folder_base + subject + '/step/'
# experiment_type = 'step'
# underscore = '_'
# file_name = experiment_type + underscore + experiment_number
# file_extension = '.bag'
# bag_input = bag_folder + file_name + file_extension
# exp_time, exp_ref_pos, exp_act_pos, exp_pos_err = bag_read(file_path=bag_input)
# W_opt, Td_opt = identify(mode=1, ref_pos_input=exp_ref_pos, act_pos_input=exp_act_pos, pos_err_input=exp_pos_err)

# #PLOTS
# G = W_opt * cnt.tf(1, [1, 0])
# (num, den) = cnt.pade(Td_opt, 4)
# G_nom = cnt.tf(num, den) * G
# #G_nom.dt = 0.008
# G_feedback = cnt.feedback(sys1=G_nom, sys2=1, sign=-1)
#
# # SET SIMULATED SIGNALS DIMENSIONS
# t_sim_0 = np.arange(0, 60, 0.008)
# U_sim_0 = exp_ref_pos[0:len(exp_ref_pos)] - exp_ref_pos[1]
# Y_meas_0 = exp_act_pos[0:len(exp_act_pos)] - exp_act_pos[1]
# U_sim = U_sim_0
# t_sim = t_sim_0
# Y_meas = Y_meas_0
# if len(U_sim_0) < len(t_sim_0):
#     t_sim = t_sim_0[0:len(U_sim_0)]
# if len(U_sim_0) > len(t_sim_0):
#     U_sim = U_sim_0[0:len(t_sim)]
#     Y_meas = Y_meas_0[0:len(t_sim)]
#
# response = cnt.forced_response(sys=G_feedback, T=t_sim, U=U_sim)

# plt.figure(1)
# plt.plot(t_sim, U_sim)
# plt.plot(t_sim, response.outputs)
# plt.grid()

# w, mag, phase = signal.bode(G_id)
# plt.figure(2)
# plt.semilogx(w, mag)    # Bode magnitude plot
# plt.figure()
# plt.semilogx(w, phase)  # Bode phase plot#plt.figure(1)
# plt.grid()

# plt.figure(3)
# plt.semilogy(f_uu, np.sqrt(P_uu))
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD')
# plt.grid()

# plt.figure(4)
# w = np.logspace(-7, 1, 701)
# mag_nom, phase_nom, omega_nom = cnt.bode(G_nom)
# plt.show()

#Iterative call of files from folder
for n in range(0, len(subjects)):
   bag_folder = bag_folder_base + subjects[n] + '/step/'
   for i in range(1, 11):
       experiment_type = 'step'
       underscore = '_'
       experiment_number = str(i)
       file_name = experiment_type + underscore + experiment_number
       file_extension = '.bag'
       bag_input = bag_folder + file_name + file_extension
       exp_time, exp_ref_pos, exp_act_pos, exp_pos_err = bag_read(file_path=bag_input)
       W_opt, Td_opt = identify(mode=1, ref_pos_input=exp_ref_pos, act_pos_input=exp_act_pos, pos_err_input=exp_pos_err)

       outfile_name = bag_folder + "time_param_id"
       outfile_extension = ".csv"
       outfile = open(outfile_name + outfile_extension, 'a')

       writer= csv.writer(outfile)
       if i == 1:
           row_title = ["experiment_n", "W_opt", "Td_opt"]
           writer.writerow(row_title)
       row = [experiment_number, W_opt, Td_opt]
       writer.writerow(row)
       outfile.close()
