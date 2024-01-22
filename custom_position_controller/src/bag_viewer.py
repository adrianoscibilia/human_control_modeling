#!/usr/bin/env python3

from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_bag(filename, actpos, refpos, force, vel, acc, command, poserr, humanref, expwin, virtref, pred_vel, adjust):
    b = bagreader(bagfile=filename, verbose=False)
    ref_pos_plt = 0
    act_pos_plt = 0
    pos_err_plt = 0
    ref_pos_plt = 0
    act_pos_plt = 0
    pos_err_plt = 0
    force_plt = 0
    acc_plt = 0
    command_plt = 0
    vel_plt = 0
    error_plt = 0
    human_est_plt = 0
    window_vector = 0
    force_adj_vec = 0
    human_est_adj_vec = 0
    ref_pos_adj_vec = 0
    pred_vel_plt = 0

    if virtref:
        VIRT_REF_MSG = b.message_by_topic('/custom_position_controller/virt_ref_enable')
        virt_ref_pd = pd.read_csv(VIRT_REF_MSG)
        virt_ref = virt_ref_pd['data']
        virt_ref_vector = np.zeros(len(virt_ref))
        for i in range(0, len(virt_ref)):
            if virt_ref[i]:
                virt_ref_vector[i] = 1

    if expwin:
        EXP_WIN_MSG = b.message_by_topic('/custom_position_controller/experiment_window')
        exp_win_pd = pd.read_csv(EXP_WIN_MSG)
        exp_win = exp_win_pd['data']
        window_vector = np.zeros(len(exp_win))
        for i in range(0, len(exp_win)):
            if exp_win[i]:
                window_vector[i] = 1

    if actpos:
        ACT_POS_MSG = b.message_by_topic('/custom_position_controller/actual_position')
        act_pos_pd = pd.read_csv(ACT_POS_MSG)
        act_pos_x = act_pos_pd['pose.position.x']
        act_pos_y = act_pos_pd['pose.position.y']
        act_pos_z = act_pos_pd['pose.position.z']
        act_pos_plt = act_pos_y
        act_pos_plt = act_pos_plt - act_pos_plt[0]

    if refpos:
        REF_POS_MSG = b.message_by_topic('/custom_position_controller/reference_position')
        ref_pos_pd = pd.read_csv(REF_POS_MSG)
        ref_pos_x = ref_pos_pd['pose.position.x']
        ref_pos_y = ref_pos_pd['pose.position.y']
        ref_pos_z = ref_pos_pd['pose.position.z']
        ref_pos_plt = ref_pos_y
        ref_pos_plt = ref_pos_plt - ref_pos_plt[0]

    if force: 
        FORCE_MSG = b.message_by_topic('/custom_position_controller/wrench')
        force_pd = pd.read_csv(FORCE_MSG)
        force_x = force_pd['wrench.force.x']
        force_y = force_pd['wrench.force.y']
        force_z = force_pd['wrench.force.z']
        force_plt = force_y
        force_plt = force_plt - force_plt[0]

    if vel: 
        VEL_FILT_MSG = b.message_by_topic('/custom_position_controller/cart_vel_filtered')
        vel_filt_pd = pd.read_csv(VEL_FILT_MSG)
        vel_filt_x = vel_filt_pd['pose.position.x']
        vel_filt_y = vel_filt_pd['pose.position.y']
        vel_filt_z = vel_filt_pd['pose.position.z']
        vel_plt = vel_filt_x

    if acc:
        ACC_MSG = b.message_by_topic('/custom_position_controller/acceleration')
        acc_pd = pd.read_csv(ACC_MSG)
        acc_x = acc_pd['pose.position.x']
        acc_y = acc_pd['pose.position.y']
        acc_z = acc_pd['pose.position.z']
        acc_plt = acc_y

    if command:
        CART_COMMAND_MSG = b.message_by_topic('/custom_position_controller/cart_command')
        CART_COMMAND_VEL_MSG = b.message_by_topic('/custom_position_controller/cart_command_vel')
        cart_command_pd = pd.read_csv(CART_COMMAND_MSG)
        cart_command_vel_pd = pd.read_csv(CART_COMMAND_VEL_MSG)
        cart_command_x = cart_command_pd['pose.position.x']
        cart_command_y = cart_command_pd['pose.position.y']
        cart_command_z = cart_command_pd['pose.position.z']
        cart_command_vel_x = cart_command_vel_pd['pose.position.x']
        cart_command_vel_y = cart_command_vel_pd['pose.position.y']
        cart_command_vel_z = cart_command_vel_pd['pose.position.z']
        command_plt = cart_command_vel_y

    if poserr:
        POS_ERR_MSG = b.message_by_topic('/custom_position_controller/position_error')
        pos_err_pd = pd.read_csv(POS_ERR_MSG)
        pos_err_x = pos_err_pd['pose.position.x']
        pos_err_y = pos_err_pd['pose.position.y']
        pos_err_z = pos_err_pd['pose.position.z']
        pos_err_plt = pos_err_y
        pos_err_plt = pos_err_plt - pos_err_plt[0]

    if humanref:
        HUMAN_EST_MSG = b.message_by_topic('/custom_position_controller/human_estimated_ref')
        human_est_pd = pd.read_csv(HUMAN_EST_MSG)
        human_est_x = human_est_pd['wrench.force.x']
        human_est_y = human_est_pd['wrench.force.y']
        human_est_z = human_est_pd['wrench.force.z']
        human_est_plt = human_est_y
        human_est_plt = human_est_plt - human_est_plt[0]

    if pred_vel:
        PRED_VEL_MSG = b.message_by_topic('/custom_position_controller/human_estimated_vel')
        pred_vel_pd = pd.read_csv(PRED_VEL_MSG)
        pred_vel_x = pred_vel_pd['pose.position.x']
        pred_vel_y = pred_vel_pd['pose.position.y']
        pred_vel_z = pred_vel_pd['pose.position.z']
        pred_vel_plt = pred_vel_x
        pred_vel_plt = pred_vel_plt - pred_vel_plt[0]


    if adjust:
        force_adj = []
        human_est_adj = []
        ref_pos_adj = []
        lenghts = []
        lenghts.append(len(human_est_plt))
        lenghts.append(len(force_plt))
        lenghts.append(len(ref_pos_plt))
        minimum_lenght = np.amin(lenghts)

        for idx in range(0, minimum_lenght):
            if virt_ref_vector[idx] == 1:
                force_adj.append(force_plt[idx])
                human_est_adj.append(human_est_plt[idx])
                ref_pos_adj.append(ref_pos_plt[idx])
        force_adj_vec = np.array(force_adj)
        human_est_adj_vec = np.array(human_est_adj)
        ref_pos_adj_vec = np.array(ref_pos_adj)

    # JOINT_STATE_MSG = b.message_by_topic('/joint_states')
    # joint_state_pd = pd.read_csv(JOINT_STATE_MSG)
    # joint_1 = joint_state_pd['position_0']
    # joint_2 = joint_state_pd['position_1']
    # joint_3 = joint_state_pd['position_2']
    # joint_4 = joint_state_pd['position_3']
    # joint_5 = joint_state_pd['position_4']
    # joint_6 = joint_state_pd['position_5']
    # EXPERIMENT_WINDOW_MSG = b.message_by_topic("/custom_position_controller/experiment_window")
    # experiment_window_pd = pd.read_csv(EXPERIMENT_WINDOW_MSG)
    # experiment_window = experiment_window_pd['data']
    # time = act_pos_pd['Time'] - act_pos_pd.loc[0, 'Time']

    error_plt = ref_pos_plt - act_pos_plt

    return ref_pos_plt, act_pos_plt, force_plt, pos_err_plt, acc_plt, vel_plt, command_plt, error_plt, human_est_plt, pred_vel_plt, window_vector, force_adj_vec, human_est_adj_vec, ref_pos_adj_vec


bag_home_dir = "/home/adriano/projects/bag/custom_position_controller/test_buoni/"
bag_filename = ["adriano_est_4.bag"]
ref_pos = []
act_pos = []
force = []
pos_err = []
acc = []
velocity = []
command = []
error = []
human_ref = []
pred_vel = []
window = []
force_adj = []
human_ref_adj = []
ref_pos_adj = []
for name_ in bag_filename:
    file_name = bag_home_dir + name_
    ref_pos_tmp, act_pos_tmp, force_tmp, pos_err_tmp, acc_tmp, vel_tmp, comm_tmp, error_tmp, human_ref_tmp, pred_vel_tmp, win_tmp, force_adj_tmp, href_tmp, ref_adj_tmp = load_bag(file_name, actpos=True, refpos=True, 
                                                                                                       force=True, vel=False, acc=False, 
                                                                                                       command=False, poserr=False, humanref=True, expwin=False, virtref=False, pred_vel=False, adjust=False)
    ref_pos.append(ref_pos_tmp)
    act_pos.append(act_pos_tmp)
    force.append(force_tmp)
    pos_err.append(pos_err_tmp)
    acc.append(acc_tmp)
    velocity.append(vel_tmp)
    command.append(comm_tmp)
    error.append(error_tmp)
    human_ref.append(human_ref_tmp)
    window.append(win_tmp)
    force_adj.append(force_adj_tmp)
    human_ref_adj.append(href_tmp)
    ref_pos_adj.append(ref_adj_tmp)
    pred_vel.append(pred_vel_tmp)

lenghts = []
lenghts.append(len(force[0]))
lenghts.append(len(human_ref[0]))
lenghts.append(len(error[0]))
min_len = np.amin(lenghts)

time = np.linspace(start=0, stop=(min_len*0.00125), num=min_len)

# plt.figure(1)
# legend_str = []
# for idx in range(0, len(bag_filename)):
#     plt.plot(ref_pos[idx])
#     str_ = "ref_pos " + bag_filename[idx]
#     legend_str.append(str_)
#     plt.plot(act_pos[idx])
#     str_ = "act_pos " + bag_filename[idx]
#     legend_str.append(str_)
# plt.xlabel("n_of_sample")
# plt.legend(legend_str)
# plt.grid()

# print("window len: ", len(window[0]))
# print("force len: ", len(force[0]))
# print("estimated force len: ", len(human_ref[0]))
# len_diff = len(force[0]) - len(human_ref[0])

plt.figure(1)
plt.plot(time, error[0][(len(error[0])-min_len):])
plt.xlabel("time")
plt.legend(['position error'])
plt.grid()

plt.figure(2)
plt.plot(time, force[0][(len(force[0])-min_len):], alpha=0.5)
plt.plot(time, human_ref[0][(len(human_ref[0])-min_len):], alpha=0.5)
plt.xlabel("time")
plt.legend(['measured force', 'predicted force'])
plt.grid()

# plt.figure(5)
# plt.plot(joint_1)
# plt.plot(joint_2)
# plt.plot(joint_3)
# plt.plot(joint_4)
# plt.plot(joint_5)
# plt.plot(joint_6)
# plt.xlabel("n_of_sample")
# plt.legend(['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
# plt.grid()

plt.show()
plt.close()
