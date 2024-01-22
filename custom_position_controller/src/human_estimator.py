#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import rospy
import collections
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from std_msgs.msg import Bool


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
HOME_DIR = '/home/adriano/PycharmProjects/human_reaction_delay_nn/'
FILE_NAME = HOME_DIR + 'narmax10_xy_tr50_summary.pkl'


class MLP(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, 2 * H)
        self.linear5 = torch.nn.Linear(2 * H, 4 * H)
        self.linear6 = torch.nn.Linear(4 * H, 2 * H)
        self.linear7 = torch.nn.Linear(2 * H, H)
        self.linear8 = torch.nn.Linear(H, H)
        self.linear9 = torch.nn.Linear(H, H)
        self.linear10 = torch.nn.Linear(H, D_out)
        self.to(DEVICE)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        x = self.linear4(F.relu(x))
        x = self.linear5(F.relu(x))
        x = self.linear6(F.relu(x))
        x = self.linear7(F.relu(x))
        x = self.linear8(F.relu(x))
        x = self.linear9(F.relu(x))
        y_pred = self.linear10(torch.tanh(x))
        return y_pred


QUEUE_SIZE = 50
force_x = 0.0
force_y = 0.0
ref_pos_x = 0.0
ref_pos_y = 0.0
forces_x = collections.deque(maxlen=QUEUE_SIZE)
forces_y = collections.deque(maxlen=QUEUE_SIZE)
ref_positions_x = collections.deque(maxlen=QUEUE_SIZE)
ref_positions_y = collections.deque(maxlen=QUEUE_SIZE)
norm_list = collections.deque(maxlen=QUEUE_SIZE)
pos_z = 0
time_rcv = rospy.Time()


def force_callback(msg):
    global force_x 
    force_x = msg.wrench.force.x
    global force_y
    force_y = msg.wrench.force.y
    
    global time_rcv
    time_rcv = msg.header.stamp
    




def ref_pos_callback(msg):
    global ref_pos_x
    ref_pos_x = msg.pose.position.x
    global ref_pos_y
    ref_pos_y = msg.pose.position.y
    



def normalize(input_value, input_vector):
    if (np.abs(np.max(input_vector) - np.min(input_vector))) < 0.00001:
        norm_list.append(0.0)
    else:
        norm_list.append((2*(input_value - np.min(input_vector))/(np.max(input_vector) - np.min(input_vector)))-1)
    norm_array = np.array(norm_list)
    return norm_array
    
    
def narmax():
    rospy.init_node('human_estimator', anonymous=True)
    rate_hz = 800
    rate = rospy.Rate(rate_hz)

    rospy.Subscriber("/custom_position_controller/wrench", WrenchStamped, force_callback)
    rospy.Subscriber("/custom_position_controller/virtual_ref_position", PoseStamped, ref_pos_callback)
    # rospy.Subscriber("/custom_position_controller/actual_position", PoseStamped, act_pos_callback)
    pub = rospy.Publisher('/custom_position_controller/human_estimated_ref', WrenchStamped, queue_size=1)
    exp_win_pub = rospy.Publisher('/custom_position_controller/experiment_window', Bool, queue_size=1)
    # loop_pub = rospy.Publisher('/human_estimator/loop_freq', Float64, queue_size=1)
    time = rospy.Time()

    # define NARMAX parameters
    n_b = 10
    n_a = n_c = 11
    n_k = 38
    # noise_mean = 0
    # noise_std = 0.4127
    # noise = np.random.normal(noise_mean, noise_std, size=QUEUE_SIZE)

    # positions_x = np.empty(QUEUE_SIZE)
    # positions_y = np.empty(QUEUE_SIZE)

    pred_msg = WrenchStamped()
    # loop_freq_msg = Float64()
    exp_win_msg = Bool()

    # load model
    model = pickle.load(open(FILE_NAME, 'rb'))
    rospy.sleep(2)

    # start_time = time.now().to_sec()
    # loop_cnt = 0

    while not rospy.is_shutdown():

        while True:
            forces_x.append(force_x)
            forces_y.append(force_y)
            ref_positions_x.append(ref_pos_x)
            ref_positions_y.append(ref_pos_y)
            if (len(ref_positions_x) == QUEUE_SIZE) and (len(ref_positions_y) == QUEUE_SIZE) and (len(forces_x) == QUEUE_SIZE) and (len(forces_y) == QUEUE_SIZE):
                break
            # rospy.logwarn("Waiting for queues to fill up!!")

        positions_x_norm = normalize(ref_pos_x, ref_positions_x)
        positions_y_norm = normalize(ref_pos_y, ref_positions_y)
        forces_x_norm = normalize(force_x, forces_x)
        forces_y_norm = normalize(force_y, forces_y)
       
        if (len(positions_x_norm) == QUEUE_SIZE) and (len(positions_y_norm) == QUEUE_SIZE) and (len(forces_x_norm) == QUEUE_SIZE) and (len(forces_y_norm) == QUEUE_SIZE):
            time_diff = time_rcv.to_nsec() - time.now().to_nsec()
            # print("time diff human estimator py: ", time_diff)
            if time_diff > 1e+6:
                rospy.logwarn("waiting, time difference < 0")
                d = rospy.Duration(nsecs=time_diff)
                rospy.sleep(d)
            invals1 = []
            invals2 = []
            for k in range(0, n_a-1): invals1.append(forces_x_norm[-k])
            for k in range(n_k-1, n_k + n_b-1): invals1.append(positions_x_norm[-k])
            for k in range(0, n_a-1): invals2.append(forces_y_norm[-k])
            for k in range(n_k-1, n_k + n_b-1): invals2.append(positions_y_norm[-k])
            invals = np.vstack((np.array(invals1), np.array(invals2)))
            narmax_input_test = invals.reshape(-1, 1).transpose()
            pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
            pred_x = pred_tensor[0, 0].cpu().detach().numpy()
            pred_y = pred_tensor[0, 1].cpu().detach().numpy()

            # Denormalize predictions
            max_Yx = np.max(forces_x)
            max_Yy = np.max(forces_y)
            min_Yx = np.min(forces_x)
            min_Yy = np.min(forces_y)
            prediction_x = (pred_x + 1) * (max_Yx - min_Yx)/2 + min_Yx
            prediction_y = (pred_y + 1) * (max_Yy - min_Yy)/2 + min_Yy

            pred_msg.wrench.force.x = prediction_x
            pred_msg.wrench.force.y = prediction_y
            # pred_msg.wrench.force.x = forces_x_norm
            # pred_msg.wrench.force.y = forces_y_norm
            pred_msg.header.stamp = time.now()
            pred_msg.header.frame_id = "base_link"
            pub.publish(pred_msg)

            exp_win_msg.data = True
            exp_win_pub.publish(exp_win_msg)
        # else:
        #     rospy.logwarn("wating for normalized queues to fill up")

        # loop_cnt += 1
        # loop_freq = loop_cnt/(time.now().to_sec() - start_time)
        # loop_freq_msg.data = loop_freq
        # loop_pub.publish(loop_freq_msg)

        rate.sleep()
    exp_win_msg.data = False
    exp_win_pub.publish(exp_win_msg)

if __name__ == '__main__':
    try:
        narmax()
    except rospy.ROSInterruptException:
        pass

    