#!/usr/bin/env python3

import numpy as np
import rospy
import pandas as pd
from bagpy import bagreader
from geometry_msgs.msg import WrenchStamped


HOME_DIR = '/home/adriano/projects/bag/controller_adriano/'
FILE_NAME = HOME_DIR + 'narmax_xy_10_tr50_norm.pkl'


def bag_read(file_path):
    b = bagreader(file_path)

    # read messages of force topic
    FORCE_MSG = b.message_by_topic('/human_force')
    EXPERIMENT_WINDOW_MSG = b.message_by_topic("/experiment_window")

    # import csv as panda dataframe
    force_pd = pd.read_csv(FORCE_MSG)
    experiment_window_pd = pd.read_csv(EXPERIMENT_WINDOW_MSG)

    # create vectors
    h_force_x = force_pd['wrench.force.x']
    h_force_y = force_pd['wrench.force.y']
    h_force_z = force_pd['wrench.force.z']

    experiment_window = experiment_window_pd['data']

    force_adj_x = []
    force_adj_y = []
    force_adj_z = []
    lenghts = []

    lenghts.append(len(h_force_x))
    lenghts.append(len(h_force_y))
    lenghts.append(len(h_force_z))
    lenghts.append(len(experiment_window))
    minimum_lenght = np.amin(lenghts)

    window_vector = np.zeros(len(experiment_window))
    for i in range(0, minimum_lenght):
        if experiment_window[i]:
            window_vector[i] = 1

    for idx in range(0, minimum_lenght):
        if window_vector[idx] == 1:
            # time_adj.append(time[idx])
            force_adj_x.append(h_force_x[idx])
            force_adj_y.append(h_force_y[idx])
            force_adj_z.append(h_force_z[idx])

    # CONVERT TO NP ARRAYS
    output1_array = np.array(force_adj_x)
    output2_array = np.array(force_adj_y)
    output3_array = np.array(force_adj_z)

    return output1_array, output2_array, output3_array
    
    
def bag_simulator():
    rospy.init_node('bag_simulator', anonymous=True)
    rate = rospy.Rate(62)

    force_pub = rospy.Publisher('/robotiq_ft_wrench', WrenchStamped, queue_size=10)

    force_msg = (WrenchStamped)

    time = rospy.Time()
    idx = 0

    wrench_x, wrench_y, wrench_z = bag_read(FILE_NAME)

    while not rospy.is_shutdown():
        force_x = wrench_x[idx]
        force_y = wrench_y[idx]
        force_z = wrench_z[idx]

        force_msg.wrench.force.x = force_x
        force_msg.wrench.force.y = force_y
        force_msg.wrench.force.z = force_z
        force_msg.header.stamp = time.now()
        force_msg.header.frame_id = "base_link"
        force_pub.publish(force_msg)

        idx += 1

        rate.sleep()

if __name__ == '__main__':
    try:
        bag_simulator()
    except rospy.ROSInterruptException:
        pass
