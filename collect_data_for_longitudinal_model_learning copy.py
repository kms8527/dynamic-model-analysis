#!/usr/bin/env python

import rospy
from cbnu_msgs.msg import VehicleTlm, VehicleCmd  # Assuming the message types are defined in cbnu_msgs
from geometry_msgs.msg import Twist
import pandas as pd
import os

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector', anonymous=True)

        # Subscriber to Ego topic
        self.sim_vehicle_state_sub_ = rospy.Subscriber("/Ego_topic", VehicleTlm, self.sim_vehicle_state_callback)

        # Subscriber to control command topic
        self.control_sub_ = rospy.Subscriber("ctrl_cmd", VehicleCmd, self.control_callback)

        # Member variables
        self.vhcl_vel_ = Twist()  # Vehicle velocity
        self.accel_ = 0.0         # Control input (acceleration)

        # CSV file setup
        self.file_name = './morai_model_identification_data.csv'
        if not os.path.exists(self.file_name):
            # Create a new file with headers if it doesn't exist
            df = pd.DataFrame(columns=['time', 'velocity_x', 'acceleration'])
            df.to_csv(self.file_name, index=False)

        self.start_time = rospy.get_time()

        # Set loop rate to 50 Hz (every 0.02 seconds)
        self.rate = rospy.Rate(50)

    def sim_vehicle_state_callback(self, msg):
        # Ego vehicle velocity received and saved
        self.vhcl_vel_.linear.x = msg.velocity_x

    def control_callback(self, msg):
        # Subscribe to control input and save acceleration
        self.accel_ = msg.accel

        # Record data in real-time
        current_time = rospy.get_time() - self.start_time

        # Prepare new data row
        new_data = pd.DataFrame([{
            'time': current_time,
            'velocity_x': self.vhcl_vel_.linear.x,  # m/s
            'acceleration': self.accel_,  # m/s^2
        }])

        # Append to CSV file
        new_data.to_csv(self.file_name, mode='a', header=False, index=False)

    def spin(self):
        try:
            while not rospy.is_shutdown():
                self.rate.sleep()
        except rospy.ROSInterruptException:
            pass


if __name__ == '__main__':
    data_collector = DataCollector()
    data_collector.spin()
