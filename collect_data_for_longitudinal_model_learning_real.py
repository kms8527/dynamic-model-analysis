#!/usr/bin/env python

import rospy
from avante_cn7_msgs.msg import AvanteInfoCn7  # Assuming the message type is defined in avante_cn7_msgs
from geometry_msgs.msg import Twist
import pandas as pd
import os

save_dir = './avante_cn7_info_cmd_vel_data2.csv'

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector', anonymous=True)

        # Subscriber to avante_cn7_info topic
        self.avante_info_sub_ = rospy.Subscriber("/cn7/can/avante_cn7_info", AvanteInfoCn7, self.avante_info_callback)

        # Subscriber to cmd_vel topic
        self.cmd_vel_sub_ = rospy.Subscriber("/cn7/can/cmd_vel", Twist, self.cmd_vel_callback)

        # Member variables
        self.vehicle_speed_ = 0.0  # Vehicle speed from avante_cn7_info
        self.long_accel_ = 0.0     # Longitudinal acceleration from avante_cn7_info
        self.cmd_velocity_ = 0.0   # Velocity command from cmd_vel

        # CSV file setup
        self.file_name = save_dir
        if not os.path.exists(self.file_name):
            # Create a new file with headers if it doesn't exist
            df = pd.DataFrame(columns=['time', 'velocity_x', 'acceleration'])
            df.to_csv(self.file_name, index=False)

        self.start_time = rospy.get_time()

        # Set loop rate to 50 Hz (every 0.02 seconds)
        self.rate = rospy.Rate(100)

    def avante_info_callback(self, msg):
        # double wheel_velo_fl = msg->wheel_speed.front_left;
        # double wheel_velo_fr = msg->wheel_speed.front_right;
        # double wheel_velo_rl = msg->wheel_speed.rear_left;
        # double wheel_velo_rr = msg->wheel_speed.rear_right;
        # double vel = (wheel_velo_rl + wheel_velo_rr + wheel_velo_fr + wheel_velo_fl) / 4.0;
        front_left = msg.wheel_speed.front_left
        front_right = msg.wheel_speed.front_right
        rear_left = msg.wheel_speed.rear_left
        rear_right = msg.wheel_speed.rear_right
        vehicle_speed = (front_left + front_right + rear_left + rear_right) / 4.0
        
        self.vehicle_speed_ = vehicle_speed / 3.6 # Convert km/h to m/s
        
    def cmd_vel_callback(self, msg):
        # Subscribe to cmd_vel and save the velocity command
        self.cmd_accel = msg.linear.x  # acceleration from cmd_vel (m/s^2)

        # Record data in real-time
        current_time = rospy.get_time() - self.start_time

        # Prepare new data row
        new_data = pd.DataFrame([{
            'time': current_time,
            'vehicle_speed': self.vehicle_speed_,  # m/s
            'long_accel': self.cmd_accel,  # m/s^2
        }])
        print(f"Time: {current_time}, Vehicle Speed: {self.vehicle_speed_}, Longitudinal Acceleration: {self.cmd_accel}")

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
