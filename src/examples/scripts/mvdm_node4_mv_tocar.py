#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import os
import csv
import time
from datetime import datetime
import math
from collections import deque
from functools import partial
from std_msgs.msg import Float32

params = {
    "robot_ids": ["9289"],
    # "robot_ids": ["9293", "9298"],
    "Kp": 0.0003,  # 降低比例增益
    "Ki": 0.00001,  # 添加积分项
    "Kd": 0.0008,  # 调整微分增益
    "alpha": 0.85,  # 增加滤波强度
    "max_angular": 0.4,  # 降低最大角速度
    "min_angular": -0.4,
    "control_period": 0.15,
    "expand_num": 40,
    "control_angular": 0.15,  # 降低控制角速度
    "control_angular_max": 0.12,
    "deadzone": 5,  # 添加死区，单位：像素
    "smooth_window": 5,  # 滑动平均窗口大小
    "max_deviation_change": 20,  # 最大偏差变化量
    "angle_compensation": True,  # 是否启用角度补偿
}
vehicle_params = {
    "9289": {
        "first_pos_delta": -10,
        "angular_delta": 0,
    },
    "9293": {
        "first_pos_delta": -10,
        "angular_delta": -0.005,
    },
    "9298": {
        "first_pos_delta": -80,
        "angular_delta": 0,
    }
}


class EnhancedMvdmSystem:
    def __init__(self, speed_action, lane_action):

        # 原有变量
        self.robot_ids = params["robot_ids"]

        self.speed_expand = {id: [0] * 10 + np.repeat(speed_action[id], params["expand_num"]).tolist() for id in
                             self.robot_ids}
        self.lane_expand = {id: [0] * 10 + np.repeat(lane_action[id], params["expand_num"]).tolist() for id in
                            self.robot_ids}
        self.exc_speed = {id: 0 for id in self.robot_ids}
        # 增强的控制变量
        self.deviation_history = {id: deque(maxlen=params["smooth_window"]) for id in self.robot_ids}
        self.angular_history = {id: deque(maxlen=3) for id in self.robot_ids}  # 用于角速度平滑
        self.integral_error = {id: 0 for id in self.robot_ids}  # PID积分项
        self.last_deviation = {id: 0 for id in self.robot_ids}
        self.last_time = {id: time.time() for id in self.robot_ids}
        self.angular = {id: 0 for id in self.robot_ids}
        self.filtered_deviation = {id: 0 for id in self.robot_ids}
        self.first_pos = {id: 0 for id in self.robot_ids}
        self.first = {id: True for id in self.robot_ids}

        # 预测控制相关
        self.lane_curvature = {id: 0 for id in self.robot_ids}
        self.speed_factor = {id: 1.0 for id in self.robot_ids}

        self.bridge = CvBridge()

        self.count = 0

        # 控制参数
        self.Kp = params["Kp"]
        self.Ki = params["Ki"]
        self.Kd = params["Kd"]
        self.alpha = params["alpha"]
        self.min_angular = params["min_angular"]
        self.max_angular = params["max_angular"]

        # ROS订阅和发布
        self.lane_pubs = {
            robot_id: rospy.Publisher(
                f'/limo{robot_id}/lane',
                Float32,
                queue_size=1
            )
            for robot_id in self.robot_ids
        }

        self.vel_pubs = {
            robot_id: rospy.Publisher(
                f'/limo{robot_id}/vel',
                Float32,
                queue_size=1
            )
            for robot_id in self.robot_ids
        }

        self.control_timer = rospy.Timer(rospy.Duration(params["control_period"]), self.cbControl)


    def publish_twist(self, limo_id,lane_pub, vel_pub, exc_lane, exc_speed):
        """发布控制命令（改进版）"""

        lane_pub.publish(exc_lane)
        vel_pub.publish(exc_speed)
        rospy.loginfo(
            f"publish to {limo_id} - Speed: {exc_speed:.3f}, Lane: {exc_lane:.3f} (Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]})")

    def cbControl(self, event):
        """控制回调函数（保持原有逻辑但使用改进的发布函数）"""

        for limo_id in self.robot_ids:
            self.publish_twist(limo_id,self.lane_pubs[limo_id],self.vel_pubs[limo_id],self.lane_expand[limo_id][self.count], self.speed_expand[limo_id][self.count])
        self.count += 1



def run():
    rospy.init_node('enhanced_mvdm_system', anonymous=False)
    speed_action = {'9293': [0.5, 0.5, 0.5, 0.5, 0.5], "9298": [0.5, 0.5, 0.5, 0.5, 0.5], "9289": [0.5, 0.5, 0.5, 0.5, 0.5]}
    lane_action = {"9293": [0, 0, 0, 0, 0], "9298": [0, 0, 0, 0, 0], "9289": [0, 0, 0, 0, 0]}
    mvdm_system = EnhancedMvdmSystem(speed_action, lane_action)
    rospy.spin()


if __name__ == '__main__':
    run()