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
from threading import Thread, Lock
from std_msgs.msg import Float32

class Mvdm_System:
    def __init__(self, speed_action, lane_action):
        # 存储信息
        now = datetime.now()
        time_str = now.strftime("%m%d%H%M%S")
        self.main_folder = os.path.join(os.getcwd(), time_str)
        self.video_folder = os.path.join(self.main_folder, "video")
        self._create_folders()

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        # 补充信息
        self.bridge = CvBridge()
        self.angular = 0
        self.i = 0
        self.count = 0
        self.Kp = 0.0005
        self.Kd = 0.001
        self.filtered_deviation = 0
        self.alpha = 0.8
        self.speed_expand =  np.repeat(speed_action, 40).tolist()
        self.lane_expand =  np.repeat(lane_action, 40).tolist()
        self.exc_speed = 0
        self.last_deviation = 0
        self.first = True
        self.min_angular = -0.6
        self.max_angular = 0.6
        self.video_folder = os.path.join(self.main_folder, "video")
        os.makedirs(self.video_folder, exist_ok=True)



        # 接收订阅
        self.angular = 0.0

        # 处理订阅
        self.lock = Lock()
        self.latest_image = None
        self.processed_result = None  # 控制用的处理结果

        self.angular_sub = rospy.Subscriber("/limo9293/angular_info", Float32, self.angular_callback)

        # 发布指令
        self.cmd_vel_pub = rospy.Publisher('/limo9293/cmd_vel', Twist, queue_size=1)
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.cbControl)

    def _create_folders(self):
        os.makedirs(self.video_folder, exist_ok=True)

    def angular_callback(self, angular):
        print(angular.data)
        self.angular = angular.data

    def cbControl(self, event):
        index_lane = 0

        if self.lane_expand[self.count] == 0:
            index_lane = 0
            self.exc_speed = self.speed_expand[self.count]
        else:
            progress = min(index_lane / 40, 1.0)
            phase = progress * math.pi * 2  # 0~2π
            angular = self.lane_expand[self.count]  * math.sin(phase)
            # 计算速度补偿以保持x方向分量
            try:
                # 限制最大转向角度为60度(约1.047弧度)
                max_angle = math.pi / 3
                current_angular = max(min(angular, max_angle), -max_angle)

                # 计算补偿速度: v = v_desired / cos(θ)
                speed_compensation = 1.0 / math.cos(abs(current_angular))
                # 限制最大补偿速度为初始值的2倍
                max_speed = 2.0 * self.speed_expand[self.count]
                actual_speed = min(self.speed_expand[self.count] * speed_compensation, max_speed)
            except:
                # 如果计算出错(如cos(90°))，使用默认速度
                actual_speed = self.speed_expand[self.count]
                current_angular = 0
            index_lane += 1
            self.exc_speed = actual_speed
            self.angular = current_angular * (1 / actual_speed) / 3.5
        twist = Twist()
        exc_angular = max(self.angular,self.min_angular)
        exc_angular = min(exc_angular,self.max_angular)
        twist.angular.z = exc_angular
        twist.linear.x = self.exc_speed
        print(self.angular,self.exc_speed)
        self.cmd_vel_pub.publish(twist)
        self.count += 1

        # 在图像上添加文字
        text_frame = f"Frame: {self.i:.2f}"
        text_time = f"Time: {datetime.now().time()}"
        text_angular = f"Angular: {self.angular}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # 白色文字
        thickness = 2
        ret, frame = self.cap.read()
        frame_path = os.path.join(self.video_folder, f"video_{self.i:04d}.jpg")
        cv2.putText(frame, text_frame, (20, 30), font, font_scale, font_color, thickness)
        cv2.putText(frame, text_time, (20, 60), font, font_scale, font_color, thickness)
        cv2.putText(frame, text_angular, (20, 90), font, font_scale, font_color, thickness)
        cv2.imwrite(frame_path, frame)
        self.i += 1

        return

def run():
    rospy.init_node('mvdm_ststem', anonymous=False)
    speed_action = [0.25, 0.25, 0.25, 0.25, 0.25]
    lane_action = [0, 0, 0, 0, 0]
    mvdm_system = Mvdm_System(speed_action, lane_action)
    rospy.spin()

if __name__ == '__main__':

    run()
