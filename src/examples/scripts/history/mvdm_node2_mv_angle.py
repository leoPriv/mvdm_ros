#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import os
from std_msgs.msg import Float32
from datetime import datetime
import math
from functools import partial
# from arguments import params
params = {
          "robot_ids":["9293", "9298", "9289"],
          "max_angular": 0.6,
          "min_angular": -0.6,
          "control_period": 0.1,
          "expand_num": 40,
          "control_angular": 0.2,
          "control_angular_max": 0.15,
          }
class Mvdm_System:
    def __init__(self, speed_action, lane_action):
        # 存储信息
        # now = datetime.now()
        # time_str = now.strftime("%m%d%H%M%S")
        # self.main_folder = os.path.join(os.getcwd(), time_str)
        # self.video_folder = os.path.join(self.main_folder, "video")
        # self._create_folders()

        # self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        # for j in range(20):
        #     self.cap.read()

        # 补充信息
        self.robot_ids = params["robot_ids"]
        self.min_angular = params["min_angular"]
        self.max_angular = params["max_angular"]
        self.speed_expand = {id: np.repeat(speed_action[id], params["expand_num"]).tolist() for id in self.robot_ids}
        self.lane_expand = {id: np.repeat(lane_action[id], params["expand_num"]).tolist() for id in self.robot_ids}
        self.exc_speed = {id: 0 for id in self.robot_ids}
        self.angular = {id: 0 for id in self.robot_ids}
        self.index_lane = {id: 0 for id in self.robot_ids}
        self.i = 0
        self.count = 0

        # 接收订阅
        self.angular_subs = {
            robot_id: rospy.Subscriber(
                f'/limo{robot_id}/angular_info',  # 动态生成话题名
                Float32,
                partial(self.angular_callback, extra_param=robot_id)  # 传递当前robot_id
            )
            for robot_id in self.robot_ids
        }
        # 处理发布
        self.pubs = {
            robot_id: rospy.Publisher(
                f'/limo{robot_id}/cmd_vel',
                Twist,
                queue_size=1
            )
            for robot_id in self.robot_ids
        }
        self.control_timer = rospy.Timer(rospy.Duration(params["control_period"]), self.cbControl)

    # def _create_folders(self):
    #     """创建主文件夹和子文件夹"""
    #     os.makedirs(self.video_folder, exist_ok=True)

    def angular_callback(self, angular,extra_param=None):
        limo_id = extra_param
        self.angular[limo_id] = angular.data

    def publish_keep_twist(self,publisher, linear_x, angular_z):
        twist = Twist()
        exc_angular = max(angular_z,self.min_angular)
        exc_angular = min(exc_angular,self.max_angular)
        if exc_angular > params["control_angular_max"]:
            angular = params["control_angular"]
        elif exc_angular < -params["control_angular_max"]:
            angular = -params["control_angular"]
        else:
            angular = 0
        twist.linear.x = linear_x
        twist.angular.z = angular
        publisher.publish(twist)
        rospy.loginfo(f"Published to {publisher.name}: {twist}")
    def publish_turn_twist(self,publisher, linear_x, angular_z):
        twist = Twist()
        exc_angular = max(angular_z,self.min_angular)
        exc_angular = min(exc_angular,self.max_angular)
        twist.linear.x = linear_x
        twist.angular.z = exc_angular
        publisher.publish(twist)
        rospy.loginfo(f"Published to {publisher.name}: {twist}")


    def cbControl(self, event):
        for limo_id in self.robot_ids:
            if self.lane_expand[limo_id][self.count] == 0:
                self.index_lane[limo_id] = 0
                self.exc_speed[limo_id] = self.speed_expand[limo_id][self.count]
                self.publish_keep_twist(self.pubs[limo_id], self.exc_speed[limo_id], self.angular[limo_id])
            elif self.lane_expand[limo_id][self.count] != 0:
                progress = min(self.index_lane[limo_id] / params["expand_num"], 1.0)
                phase = progress * math.pi * 2  # 0~2π
                angular = self.lane_expand[limo_id][self.count]  * math.sin(phase)
                # 计算速度补偿以保持x方向分量
                try:
                    # 限制最大转向角度为60度(约1.047弧度)
                    max_angle = math.pi / 3
                    current_angular = max(min(angular, max_angle), -max_angle)

                    # 计算补偿速度: v = v_desired / cos(θ)
                    speed_compensation = 1.0 / math.cos(abs(current_angular))
                    # 限制最大补偿速度为初始值的2倍
                    max_speed = 2.0 * self.speed_expand[limo_id][self.count]
                    actual_speed = min(self.speed_expand[limo_id][self.count] * speed_compensation, max_speed)
                except:
                    # 如果计算出错(如cos(90°))，使用默认速度
                    actual_speed = self.speed_expand[limo_id][self.count]
                    current_angular = 0
                self.index_lane[limo_id] += 1
                self.exc_speed[limo_id] = actual_speed
                self.angular[limo_id] = current_angular * (1 / actual_speed) / 4
                self.publish_turn_twist(self.pubs[limo_id], self.exc_speed[limo_id], self.angular[limo_id])
        self.count += 1
        # 在图像上添加文字
        # text_frame = f"Frame: {self.i:.2f}"
        # text_time = f"Time: {datetime.now().time()}"
        # text_angular = f"Angular: {self.angular}"
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        # font_color = (255, 255, 255)  # 白色文字
        # thickness = 2
        # ret, frame = self.cap.read()
        # frame_path = os.path.join(self.video_folder, f"video_{self.i:04d}.jpg")
        # cv2.putText(frame, text_frame, (20, 30), font, font_scale, font_color, thickness)
        # cv2.putText(frame, text_time, (20, 60), font, font_scale, font_color, thickness)
        # cv2.putText(frame, text_angular, (20, 90), font, font_scale, font_color, thickness)
        # cv2.imwrite(frame_path, frame)
        # self.i += 1
        return

def run():
    rospy.init_node('mvdm_ststem', anonymous=False)
    speed_action = {'9293':[0.3, 0.3, 0.3, 0.3, 0.3],"9298":[0.3, 0.3, 0.3, 0.3, 0.3],"9289":[ 0.3, 0.3, 0.3, 0.3, 0.3]}
    lane_action = {"9293": [0, 0, 0, 0, 0],"9298":[0, 0, 0, 0, 0],"9289":[ 0, 0, 0, 0, 0]}
    mvdm_system = Mvdm_System(speed_action, lane_action)
    rospy.spin()

if __name__ == '__main__':
    run()
