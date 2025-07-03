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
import threading # 1. 引入线程模块

params = {
    # "robot_ids":["9293", "9298", "9289"],
    # "robot_ids": ["9289"],
    "robot_ids":["9293"],
    "Kp": 0.0003,
    "Ki": 0.00001,
    "Kd": 0.0008,
    "alpha": 0.85,
    "max_angular": 0.4,
    "min_angular": -0.4,
    "control_period": 0.1,
    "expand_num": 40,
    "control_angular": 0.15,
    "control_angular_max": 0.12,
    "deadzone": 5,
    "smooth_window": 5,
    "max_deviation_change": 20,
    "angle_compensation": True,
}


class EnhancedMvdmSystem:
    def __init__(self, speed_action, lane_action):
        # ... (保持原有的初始化代码，文件夹创建等)

        self.robot_ids = params["robot_ids"]

        # 2. 为每个机器人创建一个线程锁
        self.locks = {robot_id: threading.Lock() for robot_id in self.robot_ids}

        # 原有变量
        self.speed_expand = {id: [0] * 5 + np.repeat(speed_action[id], params["expand_num"]).tolist() for id in self.robot_ids}
        self.lane_expand = {id: [0] * 5 + np.repeat(lane_action[id], params["expand_num"]).tolist() for id in self.robot_ids}
        self.exc_speed = {id: 0 for id in self.robot_ids}

        # 增强的控制变量
        self.deviation_history = {id: deque(maxlen=params["smooth_window"]) for id in self.robot_ids}
        self.angular_history = {id: deque(maxlen=3) for id in self.robot_ids}
        self.integral_error = {id: 0 for id in self.robot_ids}
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
        self.subs = {
            robot_id: rospy.Subscriber(
                f'/limo{robot_id}/camera/image_raw',
                Image,
                self.preprocess_image, # 3. 回调函数现在用于启动线程
                callback_args=robot_id, # 将robot_id传递给回调
                queue_size=1, # 仅保留最新的消息
                buff_size=2**24 # 增加缓冲区大小以防万一
            )
            for robot_id in self.robot_ids
        }

        self.angular_pubs = {
            robot_id: rospy.Publisher(f'/limo{robot_id}/angular', Float32, queue_size=1)
            for robot_id in self.robot_ids
        }

        self.vel_pubs = {
            robot_id: rospy.Publisher(f'/limo{robot_id}/vel', Float32, queue_size=1)
            for robot_id in self.robot_ids
        }

        self.control_timer = rospy.Timer(rospy.Duration(params["control_period"]), self.cbControl)


    def preprocess_image(self, image_msg, limo_id):
        """
        ROS图像消息回调函数。
        接收到消息后，立即创建一个新线程来处理图像，避免阻塞回调队列。
        """
        # 创建并启动一个守护线程(daemon)来处理图像
        # 守护线程会随主程序退出而退出
        thread = threading.Thread(target=self._thread_safe_image_processing, args=(image_msg, limo_id))
        thread.setDaemon(True)
        thread.start()


    def _thread_safe_image_processing(self, image_msg, limo_id):
        """
        在独立线程中执行的图像处理和控制计算。
        所有对共享资源的访问都通过线程锁来保护。
        """
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image for robot {limo_id}: {e}")
            return

        # --- 图像处理逻辑 (与原版相同) ---
        height, width = image.shape[:2]
        lower_half = image[round(height // 1.75): height, :]

        height, width = lower_half.shape[:2]
        lab = cv2.cvtColor(lower_half, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)

        left_points, right_points = [], []
        line_image = np.zeros_like(image)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0: continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5: continue
                if slope < 0:
                    left_points.extend([(x1, y1), (x2, y2)])
                else:
                    right_points.extend([(x1, y1), (x2, y2)])

        y_max = height
        y_min = int(height * 0.1)

        left_line = self.fit_line(left_points, y_min, y_max, (255, 0, 0), line_image)
        right_line = self.fit_line(right_points, y_min, y_max, (0, 0, 255), line_image)

        # --- 线程安全的控制计算 ---
        with self.locks[limo_id]: # 4. 在修改共享变量前加锁
            deviation = 0
            if left_line is not None and right_line is not None:
                deviation = self.calculate_deviation(left_line, right_line, width, limo_id)
                smoothed_deviation = self.smooth_deviation(deviation, limo_id)
                self.filtered_deviation[limo_id] = self.alpha * smoothed_deviation + (1 - self.alpha) * self.filtered_deviation[limo_id]
                raw_angular = self.enhanced_pid_control(self.filtered_deviation[limo_id], limo_id)
                self.angular[limo_id] = self.smooth_angular(raw_angular, limo_id)
                self.lane_curvature[limo_id] = self.calculate_lane_curvature(left_line, right_line)
            else:
                self.adaptive_control(left_line, right_line, limo_id)
        # 锁在此处自动释放

    # 其他方法 (smooth_deviation, enhanced_pid_control, etc.) 保持不变
    # 因为它们被_thread_safe_image_processing调用，并且已经处于锁的保护之下。
    # ... (此处省略未改动的函数: smooth_deviation, enhanced_pid_control, smooth_angular,
    # adaptive_control, calculate_lane_curvature, fit_line, calculate_deviation, publish_twist)
    # 你可以将它们原封不动地复制到这里

    def smooth_deviation(self, deviation, limo_id):
        """平滑偏差值，减少噪声影响"""
        # 添加到历史记录
        self.deviation_history[limo_id].append(deviation)

        # 计算滑动平均
        if len(self.deviation_history[limo_id]) >= 2:
            smoothed = np.mean(list(self.deviation_history[limo_id]))

            # 限制偏差变化率，防止突变
            if abs(smoothed - self.filtered_deviation[limo_id]) > params["max_deviation_change"]:
                if smoothed > self.filtered_deviation[limo_id]:
                    smoothed = self.filtered_deviation[limo_id] + params["max_deviation_change"]
                else:
                    smoothed = self.filtered_deviation[limo_id] - params["max_deviation_change"]

            return smoothed
        else:
            return deviation

    def enhanced_pid_control(self, deviation, limo_id):
        """增强的PID控制器"""
        current_time = time.time()
        dt = current_time - self.last_time[limo_id]

        if dt <= 0: dt = 0.1

        if abs(deviation) < params["deadzone"]: deviation = 0

        proportional = self.Kp * deviation

        self.integral_error[limo_id] += deviation * dt
        max_integral = 100
        self.integral_error[limo_id] = max(min(self.integral_error[limo_id], max_integral), -max_integral)
        integral = self.Ki * self.integral_error[limo_id]

        derivative = self.Kd * (deviation - self.last_deviation[limo_id]) / dt

        pid_output = proportional + integral + derivative

        current_speed = self.exc_speed.get(limo_id, 0.5)
        if current_speed > 0:
            speed_factor = min(current_speed / 0.5, 2.0)
            pid_output *= speed_factor

        if params["angle_compensation"] and len(self.angular_history[limo_id]) >= 2:
            angular_trend = np.mean(np.diff(list(self.angular_history[limo_id])))
            prediction_compensation = -0.1 * angular_trend
            pid_output += prediction_compensation

        pid_output = max(min(pid_output, self.max_angular), self.min_angular)

        self.last_deviation[limo_id] = deviation
        self.last_time[limo_id] = current_time

        return pid_output

    def smooth_angular(self, angular, limo_id):
        """平滑角速度输出"""
        self.angular_history[limo_id].append(angular)

        if len(self.angular_history[limo_id]) >= 2:
            weights = np.array([0.2, 0.3, 0.5])[:len(self.angular_history[limo_id])]
            weights = weights / weights.sum()
            smoothed_angular = np.average(list(self.angular_history[limo_id]), weights=weights)
            return smoothed_angular
        else:
            return angular

    def adaptive_control(self, left_line, right_line, limo_id):
        """自适应控制策略"""
        if left_line is None and right_line is None:
            self.angular[limo_id] *= 0.8
            return

        if left_line is None or right_line is None:
            if left_line is not None:
                slope_left, _ = left_line
                self.angular[limo_id] = -0.1 * np.sign(slope_left)
            else:
                slope_right, _ = right_line
                self.angular[limo_id] = 0.1 * np.sign(slope_right)
            return

    def calculate_lane_curvature(self, left_line, right_line):
        """计算车道曲率，用于前瞻控制"""
        if left_line is None or right_line is None: return 0
        slope_left, _ = left_line
        slope_right, _ = right_line
        curvature = abs(slope_left - slope_right)
        return curvature

    def fit_line(self, points, y_min, y_max, color, line_image):
        """拟合直线（保持原有逻辑）"""
        if len(points) < 2: return None
        points = np.array(points)
        x, y = points[:, 0], points[:, 1]
        fit = np.polyfit(y, x, 1)
        slope = 1 / fit[0]
        x_min = int(fit[0] * y_min + fit[1])
        x_max = int(fit[0] * y_max + fit[1])
        cv2.line(line_image, (x_min, y_min), (x_max, y_max), color, 4)
        return slope, fit

    def calculate_deviation(self, left_line, right_line, image_width, limo_id):
        """计算偏差（保持原有逻辑）"""
        if left_line is None or right_line is None: return 0
        try:
            slope_left, fit_left = left_line
            slope_right, fit_right = right_line
        except (TypeError, IndexError):
            return 0

        y_vehicle = image_width - 1
        x_left = fit_left[0] * y_vehicle + fit_left[1]
        x_right = fit_right[0] * y_vehicle + fit_right[1]
        lane_center = (x_left + x_right) / 2
        self.first_pos[limo_id] = image_width // 2 - 10
        deviation = self.first_pos[limo_id] - lane_center
        return deviation

    def publish_twist(self, limo_id, ang_pub, vel_pub, linear_x, angular_z):
        """发布控制命令（改进版）"""
        if abs(angular_z) > params["control_angular_max"]:
            angular = params["control_angular"] if angular_z > 0 else -params["control_angular"]
        else:
            angular = angular_z * (params["control_angular"] / params["control_angular_max"])

        vel_pub.publish(linear_x)
        ang_pub.publish(angular)
        rospy.loginfo(f"Enhanced Control to {limo_id} - Speed: {linear_x:.3f}, Angular: {angular:.3f}")


    def cbControl(self, event):
        """控制回调函数，定时发布控制指令"""
        index_lane = 0
        for limo_id in self.robot_ids:
            with self.locks[limo_id]: # 5. 读取共享数据前加锁
                # 复制需要使用的共享变量，以尽快释放锁
                current_angular = self.angular[limo_id]

            # 在锁外部执行后续逻辑
            if self.lane_expand[limo_id][self.count] == 0:
                self.exc_speed[limo_id] = self.speed_expand[limo_id][self.count]
                self.publish_twist(limo_id, self.angular_pubs[limo_id], self.vel_pubs[limo_id], self.exc_speed[limo_id], current_angular)
            else:
                # 车道变换逻辑
                progress = min(index_lane / 40, 1.0)
                phase = progress * math.pi * 2
                angular_lane_change = self.lane_expand[limo_id][self.count] * math.sin(phase)

                try:
                    max_angle = math.pi / 3
                    current_angular_lc = max(min(angular_lane_change, max_angle), -max_angle)
                    speed_compensation = 1.0 / math.cos(abs(current_angular_lc))
                    max_speed = 2.0 * self.speed_expand[limo_id][self.count]
                    actual_speed = min(self.speed_expand[limo_id][self.count] * speed_compensation, max_speed)
                except:
                    actual_speed = self.speed_expand[limo_id][self.count]
                    current_angular_lc = 0

                index_lane += 1
                self.exc_speed[limo_id] = actual_speed
                # 注意：这里的车道变换角速度直接覆盖了PID的计算结果
                final_angular = current_angular_lc * (1 / actual_speed) / 3.5
                self.publish_twist(limo_id, self.angular_pubs[limo_id], self.vel_pubs[limo_id], self.exc_speed[limo_id], final_angular)

        self.count += 1


def run():
    rospy.init_node('enhanced_mvdm_system', anonymous=False)
    speed_action = {'9293': [0.5, 0.5, 0.5, 0.5, 0.5], "9298": [0.5, 0.5, 0.5, 0.5, 0.5],"9289": [0.5, 0.5, 0.5, 0.5, 0.5]}
    lane_action = {"9293": [0, 0, 0, 0, 0], "9298": [0, 0, 0, 0, 0], "9289": [0, 0, 0, 0, 0]}
    mvdm_system = EnhancedMvdmSystem(speed_action, lane_action)
    rospy.spin()


if __name__ == '__main__':
    run()