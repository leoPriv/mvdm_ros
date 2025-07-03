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
    # "robot_ids": ["9293","9298"],
    "Kp": 0.0003,  # 降低比例增益
    "Ki": 0.00001,  # 添加积分项
    "Kd": 0.0008,  # 调整微分增益
    "alpha": 0.85,  # 增加滤波强度
    "max_angular": 0.4,  # 降低最大角速度
    "min_angular": -0.4,
    "control_period": 0.1,
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
        "first_pos_delta": 0,
        "angular_delta": 0,
    },
    "9298": {
        "first_pos_delta": 0,
        "angular_delta": 0,
    }
}


class EnhancedMvdmSystem:
    def __init__(self, speed_action, lane_action):
        now = datetime.now()
        time_str = now.strftime("%m%d%H%M%S")
        self.main_folder = os.path.join(os.getcwd(), time_str)
        self.csv_folder = os.path.join(self.main_folder, "csv")

        # 创建每个机器人的子文件夹
        self.robot_folders = {}
        self.picture_folders = {}
        self.picture_raw_folders = {}
        self.video_folders = {}
        self.picture_video_folders = {}
        self.csv_paths = {}

        # 原有变量
        self.robot_ids = params["robot_ids"]

        # 为每个机器人创建独立的文件夹结构
        self._create_folders()

        # 为每个机器人创建独立的CSV文件
        for robot_id in self.robot_ids:
            self.csv_paths[robot_id] = os.path.join(self.robot_folders[robot_id], f"slope_data_{robot_id}.csv")
            with open(self.csv_paths[robot_id], 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame_num', 'left_slope', 'right_slope', 'deviation', 'angular'])


        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        for j in range(20):
            self.cap.read()

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
        # 为每个机器人创建独立的帧计数器
        self.frame_counters = {id: 0 for id in self.robot_ids}
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
                partial(self.preprocess_image, extra_param=robot_id),
                queue_size=1
            )
            for robot_id in self.robot_ids
        }

        self.angular_pubs = {
            robot_id: rospy.Publisher(
                f'/limo{robot_id}/angular',
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

    def _create_folders(self):
        """创建主文件夹和每个机器人的子文件夹"""
        for robot_id in self.robot_ids:
            # 为每个机器人创建主文件夹
            robot_folder = os.path.join(self.main_folder, f"limo_{robot_id}")
            self.robot_folders[robot_id] = robot_folder
            os.makedirs(robot_folder, exist_ok=True)

            # 创建各种子文件夹
            self.picture_folders[robot_id] = os.path.join(robot_folder, "picture")
            self.picture_raw_folders[robot_id] = os.path.join(robot_folder, "raw_picture")
            self.video_folders[robot_id] = os.path.join(robot_folder, "video")
            self.picture_video_folders[robot_id] = os.path.join(robot_folder, "picture_video")

            os.makedirs(self.picture_folders[robot_id], exist_ok=True)
            os.makedirs(self.picture_raw_folders[robot_id], exist_ok=True)
            os.makedirs(self.video_folders[robot_id], exist_ok=True)
            os.makedirs(self.picture_video_folders[robot_id], exist_ok=True)

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

        if dt <= 0:
            dt = 0.1

        # 死区处理，减少小幅震荡
        if abs(deviation) < params["deadzone"]:
            deviation = 0

        # PID计算
        # 比例项
        proportional = self.Kp * deviation

        # 积分项（带积分饱和限制）
        self.integral_error[limo_id] += deviation * dt
        # 积分饱和限制
        max_integral = 100
        self.integral_error[limo_id] = max(min(self.integral_error[limo_id], max_integral), -max_integral)
        integral = self.Ki * self.integral_error[limo_id]

        # 微分项
        derivative = self.Kd * (deviation - self.last_deviation[limo_id]) / dt

        # PID输出
        pid_output = proportional + integral + derivative

        # 速度自适应调整
        current_speed = self.exc_speed.get(limo_id, 0.5)
        if current_speed > 0:
            # 速度越快，控制增益越大
            speed_factor = min(current_speed / 0.5, 2.0)  # 最大2倍增益
            pid_output *= speed_factor

        # 角度预测补偿
        if params["angle_compensation"] and len(self.angular_history[limo_id]) >= 2:
            # 基于历史角速度预测未来趋势
            angular_trend = np.mean(np.diff(list(self.angular_history[limo_id])))
            prediction_compensation = -0.1 * angular_trend  # 预测补偿系数
            pid_output += prediction_compensation

        # 输出限制
        pid_output = max(min(pid_output, self.max_angular), self.min_angular)

        # 更新历史记录
        self.last_deviation[limo_id] = deviation
        self.last_time[limo_id] = current_time

        return pid_output

    def smooth_angular(self, angular, limo_id):
        """平滑角速度输出"""
        self.angular_history[limo_id].append(angular)

        if len(self.angular_history[limo_id]) >= 2:
            # 使用加权平均，最新值权重更大
            weights = np.array([0.2, 0.3, 0.5])[:len(self.angular_history[limo_id])]
            weights = weights / weights.sum()
            smoothed_angular = np.average(list(self.angular_history[limo_id]), weights=weights)
            return smoothed_angular
        else:
            return angular

    def adaptive_control(self, left_line, right_line, limo_id):
        """自适应控制策略"""
        if left_line is None and right_line is None:
            # 两条线都丢失，保持最后的控制量但逐渐减小
            self.angular[limo_id] *= 0.8
            return

        if left_line is None or right_line is None:
            # 只有一条线，使用单边控制
            if left_line is not None:
                # 只有左边线，倾向于向右调整
                slope_left, _ = left_line
                self.angular[limo_id] = -0.1 * np.sign(slope_left)
            else:
                # 只有右边线，倾向于向左调整
                slope_right, _ = right_line
                self.angular[limo_id] = 0.1 * np.sign(slope_right)
            return

        # 正常双线控制已在主函数中处理

    def calculate_lane_curvature(self, left_line, right_line):
        """计算车道曲率，用于前瞻控制"""
        if left_line is None or right_line is None:
            return 0

        slope_left, _ = left_line
        slope_right, _ = right_line

        # 简单的曲率估计：左右车道线斜率差异
        curvature = abs(slope_left - slope_right)
        return curvature

    def preprocess_image(self, image, extra_param=None):
        """图像处理主函数（保持原有逻辑，但改进控制部分）"""
        limo_id = extra_param
        image = self.bridge.imgmsg_to_cv2(image, "bgr8")

        # 使用每个机器人独立的帧计数器
        frame_num = self.frame_counters[limo_id]

        # 保存原始图像到对应机器人的文件夹
        output_path = os.path.join(self.picture_raw_folders[limo_id], f"picture_{frame_num:04d}.jpg")
        cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 图像处理（保持原有逻辑）
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

        left_points = []
        right_points = []
        line_image = np.zeros_like(lower_half)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:
                    continue
                if slope < 0:
                    left_points.append((x1, y1))
                    left_points.append((x2, y2))
                else:
                    right_points.append((x1, y1))
                    right_points.append((x2, y2))

        y_max = height
        y_min = int(height * 0.1)

        left_line = self.fit_line(left_points, y_min, y_max, (255, 0, 0), line_image)
        right_line = self.fit_line(right_points, y_min, y_max, (0, 0, 255), line_image)

        # 改进的控制逻辑
        deviation = 0
        if left_line is not None and right_line is not None:
            deviation = self.calculate_deviation(left_line, right_line,height,width, limo_id)

            # 平滑偏差
            smoothed_deviation = self.smooth_deviation(deviation, limo_id)
            self.filtered_deviation[limo_id] = self.alpha * smoothed_deviation + (1 - self.alpha) * \
                                               self.filtered_deviation[limo_id]

            # 使用增强PID控制
            raw_angular = self.enhanced_pid_control(self.filtered_deviation[limo_id], limo_id)

            # 平滑角速度输出
            self.angular[limo_id] = self.smooth_angular(raw_angular, limo_id)

            # 计算车道曲率用于自适应
            self.lane_curvature[limo_id] = self.calculate_lane_curvature(left_line, right_line)

        else:
            # 车道线丢失时的处理
            self.adaptive_control(left_line, right_line, limo_id)

        # 保存数据到对应机器人的CSV文件
        left_slope = left_line[0] if left_line is not None else 0
        right_slope = right_line[0] if right_line is not None else 0

        with open(self.csv_paths[limo_id], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [frame_num, left_slope, right_slope, self.filtered_deviation[limo_id], self.angular[limo_id]])

        # 可视化部分（保持原有逻辑）
        if left_line is not None and right_line is not None:
            _, fit_left = left_line
            _, fit_right = right_line
            x_left = int(fit_left[0] * y_max + fit_left[1])
            x_right = int(fit_right[0] * y_max + fit_right[1])
            lane_center = (x_left + x_right) // 2

            cv2.circle(line_image, (lane_center, y_max), 5, (0, 255, 255), -1)
            cv2.circle(line_image, (width // 2, y_max), 5, (0, 255, 0), -1)

        # 添加文字信息
        text_frame = f"Frame: {frame_num} (Limo: {limo_id})"
        text_left = f"Left Slope: {left_slope:.2f}"
        text_right = f"Right Slope: {right_slope:.2f}"
        text_deviation = f"Deviation: {self.filtered_deviation[limo_id]:.2f}"
        text_angular = f"Angular: {self.angular[limo_id]:.3f}"
        text_curvature = f"Curvature: {self.lane_curvature[limo_id]:.3f}"
        text_deviation_raw = f"deviation_raw: {deviation:.3f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)
        thickness = 2

        cv2.putText(line_image, text_frame, (20, 30), font, font_scale, font_color, thickness)
        cv2.putText(line_image, text_left, (20, 55), font, font_scale, font_color, thickness)
        cv2.putText(line_image, text_right, (20, 80), font, font_scale, font_color, thickness)
        cv2.putText(line_image, text_deviation, (20, 105), font, font_scale, font_color, thickness)
        cv2.putText(line_image, text_angular, (20, 130), font, font_scale, font_color, thickness)
        cv2.putText(line_image, text_curvature, (20, 155), font, font_scale, font_color, thickness)
        cv2.putText(line_image, text_deviation_raw, (20, 180), font, font_scale, font_color, thickness)

        # 保存处理后的图像到对应机器人的文件夹
        result = cv2.addWeighted(lower_half, 0.8, line_image, 1, 0)
        output_path = os.path.join(self.picture_folders[limo_id], f"picture_{frame_num:04d}.jpg")
        cv2.imwrite(output_path, result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 处理视频帧
        ret, frame = self.cap.read()
        if ret:
            frame_path = os.path.join(self.video_folders[limo_id], f"video_{frame_num:04d}.jpg")
            cv2.putText(frame, text_frame, (20, 30), font, font_scale, font_color, thickness)
            cv2.putText(frame, text_deviation, (20, 55), font, font_scale, font_color, thickness)
            cv2.putText(frame, text_angular, (20, 80), font, font_scale, font_color, thickness)
            cv2.imwrite(frame_path, frame)

            # 合并图像
            target_width = max(result.shape[1], frame.shape[1])
            if result.shape[1] < target_width:
                result = cv2.copyMakeBorder(result, 0, 0, 0, target_width - result.shape[1],
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if frame.shape[1] < target_width:
                frame = cv2.copyMakeBorder(frame, 0, 0, 0, target_width - frame.shape[1],
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))

            combined_img = np.vstack((result, frame))
            frame_name = os.path.join(self.picture_video_folders[limo_id], f"picture_video_{frame_num:04d}.jpg")
            cv2.imwrite(frame_name, combined_img)

        # 更新该机器人的帧计数器
        self.frame_counters[limo_id] += 1

    def fit_line(self, points, y_min, y_max, color, line_image):
        """拟合直线（保持原有逻辑）"""
        if len(points) < 2:
            return None
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        fit = np.polyfit(y, x, 1)
        slope = 1 / fit[0]
        x_min = int(fit[0] * y_min + fit[1])
        x_max = int(fit[0] * y_max + fit[1])
        cv2.line(line_image, (x_min, y_min), (x_max, y_max), color, 4)
        return slope, fit

    def calculate_deviation(self, left_line, right_line,image_height,image_width, limo_id):
        """计算偏差（保持原有逻辑）"""
        if left_line is None or right_line is None:
            return 0

        try:
            slope_left, fit_left = left_line
            slope_right, fit_right = right_line
        except (TypeError, IndexError):
            return 0

        y_vehicle = image_height - 1
        x_left = fit_left[0] * y_vehicle + fit_left[1]
        x_right = fit_right[0] * y_vehicle + fit_right[1]
        lane_center = (x_left + x_right) / 2
        self.first_pos[limo_id] = image_width // 2 + vehicle_params[limo_id]["first_pos_delta"]

        deviation = self.first_pos[limo_id] - lane_center
        return deviation

    def publish_twist(self, limo_id, ang_pub, vel_pub, linear_x, angular_z):
        """发布控制命令（改进版）"""
        # # 更温和的角速度限制
        if abs(angular_z) > params["control_angular_max"]:
            if angular_z > 0:
                angular = params["control_angular"]
            else:
                angular = -params["control_angular"]
        else:
            # 按比例缩放，而不是直接置零
            angular = angular_z * (params["control_angular"] / params["control_angular_max"])
            # angular = 0
        if angular < 0:
            angular += vehicle_params[limo_id]["angular_delta"]

        vel_pub.publish(linear_x)
        ang_pub.publish(angular*1.2)

        rospy.loginfo(
            f"publish to {limo_id} - Speed: {linear_x:.3f}, Angular: {angular:.3f} (Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]})")

    def cbControl(self, event):
        """控制回调函数（保持原有逻辑但使用改进的发布函数）"""
        index_lane = 0
        for limo_id in self.robot_ids:
            if self.lane_expand[limo_id][self.count] == 0:
                self.exc_speed[limo_id] = self.speed_expand[limo_id][self.count]
                self.publish_twist(limo_id, self.angular_pubs[limo_id], self.vel_pubs[limo_id],
                                   self.exc_speed[limo_id], self.angular[limo_id])
            else:
                # 车道变换逻辑（保持原有）
                progress = min(index_lane / 40, 1.0)
                phase = progress * math.pi * 2
                angular = self.lane_expand[limo_id][self.count] * math.sin(phase)

                try:
                    max_angle = math.pi / 3
                    current_angular = max(min(angular, max_angle), -max_angle)
                    speed_compensation = 1.0 / math.cos(abs(current_angular))
                    max_speed = 2.0 * self.speed_expand[limo_id][self.count]
                    actual_speed = min(self.speed_expand[limo_id][self.count] * speed_compensation, max_speed)
                except:
                    actual_speed = self.speed_expand[limo_id][self.count]
                    current_angular = 0

                index_lane += 1
                self.exc_speed[limo_id] = actual_speed
                self.angular[limo_id] = current_angular * (1 / actual_speed) / 3.5
                self.publish_twist(limo_id, self.angular_pubs[limo_id], self.vel_pubs[limo_id],
                                   self.exc_speed[limo_id], self.angular[limo_id])

        self.count += 1




def run():
    rospy.init_node('enhanced_mvdm_system', anonymous=False)
    speed_action = {'9293': [0.5, 0.5, 0.5, 0.5, 0.5], "9298": [0.5, 0.5, 0.5, 0.5, 0.5], "9289": [0.5, 0.5, 0.5, 0.5, 0.5]}
    lane_action = {"9293": [0, 0, 0, 0, 0], "9298": [0, 0, 0, 0, 0], "9289": [0, 0, 0, 0, 0]}
    mvdm_system = EnhancedMvdmSystem(speed_action, lane_action)
    rospy.spin()


if __name__ == '__main__':
    run()