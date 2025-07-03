#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复GPU处理流水线中的错误
主要修复点：
1. 添加GPU内存状态检查
2. 修复CLAHE参数传递问题
3. 增加更详细的错误处理和调试信息
4. 添加GPU内存同步机制
"""

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

# --- 全局参数配置 ---
params = {
    "robot_ids": ["9298", "9289"],
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
    "use_gpu": True,
    "gpu_debug": True,  # 添加GPU调试开关
}

vehicle_params = {
    "9289": {"first_pos_delta": -10, "angular_delta": 0},
    "9293": {"first_pos_delta": 0, "angular_delta": 0},
    "9298": {"first_pos_delta": 0, "angular_delta": 0},
}


class GPUEnhancedMvdmSystem:
    def __init__(self, speed_action, lane_action):
        self.robot_ids = params["robot_ids"]
        self.bridge = CvBridge()

        # 检查并初始化GPU资源
        self.gpu_available = self._check_gpu_availability()
        if self.gpu_available:
            self._initialize_gpu_resources()
        else:
            rospy.logwarn("GPU not available or disabled, using CPU processing.")

        # 初始化机器人状态和控制变量
        self._initialize_control_variables()

        # ROS 订阅和发布
        self.subs = {
            robot_id: rospy.Subscriber(
                f'/limo{robot_id}/camera/image_raw',
                Image,
                partial(self.preprocess_image_callback, limo_id=robot_id),
                queue_size=1,
                buff_size=2 ** 24
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

        # 定时发布控制指令
        self.control_timer = rospy.Timer(rospy.Duration(params["control_period"]), self.cbControl)
        self.count = 0

        # 预设动作序列
        self.speed_expand = {id: [0] * 10 + np.repeat(speed_action[id], params["expand_num"]).tolist() for id in
                             self.robot_ids}
        self.lane_expand = {id: [0] * 10 + np.repeat(lane_action[id], params["expand_num"]).tolist() for id in
                            self.robot_ids}

    def _check_gpu_availability(self):
        """详细检查GPU可用性"""
        if not params["use_gpu"]:
            rospy.loginfo("GPU processing disabled in parameters.")
            return False

        try:
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count == 0:
                rospy.logwarn("No CUDA-enabled devices found.")
                return False

            # 测试基本GPU操作
            test_mat = cv2.cuda.GpuMat()
            test_img = np.zeros((100, 100), dtype=np.uint8)
            test_mat.upload(test_img)
            test_result = test_mat.download()

            rospy.loginfo(f"GPU initialization successful. Found {gpu_count} CUDA device(s).")
            return True

        except Exception as e:
            rospy.logerr(f"GPU availability check failed: {e}")
            return False

    def _initialize_gpu_resources(self):
        """为每个机器人分配独立的GPU内存缓冲区，并创建共享的处理模块。"""
        rospy.loginfo("Initializing GPU resources for each robot...")

        self.gpu_buffers = {}
        self.gpu_streams = {}  # 为每个机器人创建独立的CUDA流

        try:
            for robot_id in self.robot_ids:
                # 为每个机器人分配独立的GPU缓冲区
                self.gpu_buffers[robot_id] = {
                    "image": cv2.cuda.GpuMat(),
                    "gray": cv2.cuda.GpuMat(),
                    "enhanced": cv2.cuda.GpuMat(),
                    "blur": cv2.cuda.GpuMat(),
                    "edges": cv2.cuda.GpuMat(),
                }

                # 为每个机器人创建独立的CUDA流
                self.gpu_streams[robot_id] = cv2.cuda.Stream()

                if params["gpu_debug"]:
                    rospy.loginfo(f"GPU buffers allocated for robot {robot_id}")

            # 创建共享的GPU处理模块
            self.gpu_clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.gpu_gaussian = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
            self.gpu_canny = cv2.cuda.createCannyEdgeDetector(50, 150)

            rospy.loginfo("GPU processing modules created successfully.")

        except Exception as e:
            rospy.logerr(f"Failed to initialize GPU resources: {e}")
            self.gpu_available = False

    def _initialize_control_variables(self):
        """初始化所有与控制相关的变量"""
        self.exc_speed = {id: 0 for id in self.robot_ids}
        self.deviation_history = {id: deque(maxlen=params["smooth_window"]) for id in self.robot_ids}
        self.angular_history = {id: deque(maxlen=3) for id in self.robot_ids}
        self.integral_error = {id: 0 for id in self.robot_ids}
        self.last_deviation = {id: 0 for id in self.robot_ids}
        self.last_time = {id: time.time() for id in self.robot_ids}
        self.angular = {id: 0 for id in self.robot_ids}
        self.filtered_deviation = {id: 0 for id in self.robot_ids}
        self.first_pos = {id: 0 for id in self.robot_ids}
        self.lane_curvature = {id: 0 for id in self.robot_ids}

        # 控制参数
        self.Kp = params["Kp"]
        self.Ki = params["Ki"]
        self.Kd = params["Kd"]
        self.alpha = params["alpha"]
        self.min_angular = params["min_angular"]
        self.max_angular = params["max_angular"]

    def gpu_image_processing(self, image, limo_id):
        """修复后的GPU图像处理流水线"""
        try:
            # 获取当前机器人专用的缓冲区和流
            buffers = self.gpu_buffers[limo_id]
            stream = self.gpu_streams[limo_id]

            # 确保输入图像格式正确
            if len(image.shape) != 2:  # 如果不是灰度图，需要先转换
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 确保图像数据类型正确
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            if params["gpu_debug"]:
                rospy.logdebug(f"Processing image for robot {limo_id}, shape: {image.shape}, dtype: {image.dtype}")

            # 1. 上传图像到GPU
            buffers["gray"].upload(image, stream)

            # 等待上传完成
            stream.waitForCompletion()

            # 2. CLAHE 增强对比度 - 修复参数传递
            self.gpu_clahe.apply(buffers["gray"], buffers["enhanced"], stream)

            # 3. 高斯模糊
            self.gpu_gaussian.apply(buffers["enhanced"], buffers["blur"], stream)

            # 4. Canny 边缘检测
            self.gpu_canny.detect(buffers["blur"], buffers["edges"], stream)

            # 5. 等待所有GPU操作完成
            stream.waitForCompletion()

            # 6. 下载结果到CPU
            edges = buffers["edges"].download()

            if params["gpu_debug"]:
                rospy.logdebug(f"GPU processing completed for robot {limo_id}")

            return edges

        except cv2.error as e:
            rospy.logwarn(f"GPU processing failed for robot {limo_id}: {e}")
            if params["gpu_debug"]:
                rospy.logwarn(f"GPU error details: {str(e)}")
            return None

        except Exception as e:
            rospy.logerr(f"Unexpected error in GPU processing for robot {limo_id}: {e}")
            return None

    def cpu_image_processing(self, image):
        """CPU图像处理备用方案"""
        try:
            # 确保是灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # CLAHE增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # 高斯模糊
            blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

            # Canny边缘检测
            edges = cv2.Canny(blur, 50, 150)

            return edges

        except Exception as e:
            rospy.logerr(f"CPU image processing failed: {e}")
            return None

    def preprocess_image_callback(self, ros_image, limo_id):
        """图像消息回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error for {limo_id}: {e}")
            return

        # 图像预处理：裁剪下半部分
        height, width = cv_image.shape[:2]
        lower_half = cv_image[round(height // 1.75):, :]
        height, width = lower_half.shape[:2]

        # 图像处理：优先使用GPU，失败则回退到CPU
        edges = None
        if self.gpu_available:
            # 对于GPU处理，先转换为灰度图
            gray_image = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
            edges = self.gpu_image_processing(gray_image, limo_id)

        if edges is None:
            edges = self.cpu_image_processing(lower_half)

        if edges is None:
            rospy.logwarn(f"Both GPU and CPU image processing failed for robot {limo_id}")
            return

        # 后续处理逻辑保持不变
        self._process_lane_detection(edges, height, width, limo_id)

    def _process_lane_detection(self, edges, height, width, limo_id):
        """车道线检测和控制计算"""
        # HoughLinesP进行直线检测
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)

        left_points, right_points = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:
                    continue
                if slope < 0:
                    left_points.extend([(x1, y1), (x2, y2)])
                else:
                    right_points.extend([(x1, y1), (x2, y2)])

        y_max = height
        y_min = int(height * 0.1)

        left_line = self.fit_line(left_points, y_min, y_max)
        right_line = self.fit_line(right_points, y_min, y_max)

        if left_line is not None and right_line is not None:
            deviation = self.calculate_deviation(left_line, right_line, height, width, limo_id)
            smoothed_deviation = self.smooth_deviation(deviation, limo_id)
            self.filtered_deviation[limo_id] = self.alpha * smoothed_deviation + (1 - self.alpha) * \
                                               self.filtered_deviation[limo_id]
            raw_angular = self.enhanced_pid_control(self.filtered_deviation[limo_id], limo_id)
            self.angular[limo_id] = self.smooth_angular(raw_angular, limo_id)
            self.lane_curvature[limo_id] = self.calculate_lane_curvature(left_line, right_line)
        else:
            self.adaptive_control(left_line, right_line, limo_id)

    # --- 其余方法保持不变 ---
    def fit_line(self, points, y_min, y_max):
        if len(points) < 2:
            return None
        points = np.array(points)
        x, y = points[:, 0], points[:, 1]
        try:
            fit = np.polyfit(y, x, 1)
            return fit
        except np.linalg.LinAlgError:
            return None

    def calculate_deviation(self, left_fit, right_fit, image_height, image_width, limo_id):
        if left_fit is None or right_fit is None:
            return 0
        y_vehicle = image_height - 1
        x_left = left_fit[0] * y_vehicle + left_fit[1]
        x_right = right_fit[0] * y_vehicle + right_fit[1]
        lane_center = (x_left + x_right) / 2

        if self.first_pos[limo_id] == 0:
            self.first_pos[limo_id] = image_width // 2 + vehicle_params[limo_id]["first_pos_delta"]

        return self.first_pos[limo_id] - lane_center

    def smooth_deviation(self, deviation, limo_id):
        self.deviation_history[limo_id].append(deviation)
        if len(self.deviation_history[limo_id]) >= 2:
            smoothed = np.mean(list(self.deviation_history[limo_id]))
            if abs(smoothed - self.filtered_deviation[limo_id]) > params["max_deviation_change"]:
                smoothed = self.filtered_deviation[limo_id] + np.sign(smoothed - self.filtered_deviation[limo_id]) * \
                           params["max_deviation_change"]
            return smoothed
        return deviation

    def enhanced_pid_control(self, deviation, limo_id):
        current_time = time.time()
        dt = current_time - self.last_time[limo_id]
        if dt <= 0:
            dt = 0.1

        if abs(deviation) < params["deadzone"]:
            deviation = 0

        # PID控制
        proportional = self.Kp * deviation
        self.integral_error[limo_id] += deviation * dt
        max_integral = 100
        self.integral_error[limo_id] = max(min(self.integral_error[limo_id], max_integral), -max_integral)
        integral = self.Ki * self.integral_error[limo_id]
        derivative = self.Kd * (deviation - self.last_deviation[limo_id]) / dt

        pid_output = proportional + integral + derivative

        self.last_deviation[limo_id] = deviation
        self.last_time[limo_id] = current_time

        return max(min(pid_output, self.max_angular), self.min_angular)

    def smooth_angular(self, angular, limo_id):
        self.angular_history[limo_id].append(angular)
        if len(self.angular_history[limo_id]) >= 2:
            weights = np.array([0.2, 0.3, 0.5])[:len(self.angular_history[limo_id])]
            return np.average(list(self.angular_history[limo_id]), weights=weights / weights.sum())
        return angular

    def adaptive_control(self, left_line, right_line, limo_id):
        if left_line is None and right_line is None:
            self.angular[limo_id] *= 0.8
        else:
            target_angular = params["control_angular_max"]
            if left_line is not None:
                self.angular[limo_id] = -target_angular
            else:
                self.angular[limo_id] = target_angular

    def calculate_lane_curvature(self, left_fit, right_fit):
        return 0

    def publish_twist(self, limo_id, linear_x, angular_z):
        angular_z += vehicle_params[limo_id]["angular_delta"]
        self.vel_pubs[limo_id].publish(linear_x)
        self.angular_pubs[limo_id].publish(angular_z)

    def cbControl(self, event):
        index_lane = 0
        for limo_id in self.robot_ids:
            if self.count < len(self.lane_expand[limo_id]) and self.lane_expand[limo_id][self.count] == 0:
                self.exc_speed[limo_id] = self.speed_expand[limo_id][self.count]
                self.publish_twist(limo_id, self.exc_speed[limo_id], self.angular[limo_id])
            elif self.count < len(self.lane_expand[limo_id]):
                progress = min(index_lane / 40, 1.0)
                phase = progress * math.pi * 2
                angular_lane_change = self.lane_expand[limo_id][self.count] * math.sin(phase)
                self.angular[limo_id] = angular_lane_change
                self.exc_speed[limo_id] = self.speed_expand[limo_id][self.count]
                self.publish_twist(limo_id, self.exc_speed[limo_id], self.angular[limo_id])
                index_lane += 1

        self.count += 1

    def __del__(self):
        rospy.loginfo("Shutting down GPUEnhancedMvdmSystem node.")


def run():
    rospy.init_node('gpu_enhanced_mvdm_system', anonymous=False)
    speed_action = {"9298": [0.5] * 5, "9289": [0.5] * 5}
    lane_action = {"9298": [0] * 5, "9289": [0] * 5}
    try:
        mvdm_system = GPUEnhancedMvdmSystem(speed_action, lane_action)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal(f"An unhandled exception occurred: {e}")


if __name__ == '__main__':
    run()