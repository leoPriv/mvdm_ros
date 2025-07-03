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
import threading
from queue import Queue

params = {
    "robot_ids": ["9293", "9298"],
    "Kp": 0.0003,
    "Ki": 0.00001,
    "Kd": 0.0008,
    "alpha": 0.85,
    "max_angular": 0.4,
    "min_angular": -0.4,
    "control_period": 0.15,
    "expand_num": 40,
    "control_angular": 0.15,
    "control_angular_max": 0.12,
    "deadzone": 5,
    "smooth_window": 5,
    "max_deviation_change": 20,
    "angle_compensation": True,
}

vehicle_params = {
    "9289": {"first_pos_delta": -10, "angular_delta": 0},
    "9293": {"first_pos_delta": -10, "angular_delta": -0.005},
    "9298": {"first_pos_delta": -80, "angular_delta": 0}
}


class ImageProcessorThread(threading.Thread):
    """单独的线程用于处理单个Limo小车的图像"""

    def __init__(self, robot_id, folders, csv_path, bridge, cap):
        super().__init__()
        self.robot_id = robot_id
        self.folders = folders
        self.csv_path = csv_path
        self.bridge = bridge
        self.cap = cap
        self.queue = Queue(maxsize=5)  # 限制队列大小防止内存占用过高
        self.running = True
        self.daemon = True  # 设置为守护线程，主线程退出时自动结束

        # 控制相关变量
        self.deviation_history = deque(maxlen=params["smooth_window"])
        self.angular_history = deque(maxlen=3)
        self.integral_error = 0
        self.last_deviation = 0
        self.last_time = time.time()
        self.angular = 0
        self.filtered_deviation = 0
        self.first_pos = 0
        self.first = True
        self.lane_curvature = 0
        self.frame_counter = 0

    def add_image(self, image_msg):
        """添加图像到处理队列"""
        if not self.queue.full():
            self.queue.put(image_msg)

    def run(self):
        """线程主循环"""
        while self.running or not self.queue.empty():
            try:
                image_msg = self.queue.get(timeout=1)
                self.process_image(image_msg)
            except Exception as e:
                rospy.logerr(f"Error in ImageProcessorThread for {self.robot_id}: {str(e)}")

    def stop(self):
        """停止线程"""
        self.running = False

    def process_image(self, image_msg):
        """处理单个图像"""
        start_time = time.time()
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        frame_num = self.frame_counter

        # 保存原始图像
        output_path = os.path.join(self.folders["picture_raw"], f"picture_{frame_num:04d}.jpg")
        cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 图像处理
        height, width = image.shape[:2]
        lower_half = image[round(height // 1.75): height, :]
        height, width = lower_half.shape[:2]

        # 图像增强处理
        lab = cv2.cvtColor(lower_half, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 边缘检测
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 车道线检测
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)
        left_points, right_points = [], []
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

        # 拟合车道线
        left_line = self.fit_line(left_points, y_min, y_max, (255, 0, 0), line_image)
        right_line = self.fit_line(right_points, y_min, y_max, (0, 0, 255), line_image)

        # 计算偏差和控制量
        deviation = self.calculate_deviation(left_line, right_line, width)
        self.update_control(deviation, left_line, right_line)

        # 保存数据到CSV
        left_slope = left_line[0] if left_line is not None else 0
        right_slope = right_line[0] if right_line is not None else 0
        self.save_csv_data(frame_num, left_slope, right_slope)

        # 可视化处理
        result = self.visualize_processing(lower_half, line_image, left_line, right_line,
                                           frame_num, left_slope, right_slope, y_max, width)

        # 保存处理结果
        self.save_results(result, frame_num)

        self.frame_counter += 1

    def fit_line(self, points, y_min, y_max, color, line_image):
        """拟合直线"""
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

    def calculate_deviation(self, left_line, right_line, image_width):
        """计算偏差"""
        if left_line is None or right_line is None:
            return 0

        try:
            slope_left, fit_left = left_line
            slope_right, fit_right = right_line
        except (TypeError, IndexError):
            return 0

        y_vehicle = image_width - 1
        x_left = fit_left[0] * y_vehicle + fit_left[1]
        x_right = fit_right[0] * y_vehicle + fit_right[1]
        lane_center = (x_left + x_right) / 2
        self.first_pos = image_width // 2 + vehicle_params[self.robot_id]["first_pos_delta"]

        return self.first_pos - lane_center

    def update_control(self, deviation, left_line, right_line):
        """更新控制量"""
        if left_line is not None and right_line is not None:
            # 平滑偏差
            smoothed_deviation = self.smooth_deviation(deviation)
            self.filtered_deviation = params["alpha"] * smoothed_deviation + \
                                      (1 - params["alpha"]) * self.filtered_deviation

            # PID控制
            raw_angular = self.enhanced_pid_control(self.filtered_deviation)
            self.angular = self.smooth_angular(raw_angular)

            # 计算车道曲率
            self.lane_curvature = self.calculate_lane_curvature(left_line, right_line)
        else:
            self.adaptive_control(left_line, right_line)

    def smooth_deviation(self, deviation):
        """平滑偏差值"""
        self.deviation_history.append(deviation)
        if len(self.deviation_history) >= 2:
            smoothed = np.mean(list(self.deviation_history))
            if abs(smoothed - self.filtered_deviation) > params["max_deviation_change"]:
                if smoothed > self.filtered_deviation:
                    smoothed = self.filtered_deviation + params["max_deviation_change"]
                else:
                    smoothed = self.filtered_deviation - params["max_deviation_change"]
            return smoothed
        return deviation

    def enhanced_pid_control(self, deviation):
        """增强的PID控制器"""
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.1

        # 死区处理
        if abs(deviation) < params["deadzone"]:
            deviation = 0

        # PID计算
        proportional = params["Kp"] * deviation
        self.integral_error += deviation * dt
        self.integral_error = max(min(self.integral_error, 100), -100)
        integral = params["Ki"] * self.integral_error
        derivative = params["Kd"] * (deviation - self.last_deviation) / dt
        pid_output = proportional + integral + derivative

        # 更新历史记录
        self.last_deviation = deviation
        self.last_time = current_time

        return max(min(pid_output, params["max_angular"]), params["min_angular"])

    def smooth_angular(self, angular):
        """平滑角速度输出"""
        self.angular_history.append(angular)
        if len(self.angular_history) >= 2:
            weights = np.array([0.2, 0.3, 0.5][:len(self.angular_history)])
            weights = weights / weights.sum()
            return np.average(list(self.angular_history), weights=weights)
        return angular

    def adaptive_control(self, left_line, right_line):
        """自适应控制策略"""
        if left_line is None and right_line is None:
            self.angular *= 0.8
            return

        if left_line is None or right_line is None:
            if left_line is not None:
                slope_left, _ = left_line
                self.angular = -0.1 * np.sign(slope_left)
            else:
                slope_right, _ = right_line
                self.angular = 0.1 * np.sign(slope_right)

    def calculate_lane_curvature(self, left_line, right_line):
        """计算车道曲率"""
        if left_line is None or right_line is None:
            return 0
        slope_left, _ = left_line
        slope_right, _ = right_line
        return abs(slope_left - slope_right)

    def save_csv_data(self, frame_num, left_slope, right_slope):
        """保存数据到CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_num,
                left_slope,
                right_slope,
                self.filtered_deviation,
                self.angular
            ])

    def visualize_processing(self, lower_half, line_image, left_line, right_line,
                             frame_num, left_slope, right_slope, y_max, width):
        """可视化处理结果"""
        # 绘制车道中心点
        if left_line is not None and right_line is not None:
            _, fit_left = left_line
            _, fit_right = right_line
            x_left = int(fit_left[0] * y_max + fit_left[1])
            x_right = int(fit_right[0] * y_max + fit_right[1])
            lane_center = (x_left + x_right) // 2
            cv2.circle(line_image, (lane_center, y_max), 5, (0, 255, 255), -1)
            cv2.circle(line_image, (width // 2, y_max), 5, (0, 255, 0), -1)

        # 添加文字信息
        texts = [
            f"Frame: {frame_num} (Limo: {self.robot_id})",
            f"Left Slope: {left_slope:.2f}",
            f"Right Slope: {right_slope:.2f}",
            f"Deviation: {self.filtered_deviation:.2f}",
            f"Angular: {self.angular:.3f}",
            f"Curvature: {self.lane_curvature:.3f}"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)
        thickness = 2

        for i, text in enumerate(texts):
            cv2.putText(line_image, text, (20, 30 + i * 25), font, font_scale, font_color, thickness)

        # 合并处理结果
        return cv2.addWeighted(lower_half, 0.8, line_image, 1, 0)

    def save_results(self, result, frame_num):
        """保存处理结果"""
        # 保存处理后的图像
        output_path = os.path.join(self.folders["picture"], f"picture_{frame_num:04d}.jpg")
        cv2.imwrite(output_path, result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 处理视频帧
        ret, frame = self.cap.read()
        if ret:
            # 保存视频帧
            frame_path = os.path.join(self.folders["video"], f"video_{frame_num:04d}.jpg")
            cv2.putText(frame, f"Frame: {frame_num} (Limo: {self.robot_id})",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(frame_path, frame)

            # 合并图像并保存
            target_width = max(result.shape[1], frame.shape[1])
            result = cv2.copyMakeBorder(result, 0, 0, 0, target_width - result.shape[1],
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            frame = cv2.copyMakeBorder(frame, 0, 0, 0, target_width - frame.shape[1],
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            combined_img = np.vstack((result, frame))
            frame_name = os.path.join(self.folders["picture_video"], f"picture_video_{frame_num:04d}.jpg")
            cv2.imwrite(frame_name, combined_img)


class EnhancedMvdmSystem:
    def __init__(self, speed_action, lane_action):
        now = datetime.now()
        time_str = now.strftime("%m%d%H%M%S")
        self.main_folder = os.path.join(os.getcwd(), time_str)

        self.robot_ids = params["robot_ids"]
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        for j in range(20):
            self.cap.read()

        # 初始化速度控制参数
        self.speed_expand = {id: [0] * 10 + np.repeat(speed_action[id], params["expand_num"]).tolist()
                             for id in self.robot_ids}
        self.lane_expand = {id: [0] * 10 + np.repeat(lane_action[id], params["expand_num"]).tolist()
                            for id in self.robot_ids}
        self.exc_speed = {id: 0 for id in self.robot_ids}
        self.angular = {id: 0 for id in self.robot_ids}
        self.count = 0

        # 创建文件夹和处理器线程
        self.processors = {}
        self._create_processors()

        # ROS发布器
        self.angular_pubs = {
            robot_id: rospy.Publisher(f'/limo{robot_id}/angular', Float32, queue_size=1)
            for robot_id in self.robot_ids
        }
        self.vel_pubs = {
            robot_id: rospy.Publisher(f'/limo{robot_id}/vel', Float32, queue_size=1)
            for robot_id in self.robot_ids
        }

        # ROS订阅器
        self.subs = {
            robot_id: rospy.Subscriber(
                f'/limo{robot_id}/camera/image_raw',
                Image,
                partial(self.image_callback, robot_id=robot_id),
                queue_size=1
            )
            for robot_id in self.robot_ids
        }

        # 控制定时器
        self.control_timer = rospy.Timer(rospy.Duration(params["control_period"]), self.cbControl)

    def _create_processors(self):
        """创建处理器线程"""
        self.bridge = CvBridge()

        for robot_id in self.robot_ids:
            # 创建文件夹
            robot_folder = os.path.join(self.main_folder, f"limo_{robot_id}")
            folders = {
                "main": robot_folder,
                "picture": os.path.join(robot_folder, "picture"),
                "picture_raw": os.path.join(robot_folder, "raw_picture"),
                "video": os.path.join(robot_folder, "video"),
                "picture_video": os.path.join(robot_folder, "picture_video")
            }

            for folder in folders.values():
                os.makedirs(folder, exist_ok=True)

            # 创建CSV文件
            csv_path = os.path.join(robot_folder, f"slope_data_{robot_id}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame_num', 'left_slope', 'right_slope', 'deviation', 'angular'])

            # 创建处理器线程
            processor = ImageProcessorThread(robot_id, folders, csv_path, self.bridge, self.cap)
            processor.start()
            self.processors[robot_id] = processor

    def image_callback(self, image_msg, robot_id):
        """图像回调函数，将图像分发到对应处理线程"""
        if robot_id in self.processors:
            self.processors[robot_id].add_image(image_msg)

    def publish_twist(self, limo_id, linear_x, angular):
        """发布控制命令"""
        self.vel_pubs[limo_id].publish(linear_x)
        self.angular_pubs[limo_id].publish(angular)
        rospy.loginfo(
            f"publish to {limo_id} - Speed: {linear_x:.3f}, Angular: {angular:.3f} "
            f"(Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]})")

    def cbControl(self, event):
        """控制回调函数"""
        index_lane = 0
        for limo_id in self.robot_ids:
            if self.lane_expand[limo_id][self.count] == 0:
                self.exc_speed[limo_id] = self.speed_expand[limo_id][self.count]
                angular = self.processors[limo_id].angular if limo_id in self.processors else 0
                self.publish_twist(limo_id, self.exc_speed[limo_id], angular)
            else:
                # 车道变换逻辑
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
                angular = current_angular * (1 / actual_speed) / 3.5
                self.publish_twist(limo_id, self.exc_speed[limo_id], angular)

        self.count += 1

    def __del__(self):
        """清理资源"""
        for processor in self.processors.values():
            processor.stop()
            processor.join()
        self.cap.release()


def run():
    rospy.init_node('enhanced_mvdm_system', anonymous=False)
    speed_action = {
        '9293': [0.5, 0.5, 0.5, 0.5, 0.5],
        "9298": [0.5, 0.5, 0.5, 0.5, 0.5],
        "9289": [0.5, 0.5, 0.5, 0.5, 0.5]
    }
    lane_action = {
        "9293": [0, 0, 0, 0, 0],
        "9298": [0, 0, 0, 0, 0],
        "9289": [0, 0, 0, 0, 0]
    }
    mvdm_system = EnhancedMvdmSystem(speed_action, lane_action)
    rospy.spin()


if __name__ == '__main__':
    run()