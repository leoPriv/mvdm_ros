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
import queue  # 引入线程安全的队列

params = {
    # "robot_ids":["9293", "9298", "9289"],
    "robot_ids": ["9298", "9289"], # 修改为您自己的机器人ID
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
vehicle_params = {
    "9289": { # 注意ID要与上面一致
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
        now = datetime.now()
        time_str = now.strftime("%m%d%H%M%S")
        self.main_folder = os.path.join(os.getcwd(), time_str)
        self.csv_folder = os.path.join(self.main_folder, "csv")
        self.picture_folder = os.path.join(self.main_folder, "picture")
        self.picture_raw_folder = os.path.join(self.main_folder, "raw_picture")
        self.video_folder = os.path.join(self.main_folder, "video")
        self.picture_video_folder = os.path.join(self.main_folder, "picture_video")
        self._create_folders()

        self.csv_path = os.path.join(self.csv_folder, "slope_data.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_num', 'robot_id', 'left_slope', 'right_slope', 'deviation', 'angular'])

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        for j in range(20):
            self.cap.read()

        self.robot_ids = params["robot_ids"]
        self.speed_expand = {id: [0] * 5 + np.repeat(speed_action[id], params["expand_num"]).tolist() for id in self.robot_ids}
        self.lane_expand = {id: [0] * 5 + np.repeat(lane_action[id], params["expand_num"]).tolist() for id in self.robot_ids}

        # --- 多线程改造：引入锁和共享状态变量 ---
        self.lock = threading.Lock() # 创建一个锁来保护共享数据
        self.exc_speed = {id: 0 for id in self.robot_ids}
        self.angular = {id: 0 for id in self.robot_ids}
        self.filtered_deviation = {id: 0 for id in self.robot_ids}
        self.lane_curvature = {id: 0 for id in self.robot_ids}
        self.i = 0 # 全局帧计数器

        # 增强的控制变量 (每个线程独立维护自己的状态，除了需要被cbControl读取的)
        self.deviation_history = {id: deque(maxlen=params["smooth_window"]) for id in self.robot_ids}
        self.angular_history = {id: deque(maxlen=3) for id in self.robot_ids}
        self.integral_error = {id: 0 for id in self.robot_ids}
        self.last_deviation = {id: 0 for id in self.robot_ids}
        self.last_time = {id: time.time() for id in self.robot_ids}
        self.first_pos = {id: 0 for id in self.robot_ids}

        self.bridge = CvBridge()
        self.count = 0

        # 控制参数
        self.Kp = params["Kp"]
        self.Ki = params["Ki"]
        self.Kd = params["Kd"]
        self.alpha = params["alpha"]
        self.min_angular = params["min_angular"]
        self.max_angular = params["max_angular"]

        # --- 多线程改造：为每个机器人创建队列和处理线程 ---
        self.image_queues = {robot_id: queue.Queue(maxsize=1) for robot_id in self.robot_ids}
        self.processing_threads = {}

        # ROS订阅和发布
        self.subs = {
            robot_id: rospy.Subscriber(
                f'/limo{robot_id}/camera/image_raw',
                Image,
                partial(self.image_callback, robot_id=robot_id) # 回调函数变得轻量
            )
            for robot_id in self.robot_ids
        }

        for robot_id in self.robot_ids:
            # 为每个机器人创建一个并启动一个处理线程
            thread = threading.Thread(target=self._processing_loop, args=(robot_id,))
            thread.daemon = True  # 设置为守护线程，主程序退出时线程也退出
            self.processing_threads[robot_id] = thread
            thread.start()
            rospy.loginfo(f"Processing thread started for robot {robot_id}")


        self.angular_pubs = {
            robot_id: rospy.Publisher(f'/limo{robot_id}/angular', Float32, queue_size=1)
            for robot_id in self.robot_ids
        }

        self.vel_pubs = {
            robot_id: rospy.Publisher(f'/limo{robot_id}/vel', Float32, queue_size=1)
            for robot_id in self.robot_ids
        }

        self.control_timer = rospy.Timer(rospy.Duration(params["control_period"]), self.cbControl)

    def image_callback(self, msg, robot_id):
        """
        轻量级的ROS回调函数。
        仅将ROS图像消息转换为OpenCV帧，并放入对应机器人的队列。
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 使用put_nowait，如果队列已满，则立即引发Full异常，不会阻塞回调
            self.image_queues[robot_id].put_nowait(cv_image)
        except queue.Full:
            # 如果处理线程跟不上，就丢弃这个旧帧，这对于实时控制是合理的
            pass
        except Exception as e:
            rospy.logerr(f"Error in image_callback for {robot_id}: {e}")

    def _processing_loop(self, robot_id):
        """
        每个机器人专属的工作线程循环。
        从队列中获取图像并执行所有耗时的处理。
        """
        rospy.loginfo(f"Worker thread for {robot_id} is running.")
        while not rospy.is_shutdown():
            try:
                # get()方法会阻塞线程，直到队列中有新项目
                image = self.image_queues[robot_id].get(timeout=1.0)
                self._process_single_image(image, robot_id)
            except queue.Empty:
                # 如果1秒内没有新图像，则继续等待
                continue
            except Exception as e:
                rospy.logerr(f"Error in processing loop for {robot_id}: {e}")

    def _create_folders(self):
        os.makedirs(self.csv_folder, exist_ok=True)
        os.makedirs(self.picture_folder, exist_ok=True)
        os.makedirs(self.picture_raw_folder, exist_ok=True)
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.picture_video_folder, exist_ok=True)

    def smooth_deviation(self, deviation, limo_id):
        self.deviation_history[limo_id].append(deviation)
        if len(self.deviation_history[limo_id]) >= 2:
            smoothed = np.mean(list(self.deviation_history[limo_id]))
            with self.lock: # 读取共享数据需要加锁
                filtered_dev = self.filtered_deviation[limo_id]
            if abs(smoothed - filtered_dev) > params["max_deviation_change"]:
                smoothed = filtered_dev + params["max_deviation_change"] if smoothed > filtered_dev else filtered_dev - params["max_deviation_change"]
            return smoothed
        return deviation

    def enhanced_pid_control(self, deviation, limo_id):
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

        with self.lock: # 读取共享的exc_speed需要加锁
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
        self.angular_history[limo_id].append(angular)
        if len(self.angular_history[limo_id]) >= 2:
            weights = np.array([0.2, 0.3, 0.5])[:len(self.angular_history[limo_id])]
            weights = weights / weights.sum()
            return np.average(list(self.angular_history[limo_id]), weights=weights)
        return angular

    def adaptive_control(self, left_line, right_line, limo_id):
        current_angular = 0.0
        if left_line is None and right_line is None:
            with self.lock: # 读写共享数据需要加锁
                self.angular[limo_id] *= 0.8
            return
        if left_line is None or right_line is None:
            if left_line is not None:
                slope_left, _ = left_line
                current_angular = -0.1 * np.sign(slope_left)
            else:
                slope_right, _ = right_line
                current_angular = 0.1 * np.sign(slope_right)

        with self.lock: # 写共享数据需要加锁
            self.angular[limo_id] = current_angular

    def calculate_lane_curvature(self, left_line, right_line):
        if left_line is None or right_line is None: return 0
        slope_left, _ = left_line
        slope_right, _ = right_line
        return abs(slope_left - slope_right)

    def _process_single_image(self, image, limo_id):
        """这是实际的图像处理和控制计算函数"""

        with self.lock: # 对全局帧计数器i的访问需要加锁
            current_i = self.i
            self.i += 1

        output_path_raw = os.path.join(self.picture_raw_folder, f"picture_{limo_id}_{current_i:04d}.jpg")
        cv2.imwrite(output_path_raw, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        height, width = image.shape[:2]
        lower_half = image[round(height // 1.75): height, :]

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
        line_image = np.zeros_like(lower_half)

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

        y_max, y_min = line_image.shape[0], int(line_image.shape[0] * 0.1)
        left_line = self.fit_line(left_points, y_min, y_max, (255, 0, 0), line_image)
        right_line = self.fit_line(right_points, y_min, y_max, (0, 0, 255), line_image)

        current_angular = 0
        current_deviation = 0
        current_curvature = 0

        if left_line is not None and right_line is not None:
            deviation = self.calculate_deviation(left_line, right_line, width, limo_id)
            smoothed_deviation = self.smooth_deviation(deviation, limo_id)

            with self.lock: # 读写共享数据需要加锁
                current_filtered_dev = self.filtered_deviation[limo_id]
                self.filtered_deviation[limo_id] = self.alpha * smoothed_deviation + (1 - self.alpha) * current_filtered_dev
                current_deviation = self.filtered_deviation[limo_id]

            raw_angular = self.enhanced_pid_control(current_deviation, limo_id)
            current_angular = self.smooth_angular(raw_angular, limo_id)
            current_curvature = self.calculate_lane_curvature(left_line, right_line)

            with self.lock: # 写共享数据需要加锁
                self.angular[limo_id] = current_angular
                self.lane_curvature[limo_id] = current_curvature
        else:
            self.adaptive_control(left_line, right_line, limo_id)
            with self.lock: # 读取最新的值
                current_angular = self.angular[limo_id]
                current_deviation = self.filtered_deviation[limo_id]
                current_curvature = self.lane_curvature[limo_id]

        left_slope = left_line[0] if left_line is not None else 0
        right_slope = right_line[0] if right_line is not None else 0

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_i, limo_id, left_slope, right_slope, current_deviation, current_angular])

        # --- 可视化与保存 ---
        # ... (此部分代码与原版基本相同，仅为了清晰，将变量引用为当前函数内的局部变量)
        font = cv2.FONT_HERSHEY_SIMPLEX
        result = cv2.addWeighted(lower_half, 0.8, line_image, 1, 0)
        cv2.putText(result, f"Frame: {current_i} | Robot: {limo_id}", (20, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"Deviation: {current_deviation:.2f}", (20, 55), font, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"Angular: {current_angular:.3f}", (20, 80), font, 0.7, (255, 255, 255), 2)
        output_path_processed = os.path.join(self.picture_folder, f"picture_{limo_id}_{current_i:04d}.jpg")
        cv2.imwrite(output_path_processed, result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def fit_line(self, points, y_min, y_max, color, line_image):
        if len(points) < 2: return None
        points = np.array(points)
        x, y = points[:, 0], points[:, 1]
        try:
            fit = np.polyfit(y, x, 1)
        except np.linalg.LinAlgError:
            return None
        slope = 1 / fit[0] if fit[0] != 0 else float('inf')
        x_min = int(fit[0] * y_min + fit[1])
        x_max = int(fit[0] * y_max + fit[1])
        cv2.line(line_image, (x_min, y_min), (x_max, y_max), color, 4)
        return slope, fit

    def calculate_deviation(self, left_line, right_line, image_width, limo_id):
        if left_line is None or right_line is None: return 0
        try:
            _, fit_left = left_line
            _, fit_right = right_line
        except (TypeError, IndexError):
            return 0
        y_vehicle = image_width - 1
        x_left = fit_left[0] * y_vehicle + fit_left[1]
        x_right = fit_right[0] * y_vehicle + fit_right[1]
        lane_center = (x_left + x_right) / 2

        # 第一次计算时设定目标位置
        if self.first_pos[limo_id] == 0:
            self.first_pos[limo_id] = image_width // 2 + vehicle_params[limo_id]["first_pos_delta"]

        return self.first_pos[limo_id] - lane_center

    def publish_twist(self, limo_id, ang_pub, vel_pub, linear_x, angular_z):
        # 实际发布指令
        ang_pub.publish(angular_z)
        vel_pub.publish(linear_x)
        rospy.loginfo(f"Publishing to {limo_id} -> Speed: {linear_x:.3f}, Angular: {angular_z:.3f}")


    def cbControl(self, event):
        """
        定时器回调函数，以固定频率发布控制指令。
        它现在从共享变量中读取数据，因此需要加锁。
        """
        index_lane = 0
        # 对共享数据的访问全部放入一个锁块中，确保一致性
        with self.lock:
            for limo_id in self.robot_ids:
                local_angular = self.angular[limo_id]  # 读取共享的角速度

                if self.lane_expand[limo_id][self.count] == 0:
                    # 巡线模式
                    self.exc_speed[limo_id] = self.speed_expand[limo_id][self.count]
                    linear_x = self.exc_speed[limo_id]
                    angular_z = local_angular
                else:
                    # 变道模式
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
                    linear_x = actual_speed
                    # 变道时的角速度也需要补偿
                    angular_z = current_angular * (1 / actual_speed) / 3.5 + vehicle_params[limo_id]["angular_delta"]

                # 统一发布指令
                self.publish_twist(limo_id, self.angular_pubs[limo_id], self.vel_pubs[limo_id], linear_x, angular_z)

        self.count = (self.count + 1) % len(self.speed_expand[self.robot_ids[0]])


def run():
    rospy.init_node('enhanced_mvdm_system_threaded', anonymous=False)
    # 确保action字典的key与params中的robot_ids一致
    ids = params["robot_ids"]
    speed_action = {id: [0.5] * 5 for id in ids}
    lane_action = {id: [0] * 5 for id in ids}

    mvdm_system = EnhancedMvdmSystem(speed_action, lane_action)
    rospy.spin()


if __name__ == '__main__':
    run()