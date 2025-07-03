#!/usr/bin/env python3
import rospy
from pylimo import limo
from sensor_msgs.msg import Image
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import os
import time
from datetime import datetime
import math
from threading import Thread, Lock
from std_msgs.msg import Float32
from collections import deque
limo=limo.LIMO()
limo.EnableCommand()
params = {
          "Kp": 0.0005,
          "Kd": 0.001,
          "alpha": 0.8,
          "max_angular": 0.6,
          "min_angular": -0.6,
          "control_period": 0.1,
          "expand_num": 40,
          "control_angular": 0.25,
          "control_angular_max": 0.15,
          }
class PurePythonImageConverter:
    """纯Python实现的图像转换器，替代cv_bridge"""

    @staticmethod
    def imgmsg_to_cv2(img_msg, desired_encoding="bgr8"):
        """
        将ROS Image消息转换为OpenCV图像
        """
        try:
            # 根据编码类型设置参数
            if img_msg.encoding == "bgr8":
                dtype = np.uint8
                n_channels = 3
                need_convert = False
            elif img_msg.encoding == "rgb8":
                dtype = np.uint8
                n_channels = 3
                need_convert = True  # 需要RGB转BGR
            elif img_msg.encoding == "mono8":
                dtype = np.uint8
                n_channels = 1
                need_convert = False
            elif img_msg.encoding == "8UC3":  # 某些相机的编码
                dtype = np.uint8
                n_channels = 3
                need_convert = False
            else:
                rospy.logerr(f"Unsupported encoding: {img_msg.encoding}")
                return None

            # 将消息数据转换为numpy数组
            img_data = np.frombuffer(img_msg.data, dtype=dtype)

            # 重塑数组为图像格式
            if n_channels == 1:
                cv_image = img_data.reshape((img_msg.height, img_msg.width))
            else:
                cv_image = img_data.reshape((img_msg.height, img_msg.width, n_channels))

            # 如果是RGB编码且需要BGR，进行转换
            if need_convert and desired_encoding == "bgr8":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            return cv_image

        except Exception as e:
            rospy.logerr(f"Image conversion error: {e}")
            return None


class Mvdm_System:
    def __init__(self):
        # 补充信息 - 使用纯Python图像转换器替代cv_bridge
        self.image_converter = PurePythonImageConverter()

        # 增强的控制变量
        self.deviation_history = deque(maxlen=params["smooth_window"])
        self.angular_history = deque(maxlen=3)
        self.integral_error = 0
        self.last_deviation = 0
        self.last_time = time.time()
        self.angular = 0
        self.filtered_deviation = 0
        self.first_pos = 0
        # 预测控制相关
        self.lane_curvature = 0
        self.speed_factor = 1.0
        self.exc_speed = 0


        # 控制参数
        self.Kp = params["Kp"]
        self.Ki = params["Ki"]
        self.Kd = params["Kd"]
        self.alpha = params["alpha"]
        self.min_angular = params["min_angular"]
        self.max_angular = params["max_angular"]

        # 接收订阅
        self.image = Image()

        # 处理订阅
        self.lock = Lock()
        self.latest_image = None

        self.image_sub = rospy.Subscriber('/limo9289/camera/image_raw', Image, self.image_callback)
        self.vel_sub = rospy.Subscriber('/limo9289/vel', Float32, self.vel_callback)
        self.lane_sub = rospy.Subscriber('/limo9289/lane', Float32, self.lane_callback)
        self.control_timer = rospy.Timer(rospy.Duration(params["control_period"]), self.cbControl)

    def image_callback(self, msg):
        # 开线程处理图像
        Thread(target=self.preprocess_image, args=(msg,), daemon=True).start()

    def vel_callback(self, vel):
        self.exc_speed = vel.data

    def lane_callback(self, lane):
        self.exc_lane = lane.data

    def smooth_deviation(self, deviation):
        """平滑偏差值，减少噪声影响"""
        # 添加到历史记录
        self.deviation_history.append(deviation)

        # 计算滑动平均
        if len(self.deviation_history) >= 2:
            smoothed = np.mean(list(self.deviation_history))

            # 限制偏差变化率，防止突变
            if abs(smoothed - self.filtered_deviation) > params["max_deviation_change"]:
                if smoothed > self.filtered_deviation:
                    smoothed = self.filtered_deviation + params["max_deviation_change"]
                else:
                    smoothed = self.filtered_deviation - params["max_deviation_change"]

            return smoothed
        else:
            return deviation

    def enhanced_pid_control(self, deviation ):
        """增强的PID控制器"""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0:
            dt = 0.1

        # 死区处理，减少小幅震荡
        if abs(deviation) < params["deadzone"]:
            deviation = 0

        # PID计算
        # 比例项
        proportional = self.Kp * deviation

        # 积分项（带积分饱和限制）
        self.integral_error += deviation * dt
        # 积分饱和限制
        max_integral = 100
        self.integral_error = max(min(self.integral_error, max_integral), -max_integral)
        integral = self.Ki * self.integral_error

        # 微分项
        derivative = self.Kd * (deviation - self.last_deviation) / dt

        # PID输出
        pid_output = proportional + integral + derivative

        # 速度自适应调整
        current_speed = self.exc_speed
        if current_speed > 0:
            # 速度越快，控制增益越大
            speed_factor = min(current_speed / 0.5, 2.0)  # 最大2倍增益
            pid_output *= speed_factor

        # 角度预测补偿
        if params["angle_compensation"] and len(self.angular_history) >= 2:
            # 基于历史角速度预测未来趋势
            angular_trend = np.mean(np.diff(list(self.angular_history)))
            prediction_compensation = -0.1 * angular_trend  # 预测补偿系数
            pid_output += prediction_compensation

        # 输出限制
        pid_output = max(min(pid_output, self.max_angular), self.min_angular)

        # 更新历史记录
        self.last_deviation = deviation
        self.last_time = current_time

        return pid_output

    def smooth_angular(self, angular ):
        """平滑角速度输出"""
        self.angular_history.append(angular)

        if len(self.angular_history) >= 2:
            # 使用加权平均，最新值权重更大
            weights = np.array([0.2, 0.3, 0.5])[:len(self.angular_history)]
            weights = weights / weights.sum()
            smoothed_angular = np.average(list(self.angular_history), weights=weights)
            return smoothed_angular
        else:
            return angular

    def adaptive_control(self, left_line, right_line):
        """自适应控制策略"""
        if left_line is None and right_line is None:
            # 两条线都丢失，保持最后的控制量但逐渐减小
            self.angular *= 0.8
            return

        if left_line is None or right_line is None:
            # 只有一条线，使用单边控制
            if left_line is not None:
                # 只有左边线，倾向于向右调整
                slope_left, _ = left_line
                self.angular = -0.1 * np.sign(slope_left)
            else:
                # 只有右边线，倾向于向左调整
                slope_right, _ = right_line
                self.angular = 0.1 * np.sign(slope_right)
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

    def preprocess_image(self, image):
        """图像处理主函数（保持原有逻辑，但改进控制部分）"""

        image = self.image_converter.imgmsg_to_cv2(image, "bgr8")

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
            deviation = self.calculate_deviation(left_line, right_line, width,)

            # 平滑偏差
            smoothed_deviation = self.smooth_deviation(deviation)
            self.filtered_deviation = self.alpha * smoothed_deviation + (1 - self.alpha) * \
                                               self.filtered_deviation

            # 使用增强PID控制
            raw_angular = self.enhanced_pid_control(self.filtered_deviation)

            # 平滑角速度输出
            self.angular = self.smooth_angular(raw_angular)

            # 计算车道曲率用于自适应
            self.lane_curvature = self.calculate_lane_curvature(left_line, right_line)

        else:
            # 车道线丢失时的处理
            self.adaptive_control(left_line, right_line)

        rospy.loginfo(
            f"calculate_deviation - Deviation: {self.filtered_deviation:.2f}, Angular: {self.angular:.3f}")


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

    def calculate_deviation(self, left_line, right_line, image_width):
        """计算偏差（保持原有逻辑）"""
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
        self.first_pos = image_width // 2 - 10


        deviation = self.first_pos - lane_center
        return deviation

    def publish_twist(self,linear_x, angular_z):
        """发布控制命令（改进版）"""
        # 更温和的角速度限制
        if abs(angular_z) > params["control_angular_max"]:
            if angular_z > 0:
                angular = params["control_angular"]
            else:
                angular = -params["control_angular"]
        else:
            # 按比例缩放，而不是直接置零
            angular = angular_z * (params["control_angular"] / params["control_angular_max"])
        limo.SetMotionCommand(linear_x, -0.01, 0, angular / 2)

        rospy.loginfo(f"SetMotionCommand - Speed: {linear_x:.3f}, Angular: {angular:.3f}")

    def cbControl(self, event):
        """控制回调函数（保持原有逻辑但使用改进的发布函数）"""
        index_lane = 0
        if self.exc_lane == 0:
            self.publish_twist(self.exc_speed,self.angular)
        else:
            # 车道变换逻辑（保持原有）
            progress = min(index_lane / 40, 1.0)
            phase = progress * math.pi * 2
            angular = self.exc_lane * math.sin(phase)

            try:
                max_angle = math.pi / 3
                current_angular = max(min(angular, max_angle), -max_angle)
                speed_compensation = 1.0 / math.cos(abs(current_angular))
                max_speed = 2.0 * self.exc_speed
                actual_speed = min(self.exc_speed * speed_compensation, max_speed)
            except:
                actual_speed = self.exc_speed
                current_angular = 0

            index_lane += 1
            self.exc_speed= actual_speed
            self.angular = current_angular * (1 / actual_speed) / 3.5
            self.publish_twist(self.exc_speed, self.angular)

def run():
    rospy.init_node('car_9289', anonymous=False)
    mvdm_system = Mvdm_System()
    rospy.spin()


if __name__ == '__main__':
    run()