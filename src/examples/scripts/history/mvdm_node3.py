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

class Mvdm_System:
    def __init__(self, speed_action, lane_action):

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
        self.image = Image()

        self.lock = Lock()
        self.latest_image = None
        self.processed_result = None  # 控制用的处理结果

        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

        # 发布指令
        self.cmd_vel_pub = rospy.Publisher('/limo2/cmd_vel', Twist, queue_size=1)
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.cbControl)

    def image_callback(self, msg):
        # 开线程处理图像
        Thread(target=self.preprocess_image, args=(msg,), daemon=True).start()


    def preprocess_image(self, image):
        image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        # 读取图片并裁剪
        height, width = image.shape[:2]
        lower_half = image[round(height // 1.75): height, :]

        lab = cv2.cvtColor(lower_half, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        # 灰度 + 高斯模糊 + Canny 边缘检测
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)

        left_points = []
        right_points = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 排除接近水平的线段
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:
                    continue
                # 分类：左侧 or 右侧车道
                if slope < 0:
                    left_points.append((x1, y1))
                    left_points.append((x2, y2))
                else:
                    right_points.append((x1, y1))
                    right_points.append((x2, y2))
                # 可视化所有线段
                # cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 设定绘制 y 范围
        y_max = height
        y_min = int(height * 0.1)  # 可以根据需要调整

        # 绘制拟合车道线
        line_image = np.zeros_like(image)
        left_line = self.fit_line(left_points, y_min, y_max, (255, 0, 0), line_image)
        right_line = self.fit_line(right_points, y_min, y_max, (0, 0, 255), line_image)

        # 计算偏离距离
        deviation = 0
        if left_line is not None and right_line is not None:
            deviation = self.calculate_deviation(left_line, right_line, width)
            self.filtered_deviation = self.alpha * deviation + (1 - self.alpha) * self.filtered_deviation
            print(f"偏离距离: {deviation:.2f} 像素（{'偏右' if deviation > 0 else '偏左'}）")

        self.angular = self.Kp * deviation + self.Kd * self.last_deviation
        self.last_deviation = deviation


    def fit_line(self, points, y_min, y_max, color, line_image):
        if len(points) < 2:
            return None
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        fit = np.polyfit(y, x, 1)  # x = a*y + b
        slope = 1 / fit[0]  # 转换为 y = m*x + c 的斜率
        x_min = int(fit[0] * y_min + fit[1])
        x_max = int(fit[0] * y_max + fit[1])
        cv2.line(line_image, (x_min, y_min), (x_max, y_max), color, 4)
        return slope, fit

    def calculate_deviation(self, left_line, right_line, image_width):
        if left_line is None or right_line is None:
            return 0  # 无法计算时返回0

        try:
            slope_left, fit_left = left_line  # fit_left = [a, b]（x = a*y + b）
            slope_right, fit_right = right_line
        except (TypeError, IndexError):
            return 0

        # 假设车辆位置在图像底部中心
        y_vehicle = image_width - 1  # 图像最底部
        self.vehicle_position = (image_width // 2, y_vehicle)

        # 计算左右车道线在车辆位置的x坐标（用 x = a*y + b）
        x_left = fit_left[0] * y_vehicle + fit_left[1]
        x_right = fit_right[0] * y_vehicle + fit_right[1]

        # 计算车道中心
        lane_center = (x_left + x_right) / 2

        # 计算偏离距离（正：偏右，负：偏左）
        # deviation = self.vehicle_position[0] - lane_center
        print(self.first)    
        if self.first:
            self.first_pos = lane_center
            self.first = False


        deviation = self.first_pos - lane_center


        return deviation


    def cbControl(self, event):
        start_time_img = time.time()
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
        end_time_img = time.time()
        print(f"control耗时: {end_time_img-start_time_img:.2f} ")
        return











def run():
    rospy.init_node('mvdm_ststem', anonymous=False)
    speed_action = [0.25, 0.25, 0.25, 0.25, 0.25]
    lane_action = [0, 0, 0, 0, 0]
    mvdm_system = Mvdm_System(speed_action, lane_action)
    rospy.spin()

if __name__ == '__main__':

    run()
