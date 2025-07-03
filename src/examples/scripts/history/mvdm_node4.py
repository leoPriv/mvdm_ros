#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
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
from tracker import Tracker


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
        self.angular = 0
        self.Kp = 0.0005
        self.Kd = 0.001
        self.filtered_deviation = 0
        self.alpha = 0.8
        self.last_deviation = 0
        self.first = True
        self.Tracker = Tracker()


        # 接收订阅
        self.image = Image()

        # 处理订阅
        self.lock = Lock()
        self.latest_image = None

        self.image_sub = rospy.Subscriber('/limo9298/camera/image_raw', Image, self.image_callback)
        self.angular_pub = rospy.Publisher('/limo9298/angular_info', Float32, queue_size=1)

    def image_callback(self, msg):
        # 开线程处理图像
        Thread(target=self.preprocess_image, args=(msg,), daemon=True).start()

    def preprocess_image(self, image_msg):
        """图像预处理函数 - 使用纯Python实现"""

        # 使用纯Python转换器替代cv_bridge
        image = self.image_converter.imgmsg_to_cv2(image_msg, "bgr8")

        if image is None:
            rospy.logerr("Failed to convert image message")
            return

        # 读取图片并裁剪
        height, width = image.shape[:2]
        lower_half = image[round(height // 1.75): height, :]

        height, width = lower_half.shape[:2]

        # 图像增强
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

        line_image = np.zeros_like(image)

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

        # 设定绘制 y 范围
        y_max = height
        y_min = int(height * 0.1)

        # 绘制拟合车道线
        left_line = self.fit_line(left_points, y_min, y_max, (255, 0, 0), line_image)
        right_line = self.fit_line(right_points, y_min, y_max, (0, 0, 255), line_image)


        # 计算偏离距离
        deviation = 0
        if left_line is not None and right_line is not None:
            deviation = self.calculate_deviation(left_line, right_line, width)
            self.filtered_deviation = self.alpha * deviation + (1 - self.alpha) * self.filtered_deviation

        self.angular = self.Kp * self.filtered_deviation + self.Kd * self.last_deviation
        self.last_deviation = self.filtered_deviation
        self.angular_pub.publish(self.angular)

    def fit_line(self, points, y_min, y_max, color, line_image):
        if len(points) < 2:
            return None
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]

        try:
            poly = self.Tracker.add(np.poly1d(np.polyfit(y, x, deg=1)))

            # x = a*y + b
            x_min = int(poly(y_min))
            x_max = int(poly(y_min))
            cv2.line(line_image, (x_min, y_min), (x_max, y_max), color, 4)
            return poly

        except np.RankWarning:
            rospy.logwarn("Polyfit rank warning")
            return None
        except Exception as e:
            rospy.logerr(f"Line fitting error: {e}")
            return None

    def calculate_deviation(self, left_line, right_line, image_width):
        if left_line is None or right_line is None:
            return 0  # 无法计算时返回0

        # 假设车辆位置在图像底部中心
        y_vehicle = image_width - 1  # 图像最底部
        self.vehicle_position = (image_width // 2, y_vehicle)

        # 计算左右车道线在车辆位置的x坐标（用 x = a*y + b）
        x_left = left_line(y_vehicle)
        x_right = right_line(y_vehicle)

        # 计算车道中心
        lane_center = (x_left + x_right) / 2

        # 计算偏离距离（正：偏右，负：偏左）
        if self.first:
            self.first_pos = lane_center
            self.first = False

        deviation = self.first_pos - lane_center

        return deviation


def run():
    rospy.init_node('car_9298', anonymous=False)
    mvdm_system = Mvdm_System()
    rospy.spin()


if __name__ == '__main__':
    run()