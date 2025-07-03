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
        # 存储信息
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
            writer.writerow(['frame_num', 'left_slope', 'right_slope'])

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        for j in range(20):
            self.cap.read()

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


        # 接收订阅

        self.image = Image()


        # 处理订阅
        self.lock = Lock()
        self.latest_image = None
        self.processed_result = None  # 控制用的处理结果

        self.image_sub = rospy.Subscriber('/limo9289/camera/image_raw', Image, self.image_callback)

        # 发布指令
        self.cmd_vel_pub = rospy.Publisher('/limo9289/cmd_vel', Twist, queue_size=1)
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.cbControl)

    def image_callback(self, msg):
        # 开线程处理图像
        Thread(target=self.preprocess_image, args=(msg,), daemon=True).start()

    def _create_folders(self):
        """创建主文件夹和子文件夹"""
        os.makedirs(self.csv_folder, exist_ok=True)
        os.makedirs(self.picture_folder, exist_ok=True)
        os.makedirs(self.picture_raw_folder, exist_ok=True)
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.picture_video_folder, exist_ok=True)

    def preprocess_image(self, image):
        start_time = time.time()
        image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        output_path = os.path.join(self.picture_raw_folder, f"picture_{self.i:04d}.jpg")  # Use JPG instead
        cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # 95% quality
        # 读取图片并裁剪
        height, width = image.shape[:2]
        lower_half = image[round(height // 1.75): height, :]
        cv2.imwrite('image.png', lower_half)

        # 再次读取处理
        image = cv2.imread('./image.png', cv2.IMREAD_UNCHANGED)
        height, width = image.shape[:2]

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
                # 可视化所有线段
                # cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 设定绘制 y 范围
        y_max = height
        y_min = int(height * 0.1)  # 可以根据需要调整

        # 绘制拟合车道线
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

        # 可视化车道中心和车辆位置
        if left_line is not None and right_line is not None:
            _, fit_left = left_line
            _, fit_right = right_line
            x_left = int(fit_left[0] * y_max + fit_left[1])
            x_right = int(fit_right[0] * y_max + fit_right[1])
            lane_center = (x_left + x_right) // 2

            cv2.circle(line_image, (lane_center, y_max), 5, (0, 255, 255), -1)  # 车道中心
            cv2.circle(line_image, (width // 2, y_max), 5, (0, 255, 0), -1)  # 车辆位置

        left_slope = left_line[0] if left_line is not None else 0
        right_slope = right_line[0] if right_line is not None else 0

        # 在图像上添加文字
        text_frame = f"Frame: {self.i:.2f}"
        text_left = f"Left Slope: {left_slope:.2f}"
        text_right = f"Right Slope: {right_slope:.2f}"
        text_time = f"Time: {datetime.now().time()}"
        text_angular = f"Angular: {self.angular}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # 白色文字
        thickness = 2

        # 在左上角添加文字
        cv2.putText(line_image, text_frame, (20, 30), font, font_scale, font_color, thickness)
        cv2.putText(line_image, text_left, (20, 60), font, font_scale, font_color, thickness)
        cv2.putText(line_image, text_right, (20, 90), font, font_scale, font_color, thickness)
        cv2.putText(line_image, text_time, (20, 120), font, font_scale, font_color, thickness)
        cv2.putText(line_image, text_angular, (20, 150), font, font_scale, font_color, thickness)

        # Save slopes to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.i, left_slope, right_slope])

        # 合成最终图像
        result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
        output_path = os.path.join(self.picture_folder, f"picture_{self.i:04d}.jpg")  # Use JPG instead
        cv2.imwrite(output_path, result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # 95% quality

        ret, frame = self.cap.read()
        frame_path = os.path.join(self.video_folder, f"video_{self.i:04d}.jpg")
        cv2.putText(frame, text_frame, (20, 30), font, font_scale, font_color, thickness)
        cv2.putText(frame, text_left, (20, 60), font, font_scale, font_color, thickness)
        cv2.putText(frame, text_right, (20, 90), font, font_scale, font_color, thickness)
        cv2.putText(frame, text_time, (20, 120), font, font_scale, font_color, thickness)
        cv2.putText(frame, text_angular, (20, 150), font, font_scale, font_color, thickness)
        cv2.imwrite(frame_path, frame)

        # 统一宽度（取两者中较大的宽度）
        target_width = max(result.shape[1], frame.shape[1])

        # 调整图片宽度（保持比例）
        if result.shape[1] < target_width:
            result = cv2.copyMakeBorder(result, 0, 0, 0, target_width - result.shape[1],
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))

        if frame.shape[1] < target_width:
            frame = cv2.copyMakeBorder(frame, 0, 0, 0, target_width - frame.shape[1],
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 上下拼接
        combined_img = np.vstack((result, frame))

        frame_name = os.path.join(self.picture_video_folder, f"picture_video_{self.i:04d}.jpg")
        cv2.imwrite(frame_name, combined_img)

        self.i += 1
        end_time = time.time()
        print(f"img耗时: {end_time-start_time:.2f} ")

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
