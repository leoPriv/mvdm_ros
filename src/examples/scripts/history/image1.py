#!/usr/bin/env python3
import rospy
from duplicity.globals import current_time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import os
import csv
from datetime import datetime
import math
from tqdm import tqdm
from natsort import natsorted

class LaneKeepingSystem:
    def __init__(self):

        self.tmp = 0

        self.bridge = CvBridge()
        self.last_steering = 0
        self.lane_width = 300  # 假设的车道宽度(像素)，需要根据实际情况调整
        self.vehicle_position = None  # 车辆位置（图像底部中心）
        self.Kp = 0.002
        self.cmd_vel_pub = rospy.Publisher('/limo2/cmd_vel', Twist, queue_size=1)
        self.i = 0
        now = datetime.now()
        time_str = now.strftime("%m%d%H%M%S")
        self.main_folder = os.path.join(os.getcwd(), time_str)
        self.csv_folder = os.path.join(self.main_folder, "csv")
        self.picture_folder = os.path.join(self.main_folder, "picture")
        self.picture_raw_folder = os.path.join(self.main_folder, "raw_picture")
        self.video_folder = os.path.join(self.main_folder, "video")
        self.picture_video_folder = os.path.join(self.main_folder, "picture_video")
        self.mp4_folder = os.path.join(self.main_folder, "mp4")
        self._create_folders()
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        for i in range(20):
            self.cap.read()
        self.rate = rospy.Rate(10)
        self.csv_path = os.path.join(self.csv_folder, "slope_data.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_num', 'left_slope', 'right_slope'])

        # 转向参数
        self.is_turning = False
        self.turn_start_time = 0
        self.turn_duration = 2.0  # 转向持续时间(秒)
        self.turn_angular = 0.0  # 转向角度
        #输入参数
        self.vel_seq = [0.25, 0.25, 0.25, 0.25, 0.25]
        self.lane_seq = [1, 0, 0, 0, 0]
        self.durations = [4.0, 2.0, 2.0, 2.0, 2.0]
        self.current_step = 0
        self.step_start_time = None

    def _create_folders(self):
        """创建主文件夹和子文件夹"""
        os.makedirs(self.csv_folder, exist_ok=True)
        os.makedirs(self.picture_folder, exist_ok=True)
        os.makedirs(self.picture_raw_folder, exist_ok=True)
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.picture_video_folder, exist_ok=True)
        os.makedirs(self.mp4_folder, exist_ok=True)

    def preprocess_image(self, image, time):
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
            print(f"偏离距离: {deviation:.2f} 像素（{'偏右' if deviation > 0 else '偏左'}）")

        angular = self.Kp * deviation

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
        text_time = f"Time: {time}"
        text_angular = f"Augular: {angular}"
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
        return result, angular

    def execute_current_step(self,fix_angular):
        """执行当前步骤的动作"""
        if self.current_step >= len(self.vel_seq):
            rospy.loginfo("动作序列已完成")
            return

        if self.step_start_time is None:
            self.step_start_time = rospy.get_time()

        elapsed = rospy.get_time() - self.step_start_time
        current_duration = self.durations[self.current_step]

        # 准备控制命令
        twist = Twist()


        # 计算转向角度（平滑过渡）
        if self.lane_seq[self.current_step] != 0 and elapsed < current_duration:
            progress = min(elapsed / current_duration, 1.0)
            phase = progress * math.pi * 2  # 0~2π
            angular = self.lane_seq[self.current_step] * math.sin(phase)
            # 计算速度补偿以保持x方向分量
            try:
                # 限制最大转向角度为60度(约1.047弧度)
                max_angle = math.pi / 3
                current_angular = max(min(angular, max_angle), -max_angle)

                # 计算补偿速度: v = v_desired / cos(θ)
                speed_compensation = 1.0 / math.cos(abs(current_angular))
                # 限制最大补偿速度为初始值的2倍
                max_speed = 2.0 * self.vel_seq[self.current_step]
                actual_speed = min(self.vel_seq[self.current_step] * speed_compensation, max_speed)
            except:
                # 如果计算出错(如cos(90°))，使用默认速度
                actual_speed = self.vel_seq[self.current_step]
                current_angular = 0
            twist.linear.x = actual_speed
            twist.angular.z = current_angular * (1 / actual_speed) / 5
        else:
            twist.linear.x = self.vel_seq[self.current_step]
            twist.angular.z = fix_angular

        # 发布命令
        self.cmd_vel_pub.publish(twist)

        # 检查是否切换到下一步骤
        if elapsed >= current_duration:
            self.current_step += 1
            self.step_start_time = rospy.get_time()
            if self.current_step < len(self.vel_seq):
                rospy.loginfo(f"切换到步骤 {self.current_step + 1}/{len(self.vel_seq)}")
            else:
                rospy.loginfo("动作序列执行完成")
    def publish_constant_speed(self, linear=0.2, angular=0.0):
        twist = Twist()
        desired_linear_x = linear

        if self.is_turning:
            current_time = rospy.get_time()
            elapsed_time = current_time - self.turn_start_time
            progress = min(elapsed_time / self.turn_duration, 1.0)  # 0~1

            # 使用完整正弦周期实现平滑转向 (0→π→2π)
            # 正弦函数从0开始，上升到1，下降到-1，最后回到0
            phase = progress * 2 * math.pi  # 0~2π
            current_angular = self.turn_angular * math.sin(phase)

            # 计算速度补偿以保持x方向分量
            try:
                # 限制最大转向角度为60度(约1.047弧度)
                max_angle = math.pi / 3
                current_angular = max(min(current_angular, max_angle), -max_angle)

                # 计算补偿速度: v = v_desired / cos(θ)
                speed_compensation = 1.0 / math.cos(abs(current_angular))

                # 限制最大补偿速度为初始值的2倍
                max_speed = 2.0 * desired_linear_x
                actual_speed = min(desired_linear_x * speed_compensation, max_speed)
            except:
                # 如果计算出错(如cos(90°))，使用默认速度
                actual_speed = desired_linear_x
                current_angular = 0

            # 转向完成检测
            if progress >= 1.0:
                self.is_turning = False
                current_angular = 0

            # 设置最终速度命令
            twist.linear.x = actual_speed
            twist.angular.z = current_angular

            rospy.loginfo(f"转向进度: {progress:.2f}, 角度: {current_angular:.2f}, 速度: {actual_speed:.2f}")
        else:
            # 车道保持模式
            twist.linear.x = desired_linear_x
            twist.angular.z = angular



        self.cmd_vel_pub.publish(twist)

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
        deviation = self.vehicle_position[0] - lane_center

        return deviation

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

    def start_turn(self, angular, duration=None):
        """开始转向"""
        self.is_turning = True
        self.turn_start_time = rospy.get_time()
        self.turn_angular = angular
        if duration is not None:
            self.turn_duration = duration


    def callback(self, data):
        try:
            now = datetime.now()
            time_str = now.strftime("%H:%M:%S.%f")[:-3]
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            result, angular = self.preprocess_image(image, time_str)
            # twist = Twist()
            # twist.linear.x = 0.2
            # twist.angular.z = angular
            # self.cmd_vel_pub.publish(twist)
            #
            # self.rate.sleep()
            self.execute_current_step(angular)
            # 如果不处于转向状态，才使用车道保持的角度
            # # linear = vel
            # if not self.is_turning:
            #     self.publish_constant_speed(linear=vel, angular=angular)
            # else:
            #     self.publish_constant_speed(linear=vel)

        except Exception as e:
            rospy.logwarn(f"Processing failed: {e}")


def receive_message():
    rospy.init_node('lane_keeping_system', anonymous=True)
    lks = LaneKeepingSystem()
    rospy.Subscriber('/camera/rgb/image_raw', Image, lks.callback)

    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    receive_message()
    