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
    # "robot_ids": ["9289"],
    "robot_ids": ["9298", "9289"],
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
    "use_gpu": True,  # 是否使用GPU加速
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


def gaussian_blur_gpu(img, kernel_size=5, sigma=1.0):
    # 将图像上传到 GPU
    img_gpu = img.asty

    # 创建输出缓冲区
    output_gpu = gpuarray.empty_like(img_gpu)

    # 编写简单的 CUDA 核函数 (这里简化了实际实现)
    kernel_code = """
    __global__ void gaussian_blur(float *input, float *output, int width, int height) {
        // 实际实现需要填充高斯核计算逻辑
        // 此处仅为示例框架
    }
    """
    # 调用核函数并返回结果
    return output_gpu.get()

class GPUEnhancedMvdmSystem:
    def __init__(self, speed_action, lane_action):
        # 检查CUDA支持
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0 and params["use_gpu"]
        if self.gpu_available:
            rospy.loginfo("GPU acceleration enabled")
            # 初始化GPU内存分配器
            self.gpu_mem_pool = cv2.cuda.GpuMat()
            # 预分配GPU内存
            self.gpu_image = cv2.cuda.GpuMat()
            self.gpu_gray = cv2.cuda.GpuMat()
            self.gpu_blur = cv2.cuda.GpuMat()
            self.gpu_edges = cv2.cuda.GpuMat()
            self.gpu_enhanced = cv2.cuda.GpuMat()
            self.gpu_lab = cv2.cuda.GpuMat()

            # 创建GPU处理对象
            self.gpu_clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.gpu_gaussian = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
            self.gpu_canny = cv2.cuda.createCannyEdgeDetector(50, 150)

            # HoughLines检测器 (GPU版本)
            self.gpu_hough = cv2.cuda.createHoughLinesDetector(rho=1, theta=np.pi / 180, threshold=20)
        else:
            rospy.logwarn("GPU not available, using CPU processing")

        # 原有变量
        self.robot_ids = params["robot_ids"]

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

        self.hough_detector = cv2.cuda.createHoughLinesDetector(
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            maxLines=300
        )

    def gpu_image_processing(self, image,limo_id):
        """最安全的GPU处理"""
        try:
            # 检查输入
            if image is None or image.size == 0:
                return None

            rospy.loginfo(f"Processing image: {image.shape}， id: {limo_id}")
            height, width = image.shape[:2]
            # 创建新的GPU矩阵（避免重用问题）
            gpu_src = cv2.cuda.GpuMat()
            gpu_dst = cv2.cuda_GpuMat(gpu_src.size(), cv2.CV_8UC1)

            # 上传
            gpu_src.upload(image)

            if gpu_src.empty():
                rospy.logwarn("GPU upload failed")
                return None

            # 颜色转换
            try:
                gpu_dst = cv2.cuda.cvtColor(gpu_src, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                rospy.logwarn(f"GPU cvtColor failed: {e}")
                return None

            if gpu_dst.empty():
                rospy.logwarn("GPU cvtColor produced empty result")
                return None

            # 下载
            gray_result = gpu_dst.download()
            # if gray_result is None or gray_result.size == 0:
            #     rospy.logwarn("GPU download failed")
            #     return None

            # CPU后续处理
            blur = cv2.GaussianBlur(gray_result, (5, 5), 0)
            gpu_blur = cv2.cuda_GpuMat()
            gpu_blur.upload(blur)
            canny_detector = cv2.cuda.createCannyEdgeDetector(low_thresh=50, high_thresh=150)
            edges = canny_detector.detect(gpu_blur)

            # 4. 霍夫线检测（GPU）

            lines_gpu = self.hough_detector.detect(edges)

            # 5. 下载线段结果到CPU（关键步骤！）
            lines = lines_gpu.download()  # 返回的是np.array，形状为(N, 1, 4)
            print(lines.size)


            # edges = cv2.cuda.CannyEdgeDetector(blur, 50, 150)
            # lines = cv2.cuda.HoughLinesDetector(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)

            left_points = []
            right_points = []
            line_image = np.zeros_like(image)

            if lines is not None:
                for line in lines:
                    # lines的形状是(N, 1, 4)，每个line是(1, 4)
                    if len(line.shape) > 1 and line.shape[-1] == 4:
                        # 线段端点格式 (x1, y1, x2, y2)
                        x1, y1, x2, y2 = line[0]  # 取第一个元素，它包含4个坐标
                    elif len(line.shape) > 1 and line.shape[-1] == 2:
                        # 极坐标格式 (rho, theta)
                        rho, theta = line[0]  # 注意这里也要取line[0]
                        # 转换逻辑
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                    else:
                        continue

                    # x1, y1, x2, y2 = line[0]
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
                deviation = self.calculate_deviation(left_line, right_line, height, width, limo_id)

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

            return edges

        except Exception as e:
            rospy.logwarn(f"GPU processing completely failed: {e}")
            return None

    def cpu_image_processing(self, image):
        """CPU图像处理（备用方案）"""
        # 颜色增强
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 转换为灰度图并应用高斯模糊
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny边缘检测
        edges = cv2.Canny(blur, 50, 150)

        return edges

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
        """图像处理主函数（使用GPU加速）"""
        limo_id = extra_param
        image = self.bridge.imgmsg_to_cv2(image, "bgr8")

        # 图像预处理
        height, width = image.shape[:2]
        lower_half = image[round(height // 1.75): height, :]
        height, width = lower_half.shape[:2]

        # 尝试使用GPU处理，失败则使用CPU
        if self.gpu_available:
            edges = self.gpu_image_processing(lower_half,limo_id)
            print("gpu")

            if edges is None:
                print("cpu")
                edges = self.cpu_image_processing(lower_half)
        else:
            edges = self.cpu_image_processing(lower_half)

        # HoughLines检测（仍然使用CPU，因为GPU版本的HoughLinesP不太稳定）

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

    def calculate_deviation(self, left_line, right_line, image_height, image_width, limo_id):
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
        # 更温和的角速度限制
        if abs(angular_z) > params["control_angular_max"]:
            if angular_z > 0:
                angular = params["control_angular"]
            else:
                angular = -params["control_angular"]
        else:
            # 按比例缩放，而不是直接置零
            angular = angular_z * (params["control_angular"] / params["control_angular_max"])

        if angular < 0:
            angular += vehicle_params[limo_id]["angular_delta"]

        vel_pub.publish(linear_x)
        ang_pub.publish(angular * 1.2)

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

    def __del__(self):
        """清理GPU资源"""
        if hasattr(self, 'gpu_available') and self.gpu_available:
            # GPU内存会自动释放，但显式释放是好习惯
            try:
                del self.gpu_image
                del self.gpu_gray
                del self.gpu_blur
                del self.gpu_edges
                del self.gpu_enhanced
                del self.gpu_lab
            except:
                pass


def run():
    rospy.init_node('gpu_enhanced_mvdm_system', anonymous=False)
    speed_action = {'9293': [0.5, 0.5, 0.5, 0.5, 0.5], "9298": [0.5, 0.5, 0.5, 0.5, 0.5],
                    "9289": [0.5, 0.5, 0.5, 0.5, 0.5]}
    lane_action = {"9293": [0, 0, 0, 0, 0], "9298": [0, 0, 0, 0, 0], "9289": [0, 0, 0, 0, 0]}
    mvdm_system = GPUEnhancedMvdmSystem(speed_action, lane_action)
    rospy.spin()


if __name__ == '__main__':
    run()