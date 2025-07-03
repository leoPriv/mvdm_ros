#!/usr/bin/env python3
import rospy
from pylimo import limo
import cv2
import numpy as np
import time
import math
from threading import Thread, Lock
from std_msgs.msg import Float32
from collections import deque
import signal
import sys
import atexit

limo = limo.LIMO()
limo.EnableCommand()
params = {
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


class Mvdm_System:
    def __init__(self):

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

        # 默认值和超时设置
        self.exc_speed = 0.0  # 默认速度为0
        self.exc_lane = 0.0  # 默认车道变换为0

        # 超时机制相关变量
        self.last_vel_time = 0  # 上次收到速度指令的时间，初始为0
        self.last_lane_time = 0  # 上次收到车道变换指令的时间，初始为0
        self.vel_timeout = 1.0  # 速度指令超时时间（秒）
        self.lane_timeout = 1.0  # 车道变换指令超时时间（秒）
        self.vel_ever_received = False  # 是否曾经收到过速度指令
        self.lane_ever_received = False  # 是否曾经收到过车道变换指令

        # 图像相关
        self.width = 640
        self.height = 480
        self.fps = 30

        # 控制参数
        self.Kp = params["Kp"]
        self.Ki = params["Ki"]
        self.Kd = params["Kd"]
        self.alpha = params["alpha"]
        self.min_angular = params["min_angular"]
        self.max_angular = params["max_angular"]

        # 摄像头初始化
        self.cap = None
        self.init_camera()

        # 处理订阅
        self.lock = Lock()

        self.vel_sub = rospy.Subscriber('/limo9293/vel', Float32, self.vel_callback)
        self.lane_sub = rospy.Subscriber('/limo9293/lane', Float32, self.lane_callback)
        self.control_timer = rospy.Timer(rospy.Duration(params["control_period"]), self.cbControl)
        # 读取图像
        self.img_timer = rospy.Timer(rospy.Duration(params["control_period"]), self.publish_frame)

        # 超时检查定时器
        self.timeout_timer = rospy.Timer(rospy.Duration(0.5), self.check_timeout)

        # 注册清理函数
        atexit.register(self.cleanup)

        rospy.loginfo("MVDM系统初始化完成")
        rospy.loginfo(f"速度超时设置: {self.vel_timeout}s, 车道变换超时设置: {self.lane_timeout}s")

    def check_timeout(self, event):
        """检查超时并重置值"""
        current_time = time.time()

        # 检查速度超时
        if self.vel_ever_received:
            if (current_time - self.last_vel_time) > self.vel_timeout:
                if self.exc_speed != 0:  # 只在值改变时输出日志
                    rospy.logwarn(f"速度指令超时({self.vel_timeout:.1f}s)，重置为0")
                self.exc_speed = 0.0
        else:
            # 从未收到过指令，定期提醒
            rospy.loginfo_throttle(10, "等待速度指令...")

        # 检查车道变换超时
        if self.lane_ever_received:
            if (current_time - self.last_lane_time) > self.lane_timeout:
                if self.exc_lane != 0:  # 只在值改变时输出日志
                    rospy.logwarn(f"车道变换指令超时({self.lane_timeout:.1f}s)，重置为0")
                self.exc_lane = 0.0
        else:
            # 从未收到过指令，定期提醒
            rospy.loginfo_throttle(10, "等待车道变换指令...")

    def init_camera(self):
        """初始化摄像头"""
        try:
            # 如果摄像头已经打开，先关闭
            if self.cap is not None:
                self.cap.release()
                time.sleep(0.5)  # 等待释放完成

            self.cap = cv2.VideoCapture('/dev/video0')
            if self.cap.isOpened():
                # 设置参数
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

                # 测试读取一帧
                ret, frame = self.cap.read()
                if ret:
                    rospy.loginfo("摄像头初始化成功")
                    return True
                else:
                    rospy.logerr("摄像头无法读取图像")
                    return False
            else:
                rospy.logerr("摄像头初始化失败")
                return False
        except Exception as e:
            rospy.logerr(f"摄像头初始化异常: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        rospy.loginfo("开始清理资源...")

        try:
            # 停止定时器
            if hasattr(self, 'control_timer'):
                self.control_timer.shutdown()
            if hasattr(self, 'img_timer'):
                self.img_timer.shutdown()
            if hasattr(self, 'timeout_timer'):
                self.timeout_timer.shutdown()
        except:
            pass

        # 释放摄像头
        if self.cap is not None:
            try:
                self.cap.release()
                rospy.loginfo("摄像头资源已释放")
            except Exception as e:
                rospy.logerr(f"释放摄像头时出错: {e}")

        # 停止车辆运动
        try:
            limo.SetMotionCommand(0, 0, 0, 0)
            rospy.loginfo("车辆已停止")
        except Exception as e:
            rospy.logerr(f"停止车辆时出错: {e}")

        # 销毁所有OpenCV窗口
        try:
            cv2.destroyAllWindows()
        except:
            pass

        rospy.loginfo("资源清理完成")

    def image_callback(self, msg):
        # 开线程处理图像
        Thread(target=self.preprocess_image, args=(msg,), daemon=True).start()

    def vel_callback(self, vel):
        """速度回调函数"""
        try:
            if hasattr(vel, 'data'):
                self.exc_speed = vel.data
                self.last_vel_time = time.time()
                self.vel_ever_received = True
                rospy.loginfo_throttle(2, f"收到速度指令: {self.exc_speed:.2f}")
            else:
                rospy.logwarn("速度消息格式错误，重置为0")
                self.exc_speed = 0.0
        except Exception as e:
            rospy.logerr(f"处理速度消息时出错: {e}")
            self.exc_speed = 0.0

    def lane_callback(self, lane):
        """车道变换回调函数"""
        try:
            if hasattr(lane, 'data'):
                self.exc_lane = lane.data
                self.last_lane_time = time.time()
                self.lane_ever_received = True
                rospy.loginfo_throttle(2, f"收到车道变换指令: {self.exc_lane:.2f}")
            else:
                rospy.logwarn("车道变换消息格式错误，重置为0")
                self.exc_lane = 0.0
        except Exception as e:
            rospy.logerr(f"处理车道变换消息时出错: {e}")
            self.exc_lane = 0.0

    def publish_frame(self, event):
        """定时器回调：读取摄像头并发布ROS消息"""
        try:
            if not self.cap or not self.cap.isOpened():
                rospy.logwarn_throttle(5, "摄像头未打开，尝试重新初始化...")
                self.init_camera()
                return

            ret, frame = self.cap.read()
            if ret and frame is not None:
                # 调用图像处理
                Thread(target=self.preprocess_image, args=(frame,), daemon=True).start()
            else:
                rospy.logwarn_throttle(5, "摄像头读取失败")
        except Exception as e:
            rospy.logerr(f"图像发布错误: {str(e)}")

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

    def enhanced_pid_control(self, deviation):
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

    def smooth_angular(self, angular):
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
            deviation = self.calculate_deviation(left_line, right_line, height, width)

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

        rospy.loginfo_throttle(2,
                               f"Deviation: {self.filtered_deviation:.2f}, Angular: {self.angular:.3f}, Speed: {self.exc_speed:.2f}, Lane: {self.exc_lane:.2f}")

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

    def calculate_deviation(self, left_line, right_line, image_height, image_width):
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
        self.first_pos = image_width // 2 - 10

        deviation = self.first_pos - lane_center
        return deviation

    def publish_keep_twist(self, linear_x, angular_z):
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

        #rospy.loginfo_throttle(2, f"SetMotionCommand - Speed: {linear_x:.3f}, Angular: {angular:.3f}")
        rospy.loginfo(f"SetMotionCommand - Speed: {linear_x:.3f}, Angular: {angular / 2:.3f}")

    def publish_lc_twist(self, linear_x, angular_z):
        """发布控制命令（改进版）"""

        limo.SetMotionCommand(linear_x, -0.01, 0, angular_z)

        #rospy.loginfo_throttle(2, f"SetMotionCommand - Speed: {linear_x:.3f}, Angular: {angular:.3f}")
        rospy.loginfo(f"SetMotionCommand - Speed: {linear_x:.3f}, Angular: {angular_z:.3f}")

    def cbControl(self, event):
        """控制回调函数"""
        index_lane = 0

        # 如果速度为0，直接停止车辆
        if self.exc_speed == 0:
            limo.SetMotionCommand(0, 0, 0, 0)
            rospy.loginfo_throttle(5, "速度为0，车辆静止")
            return

        if self.exc_lane == 0:
            self.publish_keep_twist(self.exc_speed, self.angular)
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
            exc_angular = current_angular * (1 / actual_speed) * 10
            self.publish_lc_twist(actual_speed, exc_angular)


def signal_handler(sig, frame):
    """信号处理函数"""
    rospy.loginfo("接收到中断信号，正在安全退出...")

    # 停止车辆
    try:
        limo.SetMotionCommand(0, 0, 0, 0)
        rospy.loginfo("车辆已停止")
    except:
        pass

    # 关闭ROS节点
    rospy.signal_shutdown("用户中断")

    # 等待一下确保资源释放
    time.sleep(0.5)

    sys.exit(0)


def run():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.init_node('car_9293', anonymous=False)

    try:
        mvdm_system = Mvdm_System()
        rospy.loginfo("系统启动完成，等待信号...")
        rospy.loginfo("注意：当前速度和车道变换默认为0，车辆将保持静止")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS节点被中断")
    except Exception as e:
        rospy.logerr(f"系统运行错误: {e}")
    finally:
        # 确保清理函数被调用
        if 'mvdm_system' in locals():
            mvdm_system.cleanup()


if __name__ == '__main__':
    run()