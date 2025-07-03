#!/usr/bin/env python3
import rospy
import cv2
import time
from datetime import datetime
from pylimo import limo
import numpy as np
import pylimo.limomsg as limomsg
from std_msgs.msg import Float32, Header
from sensor_msgs.msg import Image

limo = limo.LIMO()
limo.EnableCommand()


class PurePythonImageConverter:
    """纯Python实现的图像转换器，替代cv_bridge"""

    @staticmethod
    def cv2_to_imgmsg(cv_image, encoding="bgr8", header=None):
        """
        将OpenCV图像转换为ROS Image消息

        参数:
            cv_image: numpy.ndarray, OpenCV图像
            encoding: str, ROS图像编码格式("bgr8", "rgb8", "mono8"等)
            header: std_msgs/Header, 可选的ROS头信息

        返回:
            sensor_msgs/Image 或 None(失败时)
        """
        try:
            # 验证输入图像
            if not isinstance(cv_image, np.ndarray):
                rospy.logerr("输入必须是numpy数组")
                return None
            if cv_image.size == 0:
                rospy.logerr("输入图像为空")
                return None

            # 设置消息头
            img_msg = Image()
            if header is not None:
                img_msg.header = header
            else:
                img_msg.header = Header(stamp=rospy.Time.now())

            # 根据编码设置参数
            img_msg.encoding = encoding

            # 处理不同编码格式
            if encoding == "bgr8":
                if len(cv_image.shape) == 2:  # 灰度图转BGR
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                elif cv_image.shape[2] == 4:  # 带alpha通道
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
            elif encoding == "rgb8":
                if len(cv_image.shape) == 2:  # 灰度图转RGB
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                elif cv_image.shape[2] == 3:  # BGR转RGB
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            elif encoding == "mono8":
                if len(cv_image.shape) == 3:  # 彩色转灰度
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # 设置图像尺寸和步长
            img_msg.height = cv_image.shape[0]
            img_msg.width = cv_image.shape[1]

            if len(cv_image.shape) == 3:
                img_msg.step = cv_image.shape[1] * cv_image.shape[2]  # width * channels
            else:
                img_msg.step = cv_image.shape[1]  # width for mono

            # 确保数据类型是uint8
            if cv_image.dtype != np.uint8:
                cv_image = cv_image.astype(np.uint8)

            # 设置图像数据
            img_msg.data = cv_image.tobytes()
            img_msg.is_bigendian = 0  # 小端存储

            return img_msg

        except Exception as e:
            rospy.logerr(f"cv2_to_imgmsg转换失败: {str(e)}")
            return None


class Mvdm_System:
    def __init__(self):
        self.angular = 0
        self.vel = 0
        self.img_control = 0
        self.image = None
        self.min_angular = -0.6
        self.max_angular = 0.6
        self.width = 640
        self.height = 480
        self.fps = 30

        # 订阅器
        self.image_converter = PurePythonImageConverter()
        self.cap = cv2.VideoCapture('/dev/video0')
        if self.cap.isOpened():
            # 设置参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            rospy.loginfo("摄像头初始化成功")
        else:
            rospy.logerr("摄像头初始化失败")

        self.vel_sub = rospy.Subscriber('/limo9289/vel', Float32, self.vel_callback)
        self.ang_sub = rospy.Subscriber('/limo9289/angular', Float32, self.ang_callback)
        # self.image_sub = rospy.Subscriber('/limo9289/camera/image_raw', Image, self.image_callback)
        self.imgcontrol_sub = rospy.Subscriber('/limo9289/img_control', Float32, self.img_control_callback)
        self.image_pub = rospy.Publisher('/limo9289/image', Image, queue_size=1)

        # 发布指令
        self.img_timer = rospy.Timer(rospy.Duration(0.1), self.publish_frame)
        # self.control_timer = rospy.Timer(rospy.Duration(0.1), self.cbControl)

    def vel_callback(self, vel):
        self.vel = vel.data

    def img_control_callback(self, img_control):
        self.img_control = img_control.data
        print(f"img_control: {self.img_control}")

        # 修复：检查image是否为None
        if self.img_control == 1.0:
            if self.image is not None:
                self.image_pub.publish(self.image)
                print("图像已发布")
            else:
                rospy.logwarn("图像为空，无法发布")

    def ang_callback(self, ang):
        self.angular = ang.data
        exc_angular = max(self.angular, self.min_angular)
        exc_angular = min(exc_angular, self.max_angular)
        limo.SetMotionCommand(self.vel, -0.01, 0, exc_angular / 2)
        print(datetime.now().strftime('%H:%M:%S.%f')[:-3], self.angular, self.vel)

    # def image_callback(self, image_msg):
    #     self.image = image_msg

    def publish_frame(self, event):
        """定时器回调：读取摄像头并发布ROS消息"""
        try:
            if not self.cap.isOpened():
                rospy.logwarn("摄像头未打开")
                return

            ret, frame = self.cap.read()
            if ret and frame is not None:
                # 转换为ROS消息
                ros_image = self.image_converter.cv2_to_imgmsg(frame, encoding="bgr8")
                if ros_image is not None:
                    self.image = ros_image
                else:
                    rospy.logwarn("图像转换失败")
            else:
                rospy.logwarn("摄像头读取失败")
        except Exception as e:
            rospy.logerr(f"图像发布错误: {str(e)}")

    def cbControl(self, event):
        print(self.angular, self.vel)
        return

    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
            rospy.loginfo("摄像头资源已释放")


def run():
    rospy.init_node('car_9289')
    # limo.SetMotionCommand(0,0,0,0)
    mvdm_system = Mvdm_System()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    finally:
        mvdm_system.cleanup()


if __name__ == '__main__':
    run()