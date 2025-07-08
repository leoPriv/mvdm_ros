import cv2
import threading
import time
from datetime import datetime
import os


class DualCameraRecorder:
    def __init__(self, camera1_id=0, camera2_id=2, output_dir="camera"):
        self.camera1_id = camera1_id
        self.camera2_id = camera2_id
        self.output_dir = output_dir
        self.recording = False

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 初始化摄像头
        self.cap1 = None
        self.cap2 = None
        self.writer1 = None
        self.writer2 = None

    def initialize_cameras(self):
        """初始化两个摄像头"""
        print("正在初始化摄像头...")

        # 打开摄像头1
        self.cap1 = cv2.VideoCapture(self.camera1_id)
        if not self.cap1.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera1_id}")
            return False

        # 打开摄像头2
        self.cap2 = cv2.VideoCapture(self.camera2_id)
        if not self.cap2.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera2_id}")
            self.cap1.release()
            return False
        for j in range(20):
            self.cap1.read()
            self.cap2.read()
        # 设置摄像头参数
        width, height = 640, 480
        fps = 30
        exposure_value = -int(10)

        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap1.set(cv2.CAP_PROP_FPS, fps)
        self.cap1.set(cv2.CAP_PROP_BRIGHTNESS, 5)
        self.cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap1.set(cv2.CAP_PROP_EXPOSURE, exposure_value)



        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap2.set(cv2.CAP_PROP_FPS, fps)
        self.cap2.set(cv2.CAP_PROP_BRIGHTNESS, 5)
        self.cap2.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap2.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

        print("摄像头初始化成功")
        return True

    def setup_video_writers(self):
        """设置视频编码器"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 视频文件路径
        video1_path = os.path.join(self.output_dir, f"camera1_{timestamp}.mp4")
        video2_path = os.path.join(self.output_dir, f"camera2_{timestamp}.mp4")

        # 视频编码设置
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        frame_size = (640, 480)

        self.writer1 = cv2.VideoWriter(video1_path, fourcc, fps, frame_size)
        self.writer2 = cv2.VideoWriter(video2_path, fourcc, fps, frame_size)

        if not self.writer1.isOpened() :
            print("错误: 无法创建视频文件")
            return False

        print(f"视频将保存到:")
        print(f"摄像头1: {video1_path}")
        # print(f"摄像头2: {video2_path}")
        return True

    def record_camera(self, cap, writer, camera_name):
        """录制单个摄像头的线程函数"""
        frame_count = 0

        while self.recording:
            ret, frame = cap.read()
            if not ret:
                print(f"警告: {camera_name} 读取帧失败")
                break

            # 在帧上添加时间戳和摄像头标识
            # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # cv2.putText(frame, f"{camera_name} - {timestamp}",
            #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 写入视频文件
            writer.write(frame)
            frame_count += 1

            # 显示预览窗口
            # cv2.imshow(camera_name, frame)

            # # 检查按键
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     self.recording = False
            #     break

        print(f"{camera_name} 录制完成，共录制 {frame_count} 帧")

    def start_recording(self):
        """开始录制"""
        if not self.initialize_cameras():
            return False

        if not self.setup_video_writers():
            self.cleanup()
            return False

        print("开始录制... 按 'q' 键停止录制")
        self.recording = True

        # 创建两个线程分别录制两个摄像头
        thread1 = threading.Thread(target=self.record_camera,
                                   args=(self.cap1, self.writer1, "Camera 1"))
        thread2 = threading.Thread(target=self.record_camera,
                                   args=(self.cap2, self.writer2, "Camera 2"))

        # 启动录制线程
        thread1.start()
        thread2.start()

        # 等待线程结束
        thread1.join()
        thread2.join()

        self.cleanup()
        return True

    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")

        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        if self.writer1:
            self.writer1.release()
        if self.writer2:
            self.writer2.release()

        cv2.destroyAllWindows()
        print("资源清理完成")


def main():
    # 创建录制器实例
    recorder = DualCameraRecorder(
        camera1_id=0,  # 第一个摄像头ID，通常是0
        camera2_id=2,  # 第二个摄像头ID，通常是1
        output_dir="0708"  # 输出目录
    )

    try:
        # 开始录制
        recorder.start_recording()
        print("录制完成")

    except KeyboardInterrupt:
        print("\n用户中断录制")
        recorder.recording = False
        recorder.cleanup()
    except Exception as e:
        print(f"录制过程中出现错误: {e}")
        recorder.cleanup()


if __name__ == "__main__":
    main()