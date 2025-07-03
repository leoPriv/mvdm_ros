import cv2
import numpy as np
import threading
import time
from datetime import datetime
import os


class AutoLineUndistorter:
    def __init__(self, output_dir="debug_output"):
        # 存储参考线信息和单应性矩阵
        self.homography = None
        self.target_lines = [
            ((0, 240), (640, 240)),  # 水平中线
            ((320, 0), (320, 480)),  # 垂直中线
            ((0, 0), (640, 480))  # 对角线
        ]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_strongest_lines(self, frame, n=3, save_debug=False, camera_name=""):
        """自动检测图像中最明显的n条直线"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        # 保存边缘检测结果（调试用）
        if save_debug:
            cv2.imwrite(os.path.join(self.output_dir, f"{camera_name}_edges.jpg"), edges)

        # 使用Hough变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)

        if lines is None:
            return None

        # 修正直线数据格式处理
        lines = [line[0] for line in lines]  # 从形状(n,1,4)转换为(n,4)

        if len(lines) < n:
            return None

        # 按线长度排序，选择最长的n条线
        lines = sorted(lines, key=lambda x: np.linalg.norm(x[2:] - x[:2]), reverse=True)
        detected_lines = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in lines[:n]]

        # 保存检测结果（调试用）
        if save_debug:
            debug_frame = frame.copy()
            for i, ((x1, y1), (x2, y2)) in enumerate(detected_lines):
                cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_frame, f"Line {i + 1}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(self.output_dir, f"{camera_name}_detected_lines.jpg"), debug_frame)

        return detected_lines

    def calculate_homography(self, detected_lines):
        """根据检测到的直线计算单应性矩阵"""
        if len(detected_lines) != 3 or len(self.target_lines) != 3:
            return None

        # 从三条线中提取6个点（每条线的两个端点）
        src_pts = []
        dst_pts = []

        # 将检测到的线与目标线配对（基于角度相似性）
        detected_sorted = self._match_lines_by_angle(detected_lines, self.target_lines)

        for detected, target in zip(detected_sorted, self.target_lines):
            src_pts.append(detected[0])  # 线起点
            src_pts.append(detected[1])  # 线终点
            dst_pts.append(target[0])
            dst_pts.append(target[1])

        # 转换为numpy数组
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        # 计算单应性矩阵
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def _match_lines_by_angle(self, detected_lines, target_lines):
        """基于角度相似性匹配检测到的线和目标线"""

        # 计算每条线的角度
        def line_angle(line):
            (x1, y1), (x2, y2) = line
            return np.arctan2(y2 - y1, x2 - x1)

        detected_angles = [line_angle(line) for line in detected_lines]
        target_angles = [line_angle(line) for line in target_lines]

        # 创建角度差异矩阵
        angle_diff = np.abs(np.array(detected_angles)[:, None] - np.array(target_angles))

        # 找到最佳匹配
        matched_indices = []
        for i in range(3):
            # 找到最小差异的配对
            min_diff = np.min(angle_diff)
            loc = np.where(angle_diff == min_diff)
            row, col = loc[0][0], loc[1][0]

            matched_indices.append((row, col))

            # 将这些行和列设置为大值，避免重复选择
            angle_diff[row, :] = np.inf
            angle_diff[:, col] = np.inf

        # 按目标线顺序排序
        matched_indices.sort(key=lambda x: x[1])
        return [detected_lines[i] for i, _ in matched_indices]

    def undistort_frame(self, frame):
        """使用单应性矩阵校正图像"""
        if self.homography is None:
            return frame

        h, w = frame.shape[:2]
        return cv2.warpPerspective(frame, self.homography, (w, h))

    def calibrate(self, frame, camera_name=""):
        """使用一帧图像进行自动校准"""
        detected_lines = self.detect_strongest_lines(frame, save_debug=True, camera_name=camera_name)
        if detected_lines is None:
            print(f"{camera_name}: 无法检测到足够的直线")
            return False

        # 在图像上绘制检测到的线（调试用）
        debug_frame = frame.copy()
        for i, ((x1, y1), (x2, y2)) in enumerate(detected_lines):
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Line {i + 1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 计算单应性矩阵
        self.homography = self.calculate_homography(detected_lines)
        if self.homography is None:
            print(f"{camera_name}: 无法计算单应性矩阵")
            return False

        # 保存校正前后的对比图
        undistorted = self.undistort_frame(frame)
        combined = np.hstack((debug_frame, undistorted))
        cv2.putText(combined, "Detected Lines -> Corrected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(self.output_dir, f"{camera_name}_calibration_result.jpg"), combined)

        print(f"{camera_name}: 校准完成，结果已保存到{self.output_dir}")
        return True


class DualCameraRecorder:
    def __init__(self, camera1_id=0, camera2_id=2, output_dir="camera"):
        self.camera1_id = camera1_id
        self.camera2_id = camera2_id
        self.output_dir = output_dir
        self.recording = False

        # 创建自动校正器实例
        self.undistorter1 = AutoLineUndistorter(os.path.join(output_dir, "debug_cam1"))
        self.undistorter2 = AutoLineUndistorter(os.path.join(output_dir, "debug_cam2"))

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 初始化摄像头
        self.cap1 = None
        self.cap2 = None
        self.writer1 = None
        self.writer2 = None

    def initialize_cameras(self):
        """初始化两个摄像头并自动校准"""
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

        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap1.set(cv2.CAP_PROP_FPS, fps)
        self.cap1.set(cv2.CAP_PROP_BRIGHTNESS, 10)

        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap2.set(cv2.CAP_PROP_FPS, fps)
        self.cap2.set(cv2.CAP_PROP_BRIGHTNESS, 10)

        # 从摄像头1获取一帧进行自动校准
        print("正在校准摄像头1...")
        ret, frame = self.cap1.read()
        if ret:
            if not self.undistorter1.calibrate(frame, "camera1"):
                print("摄像头1自动校准失败")
                return False
        else:
            print("无法从摄像头1读取帧")
            return False

        # 从摄像头2获取一帧进行自动校准
        print("正在校准摄像头2...")
        ret, frame = self.cap2.read()
        if ret:
            if not self.undistorter2.calibrate(frame, "camera2"):
                print("摄像头2自动校准失败")
                return False
        else:
            print("无法从摄像头2读取帧")
            return False

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

        if not self.writer1.isOpened():
            print("错误: 无法创建视频文件")
            return False

        print(f"视频将保存到:")
        print(f"摄像头1: {video1_path}")
        print(f"摄像头2: {video2_path}")
        return True

    def record_camera(self, cap, writer, undistorter, camera_name):
        """录制单个摄像头的线程函数"""
        frame_count = 0

        while self.recording:
            ret, frame = cap.read()
            if not ret:
                print(f"警告: {camera_name} 读取帧失败")
                break

            # 校正畸变
            frame = undistorter.undistort_frame(frame)

            # 调整大小以匹配视频写入器
            frame = cv2.resize(frame, (640, 480))

            # 在帧上添加时间戳和摄像头标识
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"{camera_name} - {timestamp}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 写入视频文件
            writer.write(frame)
            frame_count += 1

        print(f"{camera_name} 录制完成，共录制 {frame_count} 帧")

    def start_recording(self):
        """开始录制"""
        if not self.initialize_cameras():
            return False

        if not self.setup_video_writers():
            self.cleanup()
            return False

        print("开始录制... 按 Ctrl+C 停止录制")
        self.recording = True

        # 创建两个线程分别录制两个摄像头
        thread1 = threading.Thread(target=self.record_camera,
                                   args=(self.cap1, self.writer1,
                                         self.undistorter1, "Camera 1"))
        thread2 = threading.Thread(target=self.record_camera,
                                   args=(self.cap2, self.writer2,
                                         self.undistorter2, "Camera 2"))

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
        output_dir="camera"  # 输出目录
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