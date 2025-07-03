#!/usr/bin/env python3
"""
简化版摄像头打开程序
"""
import cv2
import time


class SimpleCamera:
    def __init__(self, device_path='/dev/video0', width=640, height=480, fps=30):
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def initialize_camera(self):
        """初始化摄像头"""
        print(f"正在打开摄像头: {self.device_path}")

        try:
            self.cap = cv2.VideoCapture(self.device_path)

            if self.cap.isOpened():
                # 设置参数
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

                # 测试读取
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    actual_height, actual_width = frame.shape[:2]
                    print(f"✓ 摄像头初始化成功: {actual_width}x{actual_height}")
                    return True
                else:
                    print("✗ 摄像头无法读取图像")
                    return False
            else:
                print("✗ 摄像头无法打开")
                return False

        except Exception as e:
            print(f"✗ 摄像头初始化失败: {e}")
            return False

    def show_camera(self):
        """显示摄像头画面"""
        if not self.cap or not self.cap.isOpened():
            print("摄像头未初始化")
            return

        print("摄像头已启动，按 'q' 退出...")

        try:
            while True:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # 显示帧率信息
                    cv2.putText(frame, f"Camera: {self.device_path}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Size: {self.width}x{self.height}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow('Camera View', frame)

                    # 按 'q' 退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("无法读取摄像头画面")
                    break

        except KeyboardInterrupt:
            print("\n程序被中断")
        finally:
            cv2.destroyAllWindows()

    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
            print("摄像头已关闭")


def main():
    """主函数"""
    print("简化版摄像头程序")
    print("=" * 30)

    # 创建摄像头对象 (使用你已经确认的设备)
    camera = SimpleCamera(device_path='/dev/video0', width=640, height=480, fps=30)

    try:
        # 初始化摄像头
        if camera.initialize_camera():
            # 显示摄像头画面
            camera.show_camera()
        else:
            print("摄像头初始化失败")

    except Exception as e:
        print(f"程序错误: {e}")

    finally:
        # 清理资源
        camera.cleanup()


if __name__ == '__main__':
    main()