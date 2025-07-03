# launcher.py (修改版本 - 支持优雅退出)
import os
import signal
import subprocess
import sys
from multiprocessing import Process
import time


class RobotLauncher:
    def __init__(self):
        self.processes = []
        self.subprocesses = []

    def signal_handler(self, signum, frame):
        """处理Ctrl+C信号"""
        print("\n正在停止所有机器人节点...")
        self.cleanup()
        sys.exit(0)

    def start_robot_node(self, robot_id):
        """为每个机器人启动完全独立的Python解释器"""
        cmd = f"python3 robot_node.py --robot_id={robot_id}"
        try:
            # 使用Popen并保存进程引用
            proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
            self.subprocesses.append(proc)
            print(f"机器人 {robot_id} 已启动，PID: {proc.pid}")

            # 等待子进程结束
            proc.wait()
        except Exception as e:
            print(f"启动机器人 {robot_id} 失败: {e}")

    def cleanup(self):
        """清理所有进程"""
        # 终止所有subprocess
        for proc in self.subprocesses:
            try:
                if proc.poll() is None:  # 进程还在运行
                    # 终止整个进程组
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    print(f"已终止进程组 {proc.pid}")
            except Exception as e:
                print(f"终止进程时出错: {e}")

        # 终止所有multiprocessing进程
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                print(f"已终止多进程 {p.pid}")

    def run(self):
        """主运行函数"""
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # 启动所有机器人节点
            # robot_ids = ["9293","9298"]  # 可以添加更多机器人ID

            for rid in robot_ids:
                p = Process(target=self.start_robot_node, args=(rid,))
                p.start()
                self.processes.append(p)
                print(f"为机器人 {rid} 创建了主进程")

            print("所有机器人节点已启动。按 Ctrl+C 停止...")

            # 等待所有进程完成
            for p in self.processes:
                p.join()

        except KeyboardInterrupt:
            print("\n接收到中断信号...")
        finally:
            self.cleanup()


# 简化版本（如果你不需要复杂的进程管理）
def simple_launcher():
    """简化的启动器版本"""
    processes = []

    def signal_handler(signum, frame):
        print("\n正在停止所有进程...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)
                if p.is_alive():
                    p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def start_robot_node(robot_id):
        cmd = f"python3 robot_node.py --robot_id={robot_id}"
        subprocess.run(cmd, shell=True)

    try:
        for rid in ["9289","9298"]:
            p = Process(target=start_robot_node, args=(rid,))
            p.start()
            processes.append(p)
            print(f"为机器人 {rid} 创建了主进程")

        print("机器人节点已启动。按 Ctrl+C 停止...")
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n接收到中断信号...")


if __name__ == '__main__':
    # 使用完整版本
    # launcher = RobotLauncher()
    # launcher.run()

    # 或者使用简化版本
    simple_launcher()