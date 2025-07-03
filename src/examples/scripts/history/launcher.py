# launcher.py (主启动脚本)
import os
import subprocess
from multiprocessing import Process

def start_robot_node(robot_id):
    """为每个机器人启动完全独立的Python解释器"""
    cmd = f"python3 robot_node.py --robot_id={robot_id}"
    subprocess.Popen(cmd, shell=True)

if __name__ == '__main__':
    for rid in ["9293"]:
        Process(target=start_robot_node, args=(rid,)).start()