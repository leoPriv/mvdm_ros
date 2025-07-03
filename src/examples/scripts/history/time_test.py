#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt


class ImageIntervalChecker:
    def __init__(self):
        rospy.init_node('image_interval_checker')

        # 存储最近两次接收时间 {robot_id: [last_time, current_time]}
        self.time_records = {
            1: [None, None],
            2: [None, None]
        }

        # 存储所有间隔数据 {robot_id: [intervals]}
        self.interval_data = {1: [], 2: []}

        # 设置订阅者
        rospy.Subscriber('/limo9298/camera/image_raw', Image, self.callback, callback_args=1)
        rospy.Subscriber('/limo9289/camera/image_raw', Image, self.callback, callback_args=2)

        # 结束时绘制结果
        rospy.on_shutdown(self.plot_results)

    def callback(self, msg, robot_id):
        current_time = rospy.Time.now().to_sec()

        # 更新时间记录
        last_time = self.time_records[robot_id][1]
        self.time_records[robot_id][0] = last_time
        self.time_records[robot_id][1] = current_time

        # 计算间隔（如果是第一条消息则跳过）
        if last_time is not None:
            interval = current_time - last_time
            self.interval_data[robot_id].append(interval)

            rospy.loginfo(
                f"Robot {robot_id} 接收间隔: {interval * 1000:.2f}ms | "
                f"预期频率: {1 / np.mean(self.interval_data[robot_id]):.1f}Hz"
            )

    def plot_results(self):
        plt.figure(figsize=(12, 6))

        for robot_id, intervals in self.interval_data.items():
            if intervals:
                # 转换为毫秒
                intervals_ms = np.array(intervals) * 1000

                plt.plot(intervals_ms, label=(
                    f"Robot {robot_id} | "
                    f"平均: {np.mean(intervals_ms):.2f}±{np.std(intervals_ms):.2f}ms"
                ))

        plt.title("连续图像接收间隔分析")
        plt.xlabel("图像序列号")
        plt.ylabel("接收间隔 (ms)")
        plt.legend()
        plt.grid()
        plt.savefig('receive_interval.png')
        rospy.loginfo("已保存间隔分析图: receive_interval.png")


if __name__ == '__main__':
    ImageIntervalChecker()
    rospy.spin()