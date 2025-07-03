#!/usr/bin/env python3
import rospy
import numpy as np
from datetime import datetime
from std_msgs.msg import Float32
import pandas as pd

params = {
    # "robot_ids": ["9298"],
    "robot_ids": ["9293","9298", "9289"],
    "control_period": 0.1,
    "expand_num": 21,
}

class EnhancedMvdmSystem:
    def __init__(self, speed_action, lane_action):

        self.robot_ids = params["robot_ids"]
        self.speed_expand = {id: [0] * 10 + speed_action[id] for id in self.robot_ids}
        # self.speed_expand = {id: [0] * 10 + np.repeat(speed_action[id], params["expand_num"]).tolist() for id in
        #                      self.robot_ids}
        self.lane_expand = {
            id: [0] * 10 + (  # 前导的10个0
                [0] if lane_action[id] == 0 else  # 如果等于0就不重复
                np.repeat(lane_action[id], params["expand_num"]).tolist()  # 不等于0就重复
            )
            for id in self.robot_ids
        }
        self.count = 0

        # ROS订阅和发布
        self.lane_pubs = {
            robot_id: rospy.Publisher(
                f'/limo{robot_id}/lane',
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


    def publish_twist(self, limo_id,lane_pub, vel_pub, exc_lane, exc_speed):
        """发布控制命令（改进版）"""

        lane_pub.publish(exc_lane)
        if exc_lane !=0 :
            exc_speed = 0.5
        vel_pub.publish(exc_speed)
        rospy.loginfo(
            f"publish to {limo_id} - Speed: {exc_speed:.3f}, Lane: {exc_lane:.3f} (Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]})")

    def cbControl(self, event):
        """控制回调函数（保持原有逻辑但使用改进的发布函数）"""

        for limo_id in self.robot_ids:
            self.publish_twist(limo_id,self.lane_pubs[limo_id],self.vel_pubs[limo_id],self.lane_expand[limo_id][self.count], self.speed_expand[limo_id][self.count])
        self.count += 1



def run():
    rospy.init_node('enhanced_mvdm_system', anonymous=False)

    csv_files = [
        "/home/mvdm-test/ros2_real/src/examples/scripts/vehicle_data/flow_0012_4.csv",  # 替换为实际路径
        "/home/mvdm-test/ros2_real/src/examples/scripts/vehicle_data/flow_101_13.csv",
        "/home/mvdm-test/ros2_real/src/examples/scripts/vehicle_data/flow_1012_6.csv"
    ]
    speed_action = {}
    lane_action = {}

    for index , file in enumerate(csv_files):
        # 从路径中提取车辆ID（假设文件名是"车辆ID.csv"）
        vehicle_id = params["robot_ids"][index]  # 提取文件名并去掉.csv

        # 读取CSV文件
        df = pd.read_csv(file)
        df.head(80)

        # 提取数据并存入字典
        speed_action[vehicle_id] = (df["velocity"]/50).tolist()
        lane_action[vehicle_id] = (df["lane_change_action"]).tolist()
    # 假设我们设置采样频率为每10个数据点取一个（可根据需要调整）
    # SAMPLING_INTERVAL = 1
    #
    # for index, file in enumerate(csv_files):
    #     # 从路径中提取车辆ID
    #     vehicle_id = params["robot_ids"][index]
    #
    #     # 读取CSV文件
    #     df = pd.read_csv(file)
    #     df = pd.read_csv(file).iloc[2:80]
    #
    #     # 按固定频率采样数据
    #     sampled_df = df.iloc[::SAMPLING_INTERVAL, :]
    #
    #     # 提取采样后的数据存入字典
    #     speed_action[vehicle_id] = (sampled_df["velocity"] / 25).tolist()
    #     lane_action[vehicle_id] = sampled_df["lane_change_action"].tolist()

        # # 打印采样信息（可选）
        # print(f"车辆 {vehicle_id} 原始数据点数: {len(df)}，采样后数据点数: {len(sampled_df)}")

    # speed_action = {'9293': [0.5, 0.5, 0.5, 0.5,0.5, 0.5,0], "9298": [0.5, 0.5, 0.5, 0.5,0.5, 0.5,0], "9289": [0.5, 0.5, 0.5, 0.5,0.5, 0.5,0]}
    # lane_action = {"9293": [0, 0, 0, 0, 0, 0, 0], "9298": [0, 1, 0, -1, 0, 0, 0], "9289": [0, 0, 0, 0, 0, 0, 0]}
    mvdm_system = EnhancedMvdmSystem(speed_action, lane_action)
    rospy.spin()


if __name__ == '__main__':
    run()