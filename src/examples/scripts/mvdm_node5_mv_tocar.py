#!/usr/bin/env python3
import rospy
import numpy as np
from datetime import datetime
from std_msgs.msg import Float32

params = {
    "robot_ids": ["9293"],
    # "robot_ids": ["9293", "9289","9298"],
    "control_period": 0.1,
    "expand_num": 40,
}

class EnhancedMvdmSystem:
    def __init__(self, speed_action, lane_action):

        self.robot_ids = params["robot_ids"]
        self.speed_expand = {id: [0] * 10 + np.repeat(speed_action[id], params["expand_num"]).tolist() for id in
                             self.robot_ids}
        self.lane_expand = {id: [0] * 10 + np.repeat(lane_action[id], params["expand_num"]).tolist() for id in
                            self.robot_ids}
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
    speed_action = {'9293': [0.5, 0.5, 0.5, 0.5, 0.5], "9298": [0.5, 0.5, 0.5, 0.5, 0.5], "9289": [0.5, 0.5, 0.5, 0.5, 0.5]}
    lane_action = {"9293": [0, 0, 0, 0, 0], "9298": [0, 0, 0, 0, 0], "9289": [0, 0, 0, 0, 0]}
    mvdm_system = EnhancedMvdmSystem(speed_action, lane_action)
    rospy.spin()


if __name__ == '__main__':
    run()