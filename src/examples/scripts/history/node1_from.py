#!/usr/bin/env python3
import rospy
import time
from pylimo import limo
import pylimo.limomsg as limomsg
from std_msgs.msg import Float32
limo=limo.LIMO()
limo.EnableCommand()


class Mvdm_System:
    def __init__(self):

        self.angular = 0
        self.vel = 0
        self.min_angular = -0.6
        self.max_angular = 0.6


        self.vel_sub = rospy.Subscriber('/limo9289/vel', Float32, self.vel_callback)
        self.ang_sub = rospy.Subscriber('/limo9289/angular', Float32, self.ang_callback)

        # 发布指令
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.cbControl)

    def vel_callback(self, vel):
        self.vel = vel.data
    def ang_callback(self, ang):
        self.angular = ang.data
    def cbControl(self, event):
        start_time_img = time.time()
        exc_angular = max(self.angular, self.min_angular)
        exc_angular = min(exc_angular, self.max_angular)
        print(self.angular, self.vel)
        limo.SetMotionCommand(self.vel,-0.01,0,exc_angular/2)
        end_time_img = time.time()
        print(f"control耗时: {end_time_img - start_time_img:.2f} ")
        return


def run():
    rospy.init_node('car_9289', anonymous=False)
    #limo.SetMotionCommand(0,0,0,0)
    mvdm_system = Mvdm_System()
    rospy.spin()


if __name__ == '__main__':
    run()