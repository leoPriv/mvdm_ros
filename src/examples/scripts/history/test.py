import rospy
from std_msgs.msg import Float32


def main():
    rospy.init_node('test_pub')
    pub = rospy.Publisher('/limo9289/angular', Float32, queue_size=10)
    rate = rospy.Rate(0.1)  # 1Hz = 每秒1次

    toggle = True  # 用于切换正负值的标志位

    while not rospy.is_shutdown():
        # 交替发布 +0.3 和 -0.3
        if toggle:
            data = 0.3
        else:
            print("--------")
            data = -0.3

        pub.publish(data)
        rospy.loginfo(f"Published: {data} to /limo9289/angular")

        toggle = not toggle  # 切换标志位
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass