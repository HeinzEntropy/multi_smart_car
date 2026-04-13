#!/usr/bin/env python
# multi_smart_car/scripts/formation_controller.py

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class FormationController:
    def __init__(self):
        rospy.init_node('formation_controller')

        # 编队参数
        self.formation_type = rospy.get_param('~formation_type', 'v_shape')
        self.formation_distance = rospy.get_param('~formation_distance', 1.5)
        self.num_robots = rospy.get_param('~num_robots', 3)

        # 机器人命名空间列表
        self.robot_namespaces = [f'robot{i+1}' for i in range(self.num_robots)]

        # 机器人位姿存储
        self.robot_poses = {}
        self.robot_velocities = {}

        # 订阅所有机器人的位姿（使用PoseWithCovarianceStamped）
        for ns in self.robot_namespaces:
            rospy.Subscriber(f'/{ns}/amcl_pose', PoseWithCovarianceStamped,
                           lambda msg, n=ns: self.amcl_pose_callback(msg, n))
            rospy.Subscriber(f'/{ns}/odom', Odometry,
                           lambda msg, n=ns: self.odom_callback(msg, n))

        # 为每个机器人发布目标点（使用PoseStamped，move_base期望这个类型）
        self.goal_publishers = {}
        for ns in self.robot_namespaces:
            self.goal_publishers[ns] = rospy.Publisher(
                f'/{ns}/move_base_simple/goal', PoseStamped, queue_size=10)

        # 主机器人目标点订阅（由rviz的2D Nav Goal发布，使用PoseStamped）
        rospy.Subscriber('/robot1/move_base_simple/goal', PoseStamped, self.master_goal_cb)

        # 存储主机器人的目标
        self.master_goal = None

        # 控制频率
        self.rate = rospy.Rate(5)  # 5Hz

        rospy.loginfo(f"Formation Controller Started")
        rospy.loginfo(f"Formation Type: {self.formation_type}")
        rospy.loginfo(f"Formation Distance: {self.formation_distance}m")
        rospy.loginfo(f"Number of Robots: {self.num_robots}")

    def amcl_pose_callback(self, msg, namespace):
        """AMCL位姿回调（PoseWithCovarianceStamped）"""
        # 提取位姿部分，转换为PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose  # 提取位姿（忽略协方差）

        self.robot_poses[namespace] = pose_stamped
        rospy.logdebug(f"Received pose for {namespace}: ({pose_stamped.pose.position.x:.2f}, {pose_stamped.pose.position.y:.2f})")

    def odom_callback(self, msg, namespace):
        """机器人速度回调"""
        self.robot_velocities[namespace] = msg.twist.twist

    def master_goal_cb(self, msg):
        """主机器人目标回调（PoseStamped）"""
        self.master_goal = msg
        rospy.loginfo(f"New formation goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def calculate_follower_goal(self, master_pose, follower_index):
        """
        根据编队类型计算从机器人的目标点

        参数:
            master_pose: 主机器人位姿（PoseStamped）
            follower_index: 从机器人索引（0, 1, 2...）

        返回:
            follower_pose: 从机器人的目标位姿（PoseStamped）
        """
        # 获取主机器人的位置和朝向
        x_m = master_pose.pose.position.x
        y_m = master_pose.pose.position.y
        q = master_pose.pose.orientation
        _, _, theta_m = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # 根据编队类型计算偏移
        if self.formation_type == 'line':
            # 纵队：所有机器人在主机器人后方
            offset_x = -(follower_index + 1) * self.formation_distance
            offset_y = 0.0
            offset_theta = 0.0

        elif self.formation_type == 'v_shape':
            # V字形：左右交替
            side = 1 if follower_index % 2 == 0 else -1
            row = (follower_index // 2) + 1
            offset_x = -row * self.formation_distance
            offset_y = side * row * self.formation_distance * 0.5
            offset_theta = 0.0

        elif self.formation_type == 'circle':
            # 圆形：围绕主机器人
            angle = 2 * np.pi * follower_index / self.num_robots
            offset_x = self.formation_distance * np.cos(angle)
            offset_y = self.formation_distance * np.sin(angle)
            offset_theta = angle + np.pi

        else:  # 默认：单列纵队
            offset_x = -(follower_index + 1) * self.formation_distance
            offset_y = 0.0
            offset_theta = 0.0

        # 将偏移转换到全局坐标系
        global_x = x_m + offset_x * np.cos(theta_m) - offset_y * np.sin(theta_m)
        global_y = y_m + offset_x * np.sin(theta_m) + offset_y * np.cos(theta_m)
        global_theta = theta_m + offset_theta

        # 创建目标位姿（PoseStamped）
        follower_pose = PoseStamped()
        follower_pose.header.stamp = rospy.Time.now()
        follower_pose.header.frame_id = "map"
        follower_pose.pose.position.x = global_x
        follower_pose.pose.position.y = global_y
        follower_pose.pose.position.z = 0.0

        # 将欧拉角转换为四元数
        q = quaternion_from_euler(0, 0, global_theta)
        follower_pose.pose.orientation.x = q[0]
        follower_pose.pose.orientation.y = q[1]
        follower_pose.pose.orientation.z = q[2]
        follower_pose.pose.orientation.w = q[3]

        return follower_pose

    def update_formation(self):
        """更新编队"""
        if self.master_goal is None:
            return

        # 获取主机器人的当前位姿
        master_namespace = self.robot_namespaces[0]
        if master_namespace not in self.robot_poses:
            rospy.logwarn_throttle(5, "Master robot pose not available")
            return

        master_pose = self.robot_poses[master_namespace]

        # 为主机器人发布目标（直接使用master_goal，已经是PoseStamped）
        self.goal_publishers[master_namespace].publish(self.master_goal)

        # 为每个从机器人计算并发布目标
        for i, namespace in enumerate(self.robot_namespaces[1:], start=1):
            follower_goal = self.calculate_follower_goal(self.master_goal, i - 1)
            self.goal_publishers[namespace].publish(follower_goal)

            rospy.logdebug(f"Published goal for {namespace}: "
                          f"({follower_goal.pose.position.x:.2f}, "
                          f"{follower_goal.pose.position.y:.2f})")

    def run(self):
        """主循环"""
        rospy.loginfo("Formation controller running...")

        while not rospy.is_shutdown():
            self.update_formation()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = FormationController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
