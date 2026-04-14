#!/usr/bin/env python
# multi_smart_car/scripts/formation_controller.py

import rospy
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf

# 用于 ProxemicLayer
from people_msgs.msg import PositionMeasurementArray, PositionMeasurement

class FormationController:
    def __init__(self):
        rospy.init_node('formation_controller')

        # 编队参数
        self.formation_type = rospy.get_param('~formation_type', 'v_shape')
        self.formation_distance = rospy.get_param('~formation_distance', 1.5)
        self.num_robots = rospy.get_param('~num_robots', 3)

        # 机器人参数
        self.robot_radius = rospy.get_param('~robot_radius', 0.175)

        # 机器人命名空间列表
        self.robot_namespaces = [f'robot{i+1}' for i in range(self.num_robots)]

        # 机器人位姿存储
        self.robot_poses = {}
        self.robot_velocities = {}

        # 订阅所有机器人的位姿
        for ns in self.robot_namespaces:
            rospy.Subscriber(f'/{ns}/amcl_pose', PoseWithCovarianceStamped,
                           lambda msg, n=ns: self.amcl_pose_callback(msg, n))
            rospy.Subscriber(f'/{ns}/odom', Odometry,
                           lambda msg, n=ns: self.odom_callback(msg, n))

        # 为每个机器人发布目标点
        self.goal_publishers = {}
        for ns in self.robot_namespaces:
            self.goal_publishers[ns] = rospy.Publisher(
                f'/{ns}/move_base_simple/goal', PoseStamped, queue_size=10)

        # 为每个机器人发布 people 消息（给 ProxemicLayer）
        self.people_publishers = {}
        for ns in self.robot_namespaces:
            self.people_publishers[ns] = rospy.Publisher(
                f'/{ns}/people', PositionMeasurementArray, queue_size=10)

        # 主机器人目标点订阅
        rospy.Subscriber('/robot1/move_base_simple/goal', PoseStamped, self.master_goal_cb)

        # 存储主机器人的目标
        self.master_goal = None

        # TF监听器
        self.tf_listener = tf.TransformListener()

        rospy.loginfo("Formation Controller Started")
        rospy.loginfo(f"Formation Type: {self.formation_type}")
        rospy.loginfo(f"Formation Distance: {self.formation_distance}m")
        rospy.loginfo(f"Number of Robots: {self.num_robots}")
        rospy.loginfo(f"Robot Radius: {self.robot_radius}m")

    def amcl_pose_callback(self, msg, namespace):
        """AMCL位姿回调"""
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.robot_poses[namespace] = pose_stamped

    def odom_callback(self, msg, namespace):
        """机器人速度回调"""
        self.robot_velocities[namespace] = msg.twist.twist

    def master_goal_cb(self, msg):
        """主机器人目标回调"""
        self.master_goal = msg
        rospy.loginfo(f"New formation goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def calculate_follower_goal(self, master_pose, follower_index):
        """计算从机器人的目标点"""
        x_m = master_pose.pose.position.x
        y_m = master_pose.pose.position.y
        q = master_pose.pose.orientation
        _, _, theta_m = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if self.formation_type == 'line':
            offset_x = -(follower_index + 1) * self.formation_distance
            offset_y = 0.0
            offset_theta = 0.0
        elif self.formation_type == 'v_shape':
            side = 1 if follower_index % 2 == 0 else -1
            row = (follower_index // 2) + 1
            offset_x = -row * self.formation_distance
            offset_y = side * row * self.formation_distance * 0.5
            offset_theta = 0.0
        elif self.formation_type == 'circle':
            angle = 2 * np.pi * follower_index / self.num_robots
            offset_x = self.formation_distance * np.cos(angle)
            offset_y = self.formation_distance * np.sin(angle)
            offset_theta = angle + np.pi
        else:
            offset_x = -(follower_index + 1) * self.formation_distance
            offset_y = 0.0
            offset_theta = 0.0

        global_x = x_m + offset_x * np.cos(theta_m) - offset_y * np.sin(theta_m)
        global_y = y_m + offset_x * np.sin(theta_m) + offset_y * np.cos(theta_m)
        global_theta = theta_m + offset_theta

        follower_pose = PoseStamped()
        follower_pose.header.stamp = rospy.Time.now()
        follower_pose.header.frame_id = "map"
        follower_pose.pose.position.x = global_x
        follower_pose.pose.position.y = global_y
        follower_pose.pose.position.z = 0.0

        q = quaternion_from_euler(0, 0, global_theta)
        follower_pose.pose.orientation.x = q[0]
        follower_pose.pose.orientation.y = q[1]
        follower_pose.pose.orientation.z = q[2]
        follower_pose.pose.orientation.w = q[3]

        return follower_pose

    def update_formation(self):
        """更新编队（发布目标点）"""
        if self.master_goal is None:
            return

        master_namespace = self.robot_namespaces[0]
        if master_namespace not in self.robot_poses:
            rospy.logwarn_throttle(5, "Master robot pose not available")
            return

        self.goal_publishers[master_namespace].publish(self.master_goal)

        for i, namespace in enumerate(self.robot_namespaces[1:], start=1):
            follower_goal = self.calculate_follower_goal(self.master_goal, i - 1)
            self.goal_publishers[namespace].publish(follower_goal)

    def publish_teammates_as_people(self):
        """为每个机器人发布队友位置，供 ProxemicLayer 使用"""
        for target_ns in self.robot_namespaces:
            if target_ns not in self.robot_poses:
                continue

            people_array = PositionMeasurementArray()
            people_array.header.stamp = rospy.Time.now()
            people_array.header.frame_id = "map"

            for other_ns in self.robot_namespaces:
                if other_ns == target_ns:
                    continue
                if other_ns not in self.robot_poses:
                    continue

                other_pose = self.robot_poses[other_ns]

                person = PositionMeasurement()
                person.header.stamp = rospy.Time.now()
                person.header.frame_id = "map"
                person.name = other_ns
                person.object_id = other_ns
                person.pos = other_pose.pose.position
                person.covariance = [0.1, 0.0, 0.0,
                                     0.0, 0.1, 0.0,
                                     0.0, 0.0, 0.1]
                person.reliability = 0.9

                people_array.people.append(person)

            if len(people_array.people) > 0:
                self.people_publishers[target_ns].publish(people_array)

    def run(self):
        """主循环"""
        rospy.loginfo("Formation controller running...")

        last_formation_update = rospy.Time.now()
        last_people_update = rospy.Time.now()

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()

            # 更新编队目标点（5Hz）
            if (current_time - last_formation_update).to_sec() >= 0.2:
                self.update_formation()
                last_formation_update = current_time

            # 发布 people 消息（10Hz）
            if (current_time - last_people_update).to_sec() >= 0.1:
                self.publish_teammates_as_people()
                last_people_update = current_time

            rospy.sleep(0.01)

if __name__ == '__main__':
    try:
        controller = FormationController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
