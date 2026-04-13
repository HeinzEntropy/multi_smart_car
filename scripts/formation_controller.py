#!/usr/bin/env python
# multi_smart_car/scripts/formation_controller.py

import rospy
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf

class FormationController:
    def __init__(self):
        rospy.init_node('formation_controller')

        # 编队参数
        self.formation_type = rospy.get_param('~formation_type', 'v_shape')
        self.formation_distance = rospy.get_param('~formation_distance', 1.5)
        self.num_robots = rospy.get_param('~num_robots', 3)

        # 机器人参数
        self.robot_radius = rospy.get_param('~robot_radius', 0.175)  # 从你的footprint计算

        # 激光雷达参数（从你的配置文件获取）
        self.scan_angle_min = rospy.get_param('~scan_angle_min', -math.pi)
        self.scan_angle_max = rospy.get_param('~scan_angle_max', math.pi)
        self.scan_angle_increment = rospy.get_param('~scan_angle_increment', math.pi/36)  # 1度
        self.scan_range_min = rospy.get_param('~scan_range_min', 0.1)
        self.scan_range_max = rospy.get_param('~scan_range_max', 10.0)

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

        # 为每个机器人发布虚拟障碍物
        self.virtual_scan_publishers = {}
        for ns in self.robot_namespaces:
            self.virtual_scan_publishers[ns] = rospy.Publisher(
                f'/{ns}/virtual_obstacle_scan', LaserScan, queue_size=10)

        # 主机器人目标点订阅
        rospy.Subscriber('/robot1/move_base_simple/goal', PoseStamped, self.master_goal_cb)

        # 存储主机器人的目标
        self.master_goal = None

        # TF监听器
        self.tf_listener = tf.TransformListener()

        # 控制频率
        self.formation_rate = rospy.Rate(5)   # 编队控制频率 5Hz
        self.obstacle_rate = rospy.Rate(10)   # 虚拟障碍物发布频率 10Hz

        rospy.loginfo("Formation Controller with Obstacles Started")
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

    def generate_virtual_obstacle_scan(self, target_robot_namespace):
        """
        为指定机器人生成虚拟障碍物扫描（包含其他所有机器人）

        参数:
            target_robot_namespace: 目标机器人命名空间

        返回:
            scan: LaserScan消息
        """
        # 检查目标机器人位姿是否可用
        if target_robot_namespace not in self.robot_poses:
            return None

        # 获取目标机器人位姿
        target_pose = self.robot_poses[target_robot_namespace]
        target_x = target_pose.pose.position.x
        target_y = target_pose.pose.position.y
        q = target_pose.pose.orientation
        _, _, target_theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # 创建激光扫描消息
        scan = LaserScan()
        scan.header.stamp = rospy.Time.now()
        scan.header.frame_id = f"{target_robot_namespace}_laser_mount"
        scan.angle_min = self.scan_angle_min
        scan.angle_max = self.scan_angle_max
        scan.angle_increment = self.scan_angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = self.scan_range_min
        scan.range_max = self.scan_range_max

        # 初始化所有距离为无穷大
        num_readings = int((scan.angle_max - scan.angle_min) / scan.angle_increment)
        scan.ranges = [self.scan_range_max] * num_readings
        scan.intensities = [0.0] * num_readings

        # 遍历所有其他机器人，生成虚拟障碍物
        for other_namespace in self.robot_namespaces:
            # 跳过自己
            if other_namespace == target_robot_namespace:
                continue

            # 检查其他机器人位姿是否可用
            if other_namespace not in self.robot_poses:
                continue

            # 获取其他机器人位姿
            other_pose = self.robot_poses[other_namespace]
            other_x = other_pose.pose.position.x
            other_y = other_pose.pose.position.y

            # 计算相对位置（从目标机器人到其他机器人）
            dx = other_x - target_x
            dy = other_y - target_y

            # 转换到目标机器人的局部坐标系
            local_x = dx * math.cos(target_theta) + dy * math.sin(target_theta)
            local_y = -dx * math.sin(target_theta) + dy * math.cos(target_theta)

            # 计算距离和角度
            distance = math.sqrt(local_x**2 + local_y**2)
            angle = math.atan2(local_y, local_x)

            # 检查是否在扫描范围内
            if self.scan_angle_min <= angle <= self.scan_angle_max:
                # 将角度转换为索引
                angle_index = int((angle - self.scan_angle_min) / self.scan_angle_increment)

                if 0 <= angle_index < num_readings:
                    # 考虑机器人半径（障碍物在机器人边缘）
                    obstacle_distance = distance - self.robot_radius

                    if obstacle_distance > self.scan_range_min:
                        # 如果这个角度已经有更近的障碍物，保留更近的
                        if obstacle_distance < scan.ranges[angle_index]:
                            scan.ranges[angle_index] = obstacle_distance

        return scan

    def update_formation(self):
        """更新编队（发布目标点）"""
        if self.master_goal is None:
            return

        # 为主机器人发布目标
        master_namespace = self.robot_namespaces[0]
        if master_namespace not in self.robot_poses:
            rospy.logwarn_throttle(5, "Master robot pose not available")
            return

        self.goal_publishers[master_namespace].publish(self.master_goal)

        # 为每个从机器人计算并发布目标
        for i, namespace in enumerate(self.robot_namespaces[1:], start=1):
            follower_goal = self.calculate_follower_goal(self.master_goal, i - 1)
            self.goal_publishers[namespace].publish(follower_goal)

    def publish_virtual_obstacles(self):
        """为所有机器人发布虚拟障碍物"""
        for namespace in self.robot_namespaces:
            # 生成虚拟障碍物扫描
            virtual_scan = self.generate_virtual_obstacle_scan(namespace)

            if virtual_scan is not None:
                # 发布虚拟障碍物
                self.virtual_scan_publishers[namespace].publish(virtual_scan)

                # 调试信息（可选）
                obstacle_count = sum(1 for r in virtual_scan.ranges if r != float('inf'))
                if obstacle_count > 0:
                    rospy.logdebug(f"Published {obstacle_count} virtual obstacles for {namespace}")

    def run(self):
        """主循环"""
        rospy.loginfo("Formation controller with obstacles running...")

        # 初始化定时器
        last_formation_update = rospy.Time.now()
        last_obstacle_update = rospy.Time.now()

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()

            # 更新编队目标点（5Hz）
            if (current_time - last_formation_update).to_sec() >= 0.2:  # 5Hz
                self.update_formation()
                last_formation_update = current_time

            # 发布虚拟障碍物（10Hz）
            if (current_time - last_obstacle_update).to_sec() >= 0.1:  # 10Hz
                self.publish_virtual_obstacles()
                last_obstacle_update = current_time

            # 短暂休眠
            rospy.sleep(0.01)

if __name__ == '__main__':
    try:
        controller = FormationController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
