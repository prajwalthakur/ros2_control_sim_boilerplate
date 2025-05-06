#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
import math
from nav_msgs.msg import Odometry
from mppi_class import MPPI
import jax
import jax.numpy as jnp

import yaml
with open('src/mppi_planner/config/sim_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
seed            = cfg['seed']

class SimplePlanner(Node):
    def __init__(self):
        super().__init__('mppi_planner_node')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.goal = jnp.array([3,3])
        self.init = False
        self.current_pose = None
        timer_period = 0.1  # seconds
        self.control_timer = self.create_timer(timer_period, self.control_cb)

    def odom_callback(self, msg: Odometry):
        """Callback to handle incoming odometry messages."""
        current_pose = msg.pose.pose
        self.get_logger().debug(
            f'Received odom: position=({current_pose.position.x:.2f}, '
            f'{current_pose.position.y:.2f})'
        )
        self.current_pose  = jnp.array([current_pose.position.x,current_pose.position.y])
        key = jax.random.PRNGKey(seed)
        mppi_key,goal_key  = jax.random.split(key, 2)
        if self.init is False:
            start =  self.current_pose
            self.MppiObj  = MPPI(start,self.goal,mppi_key)
            self.init = True
    def control_cb(self):
        twist = Twist()
        if self.init is False:
            optimal_control, X_optimal_seq,X_rollout = self.MppiObj.compute_control(self.current_pose)
            twist.linear.x = optimal_control[0][0]
            twist.angular.z = optimal_control[1][0]
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info(f'Published cmd_vel: linear.x={twist.linear.x:.2f},angular.z={twist.angular.z:.2f}')
    
        

def main():
    rclpy.init()
    node = SimplePlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
