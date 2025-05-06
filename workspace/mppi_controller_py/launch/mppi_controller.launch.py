from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='mppi_controller_py',
             executable='mppi_node')
    ])
