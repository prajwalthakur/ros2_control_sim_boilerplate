from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import LifecycleNode
from launch_ros.actions import Node
def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('mppi_planner'),'config')
    rviz_config_dir = os.path.join(config_dir,'navigation.rviz')
    costmap_params = os.path.join(config_dir,'tb3_nav2_params.yaml')
    map_yaml = os.path.join(config_dir,'tb3_map.yaml')
    #custom_world_path = os.path.join(get_package_share_directory('mppi_planner'), 'worlds', 'turtlebot3_world.world')
    # Xacro-based TurtleBot3 URDF
    tb3_desc_pkg = get_package_share_directory('turtlebot3_description')
    urdf_file = os.path.join(tb3_desc_pkg, 'urdf', 'turtlebot3_burger.urdf.xacro')
    robot_description = Command(['xacro ', urdf_file])
    
    gazebo_launch_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'launch',
        'turtlebot3_world.launch.py'
    )
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock'
    )
    map_server = LifecycleNode(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace='',
        output='screen',
        parameters=[
            {'yaml_filename': map_yaml},
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
    )   
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace='',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
    )
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom',
        namespace='',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )
    local_costmap = LifecycleNode(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='local_costmap',
        namespace='',
        output='screen',
        parameters=[costmap_params, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )
    global_costmap = LifecycleNode(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='global_costmap',
        namespace='',
        output='screen',
        parameters=[costmap_params, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    # 4) Lifecycle manager that handles map + costmaps
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        namespace='',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'autostart': True},
            {'node_names': ['map_server', 'local_costmap', 'global_costmap']},
        ],
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_dir],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_path),
        launch_arguments={
            'TURTLEBOT3_MODEL': 'burger',
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items(),
    )

    return LaunchDescription([
        declare_use_sim_time,
        gazebo,
        map_server,
        robot_state_pub,
        static_tf,
        local_costmap,
        global_costmap,
        lifecycle_manager,
        rviz_node,
    ])