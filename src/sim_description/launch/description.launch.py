#!/usr/bin/env python3
"""
Launch robot_state_publisher for sim_description.

Converts xacro -> URDF and starts robot_state_publisher.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('sim_description')
    xacro_path = os.path.join(pkg_dir, 'urdf', 'sentry.urdf.xacro')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation time'
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': ['$(command ',
                    'xacro ', xacro_path, ')'],
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
            output='screen',
        ),
    ])
