#!/usr/bin/env python3
"""Simplified test launch — no Xvfb, just the core nodes."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _venv_python_prefix():
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return None
    python_path = os.path.join(venv, "bin", "python3")
    return python_path if os.path.exists(python_path) else None


def generate_launch_description():
    bringup_pkg = get_package_share_directory('sim_bringup')
    desc_pkg = get_package_share_directory('sim_description')
    xacro_path = os.path.join(desc_pkg, 'urdf', 'sentry.urdf.xacro')
    params_file = os.path.join(bringup_pkg, 'config', 'sim_config.yaml')
    python_prefix = _venv_python_prefix()

    robot_desc = ParameterValue(
        Command(['xacro ', xacro_path]),
        value_type=str
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
        ),
        DeclareLaunchArgument(
            'enable_viewer', default_value='true',
        ),

        Node(
            package='sim_core',
            executable='chassis',
            name='chassis',
            prefix=python_prefix,
            parameters=[
                params_file,
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                },
            ],
            output='screen',
            emulate_tty=True,
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': robot_desc,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'publish_frequency': 100.0,
            }],
            output='screen',
        ),

        Node(
            package='sim_core',
            executable='sentry_sim_node',
            name='sentry_sim_node',
            prefix=python_prefix,
            parameters=[
                params_file,
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'enable_viewer': LaunchConfiguration('enable_viewer'),
                },
            ],
            output='screen',
            emulate_tty=True,
        ),
    ])
