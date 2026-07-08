#!/usr/bin/env python3
"""Simplified test launch — no Xvfb, just the core nodes."""
import os
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


def _workspace_bringup_file(relative_path):
    return os.path.join(os.getcwd(), "src", "sim_bringup", relative_path)


def _workspace_external_file(relative_path):
    return os.path.join(os.getcwd(), "src", "external", "RM2026-sentry-ws", relative_path)


def _first_existing_path(candidates):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def _default_robot_type():
    return os.environ.get("ROBOT_TYPE", "sim_sentry_fold")


def _default_robot_xacro_path(robot_type):
    return _first_existing_path(
        [
            _workspace_external_file(os.path.join("src", "main_bringup", "urdf", f"{robot_type}.urdf.xacro")),
            _workspace_bringup_file(os.path.join("urdf", f"{robot_type}.urdf.xacro")),
            _workspace_bringup_file(os.path.join("urdf", "sentry.urdf.xacro")),
        ]
    )


def _default_sim_config_path(robot_type):
    return _first_existing_path(
        [
            _workspace_bringup_file(os.path.join("config", f"sim_config_{robot_type}.yaml")),
            _workspace_bringup_file(os.path.join("config", "sim_config.yaml")),
        ]
    )


def generate_launch_description():
    default_robot_type = _default_robot_type()
    xacro_path = _default_robot_xacro_path(default_robot_type)
    params_file = _default_sim_config_path(default_robot_type)
    python_prefix = _venv_python_prefix()
    robot_type = LaunchConfiguration("robot_type")
    sim_config_file = LaunchConfiguration("sim_config_file")
    robot_description_xacro_path = LaunchConfiguration("robot_description_xacro_path")

    robot_desc = ParameterValue(
        Command(['xacro ', robot_description_xacro_path]),
        value_type=str
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_type", default_value=default_robot_type,
        ),
        DeclareLaunchArgument(
            "sim_config_file", default_value=params_file,
        ),
        DeclareLaunchArgument(
            "robot_description_xacro_path", default_value=xacro_path,
        ),
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
                sim_config_file,
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
                sim_config_file,
                {
                    "robot_description_xacro_path": robot_description_xacro_path,
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'enable_viewer': LaunchConfiguration('enable_viewer'),
                },
            ],
            output='screen',
            emulate_tty=True,
        ),
    ])
