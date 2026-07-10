#!/usr/bin/env python3
"""Launch only keyboard_test for command-path verification."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _venv_python_prefix():
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return None
    python_path = os.path.join(venv, "bin", "python3")
    return python_path if os.path.exists(python_path) else None


def _workspace_bringup_file(relative_file):
    return os.path.join(os.getcwd(), "src", "sim_bringup", relative_file)


def generate_launch_description():
    params_file = _workspace_bringup_file(os.path.join("config", "sim_config.yaml"))
    python_prefix = _venv_python_prefix()

    return LaunchDescription([
        DeclareLaunchArgument(
            "sim_config_file",
            default_value=params_file,
            description="Keyboard node parameter file.",
        ),
        Node(
            package="sim_bringup",
            executable="keyboard_control",
            name="keyboard_test",
            prefix=python_prefix,
            parameters=[
                LaunchConfiguration("sim_config_file"),
            ],
            output="screen",
            emulate_tty=True,
        ),
    ])
