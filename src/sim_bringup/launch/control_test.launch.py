#!/usr/bin/env python3
"""Launch keyboard_test and chassis nodes for command-path verification."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_pkg = get_package_share_directory("sim_bringup")
    params_file = os.path.join(bringup_pkg, "config", "sim_config.yaml")

    return LaunchDescription([
        DeclareLaunchArgument(
            "scripted_keys",
            default_value="",
            description="Comma-separated key sequence for non-interactive keyboard node tests.",
        ),
        DeclareLaunchArgument(
            "keyboard_backend",
            default_value="auto",
            description="Keyboard backend: auto, evdev, pynput, or tty.",
        ),
        DeclareLaunchArgument(
            "keyboard_device_path",
            default_value="",
            description="Optional /dev/input/eventX path for evdev backend.",
        ),
        Node(
            package="sim_core",
            executable="keyboard_test",
            name="keyboard_test",
            parameters=[
                params_file,
                {
                    "keyboard_backend": LaunchConfiguration("keyboard_backend"),
                    "keyboard_device_path": LaunchConfiguration("keyboard_device_path"),
                    "scripted_keys": LaunchConfiguration("scripted_keys"),
                },
            ],
            output="screen",
            emulate_tty=True,
        ),
        Node(
            package="sim_core",
            executable="chassis",
            name="chassis",
            parameters=[params_file],
            output="screen",
            emulate_tty=True,
        ),
    ])
