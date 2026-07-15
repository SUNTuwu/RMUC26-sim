#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _venv_python_prefix():
    venv_dir = os.environ.get("VIRTUAL_ENV")
    if not venv_dir:
        return None

    python_file = os.path.join(venv_dir, "bin", "python3")
    return python_file if os.path.exists(python_file) else None


def _workspace_bringup_file(relative_file):
    return os.path.join(os.getcwd(), "src", "sim_bringup", relative_file)


def _workspace_external_file(relative_file):
    return os.path.join(os.getcwd(), "src", "external", "RM2026-sentry-ws", relative_file)


def _prefer_existing_file(file_candidates):
    for candidate_file in file_candidates:
        if os.path.exists(candidate_file):
            return candidate_file
    return file_candidates[-1]


def _default_robot_type():
    return os.environ.get("ROBOT_TYPE", "sim_sentry_fold")


def _default_robot_description_xacro_file(robot_type):
    return _prefer_existing_file(
        [
            _workspace_external_file(
                os.path.join("src", "main_bringup", "urdf", f"{robot_type}.urdf.xacro")
            ),
            _workspace_bringup_file(os.path.join("urdf", f"{robot_type}.urdf.xacro")),
            _workspace_bringup_file(os.path.join("urdf", "sentry.urdf.xacro")),
        ]
    )


def _default_sim_config_file(robot_type):
    return _prefer_existing_file(
        [
            _workspace_bringup_file(os.path.join("config", f"sim_config_{robot_type}.yaml")),
            _workspace_bringup_file(os.path.join("config", "sim_config.yaml")),
        ]
    )


def generate_launch_description():
    ################################Start Launch configuration variables################################
    default_robot_type = _default_robot_type()
    default_sim_config_file = _default_sim_config_file(default_robot_type)
    default_robot_description_xacro_file = _default_robot_description_xacro_file(
        default_robot_type
    )

    declare_robot_type = DeclareLaunchArgument(
        "robot_type",
        default_value=default_robot_type,
        description="Robot type",
    )
    declare_sim_config_file = DeclareLaunchArgument(
        "sim_config_file",
        default_value=default_sim_config_file,
        description="Simulation parameter file",
    )
    declare_robot_description_xacro_file = DeclareLaunchArgument(
        "robot_description_xacro_file",
        default_value=default_robot_description_xacro_file,
        description="Robot xacro file",
    )
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )
    declare_enable_viewer = DeclareLaunchArgument(
        "enable_viewer",
        default_value="true",
        description="Enable MuJoCo viewer",
    )

    declare_parameters = GroupAction(
        [
            declare_robot_type,
            declare_sim_config_file,
            declare_robot_description_xacro_file,
            declare_use_sim_time,
            declare_enable_viewer,
        ],
        scoped=False,
    )
    ################################End Launch configuration variables################################

    sim_config_file = LaunchConfiguration("sim_config_file")
    robot_description_xacro_file = LaunchConfiguration("robot_description_xacro_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_viewer = LaunchConfiguration("enable_viewer")
    python_prefix = _venv_python_prefix()

    robot_description = ParameterValue(
        Command(["xacro ", robot_description_xacro_file]),
        value_type=str,
    )

    chassis_adapter_node = Node(
        package="sim_core",
        executable="chassis_adapter",
        name="chassis_adapter",
        prefix=python_prefix,
        parameters=[
            sim_config_file,
            {
                "use_sim_time": use_sim_time,
            },
        ],
        output="screen",
        emulate_tty=True,
    )

    imu_adapter_node = Node(
        package="sim_core",
        executable="imu_adapter",
        name="imu_adapter",
        prefix=python_prefix,
        parameters=[
            sim_config_file,
            {
                "use_sim_time": use_sim_time,
            },
        ],
        output="screen",
        emulate_tty=True,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
                "publish_frequency": 100.0,
            }
        ],
        output="screen",
    )

    sim_core_node = Node(
        package="sim_core",
        executable="sentry_sim_node",
        name="sentry_sim_node",
        prefix=python_prefix,
        parameters=[
            sim_config_file,
            {
                # 仿真核心只接收自身需要的最小入口参数，具体组件配置留在 yaml 内部解耦。
                "robot_description_xacro_path": robot_description_xacro_file,
                "use_sim_time": use_sim_time,
                "enable_viewer": enable_viewer,
            },
        ],
        output="screen",
        emulate_tty=True,
    )

    all_systems = LaunchDescription(
        [
            declare_parameters,
            chassis_adapter_node,
            imu_adapter_node,
            robot_state_publisher_node,
            sim_core_node,
        ]
    )

    return all_systems
