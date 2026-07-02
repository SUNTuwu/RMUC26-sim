import os

import yaml
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node


def _venv_python_prefix():
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return None
    python_path = os.path.join(venv, "bin", "python3")
    return python_path if os.path.exists(python_path) else None


def _load_sim_lidar_topics():
    sim_bringup_pkg = get_package_share_directory("sim_bringup")
    sim_config_path = os.path.join(sim_bringup_pkg, "config", "sim_config.yaml")
    with open(sim_config_path, "r", encoding="ascii") as config_file:
        sim_config = yaml.safe_load(config_file) or {}

    sim_params = sim_config.get("sentry_sim_node", {}).get("ros__parameters", {})
    left_ip = str(sim_params.get("left_lidar_ip", "192.168.10.5"))
    right_ip = str(sim_params.get("right_lidar_ip", "192.168.10.4"))
    return [
        f"/livox/lidar_{left_ip.replace('.', '_')}/pointcloud",
        f"/livox/lidar_{right_ip.replace('.', '_')}/pointcloud",
    ]


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map_file")
    robot_type = LaunchConfiguration("robot_type")
    container_name = LaunchConfiguration("container_name")
    use_nav_rviz = LaunchConfiguration("use_nav_rviz")
    python_prefix = _venv_python_prefix()

    sim_bringup_pkg = get_package_share_directory("sim_bringup")
    main_bringup_pkg = get_package_share_directory("main_bringup")
    io_bringup_pkg = get_package_share_directory("io_bringup")
    mapping_bringup_pkg = get_package_share_directory("mapping_bringup")
    nav_bringup_pkg = get_package_share_directory("nav_bringup")
    pointcloud_preprocessor_config = PathJoinSubstitution(
        [io_bringup_pkg, "config", "pointcloud_preprocessor.yaml"]
    )
    serial_config_file = PathJoinSubstitution([io_bringup_pkg, "config", "ch343.yaml"])
    sim_lidar_topics = _load_sim_lidar_topics()

    prior_map_image_path = PythonExpression([
        "'' if '", map_file,
        "' == 'none' else '",
        main_bringup_pkg,
        "/map/' + '", map_file, "' + '.png'",
    ])
    pointcloud_preprocessor_node = Node(
        package="pointcloud_preprocessor",
        executable="pointcloud_preprocessor_node",
        name="pointcloud_preprocessor_node",
        output="screen",
        parameters=[
            pointcloud_preprocessor_config,
            {
                "use_sim_time": use_sim_time,
                "pointcloud_topics": sim_lidar_topics,
            },
        ],
    )

    mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([mapping_bringup_pkg, "launch", "mapping.launch.py"])
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "prior_map_image_path": prior_map_image_path,
            # "fixed_frame": "odom",
        }.items(),
    )

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([nav_bringup_pkg, "launch", "nav.launch.py"])
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "map_file": map_file,
            "robot_type": robot_type,
            "container_name": container_name,
        }.items(),
    )

    map_to_odom_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_static_tf",
        output="both",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
    )

    nav_feedback_adapter = Node(
        package="sim_core",
        executable="nav_feedback_adapter",
        name="nav_feedback_adapter",
        prefix=python_prefix,
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    nav_serial_plugin_node = Node(
        package="nav_serial_driver_ch343",
        executable="nav_serial_plugin_node",
        name="nav_serial_plugin_node",
        output="both",
        emulate_tty=True,
        parameters=[serial_config_file, {"use_sim_time": use_sim_time}],
        ros_arguments=["--ros-args", "--log-level", "nav_serial_plugin_node:=INFO"],
    )

    nav_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="nav_rviz2",
        arguments=[
            "-d",
            PathJoinSubstitution([sim_bringup_pkg, "rviz", "sim3d_visualization.rviz"]),
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        output="both",
        condition=IfCondition(use_nav_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("map_file", default_value="none"),
        DeclareLaunchArgument("robot_type", default_value="26_sentry_tall"),
        DeclareLaunchArgument("container_name", default_value="sim3d_nav_container"),
        DeclareLaunchArgument("use_nav_rviz", default_value="true"),
        pointcloud_preprocessor_node,
        mapping_launch,
        nav_launch,
        map_to_odom_static_tf,
        nav_feedback_adapter,
        nav_serial_plugin_node,
        nav_rviz,
    ])
