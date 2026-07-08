import os

import yaml
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node


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


def _default_sim_config_path(robot_type):
    return _first_existing_path(
        [
            _workspace_bringup_file(os.path.join("config", f"sim_config_{robot_type}.yaml")),
            _workspace_bringup_file(os.path.join("config", "sim_config.yaml")),
        ]
    )


def _default_robot_xacro_path(robot_type):
    return _first_existing_path(
        [
            _workspace_external_file(os.path.join("src", "main_bringup", "urdf", f"{robot_type}.urdf.xacro")),
            _workspace_bringup_file(os.path.join("urdf", f"{robot_type}.urdf.xacro")),
            _workspace_bringup_file(os.path.join("urdf", "sentry.urdf.xacro")),
        ]
    )


def _load_sim_lidar_topics(sim_config_path):
    with open(sim_config_path, "r", encoding="ascii") as config_file:
        sim_config = yaml.safe_load(config_file) or {}

    sim_params = sim_config.get("sentry_sim_node", {}).get("ros__parameters", {})
    left_ip = str(sim_params.get("left_lidar_ip", "192.168.10.4"))
    right_ip = str(sim_params.get("right_lidar_ip", "192.168.10.5"))
    return [
        f"/livox/lidar_{left_ip.replace('.', '_')}/pointcloud",
        f"/livox/lidar_{right_ip.replace('.', '_')}/pointcloud",
    ]


def _create_pointcloud_preprocessor_node(context, *, pointcloud_preprocessor_config, use_sim_time):
    sim_config_file = LaunchConfiguration("sim_config_file").perform(context)
    sim_lidar_topics = _load_sim_lidar_topics(sim_config_file)
    return [
        Node(
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
    ]


def generate_launch_description():
    default_robot_type = _default_robot_type()
    default_sim_config_path = _default_sim_config_path(default_robot_type)
    default_robot_xacro_path = _default_robot_xacro_path(default_robot_type)
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map_file")
    robot_type = LaunchConfiguration("robot_type")
    use_lio_rviz = LaunchConfiguration("use_lio_rviz")
    localization = LaunchConfiguration("localization")
    segmentation = LaunchConfiguration("segmentation")
    lio = LaunchConfiguration("lio")
    launch_sim = LaunchConfiguration("launch_sim")
    enable_viewer = LaunchConfiguration("enable_viewer")
    dll_df_output_prefix = LaunchConfiguration("dll_df_output_prefix")
    container_name = LaunchConfiguration("container_name")
    use_nav_rviz = LaunchConfiguration("use_nav_rviz")
    sim_config_file = LaunchConfiguration("sim_config_file")
    robot_description_xacro_path = LaunchConfiguration("robot_description_xacro_path")
    python_prefix = _venv_python_prefix()

    sim_bringup_pkg = get_package_share_directory("sim_bringup")
    main_bringup_pkg = get_package_share_directory("main_bringup")
    io_bringup_pkg = get_package_share_directory("io_bringup")
    state_estimation_bringup_pkg = get_package_share_directory("state_estimation_bringup")
    mapping_bringup_pkg = get_package_share_directory("mapping_bringup")
    nav_bringup_pkg = get_package_share_directory("nav_bringup")
    pointcloud_preprocessor_config = PathJoinSubstitution(
        [io_bringup_pkg, "config", "pointcloud_preprocessor.yaml"]
    )
    serial_config_file = PathJoinSubstitution([io_bringup_pkg, "config", "ch343.yaml"])
    prior_map_image_path = PythonExpression([
        "'' if '", map_file,
        "' == 'none' else '",
        main_bringup_pkg,
        "/map/' + '", map_file, "' + '.png'",
    ])
    pointcloud_preprocessor_node = OpaqueFunction(
        function=_create_pointcloud_preprocessor_node,
        kwargs={
            "pointcloud_preprocessor_config": pointcloud_preprocessor_config,
            "use_sim_time": use_sim_time,
        },
    )

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([sim_bringup_pkg, "launch", "sim.launch.py"])
        ),
        launch_arguments={
            "robot_type": robot_type,
            "sim_config_file": sim_config_file,
            "robot_description_xacro_path": robot_description_xacro_path,
            "use_sim_time": use_sim_time,
            "enable_viewer": enable_viewer,
        }.items(),
        condition=IfCondition(launch_sim),
    )

    disable_localization = SetLaunchConfiguration(
        "localization",
        "none",
        condition=LaunchConfigurationEquals("map_file", "none"),
    )

    state_estimation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [state_estimation_bringup_pkg, "launch", "state_estimation.launch.py"]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "map_file": map_file,
            "use_lio_rviz": use_lio_rviz,
            "localization": localization,
            "segmentation": segmentation,
            "lio": lio,
            "dll_df_output_prefix": dll_df_output_prefix,
            "robot_type": robot_type,
        }.items(),
    )

    mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([mapping_bringup_pkg, "launch", "mapping.launch.py"])
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "prior_map_image_path": prior_map_image_path,
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

    navigation_group = GroupAction(
        [
            pointcloud_preprocessor_node, # 把多路雷达 PointCloud2 按 TF 变换到统一坐标系后做近点过滤与缓存拼接，输出给建图/导航用的合并点云。
            state_estimation_launch, #当前对应laser mapping即point lio + pub静态map->odom的tf
            mapping_launch, #对应dynamic rog map
            nav_launch, #对应nav2服务器
            nav_feedback_adapter, #把 point_lio / 雷达侧输出的 云台里程计 /gimbal_Odometry ，转换成导航和外部接口更想要的 底盘里程计 /Odometry
            nav_serial_plugin_node, # 把 Nav2 产生的路径、速度和控制指令整理成更接近串口侧/上层业务需要的处理后反馈话题, 包括产出/cmd_vel_processed
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("launch_sim", default_value="true"),
        DeclareLaunchArgument("enable_viewer", default_value="true"),
        DeclareLaunchArgument("map_file", default_value="none"),
        DeclareLaunchArgument("robot_type", default_value=default_robot_type),
        DeclareLaunchArgument("sim_config_file", default_value=default_sim_config_path),
        DeclareLaunchArgument("robot_description_xacro_path", default_value=default_robot_xacro_path),
        DeclareLaunchArgument("use_lio_rviz", default_value="false"),
        DeclareLaunchArgument("localization", default_value="none"),
        DeclareLaunchArgument("segmentation", default_value="none"),
        DeclareLaunchArgument("lio", default_value="pointlio"),
        DeclareLaunchArgument(
            "dll_df_output_prefix",
            default_value="/root/sentry_sim/src/external/RM2026-sentry-ws/src/state_estimation/state_estimation_bringup/DF",
        ),
        DeclareLaunchArgument("container_name", default_value="sim3d_nav_container"),
        DeclareLaunchArgument("use_nav_rviz", default_value="true"),
        disable_localization,
        sim_launch,
        navigation_group,
        nav_rviz,
    ])
