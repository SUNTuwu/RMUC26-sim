#!/usr/bin/env python3
"""
sentry_sim_node.py — MuJoCo simulation bridge for sentry robot.

Publishes:
  /livox/lidar_192_168_10_5/pointcloud    PointCloud2 (left LiDAR, 10 Hz)
  /livox/lidar_192_168_10_4/pointcloud    PointCloud2 (right LiDAR, 10 Hz)
  /livox/imu_192_168_10_5      Imu (from left LiDAR's built-in IMU, 200 Hz)
  /livox/imu_192_168_10_4      Imu (from right LiDAR's built-in IMU, 200 Hz)
  /Odometry                    Odometry (50 Hz)
  /tf                          TF broadcast
  /joint_states                JointState

Subscribes:
  /cmd_vel_keyboard            Twist
                               linear.x / linear.y: planar velocity in gimbal frame
                               linear.z: keyboard small gyro trigger (>0.5 toggles spin mode)
                               angular.z: gimbal yaw rate command
  /cmd_vel_processed           Twist
                               linear.x / linear.y: planar velocity in base_link frame
                               angular.z: chassis yaw rate command
"""

import importlib
import json
import math
import os
import re
import time
import threading
import urllib.request
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time

import geometry_msgs.msg
import nav_msgs.msg
import rosgraph_msgs.msg
import sensor_msgs.msg
import std_msgs.msg
import tf2_ros
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import TransformStamped

import mujoco
from mujoco_lidar import MjLidarWrapper
from mujoco_lidar.scan_gen import LivoxGenerator


# ── Robot geometry (from 26_sentry_tunnel.urdf.xacro) ──────────────────────
# main_gimbal_link → left_livox_frame:  xyz=(0, 0.212, -0.051)
# main_gimbal_link → right_livox_frame: xyz=(0, -0.212, -0.051)
# main_gimbal_link → base_link:         xyz=(0, 0, -0.25)
# LiDAR positions relative to base_link:
#   left:  (0,  0.212, 0.25 - 0.051) = (0,  0.212, 0.199)
#   right: (0, -0.212, 0.25 - 0.051) = (0, -0.212, 0.199)

LIDAR_Y_OFFSET = 0.212
GIMBAL_Z = 0.25
LIDAR_Z_IN_GIMBAL = -0.051
LIDAR_Z = GIMBAL_Z + LIDAR_Z_IN_GIMBAL  # 0.199
LEFT_LIDAR_YAW = math.pi / 2.0
RIGHT_LIDAR_YAW = -math.pi / 2.0
CHASSIS_DISC_RADIUS = 0.225
CHASSIS_DISC_HALF_HEIGHT = 0.03
CHASSIS_DISC_Z = -0.055

# TF frames
FRAME_ODOM = "odom"
FRAME_BASE_FOOTPRINT = "base_footprint"
FRAME_BASE_LINK = "base_link"
FRAME_GIMBAL = "main_gimbal_link"
FRAME_LEFT_LIVOX = "left_livox_frame"
FRAME_RIGHT_LIVOX = "right_livox_frame"
FRAME_GIMBAL_ODOM = "main_gimbal_odom"
JOINT_GIMBAL_YAW = "gimbal_yaw_joint"

# LiDAR params
LIDAR_CUTOFF = 30.0   # max range (m)
LIDAR_RATE = 10.0     # Hz
IMU_RATE = 200.0      # Hz
ODOM_RATE = 50.0      # Hz
PHYSICS_DT = 0.002    # 500 Hz physics
SPAWN_X = 0.0
SPAWN_Y = 0.0
SPAWN_Z = 10.0
DEFAULT_ROBOT_INIT_LOCATION = (SPAWN_X, SPAWN_Y, SPAWN_Z)
SMALL_GYRO_MODE_THRESHOLD = 0.5

RENDER_GEOM_GROUP = 1
LIDAR_TRACE_GEOM_GROUP = 0
LIDAR_DEBUG_GEOM_GROUP = 3
COLLISION_GEOM_GROUP = 2
LIDAR_GEOMGROUP_MASK = 1 << LIDAR_TRACE_GEOM_GROUP
ENV_CONTYPE = 1
ROBOT_CONTYPE = 2
DEFAULT_BOUNDARY_X_MIN = -13.5
DEFAULT_BOUNDARY_X_MAX = 13.5
DEFAULT_BOUNDARY_Y_MIN = -7.0
DEFAULT_BOUNDARY_Y_MAX = 7.0
DEFAULT_MAX_TILT_DEG = 30.0
DEFAULT_USE_KEEP_STAND = False
DEFAULT_TILT_DOWNFORCE_THRESHOLD_DEG = 15.0
DEFAULT_TILT_DOWNFORCE_SCALE = 150.0
DEFAULT_TILT_DOWNFORCE_EXP_GAIN = 6.0

# #region debug-point A:report-helper
DEBUG_SESSION_ENV = ".dbg/pointcloud-no-output.env"
DEBUG_DEFAULT_SERVER_URL = "http://127.0.0.1:7777/event"
DEBUG_DEFAULT_SESSION_ID = "pointcloud-no-output"


def debug_report(hypothesis_id: str, location: str, msg: str, data: dict | None = None, run_id: str = "pre-fix") -> None:
    _url = DEBUG_DEFAULT_SERVER_URL
    _session = DEBUG_DEFAULT_SESSION_ID
    _run_id = os.environ.get("DEBUG_RUN_ID", run_id)
    try:
        with open(DEBUG_SESSION_ENV, "r", encoding="utf-8") as env_file:
            for line in env_file:
                if line.startswith("DEBUG_SERVER_URL="):
                    _url = line.split("=", 1)[1].strip() or _url
                elif line.startswith("DEBUG_SESSION_ID="):
                    _session = line.split("=", 1)[1].strip() or _session
    except OSError:
        pass
    payload = {
        "sessionId": _session,
        "runId": _run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": f"[DEBUG] {msg}",
        "data": data or {},
        "ts": int(time.time() * 1000),
    }
    try:
        request = urllib.request.Request(
            _url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=0.2).read()
    except Exception:
        pass

# #endregion


def _list_obj_files(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        name for name in os.listdir(directory)
        if name.lower().endswith(".obj")
    )


def resolve_assets_dir():
    """Resolve the sim_assets package root for both source and installed runs."""
    candidate_dirs = []
    try:
        candidate_dirs.append(get_package_share_directory("sim_assets"))
    except PackageNotFoundError:
        pass

    this_file = os.path.abspath(__file__)
    candidate_dirs.extend([
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(this_file))),
            "sim_assets",
        ),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(this_file))))),
            "src",
            "sim_assets",
        ),
    ])

    for candidate in candidate_dirs:
        mesh_file = os.path.join(candidate, "meshes", "mesh_view", "mesh_view.obj")
        if os.path.isfile(mesh_file):
            return candidate

    raise FileNotFoundError(
        "Could not resolve sim_assets directory containing meshes/mesh_view/mesh_view.obj"
    )


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_vector(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None
    return vec / norm


def quat_to_rpy(quat) -> tuple[float, float, float]:
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def lidar_pointcloud_topic_from_ip(ip_address: str) -> str:
    return f"/livox/lidar_{ip_address.replace('.', '_')}/pointcloud"


def lidar_imu_topic_from_ip(ip_address: str) -> str:
    return f"/livox/imu_{ip_address.replace('.', '_')}"


def make_scene_xml(
    meshdir,
    boundary_x_min,
    boundary_x_max,
    boundary_y_min,
    boundary_y_max,
    robot_init_location,
):
    """Generate MuJoCo XML with separate render/LiDAR meshes and collision assets."""
    spawn_x, spawn_y, spawn_z = robot_init_location
    view_scene_mesh_rel = "mesh_view/mesh_view.obj"
    lidar_scene_mesh_rel = "mesh_lidar/mesh_lidar.obj"
    collision_dir = os.path.join(meshdir, "mesh_collision_env")

    collision_mesh_files = _list_obj_files(collision_dir)

    if not os.path.isfile(os.path.join(meshdir, view_scene_mesh_rel)):
        raise FileNotFoundError(
            f"View scene mesh not found: {os.path.join(meshdir, view_scene_mesh_rel)}"
        )
    if not os.path.isfile(os.path.join(meshdir, lidar_scene_mesh_rel)):
        raise FileNotFoundError(
            f"LiDAR scene mesh not found: {os.path.join(meshdir, lidar_scene_mesh_rel)}"
        )
    if not collision_mesh_files:
        raise FileNotFoundError(
            f"No OBJ files found under {collision_dir}. "
            "Run scripts/convert_collision_fbx.py first."
        )

    collision_asset_xml = "\n".join(
        f'    <mesh name="env_collision_mesh_{i}" file="mesh_collision_env/{name}"/>'
        for i, name in enumerate(collision_mesh_files)
    )
    collision_geom_xml = "\n".join(
        f'      <geom name="env_collision_geom_{i}" type="mesh" mesh="env_collision_mesh_{i}" '
        f'material="mat_collision_debug" contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}" '
        f'condim="3" friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>'
        for i, _ in enumerate(collision_mesh_files)
    )
    boundary_geom_xml = f"""
      <geom name="boundary_wall_pos_x" type="box" pos="{boundary_x_max + 0.05} 0 0.6"
            size="0.05 {0.5 * (boundary_y_max - boundary_y_min)} 0.6"
            rgba="0 0 0 0" contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}"
            condim="3" friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>
      <geom name="boundary_wall_neg_x" type="box" pos="{boundary_x_min - 0.05} 0 0.6"
            size="0.05 {0.5 * (boundary_y_max - boundary_y_min)} 0.6"
            rgba="0 0 0 0" contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}"
            condim="3" friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>
      <geom name="boundary_wall_pos_y" type="box" pos="0 {boundary_y_max + 0.05} 0.6"
            size="{0.5 * (boundary_x_max - boundary_x_min)} 0.05 0.6"
            rgba="0 0 0 0" contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}"
            condim="3" friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>
      <geom name="boundary_wall_neg_y" type="box" pos="0 {boundary_y_min - 0.05} 0.6"
            size="{0.5 * (boundary_x_max - boundary_x_min)} 0.05 0.6"
            rgba="0 0 0 0" contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}"
            condim="3" friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>"""

    return f"""<mujoco model="sentry_sim">
  <compiler angle="radian" meshdir="{meshdir}"/>
  <option timestep="{PHYSICS_DT}" gravity="0 0 -9.81"/>

  <visual>
    <map force="0.1" zfar="200"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.1 0.1 0.2" width="512" height="512"/>
    <texture name="tex_grid" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"
             width="512" height="512" mark="edge" markrgb="0.5 0.5 0.5"/>
    <material name="mat_ground" texture="tex_grid" texrepeat="10 10" texuniform="true" reflectance="0.1"/>
    <material name="mat_chassis" rgba="0.4 0.4 0.5 1.0"/>
    <material name="mat_gimbal" rgba="0.2 0.2 0.3 1.0"/>
    <material name="mat_gimbal_arrow" rgba="0.9 0.1 0.1 1.0"/>
    <material name="mat_lidar" rgba="0.8 0.2 0.2 1.0"/>
    <material name="mat_arena" rgba="0.5 0.5 0.6 1.0"/>
    <material name="mat_lidar_debug" rgba="0.0 0.8 1.0 0.35"/>
    <material name="mat_collision_debug" rgba="1.0 0.55 0.1 0.28"/>

    <mesh name="arena_view_mesh" file="{view_scene_mesh_rel}"/>
    <mesh name="lidar_detect_mesh" file="{lidar_scene_mesh_rel}"/>
{collision_asset_xml}
  </asset>

  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="50 50 0.1" material="mat_ground"
          contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}" condim="3"
          friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>

    <!-- Light -->
    <light name="sun" directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"
           pos="5 5 10" dir="-0.5 -0.5 -1"/>

    <!-- ===== Arena scene ===== -->
    <body name="arena" pos="0 0 0">
      <geom name="arena_view_geom" type="mesh" mesh="arena_view_mesh" material="mat_arena"
            contype="0" conaffinity="0" group="{RENDER_GEOM_GROUP}"/>
      <!-- Keep LiDAR-only geometry in group 0 for ray casting. -->
      <geom name="lidar_geom_0" type="mesh" mesh="lidar_detect_mesh"
            rgba="0 0 0 0" contype="0" conaffinity="0" group="{LIDAR_TRACE_GEOM_GROUP}"/>
      <!-- Expose a separate viewer-toggleable debug geom for the LiDAR mesh. -->
      <geom name="lidar_debug_geom" type="mesh" mesh="lidar_detect_mesh"
            material="mat_lidar_debug" contype="0" conaffinity="0" group="{LIDAR_DEBUG_GEOM_GROUP}"/>
{collision_geom_xml}
{boundary_geom_xml}
    </body>

    <!-- ===== base_link (top-level dynamic body) ===== -->
    <body name="{FRAME_BASE_LINK}" pos="{spawn_x} {spawn_y} {spawn_z}">
      <freejoint name="base_freejoint"/>

      <geom name="chassis" type="box" size="0.225 0.225 0.01"
            material="mat_chassis" mass="0"
            contype="0" conaffinity="0" condim="3"
            friction="0 0 0" group="1"/>
      <geom name="chassis_contact_base" type="cylinder"
            size="{CHASSIS_DISC_RADIUS} {CHASSIS_DISC_HALF_HEIGHT}"
            pos="0 0 {CHASSIS_DISC_Z}" material="mat_chassis" mass="20.0"
            contype="{ROBOT_CONTYPE}" conaffinity="{ENV_CONTYPE}" condim="3"
            friction="0 0 0" group="1"/>

        <!-- ===== main_gimbal_link ===== -->
        <body name="{FRAME_GIMBAL}" pos="0 0 {GIMBAL_Z}">
          <joint name="{JOINT_GIMBAL_YAW}" type="hinge" axis="0 0 1" damping="0"/>
          <geom name="gimbal_geom" type="box" size="0.11 0.11 0.04"
                material="mat_gimbal" mass="0"
                contype="0" conaffinity="0" group="1"/>
          <!-- Two slim boxes form a backward-pointing triangular marker. -->
          <geom name="gimbal_pointer_left" type="box" size="0.055 0.012 0.04"
                pos="-0.13 0.03 0" euler="0 0 2.54159265"
                material="mat_gimbal_arrow" rgba="0.9 0.1 0.1 1.0"
                mass="0" contype="0" conaffinity="0" group="1"/>
          <geom name="gimbal_pointer_right" type="box" size="0.055 0.012 0.04"
                pos="-0.13 -0.03 0" euler="0 0 -2.54159265"
                material="mat_gimbal_arrow" rgba="0.9 0.1 0.1 1.0"
                mass="0" contype="0" conaffinity="0" group="1"/>
          <geom name="gimbal_collision" type="box" size="0.17 0.17 0.09"
                rgba="0 0 0 0" mass="0"
                contype="{ROBOT_CONTYPE}" conaffinity="{ENV_CONTYPE}" condim="3"
                friction="0 0 0" group="1"/>

          <!-- ===== main_gimbal_odom (for LIO reference) ===== -->
          <body name="{FRAME_GIMBAL_ODOM}" pos="0 0 0">
            <geom name="gimbal_odom_geom" type="sphere" size="0.001"
                  rgba="0 0 0 0" mass="0" contype="0" conaffinity="0" group="1"/>
          </body>

          <!-- ===== left_livox_frame =====
               Match the LiDAR body orientation to the URDF joint so the traced rays
               and the published frame use the same local axes. -->
          <body name="{FRAME_LEFT_LIVOX}" pos="0 {LIDAR_Y_OFFSET} {LIDAR_Z_IN_GIMBAL}"
                euler="0 0 {LEFT_LIDAR_YAW}">
            <geom name="left_lidar_vis" type="cylinder" size="0.035 0.025"
                  material="mat_lidar" mass="0.265"
                  contype="0" conaffinity="0" group="1"/>
            <inertial pos="0 0 0" mass="0.265" diaginertia="0.0001 0.0001 0.0001"/>

            <!-- LiDAR ray-casting site (identity orientation, rays in +x hemisphere) -->
            <site name="left_lidar_site" type="sphere" size="0.01" rgba="1 0 0 0.5"/>
          </body>

          <!-- ===== right_livox_frame ===== -->
          <body name="{FRAME_RIGHT_LIVOX}" pos="0 {-LIDAR_Y_OFFSET} {LIDAR_Z_IN_GIMBAL}"
                euler="0 0 {RIGHT_LIDAR_YAW}">
            <geom name="right_lidar_vis" type="cylinder" size="0.035 0.025"
                  material="mat_lidar" mass="0.265"
                  contype="0" conaffinity="0" group="1"/>
            <inertial pos="0 0 0" mass="0.265" diaginertia="0.0001 0.0001 0.0001"/>

            <!-- LiDAR ray-casting site -->
            <site name="right_lidar_site" type="sphere" size="0.01" rgba="1 0 0 0.5"/>
          </body>

        </body>
      </body>
  </worldbody>

  <sensor>
    <!-- IMU sensors (accelerometer + gyro in body frame) -->
    <accelerometer name="left_imu_acc" site="left_lidar_site"/>
    <gyro name="left_imu_gyro" site="left_lidar_site"/>
    <accelerometer name="right_imu_acc" site="right_lidar_site"/>
    <gyro name="right_imu_gyro" site="right_lidar_site"/>

    <!-- Base pose for debugging -->
    <framepos name="base_pos" objtype="body" objname="{FRAME_BASE_LINK}"/>
    <framequat name="base_quat" objtype="body" objname="{FRAME_BASE_LINK}"/>
  </sensor>
</mujoco>"""


class SentrySimNode(Node):
    def __init__(self):
        super().__init__("sentry_sim_node")
        self.enable_viewer = bool(self.declare_parameter("enable_viewer", True).value)
        self.cmd_vel_topic = str(
            self.declare_parameter("cmd_vel_topic", "/cmd_vel_keyboard").value
        )
        self.nav_cmd_vel_topic = str(
            self.declare_parameter("nav_cmd_vel_topic", "/cmd_vel_processed").value
        )
        self.keyboard_cmd_timeout_sec = max(
            float(self.declare_parameter("keyboard_cmd_timeout_sec", 0.2).value),
            0.0,
        )
        self.nav_cmd_timeout_sec = max(
            float(self.declare_parameter("nav_cmd_timeout_sec", 0.5).value),
            0.0,
        )
        self.small_gyro_spin_rate = float(
            self.declare_parameter("small_gyro_spin_rate", 6.0).value
        )
        self.boundary_x_min = float(
            self.declare_parameter("boundary_x_min", DEFAULT_BOUNDARY_X_MIN).value
        )
        self.boundary_x_max = float(
            self.declare_parameter("boundary_x_max", DEFAULT_BOUNDARY_X_MAX).value
        )
        self.boundary_y_min = float(
            self.declare_parameter("boundary_y_min", DEFAULT_BOUNDARY_Y_MIN).value
        )
        self.boundary_y_max = float(
            self.declare_parameter("boundary_y_max", DEFAULT_BOUNDARY_Y_MAX).value
        )
        self.max_tilt_deg = float(
            self.declare_parameter("max_tilt_deg", DEFAULT_MAX_TILT_DEG).value
        )
        self.use_keep_stand = bool(
            self.declare_parameter("use_keep_stand", DEFAULT_USE_KEEP_STAND).value
        )
        self.tilt_downforce_threshold_deg = max(
            float(
                self.declare_parameter(
                    "tilt_downforce_threshold_deg",
                    DEFAULT_TILT_DOWNFORCE_THRESHOLD_DEG,
                ).value
            ),
            0.0,
        )
        self.tilt_downforce_scale = max(
            float(
                self.declare_parameter(
                    "tilt_downforce_scale",
                    DEFAULT_TILT_DOWNFORCE_SCALE,
                ).value
            ),
            0.0,
        )
        self.tilt_downforce_exp_gain = max(
            float(
                self.declare_parameter(
                    "tilt_downforce_exp_gain",
                    DEFAULT_TILT_DOWNFORCE_EXP_GAIN,
                ).value
            ),
            0.0,
        )
        raw_robot_init_location = self.declare_parameter(
            "robot_init_location",
            list(DEFAULT_ROBOT_INIT_LOCATION),
        ).value
        if not isinstance(raw_robot_init_location, (list, tuple)) or len(raw_robot_init_location) != 3:
            raise ValueError("robot_init_location must be a 3-element list [x, y, z]")
        self.robot_init_location = tuple(float(value) for value in raw_robot_init_location)
        self.left_lidar_ip = str(
            self.declare_parameter("left_lidar_ip", "192.168.10.5").value
        )
        self.right_lidar_ip = str(
            self.declare_parameter("right_lidar_ip", "192.168.10.4").value
        )
        self.left_pointcloud_topic = lidar_pointcloud_topic_from_ip(self.left_lidar_ip)
        self.right_pointcloud_topic = lidar_pointcloud_topic_from_ip(self.right_lidar_ip)
        self.left_imu_topic = lidar_imu_topic_from_ip(self.left_lidar_ip)
        self.right_imu_topic = lidar_imu_topic_from_ip(self.right_lidar_ip)
        self.max_tilt_rad = math.radians(max(self.max_tilt_deg, 0.0))
        self.tilt_downforce_threshold_rad = math.radians(self.tilt_downforce_threshold_deg)
        if self.boundary_x_min >= self.boundary_x_max:
            raise ValueError("boundary_x_min must be smaller than boundary_x_max")
        if self.boundary_y_min >= self.boundary_y_max:
            raise ValueError("boundary_y_min must be smaller than boundary_y_max")
        self.viewer = None
        self._viewer_import = None
        self._viewer_sync_failed = False

        #region debug-point viewer-env
        viewer_import_ok = False
        viewer_import_error = ""
        try:
            self._viewer_import = importlib.import_module("mujoco.viewer")
            viewer_import_ok = True
        except Exception as exc:
            viewer_import_error = f"{type(exc).__name__}: {exc}"

        self.get_logger().info(
            "Viewer environment: "
            f"DISPLAY={os.environ.get('DISPLAY', '<unset>')}, "
            f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '<unset>')}, "
            f"XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE', '<unset>')}, "
            f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}, "
            f"enable_viewer={self.enable_viewer}, "
            f"mujoco_version={mujoco.__version__}, "
            f"viewer_module={'ok' if viewer_import_ok else 'failed'}"
        )
        if not viewer_import_ok:
            self.get_logger().warn(f"Viewer import failed: {viewer_import_error}")
        #endregion debug-point viewer-env

        # ── MuJoCo setup ──
        self.assets_dir = resolve_assets_dir()
        meshdir = os.path.join(self.assets_dir, "meshes")
        self.get_logger().info(f"Resolved assets directory: {self.assets_dir}")
        self.get_logger().info("Loading MuJoCo model...")
        self.model = mujoco.MjModel.from_xml_string(
            make_scene_xml(
                meshdir,
                self.boundary_x_min,
                self.boundary_x_max,
                self.boundary_y_min,
                self.boundary_y_max,
                self.robot_init_location,
            )
        )
        self.data = mujoco.MjData(self.model)
        self.gimbal_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_GIMBAL_YAW
        )
        if self.gimbal_joint_id < 0:
            raise RuntimeError(f"MuJoCo joint not found: {JOINT_GIMBAL_YAW}")
        self.gimbal_qpos_adr = self.model.jnt_qposadr[self.gimbal_joint_id]
        self.gimbal_dof_adr = self.model.jnt_dofadr[self.gimbal_joint_id]
        self.base_body_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, FRAME_BASE_LINK)
        )
        if self.base_body_id < 0:
            raise RuntimeError(f"MuJoCo body not found: {FRAME_BASE_LINK}")
        mujoco.mj_forward(self.model, self.data)
        self.keep_stand_geomgroup = np.zeros(6, dtype=np.uint8)
        self.keep_stand_geomgroup[COLLISION_GEOM_GROUP] = 1
        self._start_viewer_if_requested()

        # ── LiDAR setup ──
        self.get_logger().info("Setting up LiDAR wrappers...")
        # CPU backend follows MuJoCo's uint8[6] geomgroup filter. In this scene,
        # group 2 contains the environment collision geoms that should contribute
        # to ray hits. Group 0 keeps the LiDAR-only mesh available for backends
        # that can use it, while group 1 is excluded to avoid self hits.
        lidar_args = {
            'geomgroup': np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8),
            'bodyexclude': int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, FRAME_BASE_LINK)),
        }
        self.lidar_left = MjLidarWrapper(
            self.model, site_name="left_lidar_site",
            backend="cpu", cutoff_dist=LIDAR_CUTOFF,
            args=lidar_args
        )
        self.lidar_right = MjLidarWrapper(
            self.model, site_name="right_lidar_site",
            backend="cpu", cutoff_dist=LIDAR_CUTOFF,
            args=lidar_args
        )
        # #region debug-point A:lidar-init
        debug_report(
            "A",
            "sentry_sim_node.py:478",
            "Initialized MuJoCo LiDAR wrappers",
            {
                "backend": "cpu",
                "cutoff_dist": LIDAR_CUTOFF,
                "geomgroup_mask": int(LIDAR_GEOMGROUP_MASK),
                "left_site": "left_lidar_site",
                "right_site": "right_lidar_site",
                "ngeom": int(self.model.ngeom),
            },
        )
        # #endregion

        # Mid360 scan pattern (24000 rays)
        lg = LivoxGenerator("mid360")
        self.ray_theta, self.ray_phi = lg.sample_ray_angles()
        self.n_rays = len(self.ray_theta)
        self.get_logger().info(f"Mid360 scan pattern: {self.n_rays} rays")
        # #region debug-point B:scan-pattern
        debug_report(
            "B",
            "sentry_sim_node.py:496",
            "Loaded LiDAR scan pattern",
            {
                "n_rays": int(self.n_rays),
                "theta_shape": list(self.ray_theta.shape),
                "phi_shape": list(self.ray_phi.shape),
            },
        )
        # #endregion

        # Precompute ray directions in local frame
        cos_theta = np.cos(self.ray_theta)
        sin_theta = np.sin(self.ray_theta)
        cos_phi = np.cos(self.ray_phi)
        sin_phi = np.sin(self.ray_phi)
        self.ray_dirs = np.stack([
            cos_theta * cos_phi,
            sin_theta * cos_phi,
            sin_phi
        ], axis=1)  # (N, 3)

        # ── Odometry state ──
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.odom_origin_x = 0.0
        self.odom_origin_y = 0.0
        self.odom_origin_yaw = 0.0
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_gimbal_yaw_rate = 0.0
        self.cmd_small_gyro_mode = False
        self.nav_cmd_vx = 0.0
        self.nav_cmd_vy = 0.0
        self.nav_cmd_wz = 0.0
        self.gimbal_heading_yaw = 0.0
        self.gimbal_joint_pos = 0.0
        self.gimbal_joint_vel = 0.0
        self.last_cmd_vel_time = self.get_clock().now()
        self.last_nav_cmd_time = self.get_clock().now()
        self.active_cmd_source = "idle"
        self._lidar_debug_counter = 0
        self._initialize_odom_reference_locked()

        # ── Publishers ──
        pointcloud_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Clock publisher (required for use_sim_time:=true)
        self.clock_pub = self.create_publisher(
            rosgraph_msgs.msg.Clock, "/clock", 10
        )

        self.pc_left_pub = self.create_publisher(
            sensor_msgs.msg.PointCloud2,
            self.left_pointcloud_topic,
            pointcloud_qos,
        )
        self.pc_right_pub = self.create_publisher(
            sensor_msgs.msg.PointCloud2,
            self.right_pointcloud_topic,
            pointcloud_qos,
        )
        self.imu_left_pub = self.create_publisher(
            sensor_msgs.msg.Imu, self.left_imu_topic, sensor_qos
        )
        self.imu_right_pub = self.create_publisher(
            sensor_msgs.msg.Imu, self.right_imu_topic, sensor_qos
        )
        self.odom_pub = self.create_publisher(
            nav_msgs.msg.Odometry, "/Odometry", 10
        )
        self.joint_state_pub = self.create_publisher(
            sensor_msgs.msg.JointState, "/joint_states", 10
        )

        # ── TF broadcaster ──
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Subscriber ──
        self.cmd_vel_sub = self.create_subscription(
            geometry_msgs.msg.Twist, self.cmd_vel_topic, self.cmd_vel_cb, 10
        )
        self.nav_cmd_vel_sub = self.create_subscription(
            geometry_msgs.msg.Twist,
            self.nav_cmd_vel_topic,
            self.nav_cmd_vel_cb,
            10,
        )

        # ── Timers ──
        self.lidar_timer = self.create_timer(1.0 / LIDAR_RATE, self.lidar_callback)
        self.imu_timer = self.create_timer(1.0 / IMU_RATE, self.imu_callback)
        self.odom_timer = self.create_timer(1.0 / ODOM_RATE, self.odom_callback)

        # ── Physics loop thread ──
        self.running = True
        self.physics_lock = threading.Lock()
        self.physics_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.physics_thread.start()

        self.get_logger().info(
            f"SentrySimNode ready. keyboard_cmd_vel_topic={self.cmd_vel_topic}, "
            f"nav_cmd_vel_topic={self.nav_cmd_vel_topic}, "
            f"keyboard_cmd_timeout_sec={self.keyboard_cmd_timeout_sec:.2f}, "
            f"nav_cmd_timeout_sec={self.nav_cmd_timeout_sec:.2f}, "
            f"small_gyro_spin_rate={self.small_gyro_spin_rate:.2f}, "
            f"use_keep_stand={self.use_keep_stand}, "
            f"tilt_downforce_threshold_deg={self.tilt_downforce_threshold_deg:.1f}, "
            f"tilt_downforce_scale={self.tilt_downforce_scale:.1f}, "
            f"tilt_downforce_exp_gain={self.tilt_downforce_exp_gain:.2f}, "
            f"robot_init_location={list(self.robot_init_location)}, "
            f"left_pointcloud_topic={self.left_pointcloud_topic}, "
            f"right_pointcloud_topic={self.right_pointcloud_topic}, "
            f"boundary_x=[{self.boundary_x_min:.2f}, {self.boundary_x_max:.2f}], "
            f"boundary_y=[{self.boundary_y_min:.2f}, {self.boundary_y_max:.2f}], "
            f"max_tilt_deg={self.max_tilt_deg:.1f}"
        )

    def _start_viewer_if_requested(self):
        if not self.enable_viewer:
            self.get_logger().info("MuJoCo viewer disabled by parameter enable_viewer:=false.")
            return
        if self._viewer_import is None:
            self.get_logger().warn("MuJoCo viewer is unavailable; running without visualization.")
            return
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self.get_logger().warn(
                "MuJoCo viewer requested but no DISPLAY/WAYLAND_DISPLAY is set; "
                "running headless. Start from a graphical session or export DISPLAY."
            )
            return

        try:
            self.viewer = self._viewer_import.launch_passive(
                self.model,
                self.data,
                show_left_ui=True,
                show_right_ui=True,
            )
            self._configure_viewer_groups()
            self.get_logger().info("MuJoCo viewer launched.")
        except Exception as exc:
            self.viewer = None
            self.get_logger().error(
                f"Failed to launch MuJoCo viewer: {type(exc).__name__}: {exc}"
            )

    def _configure_viewer_groups(self):
        if self.viewer is None:
            return
        try:
            with self.viewer.lock():
                # Keep the main arena visible, while exposing LiDAR/collision debug
                # meshes as separate optional viewer groups.
                self.viewer.opt.geomgroup[RENDER_GEOM_GROUP] = 1
                self.viewer.opt.geomgroup[LIDAR_TRACE_GEOM_GROUP] = 0
                self.viewer.opt.geomgroup[LIDAR_DEBUG_GEOM_GROUP] = 0
                self.viewer.opt.geomgroup[COLLISION_GEOM_GROUP] = 0
            self.get_logger().info(
                "Viewer groups: render=on, lidar_trace=off, "
                "lidar_debug=off, collision_debug=off. "
                f"Open MuJoCo right UI -> Geom Group {COLLISION_GEOM_GROUP} "
                "to show collision meshes and boundary walls."
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to configure viewer geom groups: {type(exc).__name__}: {exc}"
            )

    def _sync_viewer(self):
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            self.viewer = None
            self.get_logger().warn("MuJoCo viewer closed; continuing headless.")
            return
        try:
            self.viewer.sync()
        except Exception as exc:
            if not self._viewer_sync_failed:
                self.get_logger().error(
                    f"MuJoCo viewer sync failed: {type(exc).__name__}: {exc}"
                )
                self._viewer_sync_failed = True
            self.viewer = None

    def _compute_surface_aligned_quat(self, base_pos: np.ndarray, yaw: float) -> np.ndarray | None:
        gravity = np.array(self.model.opt.gravity, dtype=np.float64)
        gravity_dir = normalize_vector(gravity)
        if gravity_dir is None:
            gravity_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        geomid = np.array([-1], dtype=np.int32)
        surface_normal = np.zeros(3, dtype=np.float64)
        hit_dist = mujoco.mj_ray(
            self.model,
            self.data,
            np.asarray(base_pos, dtype=np.float64),
            gravity_dir,
            self.keep_stand_geomgroup,
            True,
            self.base_body_id,
            geomid,
            surface_normal,
        )
        if hit_dist <= 0.0:
            return None

        up_axis = -gravity_dir
        z_axis = normalize_vector(surface_normal)
        if z_axis is None:
            return None
        if float(np.dot(z_axis, up_axis)) < 0.0:
            z_axis = -z_axis

        x_hint = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        x_axis = x_hint - z_axis * float(np.dot(x_hint, z_axis))
        x_axis = normalize_vector(x_axis)
        if x_axis is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(fallback, z_axis))) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = normalize_vector(fallback - z_axis * float(np.dot(fallback, z_axis)))
            if x_axis is None:
                return None

        y_axis = normalize_vector(np.cross(z_axis, x_axis))
        if y_axis is None:
            return None
        x_axis = normalize_vector(np.cross(y_axis, z_axis))
        if x_axis is None:
            return None

        rot_mat = np.column_stack((x_axis, y_axis, z_axis))
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, rot_mat.reshape(-1))
        return quat

    def _compute_tilt_downforce(self, quat: np.ndarray) -> tuple[float, float]:
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, quat)
        base_z_axis = rot_mat.reshape(3, 3)[:, 2]
        cos_tilt = clamp(float(base_z_axis[2]), -1.0, 1.0)
        tilt_rad = math.acos(cos_tilt)
        tilt_excess = max(0.0, tilt_rad - self.tilt_downforce_threshold_rad)
        if tilt_excess <= 0.0:
            return tilt_rad, 0.0
        force = self.tilt_downforce_scale * (
            math.exp(self.tilt_downforce_exp_gain * tilt_excess) - 1.0
        )
        return tilt_rad, force

    def _close_viewer(self):
        if self.viewer is None:
            return
        try:
            self.viewer.close()
        except Exception:
            pass
        self.viewer = None

    def _initialize_odom_reference_locked(self):
        """Make odom start at the current base position and initial gimbal heading."""
        base_pos = self.data.qpos[0:3]
        base_quat = self.data.qpos[3:7]
        base_yaw = math.atan2(
            2.0 * (base_quat[0] * base_quat[3] + base_quat[1] * base_quat[2]),
            1.0 - 2.0 * (base_quat[2] ** 2 + base_quat[3] ** 2),
        )
        current_gimbal_joint_pos = float(self.data.qpos[self.gimbal_qpos_adr])
        initial_gimbal_world_yaw = wrap_to_pi(base_yaw + current_gimbal_joint_pos)

        self.odom_origin_x = float(base_pos[0])
        self.odom_origin_y = float(base_pos[1])
        self.odom_origin_yaw = initial_gimbal_world_yaw
        self.gimbal_heading_yaw = initial_gimbal_world_yaw
        self.gimbal_joint_pos = current_gimbal_joint_pos
        self.gimbal_joint_vel = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = wrap_to_pi(base_yaw - self.odom_origin_yaw)

    def _is_cmd_fresh(self, stamp, timeout_sec: float, now) -> bool:
        if timeout_sec <= 0.0:
            return True
        age = (now - stamp).nanoseconds / 1e9
        return age <= timeout_sec

    # ── cmd_vel subscriber ──
    def cmd_vel_cb(self, msg: geometry_msgs.msg.Twist):
        self.cmd_vx = float(msg.linear.x)
        self.cmd_vy = float(msg.linear.y)
        self.cmd_gimbal_yaw_rate = float(msg.angular.z)
        new_small_gyro_mode = bool(msg.linear.z > SMALL_GYRO_MODE_THRESHOLD)
        if new_small_gyro_mode != self.cmd_small_gyro_mode:
            self.cmd_small_gyro_mode = new_small_gyro_mode
            self.get_logger().info(
                "keyboard small gyro set "
                f"{'on' if self.cmd_small_gyro_mode else 'off'}"
            )
        else:
            self.cmd_small_gyro_mode = new_small_gyro_mode
        self.last_cmd_vel_time = self.get_clock().now()

    def nav_cmd_vel_cb(self, msg: geometry_msgs.msg.Twist):
        self.nav_cmd_vx = float(msg.linear.x)
        self.nav_cmd_vy = float(msg.linear.y)
        self.nav_cmd_wz = float(msg.angular.z)
        self.last_nav_cmd_time = self.get_clock().now()

    # ── Physics loop (runs in background thread) ──
    def physics_loop(self):
        """High-frequency physics stepping + clock publishing."""
        last_clock = 0.0
        while self.running and rclpy.ok():
            with self.physics_lock:
                base_quat = self.data.qpos[3:7]
                base_yaw = math.atan2(
                    2.0 * (base_quat[0] * base_quat[3] + base_quat[1] * base_quat[2]),
                    1.0 - 2.0 * (base_quat[2] ** 2 + base_quat[3] ** 2)
                )
                tilt_rad, tilt_downforce = self._compute_tilt_downforce(base_quat)
                self.data.xfrc_applied[self.base_body_id, :] = 0.0
                if tilt_downforce > 0.0:
                    self.data.xfrc_applied[self.base_body_id, 2] = -tilt_downforce
                now_ros = self.get_clock().now()
                keyboard_fresh = self._is_cmd_fresh(
                    self.last_cmd_vel_time,
                    self.keyboard_cmd_timeout_sec,
                    now_ros,
                )
                nav_fresh = self._is_cmd_fresh(
                    self.last_nav_cmd_time,
                    self.nav_cmd_timeout_sec,
                    now_ros,
                )

                if keyboard_fresh:
                    # Keyboard commands are defined in the main_gimbal_link heading frame.
                    self.gimbal_heading_yaw += self.cmd_gimbal_yaw_rate * PHYSICS_DT
                    heading_yaw = self.gimbal_heading_yaw
                    world_vx = self.cmd_vx * math.cos(heading_yaw) - self.cmd_vy * math.sin(heading_yaw)
                    world_vy = self.cmd_vx * math.sin(heading_yaw) + self.cmd_vy * math.cos(heading_yaw)
                    chassis_wz = self.small_gyro_spin_rate if self.cmd_small_gyro_mode else 0.0
                    self.gimbal_joint_pos = wrap_to_pi(self.gimbal_heading_yaw - base_yaw)
                    self.gimbal_joint_vel = self.cmd_gimbal_yaw_rate - chassis_wz
                    active_cmd_source = "keyboard"
                elif self.cmd_small_gyro_mode:
                    self.gimbal_heading_yaw = wrap_to_pi(self.gimbal_heading_yaw)
                    world_vx = 0.0
                    world_vy = 0.0
                    chassis_wz = self.small_gyro_spin_rate
                    self.gimbal_joint_pos = wrap_to_pi(self.gimbal_heading_yaw - base_yaw)
                    self.gimbal_joint_vel = -chassis_wz
                    active_cmd_source = "keyboard_spin"
                elif nav_fresh:
                    # Navigation commands are already base_link-frame chassis velocities.
                    self.gimbal_heading_yaw = wrap_to_pi(base_yaw + self.gimbal_joint_pos)
                    world_vx = self.nav_cmd_vx * math.cos(base_yaw) - self.nav_cmd_vy * math.sin(base_yaw)
                    world_vy = self.nav_cmd_vx * math.sin(base_yaw) + self.nav_cmd_vy * math.cos(base_yaw)
                    chassis_wz = self.nav_cmd_wz
                    self.gimbal_joint_pos = wrap_to_pi(self.gimbal_heading_yaw - base_yaw)
                    self.gimbal_joint_vel = -chassis_wz
                    active_cmd_source = "nav"
                else:
                    self.gimbal_heading_yaw = wrap_to_pi(base_yaw + self.gimbal_joint_pos)
                    world_vx = 0.0
                    world_vy = 0.0
                    chassis_wz = 0.0
                    self.gimbal_joint_pos = wrap_to_pi(self.gimbal_heading_yaw - base_yaw)
                    self.gimbal_joint_vel = 0.0
                    active_cmd_source = "idle"

                if active_cmd_source != self.active_cmd_source:
                    self.active_cmd_source = active_cmd_source
                    self.get_logger().info(f"active command source -> {self.active_cmd_source}")

                # Preserve vertical velocity and roll/pitch dynamics so gravity and
                # contact response can act naturally on the free body.
                self.data.qpos[self.gimbal_qpos_adr] = self.gimbal_joint_pos
                self.data.qvel[0] = world_vx
                self.data.qvel[1] = world_vy
                self.data.qvel[5] = chassis_wz
                self.data.qvel[self.gimbal_dof_adr] = self.gimbal_joint_vel

                mujoco.mj_step(self.model, self.data)

                quat = self.data.qpos[3:7].copy()
                roll, pitch, yaw = quat_to_rpy(quat)
                if self.use_keep_stand:
                    aligned_quat = self._compute_surface_aligned_quat(
                        self.data.qpos[0:3].copy(),
                        yaw,
                    )
                    if aligned_quat is not None:
                        self.data.qpos[3:7] = aligned_quat
                        self.data.qvel[3] = 0.0
                        self.data.qvel[4] = 0.0
                    else:
                        clamped_roll = clamp(roll, -self.max_tilt_rad, self.max_tilt_rad)
                        clamped_pitch = clamp(pitch, -self.max_tilt_rad, self.max_tilt_rad)
                        if (
                            abs(clamped_roll - roll) > 1e-6
                            or abs(clamped_pitch - pitch) > 1e-6
                        ):
                            limited_quat = rpy_to_quat(clamped_roll, clamped_pitch, yaw)
                            self.data.qpos[3:7] = limited_quat
                            self.data.qvel[3] = 0.0
                            self.data.qvel[4] = 0.0
                else:
                    clamped_roll = clamp(roll, -self.max_tilt_rad, self.max_tilt_rad)
                    clamped_pitch = clamp(pitch, -self.max_tilt_rad, self.max_tilt_rad)
                    if (
                        abs(clamped_roll - roll) > 1e-6
                        or abs(clamped_pitch - pitch) > 1e-6
                    ):
                        limited_quat = rpy_to_quat(clamped_roll, clamped_pitch, yaw)
                        self.data.qpos[3:7] = limited_quat
                        self.data.qvel[3] = 0.0
                        self.data.qvel[4] = 0.0

                pos = self.data.qpos[0:3]
                if pos[0] < self.boundary_x_min:
                    self.data.qpos[0] = self.boundary_x_min
                    self.data.qvel[0] = max(0.0, self.data.qvel[0])
                elif pos[0] > self.boundary_x_max:
                    self.data.qpos[0] = self.boundary_x_max
                    self.data.qvel[0] = min(0.0, self.data.qvel[0])
                if pos[1] < self.boundary_y_min:
                    self.data.qpos[1] = self.boundary_y_min
                    self.data.qvel[1] = max(0.0, self.data.qvel[1])
                elif pos[1] > self.boundary_y_max:
                    self.data.qpos[1] = self.boundary_y_max
                    self.data.qvel[1] = min(0.0, self.data.qvel[1])

                pos = self.data.qpos[0:3]
                quat = self.data.qpos[3:7]
                base_yaw = math.atan2(
                    2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                    1.0 - 2.0 * (quat[2]**2 + quat[3]**2)
                )
                dx = float(pos[0]) - self.odom_origin_x
                dy = float(pos[1]) - self.odom_origin_y
                cos_origin = math.cos(self.odom_origin_yaw)
                sin_origin = math.sin(self.odom_origin_yaw)
                self.odom_x = cos_origin * dx + sin_origin * dy
                self.odom_y = -sin_origin * dx + cos_origin * dy
                self.odom_yaw = wrap_to_pi(base_yaw - self.odom_origin_yaw)
                self._sync_viewer()

            # Publish clock at ~100Hz (from physics thread, not ROS timer)
            now = time.time()
            if now - last_clock > 0.01:
                try:
                    msg = rosgraph_msgs.msg.Clock()
                    msg.clock.sec = int(now)
                    msg.clock.nanosec = int((now - int(now)) * 1e9)
                    self.clock_pub.publish(msg)
                    last_clock = now
                except Exception:
                    pass  # context shutting down

            time.sleep(PHYSICS_DT)

    # ── LiDAR callback ──
    def lidar_callback(self):
        """Publish PointCloud2 for both LiDARs."""
        with self.physics_lock:
            ranges_left = self.lidar_left.trace_rays(
                self.data, self.ray_theta, self.ray_phi
            )
            ranges_right = self.lidar_right.trace_rays(
                self.data, self.ray_theta, self.ray_phi
            )
        self._lidar_debug_counter += 1
        left_valid = int(np.sum((ranges_left > 0.1) & (ranges_left < LIDAR_CUTOFF)))
        right_valid = int(np.sum((ranges_right > 0.1) & (ranges_right < LIDAR_CUTOFF)))
        if self._lidar_debug_counter <= 3 or self._lidar_debug_counter % 20 == 0:
            # #region debug-point C:trace-rays
            debug_report(
                "C",
                "sentry_sim_node.py:768",
                "Traced LiDAR rays",
                {
                    "iteration": int(self._lidar_debug_counter),
                    "left_valid": left_valid,
                    "right_valid": right_valid,
                    "left_min": float(np.min(ranges_left)) if len(ranges_left) else None,
                    "left_max": float(np.max(ranges_left)) if len(ranges_left) else None,
                    "right_min": float(np.min(ranges_right)) if len(ranges_right) else None,
                    "right_max": float(np.max(ranges_right)) if len(ranges_right) else None,
                },
            )
            # #endregion

        # Convert to PointCloud2 in LiDAR's local frame
        pc_left = self._ranges_to_pointcloud(ranges_left, FRAME_LEFT_LIVOX)
        pc_right = self._ranges_to_pointcloud(ranges_right, FRAME_RIGHT_LIVOX)

        now = self.get_clock().now().to_msg()
        pc_left.header.stamp = now
        pc_right.header.stamp = now

        self.pc_left_pub.publish(pc_left)
        self.pc_right_pub.publish(pc_right)
        if self._lidar_debug_counter <= 3 or self._lidar_debug_counter % 20 == 0:
            # #region debug-point D:pointcloud-publish
            debug_report(
                "D",
                "sentry_sim_node.py:792",
                "Published pointcloud messages",
                {
                    "iteration": int(self._lidar_debug_counter),
                    "left_width": int(pc_left.width),
                    "right_width": int(pc_right.width),
                    "left_frame": pc_left.header.frame_id,
                    "right_frame": pc_right.header.frame_id,
                    "left_subscribers": int(self.pc_left_pub.get_subscription_count()),
                    "right_subscribers": int(self.pc_right_pub.get_subscription_count()),
                },
            )
            # #endregion

    def _ranges_to_pointcloud(self, ranges, frame_id):
        """Convert range array + precomputed ray directions to PointCloud2."""
        from sensor_msgs_py import point_cloud2

        valid = (ranges > 0.1) & (ranges < LIDAR_CUTOFF)
        n_valid = np.sum(valid)

        if n_valid == 0:
            # #region debug-point E:empty-cloud
            debug_report(
                "E",
                "sentry_sim_node.py:803",
                "Generated empty pointcloud from ranges",
                {
                    "frame_id": frame_id,
                    "range_count": int(len(ranges)),
                },
            )
            # #endregion
            pc = sensor_msgs.msg.PointCloud2()
            pc.header.frame_id = frame_id
            pc.height = 1
            pc.width = 0
            pc.is_dense = True
            pc.fields = [
                sensor_msgs.msg.PointField(name='x', offset=0, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
                sensor_msgs.msg.PointField(name='y', offset=4, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
                sensor_msgs.msg.PointField(name='z', offset=8, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
                sensor_msgs.msg.PointField(name='intensity', offset=12, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
            ]
            pc.point_step = 16
            pc.row_step = 0
            return pc

        points_local = self.ray_dirs[valid] * ranges[valid, np.newaxis]  # (N, 3)
        intensities = np.ones(n_valid, dtype=np.float32)

        # Build structured array
        cloud_data = np.zeros(n_valid, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
        ])
        cloud_data['x'] = points_local[:, 0].astype(np.float32)
        cloud_data['y'] = points_local[:, 1].astype(np.float32)
        cloud_data['z'] = points_local[:, 2].astype(np.float32)
        cloud_data['intensity'] = intensities

        fields = [
            sensor_msgs.msg.PointField(name='x', offset=0, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
            sensor_msgs.msg.PointField(name='y', offset=4, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
            sensor_msgs.msg.PointField(name='z', offset=8, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
            sensor_msgs.msg.PointField(name='intensity', offset=12, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
        ]
        pc = point_cloud2.create_cloud(
            std_msgs.msg.Header(),
            fields, cloud_data
        )
        pc.header.frame_id = frame_id
        return pc

    # ── IMU callback ──
    def _read_imu_sensor(self, acc_sensor_name: str, gyro_sensor_name: str) -> tuple[np.ndarray, np.ndarray]:
        with self.physics_lock:
            acc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, acc_sensor_name)
            gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, gyro_sensor_name)

            if acc_id >= 0 and gyro_id >= 0:
                acc_adr = self.model.sensor_adr[acc_id]
                gyro_adr = self.model.sensor_adr[gyro_id]
                acc = self.data.sensordata[acc_adr:acc_adr+3].copy()
                gyro = self.data.sensordata[gyro_adr:gyro_adr+3].copy()
            else:
                acc = np.zeros(3)
                gyro = np.zeros(3)

        return acc, gyro

    def _build_imu_message(self, stamp, frame_id: str, acc: np.ndarray, gyro: np.ndarray) -> sensor_msgs.msg.Imu:
        msg = sensor_msgs.msg.Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.linear_acceleration.x = float(acc[0])
        msg.linear_acceleration.y = float(acc[1])
        msg.linear_acceleration.z = float(acc[2])

        msg.angular_velocity.x = float(gyro[0])
        msg.angular_velocity.y = float(gyro[1])
        msg.angular_velocity.z = float(gyro[2])

        # Orientation: identity (IMU doesn't know orientation without fusion)
        msg.orientation.w = 1.0

        # Covariance: unknown
        for i in range(9):
            msg.orientation_covariance[i] = 0.0
            msg.angular_velocity_covariance[i] = 0.0
            msg.linear_acceleration_covariance[i] = 0.0

        return msg

    def imu_callback(self):
        """Publish IMU data from both LiDAR built-in IMUs."""
        stamp = self.get_clock().now().to_msg()
        left_acc, left_gyro = self._read_imu_sensor("left_imu_acc", "left_imu_gyro")
        right_acc, right_gyro = self._read_imu_sensor("right_imu_acc", "right_imu_gyro")

        self.imu_left_pub.publish(
            self._build_imu_message(stamp, FRAME_LEFT_LIVOX, left_acc, left_gyro)
        )
        self.imu_right_pub.publish(
            self._build_imu_message(stamp, FRAME_RIGHT_LIVOX, right_acc, right_gyro)
        )

    # ── Odometry + TF callback ──
    def odom_callback(self):
        """Publish odometry and TF tree."""
        now = self.get_clock().now()
        now_msg = now.to_msg()

        with self.physics_lock:
            x = self.odom_x
            y = self.odom_y
            yaw = self.odom_yaw
            vx = self.data.qvel[0]
            vy = self.data.qvel[1]
            omega = self.data.qvel[5]
            gimbal_joint_pos = self.gimbal_joint_pos
            gimbal_joint_vel = self.gimbal_joint_vel

        # Quaternion from yaw
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)

        # ── TF: only odom → base_footprint (dynamic).
        #     All other TFs (base_footprint→base_link→gimbal→livox) come from
        #     robot_state_publisher via the URDF description.
        tf_odom_fp = TransformStamped()
        tf_odom_fp.header.stamp = now_msg
        tf_odom_fp.header.frame_id = FRAME_ODOM
        tf_odom_fp.child_frame_id = FRAME_BASE_FOOTPRINT
        tf_odom_fp.transform.translation.x = float(x)
        tf_odom_fp.transform.translation.y = float(y)
        tf_odom_fp.transform.translation.z = 0.0
        tf_odom_fp.transform.rotation.w = float(qw)
        tf_odom_fp.transform.rotation.z = float(qz)

        self.tf_broadcaster.sendTransform(tf_odom_fp)

        # ── Odometry message ──
        odom = nav_msgs.msg.Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = FRAME_ODOM
        odom.child_frame_id = FRAME_BASE_FOOTPRINT
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.orientation.w = float(qw)
        odom.pose.pose.orientation.z = float(qz)
        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        odom.twist.twist.angular.z = float(omega)

        self.odom_pub.publish(odom)

        joint_state = sensor_msgs.msg.JointState()
        joint_state.header.stamp = now_msg
        joint_state.name = [JOINT_GIMBAL_YAW]
        joint_state.position = [float(gimbal_joint_pos)]
        joint_state.velocity = [float(gimbal_joint_vel)]
        self.joint_state_pub.publish(joint_state)

    def destroy_node(self):
        self.running = False
        if hasattr(self, 'physics_thread'):
            self.physics_thread.join(timeout=1.0)
        self._close_viewer()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SentrySimNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
