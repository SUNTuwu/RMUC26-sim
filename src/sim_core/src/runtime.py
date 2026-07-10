from __future__ import annotations

import importlib
import math
import os
import threading
import time

import builtin_interfaces.msg
import mujoco
import numpy as np
import rclpy
import rosgraph_msgs.msg
from rclpy.node import Node

from .component_manager import ComponentManager
from .frame_tree import (
    FRAME_BASE_LINK,
    FRAME_GIMBAL,
    FRAME_LEFT_LIVOX,
    FRAME_RIGHT_LIVOX,
    JOINT_GIMBAL_YAW,
    load_robot_frame_tree,
)
from .scene_builder import (
    COLLISION_GEOM_GROUP,
    build_scene_xml,
    frame_resource,
    livox_accel_debug_body_name,
    livox_accel_debug_geom_name,
    livox_accel_sensor_name,
    livox_gyro_sensor_name,
    livox_imu_site_name,
    load_scene_geometry_params,
    resolve_assets_dir,
)


GRAVITY_M_S2 = 9.81
DEFAULT_BOUNDARY_X_MIN = -13.5
DEFAULT_BOUNDARY_X_MAX = 13.5
DEFAULT_BOUNDARY_Y_MIN = -7.0
DEFAULT_BOUNDARY_Y_MAX = 7.0
DEFAULT_USE_KEEP_STAND = False
DEFAULT_TILT_DOWNFORCE_THRESHOLD_DEG = 15.0
DEFAULT_TILT_DOWNFORCE_SCALE = 300.0
DEFAULT_TILT_DOWNFORCE_EXP_GAIN = 6.0
DEFAULT_PHYSICS_DT = 0.002
DEFAULT_ROBOT_INIT_LOCATION = (0.0, 0.0, 10.0)


def _coerce_vector3_param(raw_value, param_name: str) -> tuple[float, float, float]:
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 3:
        raise ValueError(f"{param_name} must be a 3-element list [x, y, z]")
    return tuple(float(value) for value in raw_value)


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


def quat_from_local_z_axis(direction: np.ndarray) -> tuple[float, float, float, float]:
    z_axis = normalize_vector(np.asarray(direction, dtype=np.float64))
    if z_axis is None:
        return (1.0, 0.0, 0.0, 0.0)
    x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(x_hint, z_axis))) > 0.9:
        x_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = normalize_vector(x_hint - z_axis * float(np.dot(x_hint, z_axis)))
    if x_axis is None:
        return (1.0, 0.0, 0.0, 0.0)
    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    if y_axis is None:
        return (1.0, 0.0, 0.0, 0.0)
    x_axis = normalize_vector(np.cross(y_axis, z_axis))
    if x_axis is None:
        return (1.0, 0.0, 0.0, 0.0)
    rot_mat = np.column_stack((x_axis, y_axis, z_axis))
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rot_mat.reshape(-1))
    return tuple(float(value) for value in quat)


class SimulationRuntime:
    def __init__(
        self,
        node: Node,
        scene_xml: str,
        physics_dt: float,
        enable_viewer: bool,
        use_keep_stand: bool,
        tilt_downforce_threshold_deg: float,
        tilt_downforce_scale: float,
        tilt_downforce_exp_gain: float,
        boundary_x_min: float,
        boundary_x_max: float,
        boundary_y_min: float,
        boundary_y_max: float,
        enabled_livox_frames: list[str],
        scene_geometry: dict[str, object],
    ) -> None:
        self.node = node
        self.physics_dt = max(float(physics_dt), 1e-6)
        self.enable_viewer = bool(enable_viewer)
        self.use_keep_stand = bool(use_keep_stand)
        self.tilt_downforce_threshold_rad = math.radians(
            max(float(tilt_downforce_threshold_deg), 0.0)
        )
        self.tilt_downforce_scale = max(float(tilt_downforce_scale), 0.0)
        self.tilt_downforce_exp_gain = max(float(tilt_downforce_exp_gain), 0.0)
        self.boundary_x_min = float(boundary_x_min)
        self.boundary_x_max = float(boundary_x_max)
        self.boundary_y_min = float(boundary_y_min)
        self.boundary_y_max = float(boundary_y_max)
        self.scene_geometry = scene_geometry
        self.enabled_livox_frames = list(enabled_livox_frames)
        self.model = mujoco.MjModel.from_xml_string(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewer_import = None
        self._viewer_sync_failed = False
        self.running = False
        self.physics_lock = threading.Lock()
        self.physics_thread = None
        self.motion_provider = None
        self.latest_sim_time_ns = 0
        self.odom_origin_x = 0.0
        self.odom_origin_y = 0.0
        self.odom_origin_yaw = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.gimbal_joint_pos = 0.0
        self.gimbal_joint_vel = 0.0
        self.clock_pub = node.create_publisher(rosgraph_msgs.msg.Clock, "/clock", 10)
        self.keep_stand_geomgroup = np.zeros(6, dtype=np.uint8)
        self.keep_stand_geomgroup[COLLISION_GEOM_GROUP] = 1
        self.gimbal_joint_id = self._require_name(
            mujoco.mjtObj.mjOBJ_JOINT,
            JOINT_GIMBAL_YAW,
        )
        self.gimbal_qpos_adr = self.model.jnt_qposadr[self.gimbal_joint_id]
        self.gimbal_dof_adr = self.model.jnt_dofadr[self.gimbal_joint_id]
        self.gimbal_body_id = self._require_name(
            mujoco.mjtObj.mjOBJ_BODY,
            FRAME_GIMBAL,
        )
        self.base_body_id = self._require_name(
            mujoco.mjtObj.mjOBJ_BODY,
            FRAME_BASE_LINK,
        )
        self.livox_handles: dict[str, dict[str, int]] = {}
        for frame_name in self.enabled_livox_frames:
            self.livox_handles[frame_name] = self._bind_livox_frame(frame_name)
        mujoco.mj_forward(self.model, self.data)
        self._initialize_odom_reference_locked()
        self._start_viewer_if_requested()

    def _require_name(self, obj_type, name: str) -> int:
        obj_id = int(mujoco.mj_name2id(self.model, obj_type, name))
        if obj_id < 0:
            raise RuntimeError(f"MuJoCo object not found: {name}")
        return obj_id

    def _bind_livox_frame(self, frame_name: str) -> dict[str, int]:
        body_id = self._require_name(
            mujoco.mjtObj.mjOBJ_BODY,
            livox_accel_debug_body_name(frame_name),
        )
        imu_site_id = self._require_name(
            mujoco.mjtObj.mjOBJ_SITE,
            livox_imu_site_name(frame_name),
        )
        accel_geom_id = self._require_name(
            mujoco.mjtObj.mjOBJ_GEOM,
            livox_accel_debug_geom_name(frame_name),
        )
        acc_sensor_id = self._require_name(
            mujoco.mjtObj.mjOBJ_SENSOR,
            livox_accel_sensor_name(frame_name),
        )
        gyro_sensor_id = self._require_name(
            mujoco.mjtObj.mjOBJ_SENSOR,
            livox_gyro_sensor_name(frame_name),
        )
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError(f"MuJoCo mocap body not found: {frame_name}")
        return {
            "imu_site_id": imu_site_id,
            "accel_geom_id": accel_geom_id,
            "mocap_id": mocap_id,
            "acc_sensor_adr": int(self.model.sensor_adr[acc_sensor_id]),
            "gyro_sensor_adr": int(self.model.sensor_adr[gyro_sensor_id]),
        }

    def set_motion_provider(self, motion_provider) -> None:
        self.motion_provider = motion_provider

    def make_lidar_args(self) -> dict[str, object]:
        return {
            "geomgroup": np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8),
            "bodyexclude": self.gimbal_body_id,
        }

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.physics_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.physics_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.physics_thread is not None:
            self.physics_thread.join(timeout=1.0)
        self._close_viewer()

    def _start_viewer_if_requested(self) -> None:
        if not self.enable_viewer:
            self.node.get_logger().info(
                "MuJoCo viewer disabled by parameter enable_viewer:=false."
            )
            return
        try:
            self._viewer_import = importlib.import_module("mujoco.viewer")
        except Exception:
            self._viewer_import = None
        if self._viewer_import is None:
            self.node.get_logger().warn("MuJoCo viewer unavailable; running headless.")
            return
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self.node.get_logger().warn("MuJoCo viewer disabled: no DISPLAY/WAYLAND_DISPLAY.")
            return
        try:
            self.viewer = self._viewer_import.launch_passive(
                self.model,
                self.data,
                show_left_ui=True,
                show_right_ui=True,
            )
            with self.viewer.lock():
                self.viewer.opt.geomgroup[1] = 1
                self.viewer.opt.geomgroup[0] = 0
                self.viewer.opt.geomgroup[2] = 0
                self.viewer.opt.geomgroup[3] = 0
            self.node.get_logger().info(
                "MuJoCo viewer launched. geom groups: render=on, lidar_trace=off, "
                "collision_debug=off, lidar_debug=off."
            )
        except Exception as exc:
            self.viewer = None
            self.node.get_logger().warn(
                f"Failed to launch MuJoCo viewer: {type(exc).__name__}: {exc}"
            )

    def _sync_viewer(self) -> None:
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            self.viewer = None
            self.node.get_logger().warn("MuJoCo viewer closed; continuing headless.")
            return
        try:
            self.viewer.sync()
        except Exception as exc:
            if not self._viewer_sync_failed:
                self.node.get_logger().warn(
                    f"MuJoCo viewer sync failed: {type(exc).__name__}: {exc}"
                )
                self._viewer_sync_failed = True
            self.viewer = None

    def _close_viewer(self) -> None:
        if self.viewer is None:
            return
        try:
            self.viewer.close()
        except Exception:
            pass
        self.viewer = None

    def _sim_time_ns_locked(self) -> int:
        sim_elapsed_ns = int(round(float(self.data.time) * 1_000_000_000.0))
        return max(sim_elapsed_ns, 0)

    def _stamp_from_ns(self, timestamp_ns: int) -> builtin_interfaces.msg.Time:
        stamp = builtin_interfaces.msg.Time()
        safe_ns = max(int(timestamp_ns), 0)
        stamp.sec = safe_ns // 1_000_000_000
        stamp.nanosec = safe_ns % 1_000_000_000
        return stamp

    def capture_sim_stamp_locked(self) -> builtin_interfaces.msg.Time:
        self.latest_sim_time_ns = self._sim_time_ns_locked()
        return self._stamp_from_ns(self.latest_sim_time_ns)

    def read_imu_for_frame_locked(self, frame_name: str) -> tuple[np.ndarray, np.ndarray]:
        handles = self.livox_handles[frame_name]
        acc_adr = handles["acc_sensor_adr"]
        gyro_adr = handles["gyro_sensor_adr"]
        acc = self.data.sensordata[acc_adr : acc_adr + 3].copy()
        gyro = self.data.sensordata[gyro_adr : gyro_adr + 3].copy()
        return acc, gyro

    def _update_imu_accel_debug_geom(self, frame_name: str, acc: np.ndarray) -> None:
        handles = self.livox_handles[frame_name]
        accel_xy_imu = np.array([float(acc[0]), float(acc[1]), 0.0], dtype=np.float64)
        accel_xy_norm = float(np.linalg.norm(accel_xy_imu))
        geom_length = accel_xy_norm * float(self.scene_geometry["imu_accel_debug_scale"])
        min_length = float(self.scene_geometry["imu_accel_debug_min_length"])
        if accel_xy_norm <= 1e-9 or geom_length <= min_length:
            direction_imu = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            geom_length = min_length
        else:
            direction_imu = accel_xy_imu / accel_xy_norm
        site_id = handles["imu_site_id"]
        site_rot = self.data.site_xmat[site_id].reshape(3, 3).copy()
        direction_world = site_rot @ direction_imu
        direction_unit = normalize_vector(direction_world)
        if direction_unit is None:
            direction_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        geom_half_height = max(geom_length * 0.5, min_length * 0.5)
        geom_center = self.data.site_xpos[site_id].copy() + direction_unit * geom_half_height
        mocap_id = handles["mocap_id"]
        self.data.mocap_pos[mocap_id, :] = geom_center
        self.data.mocap_quat[mocap_id, :] = np.asarray(
            quat_from_local_z_axis(direction_unit),
            dtype=np.float64,
        )
        self.model.geom_size[handles["accel_geom_id"], 0] = float(
            self.scene_geometry["imu_accel_debug_radius"]
        )
        self.model.geom_size[handles["accel_geom_id"], 1] = geom_half_height

    def _initialize_odom_reference_locked(self) -> None:
        gimbal_pos = self.data.qpos[0:3]
        gimbal_quat = self.data.qpos[3:7]
        gimbal_yaw = math.atan2(
            2.0 * (gimbal_quat[0] * gimbal_quat[3] + gimbal_quat[1] * gimbal_quat[2]),
            1.0 - 2.0 * (gimbal_quat[2] ** 2 + gimbal_quat[3] ** 2),
        )
        self.odom_origin_x = float(gimbal_pos[0])
        self.odom_origin_y = float(gimbal_pos[1])
        self.odom_origin_yaw = gimbal_yaw
        self.gimbal_joint_pos = float(self.data.qpos[self.gimbal_qpos_adr])
        self.gimbal_joint_vel = 0.0

    def _compute_surface_aligned_quat(
        self,
        base_pos: np.ndarray,
        yaw: float,
    ) -> np.ndarray | None:
        gravity_dir = normalize_vector(np.array(self.model.opt.gravity, dtype=np.float64))
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
            self.gimbal_body_id,
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
        x_axis = normalize_vector(x_hint - z_axis * float(np.dot(x_hint, z_axis)))
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

    def _compute_tilt_downforce(self, quat: np.ndarray) -> float:
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, quat)
        body_z_axis = rot_mat.reshape(3, 3)[:, 2]
        cos_tilt = clamp(float(body_z_axis[2]), -1.0, 1.0)
        tilt_rad = math.acos(cos_tilt)
        tilt_excess = max(0.0, tilt_rad - self.tilt_downforce_threshold_rad)
        if tilt_excess <= 0.0:
            return 0.0
        return self.tilt_downforce_scale * (
            math.exp(self.tilt_downforce_exp_gain * tilt_excess) - 1.0
        )

    def read_joint_state(self) -> tuple[builtin_interfaces.msg.Time, float, float]:
        with self.physics_lock:
            stamp = self.capture_sim_stamp_locked()
            return stamp, float(self.gimbal_joint_pos), float(self.gimbal_joint_vel)

    def physics_loop(self) -> None:
        last_clock_ns = 0
        while self.running and rclpy.ok():
            current_sim_time_ns = self.latest_sim_time_ns
            with self.physics_lock:
                gimbal_quat = self.data.qpos[3:7]
                gimbal_yaw = math.atan2(
                    2.0 * (gimbal_quat[0] * gimbal_quat[3] + gimbal_quat[1] * gimbal_quat[2]),
                    1.0 - 2.0 * (gimbal_quat[2] ** 2 + gimbal_quat[3] ** 2),
                )
                target_local_vx = 0.0
                target_local_vy = 0.0
                target_chassis_yaw_rate = 0.0
                target_gimbal_yaw_rate = 0.0
                if self.motion_provider is not None:
                    now_ros = self.node.get_clock().now()
                    (
                        target_local_vx,
                        target_local_vy,
                        target_chassis_yaw_rate,
                        target_gimbal_yaw_rate,
                    ) = self.motion_provider(now_ros, self.physics_dt)
                target_world_velocity = np.array(
                    [
                        target_local_vx * math.cos(gimbal_yaw) - target_local_vy * math.sin(gimbal_yaw),
                        target_local_vx * math.sin(gimbal_yaw) + target_local_vy * math.cos(gimbal_yaw),
                    ],
                    dtype=np.float64,
                )
                self.data.xfrc_applied[self.gimbal_body_id, :] = 0.0
                self.data.xfrc_applied[self.base_body_id, :] = 0.0
                tilt_downforce = self._compute_tilt_downforce(
                    self.data.xquat[self.base_body_id].copy()
                )
                if tilt_downforce > 0.0:
                    self.data.xfrc_applied[self.base_body_id, 2] = -tilt_downforce
                self.gimbal_joint_pos = float(self.data.qpos[self.gimbal_qpos_adr])
                self.gimbal_joint_vel = target_chassis_yaw_rate - target_gimbal_yaw_rate
                self.data.qpos[self.gimbal_qpos_adr] = self.gimbal_joint_pos
                self.data.qvel[0] = float(target_world_velocity[0])
                self.data.qvel[1] = float(target_world_velocity[1])
                self.data.qvel[5] = float(target_gimbal_yaw_rate)
                self.data.qvel[self.gimbal_dof_adr] = float(self.gimbal_joint_vel)
                mujoco.mj_step(self.model, self.data)
                quat = self.data.qpos[3:7].copy()
                _, _, yaw = quat_to_rpy(quat)
                if self.use_keep_stand:
                    aligned_quat = self._compute_surface_aligned_quat(
                        self.data.qpos[0:3].copy(),
                        yaw,
                    )
                    if aligned_quat is not None:
                        self.data.qpos[3:7] = aligned_quat
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
                for frame_name in self.enabled_livox_frames:
                    acc, _ = self.read_imu_for_frame_locked(frame_name)
                    self._update_imu_accel_debug_geom(frame_name, acc / GRAVITY_M_S2)
                mujoco.mj_forward(self.model, self.data)
                pos = self.data.qpos[0:3]
                quat = self.data.qpos[3:7]
                gimbal_yaw = math.atan2(
                    2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                    1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
                )
                dx = float(pos[0]) - self.odom_origin_x
                dy = float(pos[1]) - self.odom_origin_y
                cos_origin = math.cos(self.odom_origin_yaw)
                sin_origin = math.sin(self.odom_origin_yaw)
                self.odom_x = cos_origin * dx + sin_origin * dy
                self.odom_y = -sin_origin * dx + cos_origin * dy
                self.odom_yaw = wrap_to_pi(gimbal_yaw - self.odom_origin_yaw)
                self.gimbal_joint_pos = float(self.data.qpos[self.gimbal_qpos_adr])
                self.gimbal_joint_vel = float(self.data.qvel[self.gimbal_dof_adr])
                current_sim_time_ns = self._sim_time_ns_locked()
                self.latest_sim_time_ns = current_sim_time_ns
                self._sync_viewer()
            if current_sim_time_ns - last_clock_ns >= 10_000_000:
                msg = rosgraph_msgs.msg.Clock()
                msg.clock = self._stamp_from_ns(current_sim_time_ns)
                self.clock_pub.publish(msg)
                last_clock_ns = current_sim_time_ns
            time.sleep(self.physics_dt)


class SentrySimNode(Node):
    def __init__(self) -> None:
        super().__init__("sentry_sim_node")
        enable_viewer = bool(self.declare_parameter("enable_viewer", True).value)
        robot_description_xacro_path = str(
            self.declare_parameter("robot_description_xacro_path", "").value
        ).strip()
        robot_init_location = _coerce_vector3_param(
            self.declare_parameter(
                "robot_init_location",
                list(DEFAULT_ROBOT_INIT_LOCATION),
            ).value,
            "robot_init_location",
        )
        boundary_x_min = float(
            self.declare_parameter("boundary_x_min", DEFAULT_BOUNDARY_X_MIN).value
        )
        boundary_x_max = float(
            self.declare_parameter("boundary_x_max", DEFAULT_BOUNDARY_X_MAX).value
        )
        boundary_y_min = float(
            self.declare_parameter("boundary_y_min", DEFAULT_BOUNDARY_Y_MIN).value
        )
        boundary_y_max = float(
            self.declare_parameter("boundary_y_max", DEFAULT_BOUNDARY_Y_MAX).value
        )
        if boundary_x_min >= boundary_x_max:
            raise ValueError("boundary_x_min must be smaller than boundary_x_max")
        if boundary_y_min >= boundary_y_max:
            raise ValueError("boundary_y_min must be smaller than boundary_y_max")
        use_keep_stand = bool(
            self.declare_parameter("use_keep_stand", DEFAULT_USE_KEEP_STAND).value
        )
        tilt_downforce_threshold_deg = max(
            float(
                self.declare_parameter(
                    "tilt_downforce_threshold_deg",
                    DEFAULT_TILT_DOWNFORCE_THRESHOLD_DEG,
                ).value
            ),
            0.0,
        )
        tilt_downforce_scale = max(
            float(
                self.declare_parameter(
                    "tilt_downforce_scale",
                    DEFAULT_TILT_DOWNFORCE_SCALE,
                ).value
            ),
            0.0,
        )
        tilt_downforce_exp_gain = max(
            float(
                self.declare_parameter(
                    "tilt_downforce_exp_gain",
                    DEFAULT_TILT_DOWNFORCE_EXP_GAIN,
                ).value
            ),
            0.0,
        )
        physics_dt = max(
            float(self.declare_parameter("physics_dt", DEFAULT_PHYSICS_DT).value),
            1e-6,
        )
        enable_left_livox = bool(
            self.declare_parameter("enable_left_livox", True).value
        )
        enable_right_livox = bool(
            self.declare_parameter("enable_right_livox", True).value
        )
        scene_geometry = load_scene_geometry_params(self)
        try:
            frame_tree = load_robot_frame_tree(robot_description_xacro_path)
            if robot_description_xacro_path:
                self.get_logger().info(
                    f"Loaded robot frame tree from xacro: {robot_description_xacro_path}"
                )
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to load xacro frame tree, fall back to defaults: {type(exc).__name__}: {exc}"
            )
            frame_tree = load_robot_frame_tree("")
        assets_dir = resolve_assets_dir()
        meshdir = os.path.join(assets_dir, "meshes")
        scene_xml = build_scene_xml(
            meshdir=meshdir,
            frame_tree=frame_tree,
            robot_init_location=robot_init_location,
            boundary_x_min=boundary_x_min,
            boundary_x_max=boundary_x_max,
            boundary_y_min=boundary_y_min,
            boundary_y_max=boundary_y_max,
            scene_geometry=scene_geometry,
            enable_left_livox=enable_left_livox,
            enable_right_livox=enable_right_livox,
            physics_dt=physics_dt,
        )
        enabled_livox_frames: list[str] = []
        if enable_left_livox:
            enabled_livox_frames.append(FRAME_LEFT_LIVOX)
        if enable_right_livox:
            enabled_livox_frames.append(FRAME_RIGHT_LIVOX)
        self.runtime = SimulationRuntime(
            node=self,
            scene_xml=scene_xml,
            physics_dt=physics_dt,
            enable_viewer=enable_viewer,
            use_keep_stand=use_keep_stand,
            tilt_downforce_threshold_deg=tilt_downforce_threshold_deg,
            tilt_downforce_scale=tilt_downforce_scale,
            tilt_downforce_exp_gain=tilt_downforce_exp_gain,
            boundary_x_min=boundary_x_min,
            boundary_x_max=boundary_x_max,
            boundary_y_min=boundary_y_min,
            boundary_y_max=boundary_y_max,
            enabled_livox_frames=enabled_livox_frames,
            scene_geometry=scene_geometry,
        )
        self.component_manager = ComponentManager(self, self.runtime)
        self.runtime.set_motion_provider(self.component_manager.compute_motion_command)
        self.runtime.start()
        self.get_logger().info(
            f"sentry_sim_node ready: enable_left_livox={enable_left_livox}, "
            f"enable_right_livox={enable_right_livox}, physics_dt={physics_dt:.4f}, "
            f"robot_init_location={list(robot_init_location)}, "
            f"boundary_x=[{boundary_x_min:.2f}, {boundary_x_max:.2f}], "
            f"boundary_y=[{boundary_y_min:.2f}, {boundary_y_max:.2f}]"
        )

    def destroy_node(self) -> None:
        if hasattr(self, "runtime"):
            self.runtime.stop()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SentrySimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
