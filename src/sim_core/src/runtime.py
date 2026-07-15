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
import std_msgs.msg
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
    base_force_debug_body_name,
    base_force_debug_geom_name,
    build_scene_xml,
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
DEFAULT_PHYSICS_DT = 0.002
DEFAULT_ROBOT_INIT_LOCATION = (0.0, 0.0, 10.0)
DEFAULT_VIEWER_CAMERA_DISTANCE = 3.0
DEFAULT_VIEWER_CAMERA_AZIMUTH = 135.0
DEFAULT_VIEWER_CAMERA_ELEVATION = -25.0
DEFAULT_VIEWER_FOCUS_TOPIC = "/sim/focus"
DEFAULT_VIEWER_FOCUS_ON_START = True
DEFAULT_VIEWER_FOCUS_YAW_OFFSET_DEG = 45.0
DEFAULT_VIEWER_FOCUS_SMOOTHING_TIME_SEC = 0.5
DEFAULT_VIEWER_MANUAL_CAMERA_TIMEOUT_SEC = 1.0
DEFAULT_VIEWER_MANUAL_CAMERA_POSITION_EPSILON = 1e-5
DEFAULT_VIEWER_MANUAL_CAMERA_ANGLE_EPSILON_DEG = 1e-3


def _coerce_vector3_param(raw_value, param_name: str) -> tuple[float, float, float]:
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 3:
        raise ValueError(f"{param_name} must be a 3-element list [x, y, z]")
    return tuple(float(value) for value in raw_value)


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def interpolate_angle_degrees(current: float, target: float, alpha: float) -> float:
    delta = (target - current + 180.0) % 360.0 - 180.0
    return current + alpha * delta


def normalize_vector(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None
    return vec / norm


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
        enabled_livox_frames: list[str],
        scene_geometry: dict[str, object],
    ) -> None:
        self.node = node
        self.physics_dt = max(float(physics_dt), 1e-6)
        self.enable_viewer = bool(enable_viewer)
        self.viewer_camera_distance = float(
            node.declare_parameter(
                "viewer_camera_distance",
                DEFAULT_VIEWER_CAMERA_DISTANCE,
            ).value
        )
        if self.viewer_camera_distance <= 0.0:
            raise ValueError("viewer_camera_distance must be positive")
        self.viewer_camera_azimuth = float(
            node.declare_parameter(
                "viewer_camera_azimuth",
                DEFAULT_VIEWER_CAMERA_AZIMUTH,
            ).value
        )
        self.viewer_camera_elevation = float(
            node.declare_parameter(
                "viewer_camera_elevation",
                DEFAULT_VIEWER_CAMERA_ELEVATION,
            ).value
        )
        self.viewer_focus_topic = str(
            node.declare_parameter(
                "viewer_focus_topic",
                DEFAULT_VIEWER_FOCUS_TOPIC,
            ).value
        ).strip()
        if not self.viewer_focus_topic:
            raise ValueError("viewer_focus_topic must not be empty")
        self.viewer_focus_on_start = bool(
            node.declare_parameter(
                "viewer_focus_on_start",
                DEFAULT_VIEWER_FOCUS_ON_START,
            ).value
        )
        self.viewer_focus_yaw_offset_deg = float(
            node.declare_parameter(
                "viewer_focus_yaw_offset_deg",
                DEFAULT_VIEWER_FOCUS_YAW_OFFSET_DEG,
            ).value
        )
        self.viewer_focus_smoothing_time_sec = float(
            node.declare_parameter(
                "viewer_focus_smoothing_time_sec",
                DEFAULT_VIEWER_FOCUS_SMOOTHING_TIME_SEC,
            ).value
        )
        if self.viewer_focus_smoothing_time_sec <= 0.0:
            raise ValueError("viewer_focus_smoothing_time_sec must be positive")
        self.viewer_manual_camera_timeout_sec = float(
            node.declare_parameter(
                "viewer_manual_camera_timeout_sec",
                DEFAULT_VIEWER_MANUAL_CAMERA_TIMEOUT_SEC,
            ).value
        )
        if self.viewer_manual_camera_timeout_sec <= 0.0:
            raise ValueError("viewer_manual_camera_timeout_sec must be positive")
        self.viewer_manual_camera_position_epsilon = float(
            node.declare_parameter(
                "viewer_manual_camera_position_epsilon",
                DEFAULT_VIEWER_MANUAL_CAMERA_POSITION_EPSILON,
            ).value
        )
        if self.viewer_manual_camera_position_epsilon <= 0.0:
            raise ValueError("viewer_manual_camera_position_epsilon must be positive")
        self.viewer_manual_camera_angle_epsilon_deg = float(
            node.declare_parameter(
                "viewer_manual_camera_angle_epsilon_deg",
                DEFAULT_VIEWER_MANUAL_CAMERA_ANGLE_EPSILON_DEG,
            ).value
        )
        if self.viewer_manual_camera_angle_epsilon_deg <= 0.0:
            raise ValueError("viewer_manual_camera_angle_epsilon_deg must be positive")
        self.scene_geometry = scene_geometry
        self.enabled_livox_frames = list(enabled_livox_frames)
        self.model = mujoco.MjModel.from_xml_string(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewer_import = None
        self._viewer_sync_failed = False
        self._viewer_focus_state_lock = threading.Lock()
        self._viewer_focus_requested = self.viewer_focus_on_start
        self._viewer_focus_applied = self.viewer_focus_on_start
        self._viewer_focus_control_active = self.viewer_focus_on_start
        self._viewer_focus_release_remaining_sec = 0.0
        self._viewer_manual_control_active = False
        self._viewer_manual_last_motion_time = 0.0
        self._viewer_camera_expected_state = None
        self._viewer_camera_observed_state = None
        self.running = False
        self.physics_lock = threading.Lock()
        self.physics_thread = None
        self.control_provider = None
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
        self.base_force_debug_body_id = self._require_name(
            mujoco.mjtObj.mjOBJ_BODY,
            base_force_debug_body_name(),
        )
        self.base_force_debug_geom_id = self._require_name(
            mujoco.mjtObj.mjOBJ_GEOM,
            base_force_debug_geom_name(),
        )
        self.base_force_debug_mocap_id = int(
            self.model.body_mocapid[self.base_force_debug_body_id]
        )
        if self.base_force_debug_mocap_id < 0:
            raise RuntimeError("MuJoCo mocap body not found: base force debug")
        self.livox_handles: dict[str, dict[str, int]] = {}
        for frame_name in self.enabled_livox_frames:
            self.livox_handles[frame_name] = self._bind_livox_frame(frame_name)
        self.viewer_focus_sub = node.create_subscription(
            std_msgs.msg.Bool,
            self.viewer_focus_topic,
            self._on_viewer_focus_command,
            10,
        )
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

    def set_control_provider(self, control_provider) -> None:
        self.control_provider = control_provider

    def _on_viewer_focus_command(self, msg: std_msgs.msg.Bool) -> None:
        if not msg.data:
            return
        with self._viewer_focus_state_lock:
            self._viewer_focus_requested = not self._viewer_focus_requested
            focus_enabled = self._viewer_focus_requested
        state = "enabled" if focus_enabled else "disabled"
        self.node.get_logger().info(
            f"Viewer focus {state} by {self.viewer_focus_topic}"
        )

    def make_lidar_args(self) -> dict[str, object]:
        return {
            "geomgroup": np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8),
            "bodyexclude": self.base_body_id,
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
                self.viewer.cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
                self.viewer.cam.trackbodyid = -1
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.lookat[:] = self.data.xpos[self.gimbal_body_id]
                self.viewer.cam.distance = self.viewer_camera_distance
                self.viewer.cam.azimuth = self.viewer_camera_azimuth
                self.viewer.cam.elevation = self.viewer_camera_elevation
                initial_camera_state = self._capture_viewer_camera_state_locked()
                self._viewer_camera_expected_state = initial_camera_state
                self._viewer_camera_observed_state = initial_camera_state
            self.node.get_logger().info(
                "MuJoCo viewer launched and aimed at main_gimbal_link. geom groups: "
                "render=on, lidar_trace=off, collision_debug=off, lidar_debug=off."
            )
        except Exception as exc:
            self.viewer = None
            self.node.get_logger().warn(
                f"Failed to launch MuJoCo viewer: {type(exc).__name__}: {exc}"
            )

    def _capture_viewer_camera_state_locked(
        self,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        camera = self.viewer.cam
        values = np.array(
            [
                *camera.lookat,
                camera.distance,
                camera.azimuth,
                camera.elevation,
            ],
            dtype=np.float64,
        )
        mode = (
            int(camera.type),
            int(camera.trackbodyid),
            int(camera.fixedcamid),
        )
        return values, mode

    def _viewer_camera_state_changed(
        self,
        previous: tuple[np.ndarray, tuple[int, int, int]] | None,
        current: tuple[np.ndarray, tuple[int, int, int]],
    ) -> bool:
        if previous is None:
            return False
        previous_values, previous_mode = previous
        current_values, current_mode = current
        if previous_mode != current_mode:
            return True
        position_delta = np.abs(current_values[:4] - previous_values[:4])
        if np.any(position_delta > self.viewer_manual_camera_position_epsilon):
            return True
        azimuth_delta = abs(
            (current_values[4] - previous_values[4] + 180.0) % 360.0 - 180.0
        )
        elevation_delta = abs(current_values[5] - previous_values[5])
        return (
            azimuth_delta > self.viewer_manual_camera_angle_epsilon_deg
            or elevation_delta > self.viewer_manual_camera_angle_epsilon_deg
        )

    def _update_viewer_focus_locked(self) -> None:
        with self._viewer_focus_state_lock:
            focus_requested = self._viewer_focus_requested

        camera_state = self._capture_viewer_camera_state_locked()

        if focus_requested != self._viewer_focus_applied:
            self._viewer_focus_applied = focus_requested
            self._viewer_manual_control_active = False
            self._viewer_camera_expected_state = camera_state
            self._viewer_camera_observed_state = camera_state
            if focus_requested:
                self._viewer_focus_control_active = True
                self._viewer_focus_release_remaining_sec = 0.0
            elif self._viewer_focus_control_active:
                self._viewer_focus_release_remaining_sec = (
                    self.viewer_focus_smoothing_time_sec
                )

        if not self._viewer_focus_control_active:
            return

        if focus_requested:
            now = time.monotonic()
            if self._viewer_manual_control_active:
                if self._viewer_camera_state_changed(
                    self._viewer_camera_observed_state,
                    camera_state,
                ):
                    self._viewer_camera_observed_state = camera_state
                    self._viewer_manual_last_motion_time = now
                elif (
                    now - self._viewer_manual_last_motion_time
                    >= self.viewer_manual_camera_timeout_sec
                ):
                    self._viewer_manual_control_active = False
                    self._viewer_camera_expected_state = camera_state
                    self.node.get_logger().info(
                        "Manual viewer camera control ended; resuming focus."
                    )
                if self._viewer_manual_control_active:
                    return
            elif self._viewer_camera_state_changed(
                self._viewer_camera_expected_state,
                camera_state,
            ):
                self._viewer_manual_control_active = True
                self._viewer_manual_last_motion_time = now
                self._viewer_camera_observed_state = camera_state
                self.node.get_logger().info(
                    "Manual viewer camera control detected; focus paused."
                )
                return

        follow_weight = 1.0
        if not focus_requested:
            follow_weight = (
                self._viewer_focus_release_remaining_sec
                / self.viewer_focus_smoothing_time_sec
            )
            if follow_weight <= 0.0:
                self._viewer_focus_control_active = False
                return
            self._viewer_focus_release_remaining_sec = max(
                self._viewer_focus_release_remaining_sec - self.physics_dt,
                0.0,
            )
            follow_weight = follow_weight * follow_weight * (3.0 - 2.0 * follow_weight)

        alpha = (
            1.0 - math.exp(-self.physics_dt / self.viewer_focus_smoothing_time_sec)
        ) * follow_weight
        gimbal_pos = self.data.xpos[self.gimbal_body_id]
        gimbal_rot = self.data.xmat[self.gimbal_body_id].reshape(3, 3)
        gimbal_yaw_deg = math.degrees(
            math.atan2(float(gimbal_rot[1, 0]), float(gimbal_rot[0, 0]))
        )
        desired_azimuth = gimbal_yaw_deg + self.viewer_focus_yaw_offset_deg

        camera = self.viewer.cam
        camera.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
        camera.trackbodyid = -1
        camera.fixedcamid = -1
        camera.lookat[:] += alpha * (gimbal_pos - camera.lookat)
        camera.distance += alpha * (self.viewer_camera_distance - camera.distance)
        camera.azimuth = interpolate_angle_degrees(
            camera.azimuth,
            desired_azimuth,
            alpha,
        )
        camera.elevation += alpha * (
            self.viewer_camera_elevation - camera.elevation
        )
        self._viewer_camera_expected_state = self._capture_viewer_camera_state_locked()
        self._viewer_camera_observed_state = self._viewer_camera_expected_state

    def _sync_viewer(self) -> None:
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            self.viewer = None
            self.node.get_logger().warn("MuJoCo viewer closed; continuing headless.")
            return
        try:
            with self.viewer.lock():
                self._update_viewer_focus_locked()
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

    def _update_imu_accel_debug_geom(
        self,
        frame_name: str,
        acc_in_g: np.ndarray,
    ) -> None:
        handles = self.livox_handles[frame_name]
        accel_imu = np.asarray(acc_in_g, dtype=np.float64).copy()
        accel_imu[2] += 1.0
        accel_norm = float(np.linalg.norm(accel_imu))
        geom_length = accel_norm * float(self.scene_geometry["imu_accel_debug_scale"])
        min_length = float(self.scene_geometry["imu_accel_debug_min_length"])
        if accel_norm <= 1e-9 or geom_length <= min_length:
            direction_imu = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            geom_length = min_length
        else:
            direction_imu = accel_imu / accel_norm
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

    def _update_base_force_debug_geom(self, base_force_world: np.ndarray) -> None:
        force_xy_world = np.array(
            [float(base_force_world[0]), float(base_force_world[1]), 0.0],
            dtype=np.float64,
        )
        force_norm = float(np.linalg.norm(force_xy_world))
        geom_length = force_norm * float(self.scene_geometry["base_force_debug_scale"])
        min_length = float(self.scene_geometry["base_force_debug_min_length"])
        if force_norm <= 1e-9 or geom_length <= min_length:
            direction_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            geom_length = min_length
        else:
            direction_world = force_xy_world / force_norm
        geom_half_height = max(geom_length * 0.5, min_length * 0.5)
        base_pos = self.data.xpos[self.base_body_id].copy()
        base_debug_height = 0.5 * float(self.scene_geometry["base_height"])
        geom_center = base_pos + np.array(
            [0.0, 0.0, base_debug_height],
            dtype=np.float64,
        ) + direction_world * geom_half_height
        self.data.mocap_pos[self.base_force_debug_mocap_id, :] = geom_center
        self.data.mocap_quat[self.base_force_debug_mocap_id, :] = np.asarray(
            quat_from_local_z_axis(direction_world),
            dtype=np.float64,
        )
        self.model.geom_size[self.base_force_debug_geom_id, 0] = float(
            self.scene_geometry["base_force_debug_radius"]
        )
        self.model.geom_size[self.base_force_debug_geom_id, 1] = geom_half_height

    def _initialize_odom_reference_locked(self) -> None:
        gimbal_pos = self.data.xpos[self.gimbal_body_id].copy()
        gimbal_quat = self.data.xquat[self.gimbal_body_id].copy()
        gimbal_yaw = math.atan2(
            2.0 * (gimbal_quat[0] * gimbal_quat[3] + gimbal_quat[1] * gimbal_quat[2]),
            1.0 - 2.0 * (gimbal_quat[2] ** 2 + gimbal_quat[3] ** 2),
        )
        self.odom_origin_x = float(gimbal_pos[0])
        self.odom_origin_y = float(gimbal_pos[1])
        self.odom_origin_yaw = gimbal_yaw
        self.gimbal_joint_pos = -float(self.data.qpos[self.gimbal_qpos_adr])
        self.gimbal_joint_vel = 0.0

    def _read_body_velocity_world(
        self,
        body_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        return velocity[0:3], velocity[3:6]

    def read_joint_state(self) -> tuple[builtin_interfaces.msg.Time, float, float]:
        with self.physics_lock:
            stamp = self.capture_sim_stamp_locked()
            return stamp, float(self.gimbal_joint_pos), float(self.gimbal_joint_vel)

    def physics_loop(self) -> None:
        while self.running and rclpy.ok():
            current_sim_time_ns = self.latest_sim_time_ns
            with self.physics_lock:
                base_rot_mat = (
                    self.data.xmat[self.base_body_id].reshape(3, 3).copy()
                )
                gimbal_rot_mat = (
                    self.data.xmat[self.gimbal_body_id].reshape(3, 3).copy()
                )
                base_angular_velocity_world, base_linear_velocity_world = (
                    self._read_body_velocity_world(self.base_body_id)
                )
                gimbal_angular_velocity_world, _ = self._read_body_velocity_world(
                    self.gimbal_body_id
                )
                chassis_yaw_rate = float(
                    np.dot(base_angular_velocity_world, base_rot_mat[:, 2])
                )
                gimbal_yaw_rate = float(
                    np.dot(gimbal_angular_velocity_world, gimbal_rot_mat[:, 2])
                )
                base_force_world = np.zeros(3, dtype=np.float64)
                chassis_yaw_torque = 0.0
                gimbal_yaw_torque = 0.0
                if self.control_provider is not None:
                    now_ros = self.node.get_clock().now()
                    (
                        base_force_world,
                        chassis_yaw_torque,
                        gimbal_yaw_torque,
                    ) = self.control_provider(
                        now_ros,
                        self.physics_dt,
                        gimbal_rot_mat,
                        base_linear_velocity_world,
                        chassis_yaw_rate,
                        gimbal_yaw_rate,
                    )
                base_force_world = np.asarray(base_force_world, dtype=np.float64)
                self.data.xfrc_applied[self.gimbal_body_id, :] = 0.0
                self.data.xfrc_applied[self.base_body_id, :] = 0.0
                self.data.xfrc_applied[self.base_body_id, 0:3] = base_force_world
                base_torque_world = base_rot_mat[:, 2] * float(chassis_yaw_torque)
                self.data.xfrc_applied[self.base_body_id, 3:6] = base_torque_world
                self.data.qfrc_applied[self.gimbal_dof_adr] = float(gimbal_yaw_torque)
                mujoco.mj_step(self.model, self.data)
                for frame_name in self.enabled_livox_frames:
                    acc, _ = self.read_imu_for_frame_locked(frame_name)
                    self._update_imu_accel_debug_geom(frame_name, acc / GRAVITY_M_S2)
                self._update_base_force_debug_geom(base_force_world)
                mujoco.mj_forward(self.model, self.data)
                pos = self.data.xpos[self.gimbal_body_id].copy()
                quat = self.data.xquat[self.gimbal_body_id].copy()
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
                self.gimbal_joint_pos = -float(self.data.qpos[self.gimbal_qpos_adr])
                self.gimbal_joint_vel = -float(self.data.qvel[self.gimbal_dof_adr])
                current_sim_time_ns = self._sim_time_ns_locked()
                self.latest_sim_time_ns = current_sim_time_ns
                self._sync_viewer()
            msg = rosgraph_msgs.msg.Clock()
            msg.clock = self._stamp_from_ns(current_sim_time_ns)
            self.clock_pub.publish(msg)
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
            enabled_livox_frames=enabled_livox_frames,
            scene_geometry=scene_geometry,
        )
        self.component_manager = ComponentManager(self, self.runtime)
        physics_rate = 1.0 / physics_dt
        imu_rate = float(self.get_parameter("imu_rate").value)
        if imu_rate > physics_rate:
            raise ValueError(
                f"imu_rate ({imu_rate:.1f} Hz) must not exceed the physics and /clock "
                f"rate ({physics_rate:.1f} Hz)"
            )
        self.runtime.set_control_provider(self.component_manager.compute_control_action)
        self.runtime.start()
        self.get_logger().info(
            f"sentry_sim_node ready: enable_left_livox={enable_left_livox}, "
            f"enable_right_livox={enable_right_livox}, physics_dt={physics_dt:.4f}, "
            f"clock_rate={physics_rate:.1f}, imu_rate={imu_rate:.1f}, "
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
