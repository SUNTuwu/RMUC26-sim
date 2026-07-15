from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import sensor_msgs.msg
from livox_ros_driver2.msg import CustomMsg
from mujoco_lidar import MjLidarWrapper
from mujoco_lidar.scan_gen import LivoxGenerator
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ._livox_bridge import (
    DEFAULT_POINT_OFFSET_STEP_NS,
    DEFAULT_REFLECTIVITY_DECAY_METERS,
    DEFAULT_REFLECTIVITY_MAX,
    ranges_to_custom_msg,
    ranges_to_pointcloud2_msg,
)

from ..scene_builder import livox_lidar_site_name


DEFAULT_LIDAR_RATE = 10.0
DEFAULT_IMU_RATE = 500.0
DEFAULT_LIDAR_CUTOFF = 30.0
DEFAULT_MID360_POINTS_PER_SCAN = 4032
GRAVITY_M_S2 = 9.81
IMU_GYRO_STATIC_DEADBAND_RAD_S = 5e-3
DEBUG_SESSION_ENV = ".dbg/custommsg-all-zero.env"
DEBUG_DEFAULT_SERVER_URL = "http://127.0.0.1:7777/event"
DEBUG_DEFAULT_SESSION_ID = "custommsg-all-zero"


def debug_report(
    hypothesis_id: str,
    location: str,
    msg: str,
    data: dict | None = None,
    run_id: str = "sentry-sim-node",
) -> None:
    server_url = DEBUG_DEFAULT_SERVER_URL
    session_id = DEBUG_DEFAULT_SESSION_ID
    current_run_id = os.environ.get("DEBUG_RUN_ID", run_id)
    try:
        with open(DEBUG_SESSION_ENV, "r", encoding="utf-8") as env_file:
            for line in env_file:
                if line.startswith("DEBUG_SERVER_URL="):
                    server_url = line.split("=", 1)[1].strip() or server_url
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip() or session_id
    except OSError:
        return
    payload = {
        "sessionId": session_id,
        "runId": current_run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": f"[DEBUG] {msg}",
        "data": data or {},
        "ts": int(time.time() * 1000),
    }
    try:
        request = urllib.request.Request(
            server_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=0.2).read()
    except Exception:
        pass


def summarize_custom_msg(msg: CustomMsg) -> dict:
    total_points = int(msg.point_num)
    zero_xyz_count = 0
    nonzero_reflectivity_count = 0
    tag_valid_count = 0
    tag_zero_count = 0
    first_valid_index = None
    first_nonzero_xyz_index = None
    sample_points = []
    for idx in range(total_points):
        point = msg.points[idx]
        if idx < min(total_points, 6):
            sample_points.append(
                {
                    "idx": int(idx),
                    "x": float(point.x),
                    "y": float(point.y),
                    "z": float(point.z),
                    "reflectivity": int(point.reflectivity),
                    "tag": int(point.tag),
                    "line": int(point.line),
                    "offset_time": int(point.offset_time),
                }
            )
        if point.tag == 0x10:
            tag_valid_count += 1
            if first_valid_index is None:
                first_valid_index = int(idx)
        elif point.tag == 0x00:
            tag_zero_count += 1
        if point.reflectivity != 0:
            nonzero_reflectivity_count += 1
        if abs(float(point.x)) <= 1e-9 and abs(float(point.y)) <= 1e-9 and abs(float(point.z)) <= 1e-9:
            zero_xyz_count += 1
        elif first_nonzero_xyz_index is None:
            first_nonzero_xyz_index = int(idx)
    return {
        "point_num": total_points,
        "tag_valid_count": tag_valid_count,
        "tag_zero_count": tag_zero_count,
        "zero_xyz_count": zero_xyz_count,
        "nonzero_reflectivity_count": nonzero_reflectivity_count,
        "first_valid_index": first_valid_index,
        "first_nonzero_xyz_index": first_nonzero_xyz_index,
        "sample_points": sample_points,
    }


def lidar_pointcloud_topic_from_ip(ip_address: str) -> str:
    return f"/livox/lidar_{ip_address.replace('.', '_')}"


def lidar_pointcloud2_topic_from_ip(ip_address: str) -> str:
    return f"{lidar_pointcloud_topic_from_ip(ip_address)}/pointcloud"


def simulator_imu_topic_from_ip(ip_address: str) -> str:
    return f"/sim/imu_{ip_address.replace('.', '_')}"


class LivoxComponent:
    def __init__(
        self,
        node,
        runtime,
        frame_name: str,
        ip_address: str,
        lidar_id: int,
    ) -> None:
        self.node = node
        self.runtime = runtime
        self.frame_name = frame_name
        self.ip_address = ip_address
        self.lidar_id = int(lidar_id)
        self.lidar_rate = max(float(node.get_parameter("lidar_rate").value), 1.0)
        self.imu_rate = max(float(node.get_parameter("imu_rate").value), 1.0)
        self.lidar_cutoff = max(float(node.get_parameter("lidar_cutoff").value), 0.1)
        self.mid360_points_per_scan = max(
            int(node.get_parameter("mid360_points_per_scan").value),
            1,
        )
        self.point_offset_step_ns = max(
            int(node.get_parameter("livox_point_offset_step_ns").value),
            0,
        )
        self.reflectivity_max = max(
            float(node.get_parameter("livox_reflectivity_max").value),
            0.0,
        )
        self.reflectivity_decay_meters = max(
            float(node.get_parameter("livox_reflectivity_decay_meters").value),
            1e-6,
        )
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publish_debug_counter = 0
        self.pc_pub = node.create_publisher(
            CustomMsg,
            lidar_pointcloud_topic_from_ip(ip_address),
            sensor_qos,
        )
        self.pc2_pub = node.create_publisher(
            sensor_msgs.msg.PointCloud2,
            lidar_pointcloud2_topic_from_ip(ip_address),
            sensor_qos,
        )
        self.imu_pub = node.create_publisher(
            sensor_msgs.msg.Imu,
            simulator_imu_topic_from_ip(ip_address),
            sensor_qos,
        )
        self.generator = LivoxGenerator("mid360")
        self.generator.samples = self.mid360_points_per_scan
        self.wrapper = MjLidarWrapper(
            runtime.model,
            site_name=livox_lidar_site_name(frame_name),
            backend="cpu",
            cutoff_dist=self.lidar_cutoff,
            args=runtime.make_lidar_args(),
        )
        self.lidar_timer = node.create_timer(1.0 / self.lidar_rate, self.publish_lidar)
        self.imu_timer = node.create_timer(1.0 / self.imu_rate, self.publish_imu)
        self.node.get_logger().info(
            f"LivoxComponent ready: frame={self.frame_name}, ip={self.ip_address}, "
            f"custom_topic={lidar_pointcloud_topic_from_ip(ip_address)}, "
            f"pointcloud2_topic={lidar_pointcloud2_topic_from_ip(ip_address)}, "
            f"imu_topic={simulator_imu_topic_from_ip(ip_address)}, "
            f"lidar_rate={self.lidar_rate:.1f}, imu_rate={self.imu_rate:.1f}, "
            f"points_per_scan={self.mid360_points_per_scan}, qos=reliable"
        )

    def _angles_to_ray_dirs(
        self,
        ray_theta: np.ndarray,
        ray_phi: np.ndarray,
    ) -> np.ndarray:
        cos_theta = np.cos(ray_theta)
        sin_theta = np.sin(ray_theta)
        cos_phi = np.cos(ray_phi)
        sin_phi = np.sin(ray_phi)
        return np.stack(
            [
                cos_theta * cos_phi,
                sin_theta * cos_phi,
                sin_phi,
            ],
            axis=1,
        )

    def _build_imu_message(self, stamp, acc: np.ndarray, gyro: np.ndarray):
        msg = sensor_msgs.msg.Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_name
        acc_in_g = acc / GRAVITY_M_S2
        gyro_filtered = gyro.copy()
        gyro_filtered[np.abs(gyro_filtered) < IMU_GYRO_STATIC_DEADBAND_RAD_S] = 0.0
        msg.linear_acceleration.x = float(acc_in_g[0])
        msg.linear_acceleration.y = float(acc_in_g[1])
        msg.linear_acceleration.z = float(acc_in_g[2])
        msg.angular_velocity.x = float(gyro_filtered[0])
        msg.angular_velocity.y = float(gyro_filtered[1])
        msg.angular_velocity.z = float(gyro_filtered[2])
        msg.orientation.w = 1.0
        for i in range(9):
            msg.orientation_covariance[i] = 0.0
            msg.angular_velocity_covariance[i] = 0.0
            msg.linear_acceleration_covariance[i] = 0.0
        return msg

    def publish_lidar(self) -> None:
        with self.runtime.physics_lock:
            stamp = self.runtime.capture_sim_stamp_locked()
            ray_theta, ray_phi = self.generator.sample_ray_angles()
            ray_dirs = self._angles_to_ray_dirs(ray_theta, ray_phi)
            ranges = self.wrapper.trace_rays(self.runtime.data, ray_theta, ray_phi)
            points_local = self.wrapper.get_hit_points()
        custom_msg = ranges_to_custom_msg(
            ranges,
            ray_dirs,
            ray_phi,
            self.frame_name,
            stamp,
            lidar_id=self.lidar_id,
            points_local=points_local,
            point_offset_step_ns=self.point_offset_step_ns,
            reflectivity_max=self.reflectivity_max,
            reflectivity_decay_meters=self.reflectivity_decay_meters,
        )
        pc2_msg = ranges_to_pointcloud2_msg(
            ranges,
            ray_dirs,
            ray_phi,
            self.frame_name,
            stamp,
            points_local=points_local,
            point_offset_step_ns=self.point_offset_step_ns,
            reflectivity_max=self.reflectivity_max,
            reflectivity_decay_meters=self.reflectivity_decay_meters,
        )
        self.pc_pub.publish(custom_msg)
        self.pc2_pub.publish(pc2_msg)
        self._publish_debug_counter += 1
        if self._publish_debug_counter <= 3 or self._publish_debug_counter % 20 == 0:
            debug_report(
                "LIVOX",
                "comp_livox.py:publish_lidar",
                "Published Livox messages",
                {
                    "frame": self.frame_name,
                    "iteration": int(self._publish_debug_counter),
                    "custom_topic": lidar_pointcloud_topic_from_ip(self.ip_address),
                    "pointcloud2_topic": lidar_pointcloud2_topic_from_ip(self.ip_address),
                    "imu_topic": simulator_imu_topic_from_ip(self.ip_address),
                    "point_num": int(custom_msg.point_num),
                    "pc2_width": int(pc2_msg.width),
                    "custom_subscribers": int(self.pc_pub.get_subscription_count()),
                    "pc2_subscribers": int(self.pc2_pub.get_subscription_count()),
                    "summary": summarize_custom_msg(custom_msg),
                },
            )

    def publish_imu(self) -> None:
        with self.runtime.physics_lock:
            stamp = self.runtime.capture_sim_stamp_locked()
            acc, gyro = self.runtime.read_imu_for_frame_locked(self.frame_name)
        self.imu_pub.publish(self._build_imu_message(stamp, acc, gyro))
