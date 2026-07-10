from __future__ import annotations

import geometry_msgs.msg
import sensor_msgs.msg

from .components.comp_chassis import ChassisComponent
from .components.comp_gimbal import GimbalComponent
from .components.comp_livox import (
    DEFAULT_IMU_RATE,
    DEFAULT_LIDAR_CUTOFF,
    DEFAULT_LIDAR_RATE,
    DEFAULT_MID360_POINTS_PER_SCAN,
    DEFAULT_POINT_OFFSET_STEP_NS,
    DEFAULT_REFLECTIVITY_DECAY_METERS,
    DEFAULT_REFLECTIVITY_MAX,
    LivoxComponent,
)
from .frame_tree import FRAME_LEFT_LIVOX, FRAME_RIGHT_LIVOX, JOINT_GIMBAL_YAW


class ComponentManager:
    def __init__(self, node, runtime) -> None:
        self.node = node
        self.runtime = runtime
        self.cmd_vel_timeout_sec = 0.0
        self.active_cmd_source = "idle"
        self.chassis_cmd_vel_topic = str(
            node.declare_parameter("chassis_cmd_vel_topic", "/cmd_vel_chassis").value
        )
        self.cmd_vel_timeout_sec = max(
            float(node.declare_parameter("cmd_vel_timeout_sec", 0.5).value),
            0.0,
        )
        chassis_linear_accel_limit = max(
            float(node.declare_parameter("chassis_linear_accel_limit", 3.0).value),
            0.0,
        )
        chassis_angular_accel_limit = max(
            float(node.declare_parameter("chassis_angular_accel_limit", 6.0).value),
            0.0,
        )
        gimbal_angular_accel_limit = max(
            float(node.declare_parameter("gimbal_angular_accel_limit", 12.0).value),
            0.0,
        )
        node.declare_parameter("left_lidar_ip", "192.168.10.4")
        node.declare_parameter("right_lidar_ip", "192.168.10.5")
        node.declare_parameter("lidar_rate", DEFAULT_LIDAR_RATE)
        node.declare_parameter("imu_rate", DEFAULT_IMU_RATE)
        node.declare_parameter("lidar_cutoff", DEFAULT_LIDAR_CUTOFF)
        node.declare_parameter("mid360_points_per_scan", DEFAULT_MID360_POINTS_PER_SCAN)
        node.declare_parameter("livox_point_offset_step_ns", DEFAULT_POINT_OFFSET_STEP_NS)
        node.declare_parameter("livox_reflectivity_max", DEFAULT_REFLECTIVITY_MAX)
        node.declare_parameter(
            "livox_reflectivity_decay_meters",
            DEFAULT_REFLECTIVITY_DECAY_METERS,
        )
        self.chassis_component = ChassisComponent(
            self.cmd_vel_timeout_sec,
            chassis_linear_accel_limit,
        )
        self.gimbal_component = GimbalComponent(
            self.cmd_vel_timeout_sec,
            chassis_angular_accel_limit,
            gimbal_angular_accel_limit,
        )
        self.livox_components: list[LivoxComponent] = []
        if bool(node.get_parameter("enable_left_livox").value):
            self.livox_components.append(
                LivoxComponent(
                    node,
                    runtime,
                    FRAME_LEFT_LIVOX,
                    str(node.get_parameter("left_lidar_ip").value),
                    lidar_id=5,
                )
            )
        if bool(node.get_parameter("enable_right_livox").value):
            self.livox_components.append(
                LivoxComponent(
                    node,
                    runtime,
                    FRAME_RIGHT_LIVOX,
                    str(node.get_parameter("right_lidar_ip").value),
                    lidar_id=3,
                )
            )
        self.joint_state_pub = node.create_publisher(
            sensor_msgs.msg.JointState,
            "/joint_states",
            10,
        )
        self.cmd_vel_sub = node.create_subscription(
            geometry_msgs.msg.Twist,
            self.chassis_cmd_vel_topic,
            self._cmd_vel_callback,
            10,
        )
        self.joint_state_timer = node.create_timer(0.01, self.publish_joint_state)
        self.node.get_logger().info(
            f"ComponentManager ready: chassis_cmd_vel_topic={self.chassis_cmd_vel_topic}, "
            f"cmd_vel_timeout_sec={self.cmd_vel_timeout_sec:.2f}, "
            f"left_livox={bool(node.get_parameter('enable_left_livox').value)}, "
            f"right_livox={bool(node.get_parameter('enable_right_livox').value)}"
        )

    def _cmd_vel_callback(self, msg: geometry_msgs.msg.Twist) -> None:
        now = self.node.get_clock().now()
        self.chassis_component.update_from_twist(msg, now)
        self.gimbal_component.update_from_twist(msg, now)
        if self.active_cmd_source != self.chassis_cmd_vel_topic:
            self.active_cmd_source = self.chassis_cmd_vel_topic
            self.node.get_logger().info(
                f"active command source -> {self.active_cmd_source}"
            )

    def compute_motion_command(
        self,
        now,
        dt: float,
    ) -> tuple[float, float, float, float]:
        is_cmd_fresh = False
        if self.chassis_component.last_cmd_time is not None:
            age = (now - self.chassis_component.last_cmd_time).nanoseconds / 1e9
            is_cmd_fresh = self.cmd_vel_timeout_sec <= 0.0 or age <= self.cmd_vel_timeout_sec
        if not is_cmd_fresh and self.active_cmd_source != "idle":
            self.active_cmd_source = "idle"
            self.node.get_logger().info(
                f"active command source -> {self.active_cmd_source}"
            )
        vx, vy = self.chassis_component.sample(now, dt)
        chassis_yaw_rate, gimbal_yaw_rate = self.gimbal_component.sample(now, dt)
        return vx, vy, chassis_yaw_rate, gimbal_yaw_rate

    def publish_joint_state(self) -> None:
        stamp, joint_pos, joint_vel = self.runtime.read_joint_state()
        msg = sensor_msgs.msg.JointState()
        msg.header.stamp = stamp
        msg.name = [JOINT_GIMBAL_YAW]
        msg.position = [joint_pos]
        msg.velocity = [joint_vel]
        self.joint_state_pub.publish(msg)
