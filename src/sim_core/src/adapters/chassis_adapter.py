#!/usr/bin/env python3
"""Adapt keyboard and processed navigation commands to simulator commands."""

from __future__ import annotations

import math

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, UInt16

SMALL_GYRO_TOGGLE_THRESHOLD = 0.5
ZERO_EPSILON = 1e-6


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def rotate_gimbal_velocity_to_base(
    gimbal_vx: float,
    gimbal_vy: float,
    gimbal_to_base_yaw: float,
) -> tuple[float, float]:
    """Express a planar main_gimbal_link velocity in base_link coordinates."""
    cos_yaw = math.cos(gimbal_to_base_yaw)
    sin_yaw = math.sin(gimbal_to_base_yaw)
    # JointState gives base_link yaw relative to main_gimbal_link, so vector
    # coordinates transform with the inverse rotation R_z(-yaw).
    return (
        cos_yaw * gimbal_vx + sin_yaw * gimbal_vy,
        -sin_yaw * gimbal_vx + cos_yaw * gimbal_vy,
    )


def rotate_nav_velocity_to_base(
    nav_vx: float,
    nav_vy: float,
    chassis_orientation: float,
) -> tuple[float, float]:
    """Apply the lower-board chassis-orientation correction."""
    cos_theta = math.cos(chassis_orientation)
    sin_theta = math.sin(chassis_orientation)
    return (
        cos_theta * nav_vx - sin_theta * nav_vy,
        sin_theta * nav_vx + cos_theta * nav_vy,
    )


def update_chassis_orientation(
    received_counter: int | None,
    saved_counter: int | None,
    saved_yaw: float | None,
    current_imu_yaw: float,
    optional_spin_offset: float,
) -> tuple[float, int | None, float]:
    """Update the lower-board Nav-frame yaw anchor."""
    if saved_yaw is None or received_counter != saved_counter:
        return 0.0, received_counter, current_imu_yaw
    return (
        wrap_to_pi(optional_spin_offset + saved_yaw - current_imu_yaw),
        saved_counter,
        saved_yaw,
    )


def nav_chassis_yaw_rate(follow_marker: float, requested_yaw_rate: float) -> float:
    """Use processed angular.x solely as the chassis-rotation switch."""
    follow = follow_marker != 1.0
    return 0.0 if follow else requested_yaw_rate


def nav_spin_offset(
    chassis_yaw_rate: float,
    gain_sec: float,
    min_yaw_rate: float,
) -> float:
    """Match the lower-board conditional spin feed-forward offset."""
    if abs(chassis_yaw_rate) <= min_yaw_rate:
        return 0.0
    return gain_sec * chassis_yaw_rate


class ChassisAdapter(Node):
    """Merge keyboard gimbal state with keyboard or Nav chassis commands."""

    def __init__(self) -> None:
        super().__init__("chassis_adapter")
        self.declare_parameter("keyboard_cmd_vel_topic", "/sim/keyboard/cmd_vel")
        self.declare_parameter("nav_cmd_vel_topic", "/cmd_vel_processed")
        self.declare_parameter("nav_update_counter_topic", "/update_counter")
        self.declare_parameter("base_yaw_topic", "/sim/base_yaw")
        self.declare_parameter("cmd_vel_out_topic", "/sim/cmd_vel")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("gimbal_joint_name", "gimbal_to_base")
        self.declare_parameter("publish_rate", 200.0)
        self.declare_parameter("small_gyro_spin_rate", 6.0)
        self.declare_parameter("small_gyro_toggle_timeout_sec", 1.0)
        self.declare_parameter("linear_cmd_timeout_sec", 0.5)
        self.declare_parameter("nav_cmd_timeout_sec", 0.5)
        self.declare_parameter("spin_offset_gain_sec", 0.0)
        self.declare_parameter("spin_offset_min_yaw_rate", 1.0)

        self.keyboard_cmd_vel_topic = str(
            self.get_parameter("keyboard_cmd_vel_topic").value
        )
        self.nav_cmd_vel_topic = str(self.get_parameter("nav_cmd_vel_topic").value)
        self.nav_update_counter_topic = str(
            self.get_parameter("nav_update_counter_topic").value
        )
        self.base_yaw_topic = str(self.get_parameter("base_yaw_topic").value)
        self.cmd_vel_out_topic = str(self.get_parameter("cmd_vel_out_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.gimbal_joint_name = str(self.get_parameter("gimbal_joint_name").value)
        self.publish_rate = max(float(self.get_parameter("publish_rate").value), 1.0)
        self.small_gyro_spin_rate = float(
            self.get_parameter("small_gyro_spin_rate").value
        )
        self.small_gyro_toggle_timeout_sec = max(
            float(self.get_parameter("small_gyro_toggle_timeout_sec").value), 0.0
        )
        self.linear_cmd_timeout_sec = max(
            float(self.get_parameter("linear_cmd_timeout_sec").value),
            0.0,
        )
        self.nav_cmd_timeout_sec = max(
            float(self.get_parameter("nav_cmd_timeout_sec").value),
            0.0,
        )
        self.spin_offset_gain_sec = float(
            self.get_parameter("spin_offset_gain_sec").value
        )
        self.spin_offset_min_yaw_rate = max(
            float(self.get_parameter("spin_offset_min_yaw_rate").value),
            0.0,
        )

        self.target_gimbal_vx = 0.0
        self.target_gimbal_vy = 0.0
        self.last_gimbal_v_yaw = 0.0
        self.gimbal_to_base_yaw = 0.0
        self.small_gyro_enabled = False
        self.received_counter = None
        self.saved_counter = None
        self.saved_yaw = None
        self.current_imu_yaw = None
        self.chassis_orientation = 0.0
        self.last_small_gyro_toggle_time = None
        self.last_linear_cmd_time = None
        self.last_nav_cmd_time = None
        self.last_publish_source = None
        self.linear_timeout_active = False
        self.nav_timeout_active = False
        self.nav_waiting_for_yaw_warning_active = False
        self.invalid_nav_warning_active = False
        self.invalid_yaw_warning_active = False

        self.publisher = self.create_publisher(Twist, self.cmd_vel_out_topic, 10)
        self.keyboard_subscription = self.create_subscription(
            Twist,
            self.keyboard_cmd_vel_topic,
            self.keyboard_cmd_callback,
            10,
        )
        self.nav_subscription = self.create_subscription(
            Twist,
            self.nav_cmd_vel_topic,
            self.nav_cmd_callback,
            10,
        )
        self.nav_update_counter_subscription = self.create_subscription(
            UInt16,
            self.nav_update_counter_topic,
            self.nav_update_counter_callback,
            10,
        )
        self.base_yaw_subscription = self.create_subscription(
            Float64,
            self.base_yaw_topic,
            self.base_yaw_callback,
            10,
        )
        self.joint_state_subscription = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            10,
        )
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.get_logger().info(
            "chassis_adapter ready: "
            f"keyboard={self.keyboard_cmd_vel_topic}, "
            f"nav={self.nav_cmd_vel_topic}, "
            f"counter={self.nav_update_counter_topic}, "
            f"base_yaw={self.base_yaw_topic}, "
            f"publish={self.cmd_vel_out_topic}, "
            f"joint_state={self.joint_state_topic}, "
            f"gimbal_joint={self.gimbal_joint_name}, "
            f"publish_rate={self.publish_rate:.1f}, "
            f"small_gyro_spin_rate={self.small_gyro_spin_rate:.2f}, "
            f"small_gyro_toggle_timeout_sec={self.small_gyro_toggle_timeout_sec:.2f}, "
            f"linear_cmd_timeout_sec={self.linear_cmd_timeout_sec:.2f}, "
            f"nav_cmd_timeout_sec={self.nav_cmd_timeout_sec:.2f}, "
            f"spin_offset_gain_sec={self.spin_offset_gain_sec:.4f}, "
            f"spin_offset_min_yaw_rate={self.spin_offset_min_yaw_rate:.2f}"
        )

    def _linear_command_is_fresh(self) -> bool:
        if self.last_linear_cmd_time is None:
            return False
        if self.linear_cmd_timeout_sec <= 0.0:
            return True
        age = (self.get_clock().now() - self.last_linear_cmd_time).nanoseconds / 1e9
        return age <= self.linear_cmd_timeout_sec

    def _nav_command_is_fresh(self) -> bool:
        if self.last_nav_cmd_time is None:
            return False
        if self.nav_cmd_timeout_sec <= 0.0:
            return True
        age = (self.get_clock().now() - self.last_nav_cmd_time).nanoseconds / 1e9
        return age <= self.nav_cmd_timeout_sec

    def _can_toggle_small_gyro(self) -> bool:
        if self.small_gyro_toggle_timeout_sec <= 0.0:
            self.last_small_gyro_toggle_time = self.get_clock().now()
            return True
        now = self.get_clock().now()
        if self.last_small_gyro_toggle_time is None:
            self.last_small_gyro_toggle_time = now
            return True
        elapsed = (now - self.last_small_gyro_toggle_time).nanoseconds / 1e9
        if elapsed < self.small_gyro_toggle_timeout_sec:
            return False
        self.last_small_gyro_toggle_time = now
        return True

    def _is_toggle_only_message(self, msg: Twist) -> bool:
        if msg.angular.y <= SMALL_GYRO_TOGGLE_THRESHOLD:
            return False
        return (
            abs(msg.linear.x) <= ZERO_EPSILON
            and abs(msg.linear.y) <= ZERO_EPSILON
            and abs(msg.linear.z) <= ZERO_EPSILON
            and abs(msg.angular.x) <= ZERO_EPSILON
            and abs(msg.angular.z) <= ZERO_EPSILON
        )

    def keyboard_cmd_callback(self, msg: Twist) -> None:
        if not self._is_toggle_only_message(msg):
            self.target_gimbal_vx = float(msg.linear.x)
            self.target_gimbal_vy = float(msg.linear.y)
            self.last_gimbal_v_yaw = float(msg.angular.z)
            self.last_linear_cmd_time = self.get_clock().now()
            self.linear_timeout_active = False

        if msg.angular.y > SMALL_GYRO_TOGGLE_THRESHOLD:
            if self._can_toggle_small_gyro():
                self.small_gyro_enabled = not self.small_gyro_enabled
                self.get_logger().info(
                    "small gyro toggled "
                    f"{'on' if self.small_gyro_enabled else 'off'}"
                )
            else:
                self.get_logger().info("small gyro toggle ignored by debounce timeout")

    def nav_update_counter_callback(self, msg: UInt16) -> None:
        self.received_counter = int(msg.data)

    def base_yaw_callback(self, msg: Float64) -> None:
        yaw = float(msg.data)
        if not math.isfinite(yaw):
            if not self.invalid_yaw_warning_active:
                self.invalid_yaw_warning_active = True
                self.get_logger().warn("ignored non-finite base yaw")
            return
        self.current_imu_yaw = wrap_to_pi(yaw)
        self.invalid_yaw_warning_active = False
        self.nav_waiting_for_yaw_warning_active = False

    def nav_cmd_callback(self, msg: Twist) -> None:
        nav_values = (
            float(msg.linear.x),
            float(msg.linear.y),
            float(msg.angular.x),
            float(msg.angular.z),
        )
        if not all(math.isfinite(value) for value in nav_values):
            if not self.invalid_nav_warning_active:
                self.invalid_nav_warning_active = True
                self.get_logger().warn("ignored non-finite Nav command")
            return
        self.invalid_nav_warning_active = False

        if self.current_imu_yaw is None:
            if not self.nav_waiting_for_yaw_warning_active:
                self.nav_waiting_for_yaw_warning_active = True
                self.get_logger().warn(
                    f"ignored Nav command before receiving {self.base_yaw_topic}"
                )
            return

        nav_vx, nav_vy, follow_marker, requested_yaw_rate = nav_values
        chassis_yaw_rate = nav_chassis_yaw_rate(
            follow_marker,
            requested_yaw_rate,
        )
        optional_spin_offset = nav_spin_offset(
            chassis_yaw_rate,
            self.spin_offset_gain_sec,
            self.spin_offset_min_yaw_rate,
        )
        (
            self.chassis_orientation,
            self.saved_counter,
            self.saved_yaw,
        ) = update_chassis_orientation(
            self.received_counter,
            self.saved_counter,
            self.saved_yaw,
            self.current_imu_yaw,
            optional_spin_offset,
        )

        output = Twist()
        output.linear.x, output.linear.y = rotate_nav_velocity_to_base(
            nav_vx,
            nav_vy,
            self.chassis_orientation,
        )
        output.linear.z = 0.0
        output.angular.x = self.last_gimbal_v_yaw
        output.angular.y = 0.0
        output.angular.z = chassis_yaw_rate

        self.last_nav_cmd_time = self.get_clock().now()
        self.nav_timeout_active = False
        self._publish_command(output, "nav")

    def joint_state_callback(self, msg: JointState) -> None:
        try:
            index = msg.name.index(self.gimbal_joint_name)
        except ValueError:
            return

        if index < len(msg.position):
            self.gimbal_to_base_yaw = float(msg.position[index])

    def _publish_command(self, msg: Twist, source: str) -> None:
        self.publisher.publish(msg)
        self.last_gimbal_v_yaw = float(msg.angular.x)
        if source == self.last_publish_source:
            return
        self.last_publish_source = source
        self.get_logger().info(f"active chassis command source -> {source}")

    def timer_callback(self) -> None:
        if self._nav_command_is_fresh():
            return
        if self.last_nav_cmd_time is not None and not self.nav_timeout_active:
            self.nav_timeout_active = True
            self.get_logger().info("Nav cmd timeout, restoring keyboard chassis control")

        msg = Twist()
        linear_cmd_is_fresh = self._linear_command_is_fresh()
        gimbal_vx = self.target_gimbal_vx if linear_cmd_is_fresh else 0.0
        gimbal_vy = self.target_gimbal_vy if linear_cmd_is_fresh else 0.0
        msg.linear.x, msg.linear.y = rotate_gimbal_velocity_to_base(
            gimbal_vx,
            gimbal_vy,
            self.gimbal_to_base_yaw,
        )
        msg.linear.z = 0.0
        msg.angular.x = self.last_gimbal_v_yaw
        msg.angular.y = 0.0
        msg.angular.z = (
            self.small_gyro_spin_rate if self.small_gyro_enabled else 0.0
        )

        if not linear_cmd_is_fresh and not self.linear_timeout_active:
            self.linear_timeout_active = True
            self.get_logger().info(
                "linear cmd timeout, zero linear velocity and keep small gyro state"
            )
        self._publish_command(msg, "keyboard")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ChassisAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
