#!/usr/bin/env python3
"""Acceleration-limited command arbiter for sentry_sim."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

SMALL_GYRO_MODE_THRESHOLD = 0.5


def move_towards(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return min(target, current + max_delta)
    return max(target, current - max_delta)


class ChassisNode(Node):
    """Arbitrates keyboard/nav commands, applies accel limits, republishes merged velocity."""

    def __init__(self) -> None:
        super().__init__("chassis")
        self.declare_parameter("keyboard_cmd_vel_topic", "/cmd_vel_keyboard")
        self.declare_parameter("nav_cmd_vel_topic", "/cmd_vel_processed")
        self.declare_parameter("cmd_vel_out_topic", "/cmd_vel_chassis")
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("max_linear_accel", 1.5)
        self.declare_parameter("max_angular_accel", 3.0)
        self.declare_parameter("keyboard_cmd_timeout_sec", 0.2)
        self.declare_parameter("nav_cmd_timeout_sec", 0.5)

        self.keyboard_cmd_vel_topic = str(
            self.get_parameter("keyboard_cmd_vel_topic").value
        )
        self.nav_cmd_vel_topic = str(self.get_parameter("nav_cmd_vel_topic").value)
        self.cmd_vel_out_topic = str(self.get_parameter("cmd_vel_out_topic").value)
        self.publish_rate = max(float(self.get_parameter("publish_rate").value), 1.0)
        self.max_linear_accel = max(float(self.get_parameter("max_linear_accel").value), 0.0)
        self.max_angular_accel = max(float(self.get_parameter("max_angular_accel").value), 0.0)
        self.keyboard_cmd_timeout_sec = max(
            float(self.get_parameter("keyboard_cmd_timeout_sec").value), 0.0
        )
        self.nav_cmd_timeout_sec = max(
            float(self.get_parameter("nav_cmd_timeout_sec").value), 0.0
        )

        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_mode_z = 0.0
        self.target_gimbal_wz = 0.0
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_mode_z = 0.0
        self.current_gimbal_wz = 0.0
        self.keyboard_target_vx = 0.0
        self.keyboard_target_vy = 0.0
        self.keyboard_target_gimbal_wz = 0.0
        self.keyboard_small_gyro_enabled = False
        self.keyboard_input_active = False
        self.nav_target_vx = 0.0
        self.nav_target_vy = 0.0
        self.nav_target_wz = 0.0
        self.last_keyboard_cmd_time = self.get_clock().now()
        self.last_nav_cmd_time = self.get_clock().now()
        self.last_step_time = self.get_clock().now()
        self.last_selected_source = "idle"
        self.last_publish_signature = None

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
        self.publisher = self.create_publisher(Twist, self.cmd_vel_out_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.get_logger().info(
            "chassis ready: "
            f"keyboard={self.keyboard_cmd_vel_topic}, nav={self.nav_cmd_vel_topic}, "
            f"publish {self.cmd_vel_out_topic}, "
            f"max_linear_accel={self.max_linear_accel:.2f}, "
            f"max_angular_accel={self.max_angular_accel:.2f}"
        )

    def _has_effective_input(self, msg: Twist) -> bool:
        return any(
            abs(value) > 1e-6
            for value in (
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.z,
            )
        )

    def _is_fresh(self, stamp, timeout_sec: float, now) -> bool:
        if timeout_sec <= 0.0:
            return True
        age = (now - stamp).nanoseconds / 1e9
        return age <= timeout_sec

    def keyboard_cmd_callback(self, msg: Twist) -> None:
        self.keyboard_target_vx = float(msg.linear.x)
        self.keyboard_target_vy = float(msg.linear.y)
        self.keyboard_target_gimbal_wz = float(msg.angular.z)
        self.keyboard_input_active = self._has_effective_input(msg)
        new_small_gyro_enabled = bool(msg.linear.z > SMALL_GYRO_MODE_THRESHOLD)
        if new_small_gyro_enabled != self.keyboard_small_gyro_enabled:
            self.keyboard_small_gyro_enabled = new_small_gyro_enabled
            self.get_logger().info(
                "keyboard small gyro set "
                f"{'on' if self.keyboard_small_gyro_enabled else 'off'}"
            )
        else:
            self.keyboard_small_gyro_enabled = new_small_gyro_enabled
        self.last_keyboard_cmd_time = self.get_clock().now()

    def nav_cmd_callback(self, msg: Twist) -> None:
        self.nav_target_vx = float(msg.linear.x)
        self.nav_target_vy = float(msg.linear.y)
        self.nav_target_wz = float(msg.angular.z)
        self.last_nav_cmd_time = self.get_clock().now()

    def _select_active_command(self, now) -> tuple[str, float, float, float, float]:
        keyboard_fresh = self.keyboard_input_active and self._is_fresh(
            self.last_keyboard_cmd_time,
            self.keyboard_cmd_timeout_sec,
            now,
        )
        if keyboard_fresh:
            return (
                "keyboard",
                self.keyboard_target_vx,
                self.keyboard_target_vy,
                1.0 if self.keyboard_small_gyro_enabled else 0.0,
                self.keyboard_target_gimbal_wz,
            )

        if self.keyboard_small_gyro_enabled:
            return (
                "keyboard_spin",
                0.0,
                0.0,
                1.0,
                0.0,
            )

        nav_fresh = self._is_fresh(self.last_nav_cmd_time, self.nav_cmd_timeout_sec, now)
        if nav_fresh:
            return (
                "nav",
                self.nav_target_vx,
                self.nav_target_vy,
                0.0,
                self.nav_target_wz,
            )

        return ("idle", 0.0, 0.0, 0.0, 0.0)

    def _should_publish(self, selected_source: str, msg: Twist) -> bool:
        if selected_source == "idle":
            return False
        if selected_source == "keyboard_spin":
            return True
        return any(
            abs(value) > 1e-6
            for value in (
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.z,
            )
        )

    def timer_callback(self) -> None:
        now = self.get_clock().now()
        selected_source, target_vx, target_vy, target_mode_z, target_gimbal_wz = (
            self._select_active_command(now)
        )
        dt = (now - self.last_step_time).nanoseconds / 1e9
        self.last_step_time = now
        if dt <= 0.0:
            dt = 1.0 / self.publish_rate
        dt = min(dt, 0.2)

        self.target_vx = target_vx
        self.target_vy = target_vy
        self.target_mode_z = target_mode_z
        self.target_gimbal_wz = target_gimbal_wz

        max_linear_delta = self.max_linear_accel * dt
        max_angular_delta = self.max_angular_accel * dt

        self.current_vx = move_towards(self.current_vx, self.target_vx, max_linear_delta)
        self.current_vy = move_towards(self.current_vy, self.target_vy, max_linear_delta)
        self.current_mode_z = self.target_mode_z
        self.current_gimbal_wz = move_towards(
            self.current_gimbal_wz, self.target_gimbal_wz, max_angular_delta
        )

        msg = Twist()
        msg.linear.x = self.current_vx
        msg.linear.y = self.current_vy
        msg.linear.z = self.current_mode_z
        msg.angular.z = self.current_gimbal_wz

        if selected_source != self.last_selected_source:
            self.last_selected_source = selected_source
            self.get_logger().info(
                "selected command source switched to "
                f"{selected_source}: vx={self.target_vx:.2f}, "
                f"vy={self.target_vy:.2f}, mode_z={self.target_mode_z:.2f}, "
                f"angular.z={self.target_gimbal_wz:.2f}"
            )

        if not self._should_publish(selected_source, msg):
            self.last_publish_signature = None
            return

        try:
            publish_signature = (
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.z,
            )
            self.publisher.publish(msg)
            if publish_signature != self.last_publish_signature:
                self.last_publish_signature = publish_signature
                self.get_logger().info(
                    f"published {selected_source}: vx={msg.linear.x:.2f}, "
                    f"vy={msg.linear.y:.2f}, mode_z={msg.linear.z:.2f}, "
                    f"angular.z={msg.angular.z:.2f}"
                )
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().warn(
                    f"failed to publish processed cmd_vel: {type(exc).__name__}: {exc}"
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ChassisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
