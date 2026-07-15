#!/usr/bin/env python3
"""Adapt keyboard commands to the simulator chassis command topic."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

SMALL_GYRO_TOGGLE_THRESHOLD = 0.5
ZERO_EPSILON = 1e-6


class ChassisAdapter(Node):
    """Maintain small-gyro state and publish normalized simulator commands."""

    def __init__(self) -> None:
        super().__init__("chassis_adapter")
        self.declare_parameter("keyboard_cmd_vel_topic", "/sim/keyboard/cmd_vel")
        self.declare_parameter("cmd_vel_out_topic", "/sim/cmd_vel")
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("small_gyro_spin_rate", 6.0)
        self.declare_parameter("small_gyro_toggle_timeout_sec", 1.0)
        self.declare_parameter("linear_cmd_timeout_sec", 0.5)

        self.keyboard_cmd_vel_topic = str(
            self.get_parameter("keyboard_cmd_vel_topic").value
        )
        self.cmd_vel_out_topic = str(self.get_parameter("cmd_vel_out_topic").value)
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

        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_gimbal_wz = 0.0
        self.small_gyro_enabled = False
        self.last_small_gyro_toggle_time = None
        self.last_linear_cmd_time = None
        self.last_publish_signature = None
        self.linear_timeout_active = False

        self.keyboard_subscription = self.create_subscription(
            Twist,
            self.keyboard_cmd_vel_topic,
            self.keyboard_cmd_callback,
            10,
        )
        self.publisher = self.create_publisher(Twist, self.cmd_vel_out_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.get_logger().info(
            "chassis_adapter ready: "
            f"keyboard={self.keyboard_cmd_vel_topic}, "
            f"publish={self.cmd_vel_out_topic}, "
            f"publish_rate={self.publish_rate:.1f}, "
            f"small_gyro_spin_rate={self.small_gyro_spin_rate:.2f}, "
            f"small_gyro_toggle_timeout_sec={self.small_gyro_toggle_timeout_sec:.2f}, "
            f"linear_cmd_timeout_sec={self.linear_cmd_timeout_sec:.2f}"
        )

    def _linear_command_is_fresh(self) -> bool:
        if self.last_linear_cmd_time is None:
            return False
        if self.linear_cmd_timeout_sec <= 0.0:
            return True
        age = (self.get_clock().now() - self.last_linear_cmd_time).nanoseconds / 1e9
        return age <= self.linear_cmd_timeout_sec

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
            self.target_vx = float(msg.linear.x)
            self.target_vy = float(msg.linear.y)
            self.target_gimbal_wz = float(msg.angular.z)
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

    def timer_callback(self) -> None:
        msg = Twist()
        linear_cmd_is_fresh = self._linear_command_is_fresh()
        msg.linear.x = self.target_vx if linear_cmd_is_fresh else 0.0
        msg.linear.y = self.target_vy if linear_cmd_is_fresh else 0.0
        msg.linear.z = 0.0
        msg.angular.x = self.small_gyro_spin_rate if self.small_gyro_enabled else 0.0
        msg.angular.y = 0.0
        msg.angular.z = self.target_gimbal_wz

        if not linear_cmd_is_fresh and not self.linear_timeout_active:
            self.linear_timeout_active = True
            self.get_logger().info(
                "linear cmd timeout, zero linear velocity and keep small gyro state"
            )

        publish_signature = (
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        )
        self.publisher.publish(msg)
        if publish_signature != self.last_publish_signature:
            self.last_publish_signature = publish_signature
            self.get_logger().info(
                "published simulator command: "
                f"linear=({msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.linear.z:.2f}), "
                f"angular=({msg.angular.x:.2f}, {msg.angular.y:.2f}, {msg.angular.z:.2f})"
            )


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
