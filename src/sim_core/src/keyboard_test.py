#!/usr/bin/env python3
"""Event-driven keyboard trigger node for sentry_sim."""

from __future__ import annotations

import threading

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None


class KeyboardTestNode(Node):
    """Publishes `/cmd_vel_keyboard` only on keyboard state changes."""

    def __init__(self) -> None:
        super().__init__("keyboard_test")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel_keyboard")
        self.declare_parameter("linear_speed", 1.0)
        self.declare_parameter("gimbal_yaw_speed", 1.5)

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.gimbal_yaw_speed = float(self.get_parameter("gimbal_yaw_speed").value)

        if pynput_keyboard is None:
            raise RuntimeError("pynput is required for keyboard_test")

        self.publisher = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self._listener = None
        self._state_lock = threading.Lock()
        self._active_keys: set[str] = set()
        self._pressed_space = False

        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

        self.get_logger().info(
            "keyboard_test ready: event-driven publish to "
            f"{self.cmd_vel_topic}, WASD=translation, "
            "up/down=forward/backward, left/right=gimbal yaw, space=toggle small gyro, q=quit."
        )

    def _normalize_key(self, key) -> str | None:
        if isinstance(key, pynput_keyboard.KeyCode):
            if key.char is None:
                return None
            return key.char.lower()

        special_keys = {
            pynput_keyboard.Key.up: "up",
            pynput_keyboard.Key.down: "down",
            pynput_keyboard.Key.left: "left",
            pynput_keyboard.Key.right: "right",
            pynput_keyboard.Key.space: "space",
        }
        return special_keys.get(key)

    def _compose_twist(self, toggle_small_gyro: bool) -> Twist:
        with self._state_lock:
            active_keys = set(self._active_keys)

        msg = Twist()
        forward_pressed = "w" in active_keys or "up" in active_keys
        backward_pressed = "s" in active_keys or "down" in active_keys
        left_pressed = "a" in active_keys
        right_pressed = "d" in active_keys
        gimbal_ccw_pressed = "left" in active_keys
        gimbal_cw_pressed = "right" in active_keys

        if forward_pressed != backward_pressed:
            msg.linear.x = self.linear_speed if forward_pressed else -self.linear_speed
        if left_pressed != right_pressed:
            msg.linear.y = self.linear_speed if left_pressed else -self.linear_speed
        if gimbal_ccw_pressed != gimbal_cw_pressed:
            msg.angular.z = self.gimbal_yaw_speed if gimbal_ccw_pressed else -self.gimbal_yaw_speed

        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 1.0 if toggle_small_gyro else 0.0
        return msg

    def _compose_toggle_only_twist(self) -> Twist:
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 1.0
        msg.angular.z = 0.0
        return msg

    def _publish_current_command(self, reason: str, toggle_small_gyro: bool = False) -> None:
        msg = self._compose_twist(toggle_small_gyro)
        self.publisher.publish(msg)
        self.get_logger().info(
            f"{reason}: linear=({msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.linear.z:.2f}), "
            f"angular=({msg.angular.x:.2f}, {msg.angular.y:.2f}, {msg.angular.z:.2f})"
        )

    def _publish_toggle_only_command(self, reason: str) -> None:
        msg = self._compose_toggle_only_twist()
        self.publisher.publish(msg)
        self.get_logger().info(
            f"{reason}: linear=({msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.linear.z:.2f}), "
            f"angular=({msg.angular.x:.2f}, {msg.angular.y:.2f}, {msg.angular.z:.2f})"
        )

    def _on_press(self, key):
        logical_key = self._normalize_key(key)
        if logical_key is None:
            return
        if logical_key == "q":
            self.get_logger().info("quit requested from keyboard")
            self.destroy_node()
            rclpy.try_shutdown()
            return False

        if logical_key == "space":
            with self._state_lock:
                if self._pressed_space:
                    return
                self._pressed_space = True
            self._publish_toggle_only_command("press space")
            return

        with self._state_lock:
            if logical_key in self._active_keys:
                return
            self._active_keys.add(logical_key)
        self._publish_current_command(f"press {logical_key}")

    def _on_release(self, key):
        logical_key = self._normalize_key(key)
        if logical_key is None:
            return
        if logical_key == "space":
            with self._state_lock:
                self._pressed_space = False
            return

        with self._state_lock:
            if logical_key not in self._active_keys:
                return
            self._active_keys.remove(logical_key)
        self._publish_current_command(f"release {logical_key}")

    def destroy_node(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KeyboardTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
