#!/usr/bin/env python3
"""Event-driven keyboard trigger node for sentry_sim."""

from __future__ import annotations

import os
import select
import threading
import tty

import termios

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None


class _TtyKeyReader:
    """Read one logical key at a time from a tty device."""

    def __init__(self, device_path: str, read_timeout_sec: float, on_key) -> None:
        self._device_path = device_path
        self._read_timeout_sec = read_timeout_sec
        self._on_key = on_key
        self._fd: int | None = None
        self._saved_attrs = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._fd = os.open(self._device_path, os.O_RDONLY | os.O_NOCTTY)
        self._saved_attrs = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        self._thread = None
        if self._fd is not None and self._saved_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
        if self._fd is not None:
            os.close(self._fd)
        self._fd = None
        self._saved_attrs = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._fd is None:
                return
            ready, _, _ = select.select([self._fd], [], [], self._read_timeout_sec)
            if not ready:
                continue
            logical_key = self._read_logical_key()
            if logical_key is None:
                continue
            if self._on_key(logical_key) is False:
                return

    def _read_logical_key(self) -> str | None:
        if self._fd is None:
            return None

        first = os.read(self._fd, 1)
        if not first:
            return None

        first_char = first.decode(errors="ignore")
        if first_char == "\x03":
            return "q"
        if first_char == " ":
            return "space"
        if first_char == "\x1b":
            sequence = first_char
            for _ in range(2):
                ready, _, _ = select.select([self._fd], [], [], self._read_timeout_sec)
                if not ready:
                    break
                sequence += os.read(self._fd, 1).decode(errors="ignore")
                if sequence in ("\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D"):
                    break
            special_keys = {
                "\x1b[A": "up",
                "\x1b[B": "down",
                "\x1b[C": "right",
                "\x1b[D": "left",
            }
            return special_keys.get(sequence)
        if first_char.isprintable():
            return first_char.lower()
        return None


class KeyboardTestNode(Node):
    """Publishes `/cmd_vel_keyboard` only on keyboard state changes."""

    def __init__(self) -> None:
        super().__init__("keyboard_test")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel_keyboard")
        self.declare_parameter("linear_speed", 1.0)
        self.declare_parameter("gimbal_yaw_speed", 1.5)
        self.declare_parameter("tty_device", "/dev/tty")
        self.declare_parameter("tty_key_timeout_sec", 0.5)

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.gimbal_yaw_speed = float(self.get_parameter("gimbal_yaw_speed").value)
        self.tty_device = str(self.get_parameter("tty_device").value)
        self.tty_key_timeout_sec = float(self.get_parameter("tty_key_timeout_sec").value)
        if self.tty_key_timeout_sec <= 0.0:
            raise ValueError("tty_key_timeout_sec must be positive")

        self.publisher = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self._listener = None
        self._tty_reader = None
        self._tty_release_timer = None
        self._state_lock = threading.Lock()
        self._active_keys: set[str] = set()
        self._pressed_space = False
        self._destroyed = False

        if pynput_keyboard is not None:
            self._listener = pynput_keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()
            input_mode = "pynput"
        else:
            self._tty_reader = _TtyKeyReader(
                device_path=self.tty_device,
                read_timeout_sec=self.tty_key_timeout_sec,
                on_key=self._on_tty_key,
            )
            try:
                self._tty_reader.start()
            except OSError as exc:
                raise RuntimeError(
                    f"pynput unavailable and failed to open tty device {self.tty_device}"
                ) from exc
            input_mode = f"tty:{self.tty_device}"

        self.get_logger().info(
            "keyboard_test ready: event-driven publish to "
            f"{self.cmd_vel_topic}, input={input_mode}, WASD=translation, "
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

    def _request_shutdown(self) -> bool:
        self.get_logger().info("quit requested from keyboard")
        self.destroy_node()
        rclpy.try_shutdown()
        return False

    def _cancel_tty_release_timer(self) -> None:
        with self._state_lock:
            timer = self._tty_release_timer
            self._tty_release_timer = None
        if timer is not None:
            timer.cancel()

    def _arm_tty_release_timer(self) -> None:
        timer = threading.Timer(self.tty_key_timeout_sec, self._on_tty_timeout)
        timer.daemon = True
        with self._state_lock:
            previous_timer = self._tty_release_timer
            self._tty_release_timer = timer
        if previous_timer is not None:
            previous_timer.cancel()
        timer.start()

    def _on_tty_timeout(self) -> None:
        with self._state_lock:
            if self._destroyed or not self._active_keys:
                self._tty_release_timer = None
                return
            logical_key = next(iter(self._active_keys))
            self._active_keys.clear()
            self._tty_release_timer = None
        self._publish_current_command(f"release {logical_key}")

    def _on_tty_key(self, logical_key: str) -> bool | None:
        if logical_key == "q":
            return self._request_shutdown()
        if logical_key == "space":
            self._publish_toggle_only_command("press space")
            return True

        with self._state_lock:
            if self._destroyed:
                return False
            state_changed = self._active_keys != {logical_key}
            self._active_keys.clear()
            self._active_keys.add(logical_key)

        self._arm_tty_release_timer()
        if state_changed:
            self._publish_current_command(f"press {logical_key}")
        return True

    def _on_press(self, key):
        logical_key = self._normalize_key(key)
        if logical_key is None:
            return
        if logical_key == "q":
            return self._request_shutdown()

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
        with self._state_lock:
            if self._destroyed:
                return
            self._destroyed = True
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._tty_reader is not None:
            self._tty_reader.stop()
            self._tty_reader = None
        self._cancel_tty_release_timer()
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
