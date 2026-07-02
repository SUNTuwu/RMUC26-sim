#!/usr/bin/env python3
"""Keyboard teleop test node for sentry_sim."""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    from evdev import InputDevice, ecodes as evdev_ecodes, list_devices
except ImportError:
    InputDevice = None
    evdev_ecodes = None
    list_devices = None

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None


KEY_ACTIONS = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    "up": "forward",
    "down": "backward",
    "left": "gimbal_ccw",
    "right": "gimbal_cw",
    "space": "toggle_small_gyro",
}

TTY_KEY_ALIASES = {
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[D": "left",
    "\x1b[C": "right",
    " ": "space",
}

MUTUALLY_EXCLUSIVE_GROUPS = (
    ("forward", "backward"),
    ("left", "right"),
    ("gimbal_ccw", "gimbal_cw"),
)

ACTION_TO_GROUP = {
    action: group
    for group in MUTUALLY_EXCLUSIVE_GROUPS
    for action in group
}

if evdev_ecodes is not None:
    EVDEV_KEY_ALIASES = {
        evdev_ecodes.KEY_W: "w",
        evdev_ecodes.KEY_S: "s",
        evdev_ecodes.KEY_A: "a",
        evdev_ecodes.KEY_D: "d",
        evdev_ecodes.KEY_UP: "up",
        evdev_ecodes.KEY_DOWN: "down",
        evdev_ecodes.KEY_LEFT: "left",
        evdev_ecodes.KEY_RIGHT: "right",
        evdev_ecodes.KEY_SPACE: "space",
        evdev_ecodes.KEY_Q: "q",
    }
else:
    EVDEV_KEY_ALIASES = {}


class KeyboardTestNode(Node):
    """Maps keyboard state to aggregated `/cmd_vel_keyboard` Twist messages."""

    def __init__(self) -> None:
        super().__init__("keyboard_test")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel_keyboard")
        self.declare_parameter("linear_speed", 1.0)
        self.declare_parameter("gimbal_yaw_speed", 1.5)
        self.declare_parameter("small_gyro_mode_value", 1.0)
        self.declare_parameter("scripted_keys", "")
        self.declare_parameter("script_step_sec", 0.2)
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("keyboard_backend", "auto")
        self.declare_parameter("keyboard_device_path", "")
        self.declare_parameter("tty_key_hold_sec", 0.18)

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.gimbal_yaw_speed = float(self.get_parameter("gimbal_yaw_speed").value)
        self.small_gyro_mode_value = float(self.get_parameter("small_gyro_mode_value").value)
        self.scripted_keys = str(self.get_parameter("scripted_keys").value).strip()
        self.script_step_sec = max(float(self.get_parameter("script_step_sec").value), 0.01)
        self.publish_rate = max(float(self.get_parameter("publish_rate").value), 1.0)
        self.keyboard_backend = str(self.get_parameter("keyboard_backend").value).strip().lower()
        self.keyboard_device_path = str(self.get_parameter("keyboard_device_path").value).strip()
        self.tty_key_hold_sec = max(float(self.get_parameter("tty_key_hold_sec").value), 0.02)

        self.publisher = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate, self._publish_current_command)
        self._stop_event = threading.Event()
        self._thread = None
        self._input_stream = None
        self._input_fd = None
        self._listener = None
        self._listener_backend = "script"
        self._state_lock = threading.Lock()
        self._small_gyro_key_active = False
        self._small_gyro_mode_enabled = False
        self._small_gyro_state_dirty = False
        self._keyboard_input_active = False
        self._active_keys = {}
        self._press_sequence = 0
        self._last_logged_signature = None
        self._tty_key_deadlines = {}
        self._evdev_devices = []

        self.get_logger().info(
            "keyboard_test ready: publish "
            f"{self.cmd_vel_topic} with WASD=chassis translation, "
            "up/down=forward/backward, left/right=gimbal yaw, "
            "space=toggle small gyro mode, q=quit. "
            "Only publishes while keyboard input is active; mutually exclusive directions keep the earliest key."
        )

        if self.scripted_keys:
            self._thread = threading.Thread(target=self._run_scripted_keys, daemon=True)
        else:
            self._configure_interactive_backend()

        if self._thread is not None:
            self._thread.start()

        self.get_logger().info(f"keyboard input backend: {self._listener_backend}")

    def _normalize_logical_key(self, raw_key: str) -> str | None:
        if not raw_key:
            return None
        if raw_key in TTY_KEY_ALIASES:
            return TTY_KEY_ALIASES[raw_key]
        if len(raw_key) == 1:
            return raw_key.lower()
        return raw_key.lower()

    def _normalize_script_key(self, raw_key: str) -> str | None:
        text = raw_key.strip()
        if not text:
            return None
        text = TTY_KEY_ALIASES.get(text, text)
        lowered = text.lower()
        aliases = {
            "arrow_up": "up",
            "arrow_down": "down",
            "arrow_left": "left",
            "arrow_right": "right",
            "spacebar": "space",
        }
        return aliases.get(lowered, lowered)

    def _normalize_pynput_key(self, key) -> tuple[str, str] | None:
        if pynput_keyboard is None:
            return None

        if isinstance(key, pynput_keyboard.KeyCode):
            if key.char is None:
                return None
            logical_key = self._normalize_logical_key(key.char)
            if logical_key is None:
                return None
            return f"char:{logical_key}", logical_key

        special_keys = {
            pynput_keyboard.Key.up: "up",
            pynput_keyboard.Key.down: "down",
            pynput_keyboard.Key.left: "left",
            pynput_keyboard.Key.right: "right",
            pynput_keyboard.Key.space: "space",
        }
        logical_key = special_keys.get(key)
        if logical_key is None:
            return None
        return f"special:{logical_key}", logical_key

    def _is_wayland_session(self) -> bool:
        session_type = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
        return session_type == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))

    def _use_tty_backend(self, reason: str) -> None:
        self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._listener_backend = "tty"
        self.get_logger().warn(reason)

    def _configure_interactive_backend(self) -> None:
        if self.keyboard_backend == "tty":
            self._use_tty_backend("keyboard_backend=tty requested; using TTY keyboard input.")
            return

        if self.keyboard_backend == "evdev":
            if self._start_evdev_listener():
                self._listener_backend = "evdev"
                return
            self._use_tty_backend(
                "keyboard_backend=evdev requested, but no usable /dev/input keyboard device was found; "
                "falling back to TTY pulse mode."
            )
            return

        if self.keyboard_backend == "pynput":
            if self._start_pynput_listener():
                self._listener_backend = "pynput"
                return
            self._use_tty_backend(
                "keyboard_backend=pynput requested, but listener startup failed; "
                "falling back to TTY pulse mode."
            )
            return

        if self.keyboard_backend not in ("", "auto"):
            self.get_logger().warn(
                f"Unknown keyboard_backend={self.keyboard_backend!r}; falling back to auto selection."
            )

        if self._start_evdev_listener():
            self._listener_backend = "evdev"
            return

        if not self._is_wayland_session() and self._start_pynput_listener():
            self._listener_backend = "pynput"
            return

        if self._is_wayland_session():
            self._use_tty_backend(
                "Wayland session detected and no evdev keyboard was available; "
                "falling back to TTY pulse mode."
            )
            return

        self._use_tty_backend("No low-latency keyboard backend was available; falling back to TTY pulse mode.")

    def _resolve_active_actions(self) -> set[str]:
        chosen_actions = set()
        grouped_candidates = {}

        for state in self._active_keys.values():
            action = state["action"]
            group = ACTION_TO_GROUP.get(action)
            if group is None:
                chosen_actions.add(action)
                continue
            current = grouped_candidates.get(group)
            if current is None or state["order"] < current["order"]:
                grouped_candidates[group] = state

        for state in grouped_candidates.values():
            chosen_actions.add(state["action"])
        return chosen_actions

    def _build_twist(self, active_actions: set[str], small_gyro_enabled: bool) -> Twist:
        msg = Twist()
        msg.linear.z = self.small_gyro_mode_value if small_gyro_enabled else 0.0
        if "forward" in active_actions:
            msg.linear.x = self.linear_speed
        elif "backward" in active_actions:
            msg.linear.x = -self.linear_speed
        if "left" in active_actions:
            msg.linear.y = self.linear_speed
        elif "right" in active_actions:
            msg.linear.y = -self.linear_speed
        if "gimbal_ccw" in active_actions:
            msg.angular.z = self.gimbal_yaw_speed
        elif "gimbal_cw" in active_actions:
            msg.angular.z = -self.gimbal_yaw_speed
        return msg

    def _compose_current_command_locked(
        self, consume_small_gyro_state_dirty: bool = False
    ) -> tuple[set[str], bool, Twist]:
        active_actions = self._resolve_active_actions()
        small_gyro_enabled = self._small_gyro_mode_enabled
        small_gyro_state_dirty = self._small_gyro_state_dirty
        if consume_small_gyro_state_dirty:
            self._small_gyro_state_dirty = False
        self._keyboard_input_active = bool(active_actions) or small_gyro_state_dirty
        msg = self._build_twist(active_actions, small_gyro_enabled)
        return active_actions, self._keyboard_input_active, msg

    def _command_signature(self, msg: Twist) -> tuple[float, float, float, float]:
        return (
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.z,
        )

    def _log_command_change(
        self,
        reason: str,
        msg: Twist,
        active_actions: set[str],
        has_input: bool,
    ) -> None:
        signature = self._command_signature(msg)
        if signature == self._last_logged_signature:
            return
        self._last_logged_signature = signature
        active_text = ",".join(sorted(active_actions)) if active_actions else "idle"
        self.get_logger().info(
            f"{reason}: active={active_text} -> linear=({msg.linear.x:.2f}, "
            f"{msg.linear.y:.2f}, {msg.linear.z:.2f}), angular.z={msg.angular.z:.2f}, "
            f"keyboard_input={'on' if has_input else 'off'}"
        )

    def _publish_state_change(self, reason: str) -> None:
        with self._state_lock:
            active_actions, has_input, msg = self._compose_current_command_locked(
                consume_small_gyro_state_dirty=True
            )
        if not has_input:
            self._last_logged_signature = None
            return
        self.publisher.publish(msg)
        self._log_command_change(reason, msg, active_actions, has_input)

    def _activate_logical_key(self, key_id: str, logical_key: str) -> None:
        action = KEY_ACTIONS.get(logical_key)
        if action is None:
            return

        publish_reason = None
        with self._state_lock:
            if action == "toggle_small_gyro":
                if not self._small_gyro_key_active:
                    self._small_gyro_key_active = True
                    self._small_gyro_mode_enabled = not self._small_gyro_mode_enabled
                    self._small_gyro_state_dirty = True
                    publish_reason = (
                        f"toggle {logical_key} -> "
                        f"{'on' if self._small_gyro_mode_enabled else 'off'}"
                    )
            elif key_id not in self._active_keys:
                self._press_sequence += 1
                self._active_keys[key_id] = {
                    "logical_key": logical_key,
                    "action": action,
                    "order": self._press_sequence,
                }
                publish_reason = f"press {logical_key}"

        if publish_reason is not None:
            self._publish_state_change(publish_reason)

    def _release_key_id(self, key_id: str) -> None:
        publish_reason = None
        with self._state_lock:
            state = self._active_keys.pop(key_id, None)
            if state is not None:
                publish_reason = f"release {state['logical_key']}"

        if publish_reason is not None:
            self._publish_state_change(publish_reason)

    def _release_small_gyro_key(self) -> None:
        with self._state_lock:
            self._small_gyro_key_active = False

    def _release_matching_key_ids(self, prefix: str) -> None:
        with self._state_lock:
            key_ids = [key_id for key_id in self._active_keys if key_id.startswith(prefix)]
        for key_id in key_ids:
            self._release_key_id(key_id)

    def _publish_current_command(self) -> None:
        with self._state_lock:
            active_actions, has_input, msg = self._compose_current_command_locked(
                consume_small_gyro_state_dirty=True
            )
        if not has_input:
            return
        self.publisher.publish(msg)
        self._log_command_change("timer publish", msg, active_actions, has_input)

    def _run_scripted_keys(self) -> None:
        for raw_key_group in self.scripted_keys.split(","):
            if self._stop_event.is_set():
                return

            group_text = raw_key_group.strip()
            if not group_text:
                continue
            if group_text.lower() == "quit":
                self.get_logger().info("scripted key sequence finished with quit")
                return

            pressed_key_ids = []
            triggered_small_gyro = False
            for index, raw_key in enumerate(group_text.split("+")):
                logical_key = self._normalize_script_key(raw_key)
                if logical_key is None:
                    continue
                if logical_key == "q":
                    self.get_logger().info("scripted key sequence finished with quit")
                    return
                key_id = f"script:{index}:{logical_key}"
                self._activate_logical_key(key_id, logical_key)
                if KEY_ACTIONS.get(logical_key) == "toggle_small_gyro":
                    triggered_small_gyro = True
                else:
                    pressed_key_ids.append(key_id)

            time.sleep(self.script_step_sec)

            for key_id in pressed_key_ids:
                self._release_key_id(key_id)
            if triggered_small_gyro:
                self._release_small_gyro_key()

    def _evdev_candidate_paths(self) -> list[str]:
        if self.keyboard_device_path:
            return [path.strip() for path in self.keyboard_device_path.split(",") if path.strip()]
        if list_devices is None:
            return []
        try:
            return sorted(list_devices())
        except Exception:
            return []

    def _is_evdev_keyboard(self, device: InputDevice) -> bool:
        if evdev_ecodes is None:
            return False
        caps = set(device.capabilities().get(evdev_ecodes.EV_KEY, []))
        required = {
            evdev_ecodes.KEY_W,
            evdev_ecodes.KEY_A,
            evdev_ecodes.KEY_S,
            evdev_ecodes.KEY_D,
            evdev_ecodes.KEY_UP,
            evdev_ecodes.KEY_DOWN,
            evdev_ecodes.KEY_LEFT,
            evdev_ecodes.KEY_RIGHT,
            evdev_ecodes.KEY_SPACE,
        }
        score = len(caps & required)
        name = (device.name or "").lower()
        if score >= 6:
            return True
        return "keyboard" in name and score >= 3

    def _start_evdev_listener(self) -> bool:
        if InputDevice is None or evdev_ecodes is None:
            return False

        selected_devices = []
        for path in self._evdev_candidate_paths():
            try:
                device = InputDevice(path)
            except OSError:
                continue

            if self.keyboard_device_path or self._is_evdev_keyboard(device):
                selected_devices.append(device)
            else:
                device.close()

        if not selected_devices:
            return False

        self._evdev_devices = selected_devices
        self._thread = threading.Thread(target=self._evdev_loop, daemon=True)
        device_text = ", ".join(f"{device.name}({device.path})" for device in self._evdev_devices)
        self.get_logger().info(f"Using evdev keyboard devices: {device_text}")
        return True

    def _close_evdev_devices(self) -> None:
        for device in self._evdev_devices:
            try:
                device.close()
            except OSError:
                pass
        self._evdev_devices = []

    def _evdev_loop(self) -> None:
        fd_to_device = {device.fd: device for device in self._evdev_devices}
        try:
            while rclpy.ok() and not self._stop_event.is_set() and fd_to_device:
                ready_fds, _, _ = select.select(list(fd_to_device), [], [], 0.01)
                if not ready_fds:
                    continue

                for fd in ready_fds:
                    device = fd_to_device.get(fd)
                    if device is None:
                        continue
                    try:
                        for event in device.read():
                            if event.type != evdev_ecodes.EV_KEY:
                                continue
                            logical_key = EVDEV_KEY_ALIASES.get(event.code)
                            if logical_key is None:
                                continue
                            key_id = f"evdev:{device.path}:{logical_key}"
                            if event.value == 1:
                                if logical_key == "q":
                                    self.get_logger().info("quit requested from keyboard")
                                    self._stop_event.set()
                                    return
                                self._activate_logical_key(key_id, logical_key)
                            elif event.value == 0:
                                if KEY_ACTIONS.get(logical_key) == "toggle_small_gyro":
                                    self._release_small_gyro_key()
                                else:
                                    self._release_key_id(key_id)
                    except OSError as exc:
                        self.get_logger().warn(
                            f"evdev keyboard device disconnected: {device.path} ({type(exc).__name__}: {exc})"
                        )
                        self._release_matching_key_ids(f"evdev:{device.path}:")
                        fd_to_device.pop(fd, None)
                        try:
                            device.close()
                        except OSError:
                            pass
        finally:
            self._close_evdev_devices()

    def _start_pynput_listener(self) -> bool:
        if pynput_keyboard is None:
            return False

        def on_press(key):
            normalized = self._normalize_pynput_key(key)
            if normalized is None:
                return
            key_id, logical_key = normalized
            if logical_key == "q":
                self.get_logger().info("quit requested from keyboard")
                self._stop_event.set()
                return False
            self._activate_logical_key(key_id, logical_key)

        def on_release(key):
            normalized = self._normalize_pynput_key(key)
            if normalized is None:
                return
            key_id, logical_key = normalized
            if KEY_ACTIONS.get(logical_key) == "toggle_small_gyro":
                self._release_small_gyro_key()
                return
            self._release_key_id(key_id)

        try:
            self._listener = pynput_keyboard.Listener(
                on_press=on_press,
                on_release=on_release,
            )
            self._listener.start()
            return True
        except Exception as exc:
            self.get_logger().warn(
                "Failed to start pynput keyboard listener; "
                f"{type(exc).__name__}: {exc}"
            )
            self._listener = None
            return False

    def _open_input_stream(self) -> bool:
        if sys.stdin.isatty():
            self._input_stream = sys.stdin
            self._input_fd = sys.stdin.fileno()
            return True

        try:
            self._input_stream = open("/dev/tty", "r", encoding="utf-8", buffering=1)
            self._input_fd = self._input_stream.fileno()
            self.get_logger().info("stdin is not a TTY; falling back to /dev/tty for keyboard input.")
            return True
        except OSError as exc:
            self.get_logger().warn(
                "No interactive TTY available for keyboard input. "
                f"stdin is not a TTY and /dev/tty open failed: {exc}. "
                "Use evdev, pynput, or scripted_keys instead."
            )
            self._input_stream = None
            self._input_fd = None
            return False

    def _close_input_stream(self) -> None:
        if self._input_stream is not None and self._input_stream is not sys.stdin:
            self._input_stream.close()
        self._input_stream = None
        self._input_fd = None

    def _read_key(self) -> str | None:
        if self._input_stream is None or self._input_fd is None:
            return None
        if not select.select([self._input_stream], [], [], 0.01)[0]:
            return None
        first = os.read(self._input_fd, 1)
        if not first:
            return None
        key = first.decode("utf-8", errors="ignore")
        if key != "\x1b":
            return key
        escape = bytearray(first)
        while len(escape) < 3 and select.select([self._input_stream], [], [], 0.002)[0]:
            chunk = os.read(self._input_fd, 1)
            if not chunk:
                break
            escape.extend(chunk)
        return escape.decode("utf-8", errors="ignore")

    def _expire_tty_keys(self) -> None:
        now = time.monotonic()
        expired_key_ids = [
            key_id
            for key_id, deadline in self._tty_key_deadlines.items()
            if now >= deadline
        ]
        for key_id in expired_key_ids:
            self._tty_key_deadlines.pop(key_id, None)
            logical_key = key_id.split(":", 1)[1] if ":" in key_id else ""
            if KEY_ACTIONS.get(logical_key) == "toggle_small_gyro":
                self._release_small_gyro_key()
            else:
                self._release_key_id(key_id)

    def _keyboard_loop(self) -> None:
        if not self._open_input_stream():
            return

        old_settings = termios.tcgetattr(self._input_fd)
        try:
            tty.setcbreak(self._input_fd)
            while rclpy.ok() and not self._stop_event.is_set():
                self._expire_tty_keys()
                key = self._read_key()
                if key is None:
                    continue
                logical_key = self._normalize_logical_key(key)
                if logical_key is None:
                    continue
                if logical_key == "q":
                    self.get_logger().info("quit requested from keyboard")
                    break
                key_id = f"tty:{logical_key}"
                self._activate_logical_key(key_id, logical_key)
                if KEY_ACTIONS.get(logical_key) != "toggle_small_gyro":
                    self._tty_key_deadlines[key_id] = time.monotonic() + self.tty_key_hold_sec
                else:
                    self._tty_key_deadlines[key_id] = time.monotonic() + self.tty_key_hold_sec
        finally:
            termios.tcsetattr(self._input_fd, termios.TCSADRAIN, old_settings)
            self._close_input_stream()
            for key_id in list(self._tty_key_deadlines):
                logical_key = key_id.split(":", 1)[1] if ":" in key_id else ""
                if KEY_ACTIONS.get(logical_key) == "toggle_small_gyro":
                    self._release_small_gyro_key()
                else:
                    self._release_key_id(key_id)
            self._tty_key_deadlines.clear()

    def destroy_node(self):
        self._stop_event.set()
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._close_evdev_devices()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KeyboardTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
