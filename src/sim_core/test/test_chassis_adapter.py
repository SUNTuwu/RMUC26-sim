"""Tests for simulator chassis command frame conversion."""

import math

from geometry_msgs.msg import Twist
import pytest

from sim_core.adapters.chassis_adapter import (
    ChassisAdapter,
    nav_chassis_yaw_rate,
    nav_spin_offset,
    rotate_gimbal_velocity_to_base,
    rotate_nav_velocity_to_base,
    update_chassis_orientation,
)


def test_aligned_frames_preserve_linear_velocity() -> None:
    base_velocity = rotate_gimbal_velocity_to_base(2.0, -1.0, 0.0)

    assert base_velocity == pytest.approx((2.0, -1.0))


def test_positive_base_yaw_rotates_gimbal_forward_toward_base_right() -> None:
    base_velocity = rotate_gimbal_velocity_to_base(1.0, 0.0, math.pi / 2.0)

    assert base_velocity == pytest.approx((0.0, -1.0), abs=1e-12)


def test_negative_base_yaw_rotates_gimbal_forward_toward_base_left() -> None:
    base_velocity = rotate_gimbal_velocity_to_base(1.0, 0.0, -math.pi / 2.0)

    assert base_velocity == pytest.approx((0.0, 1.0), abs=1e-12)


def test_nav_rotation_matches_lower_board_formula() -> None:
    base_velocity = rotate_nav_velocity_to_base(1.0, 2.0, math.pi / 2.0)

    assert base_velocity == pytest.approx((-2.0, 1.0), abs=1e-12)


def test_counter_change_reanchors_then_same_counter_compensates() -> None:
    orientation, saved_counter, saved_yaw = update_chassis_orientation(
        received_counter=8,
        saved_counter=7,
        saved_yaw=0.4,
        current_imu_yaw=1.0,
        optional_spin_offset=0.2,
    )

    assert orientation == 0.0
    assert saved_counter == 8
    assert saved_yaw == 1.0

    orientation, saved_counter, saved_yaw = update_chassis_orientation(
        received_counter=8,
        saved_counter=saved_counter,
        saved_yaw=saved_yaw,
        current_imu_yaw=1.0 + math.pi / 2.0,
        optional_spin_offset=0.0,
    )

    assert orientation == pytest.approx(-math.pi / 2.0)
    assert saved_counter == 8
    assert saved_yaw == 1.0


def test_chassis_orientation_wraps_across_pi_boundary() -> None:
    orientation, _, _ = update_chassis_orientation(
        received_counter=4,
        saved_counter=4,
        saved_yaw=math.radians(179.0),
        current_imu_yaw=math.radians(-179.0),
        optional_spin_offset=0.0,
    )

    assert orientation == pytest.approx(math.radians(-2.0))


def test_follow_marker_only_gates_chassis_rotation() -> None:
    assert nav_chassis_yaw_rate(1.0, 2.5) == 2.5
    assert nav_chassis_yaw_rate(0.0, 2.5) == 0.0
    assert nav_spin_offset(1.0, 0.2, 1.0) == 0.0
    assert nav_spin_offset(2.5, 0.2, 1.0) == pytest.approx(0.5)


def _make_nav_callback_adapter():
    adapter = ChassisAdapter.__new__(ChassisAdapter)
    adapter.received_counter = 8
    adapter.saved_counter = 7
    adapter.saved_yaw = 0.4
    adapter.current_imu_yaw = 1.0
    adapter.chassis_orientation = 0.0
    adapter.last_gimbal_v_yaw = 2.5
    adapter.last_nav_cmd_time = None
    adapter.nav_timeout_active = False
    adapter.nav_waiting_for_yaw_warning_active = False
    adapter.invalid_nav_warning_active = False
    adapter.spin_offset_gain_sec = 0.0
    adapter.spin_offset_min_yaw_rate = 1.0
    adapter.base_yaw_topic = "/sim/base_yaw"
    adapter.get_clock = lambda: type("Clock", (), {"now": lambda self: 123})()
    published = []
    adapter._publish_command = lambda msg, source: published.append((msg, source))
    return adapter, published


def test_nav_callback_preserves_gimbal_and_sets_sim_contract() -> None:
    adapter, published = _make_nav_callback_adapter()
    command = Twist()
    command.linear.x = 1.0
    command.linear.y = -2.0
    command.angular.x = 1.0
    command.angular.z = 3.0

    adapter.nav_cmd_callback(command)

    output, source = published[-1]
    assert source == "nav"
    assert (output.linear.x, output.linear.y, output.linear.z) == pytest.approx(
        (1.0, -2.0, 0.0)
    )
    assert (output.angular.x, output.angular.y, output.angular.z) == pytest.approx(
        (2.5, 0.0, 3.0)
    )
    assert adapter.saved_counter == 8
    assert adapter.saved_yaw == 1.0
    assert adapter.chassis_orientation == 0.0


def test_follow_enabled_zeros_chassis_rate_before_spin_offset() -> None:
    adapter, published = _make_nav_callback_adapter()
    adapter.saved_counter = 8
    adapter.saved_yaw = 1.0
    adapter.spin_offset_gain_sec = 0.2
    command = Twist()
    command.linear.x = 1.0
    command.angular.x = 0.0
    command.angular.z = 3.0

    adapter.nav_cmd_callback(command)

    output, _ = published[-1]
    assert output.angular.z == 0.0
    assert adapter.chassis_orientation == 0.0
    assert output.linear.x == pytest.approx(1.0)


def test_nav_command_waits_for_mujoco_base_yaw() -> None:
    adapter, published = _make_nav_callback_adapter()
    adapter.current_imu_yaw = None
    adapter.nav_waiting_for_yaw_warning_active = True

    adapter.nav_cmd_callback(Twist())

    assert published == []
    assert adapter.last_nav_cmd_time is None


def test_keyboard_timer_does_not_overwrite_fresh_nav_command() -> None:
    adapter = ChassisAdapter.__new__(ChassisAdapter)
    adapter._nav_command_is_fresh = lambda: True
    adapter._publish_command = lambda msg, source: pytest.fail(
        f"unexpected {source} publish"
    )

    adapter.timer_callback()
