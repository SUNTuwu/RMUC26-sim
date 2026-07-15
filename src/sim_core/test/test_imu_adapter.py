from __future__ import annotations

import math

import pytest
from sensor_msgs.msg import Imu

from sim_core.adapters.imu_adapter import ImuLowPassFilter


def make_imu(
    stamp_ns: int,
    angular_velocity: tuple[float, float, float],
    linear_acceleration: tuple[float, float, float],
) -> Imu:
    msg = Imu()
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    msg.header.frame_id = "imu_frame"
    msg.orientation.w = 1.0
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (
        angular_velocity
    )
    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (
        linear_acceleration
    )
    return msg


def test_first_sample_passes_through() -> None:
    low_pass = ImuLowPassFilter(cutoff_hz=10.0)
    raw = make_imu(1_000_000_000, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

    filtered = low_pass.filter(raw)

    assert filtered.angular_velocity.x == pytest.approx(1.0)
    assert filtered.angular_velocity.y == pytest.approx(2.0)
    assert filtered.angular_velocity.z == pytest.approx(3.0)
    assert filtered.linear_acceleration.x == pytest.approx(4.0)
    assert filtered.linear_acceleration.y == pytest.approx(5.0)
    assert filtered.linear_acceleration.z == pytest.approx(6.0)
    assert filtered.header.frame_id == raw.header.frame_id
    assert filtered.orientation.w == raw.orientation.w


def test_filter_uses_message_timestamp_for_both_vectors() -> None:
    cutoff_hz = 2.0
    dt = 0.1
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
    low_pass = ImuLowPassFilter(cutoff_hz=cutoff_hz)
    low_pass.filter(make_imu(1_000_000_000, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))

    filtered = low_pass.filter(
        make_imu(1_100_000_000, (1.0, -2.0, 3.0), (4.0, -5.0, 6.0))
    )

    assert filtered.angular_velocity.x == pytest.approx(alpha)
    assert filtered.angular_velocity.y == pytest.approx(-2.0 * alpha)
    assert filtered.angular_velocity.z == pytest.approx(3.0 * alpha)
    assert filtered.linear_acceleration.x == pytest.approx(4.0 * alpha)
    assert filtered.linear_acceleration.y == pytest.approx(-5.0 * alpha)
    assert filtered.linear_acceleration.z == pytest.approx(6.0 * alpha)


def test_timestamp_rollback_resets_filter_state() -> None:
    low_pass = ImuLowPassFilter(cutoff_hz=10.0)
    low_pass.filter(make_imu(2_000_000_000, (10.0, 10.0, 10.0), (10.0, 10.0, 10.0)))
    reset_sample = make_imu(1_000_000_000, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

    filtered = low_pass.filter(reset_sample)

    assert filtered.angular_velocity.x == pytest.approx(1.0)
    assert filtered.angular_velocity.y == pytest.approx(2.0)
    assert filtered.angular_velocity.z == pytest.approx(3.0)
    assert filtered.linear_acceleration.x == pytest.approx(4.0)
    assert filtered.linear_acceleration.y == pytest.approx(5.0)
    assert filtered.linear_acceleration.z == pytest.approx(6.0)
