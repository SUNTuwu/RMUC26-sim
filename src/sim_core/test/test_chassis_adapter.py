"""Tests for simulator chassis command frame conversion."""

import math

import pytest

from sim_core.adapters.chassis_adapter import rotate_gimbal_velocity_to_base


def test_aligned_frames_preserve_linear_velocity() -> None:
    base_velocity = rotate_gimbal_velocity_to_base(2.0, -1.0, 0.0)

    assert base_velocity == pytest.approx((2.0, -1.0))


def test_positive_base_yaw_rotates_gimbal_forward_toward_base_right() -> None:
    base_velocity = rotate_gimbal_velocity_to_base(1.0, 0.0, math.pi / 2.0)

    assert base_velocity == pytest.approx((0.0, -1.0), abs=1e-12)


def test_negative_base_yaw_rotates_gimbal_forward_toward_base_left() -> None:
    base_velocity = rotate_gimbal_velocity_to_base(1.0, 0.0, -math.pi / 2.0)

    assert base_velocity == pytest.approx((0.0, 1.0), abs=1e-12)
