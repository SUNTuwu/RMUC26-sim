"""Tests for chassis odometry frame conversion and TF fallback timing."""

from __future__ import annotations

import math

import pytest
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.time import Time

from sim_core.adapters.nav_feedback_adapter import (
    transform_child_pose_to_base,
    transform_stamp_is_usable,
)


def test_transform_child_pose_to_base_applies_mount_offset() -> None:
    """The child-to-base translation is composed in the odom frame."""
    half_angle = math.pi / 4.0

    position, orientation = transform_child_pose_to_base(
        child_position=(10.0, 0.0, 0.0),
        child_orientation=(
            0.0,
            0.0,
            math.sin(half_angle),
            math.cos(half_angle),
        ),
        base_to_child_translation=(1.0, 0.0, 0.0),
        base_to_child_rotation=(0.0, 0.0, 0.0, 1.0),
    )

    assert position == pytest.approx((10.0, -1.0, 0.0))
    assert orientation == pytest.approx(
        (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))
    )


def test_transform_child_pose_to_base_applies_mount_rotation() -> None:
    """The inverse mount rotation converts child orientation to base."""
    half_angle = math.pi / 4.0

    position, orientation = transform_child_pose_to_base(
        child_position=(0.0, 0.0, 0.0),
        child_orientation=(0.0, 0.0, 0.0, 1.0),
        base_to_child_translation=(0.0, 0.0, 0.0),
        base_to_child_rotation=(
            0.0,
            0.0,
            math.sin(half_angle),
            math.cos(half_angle),
        ),
    )

    assert position == pytest.approx((0.0, 0.0, 0.0))
    assert orientation == pytest.approx(
        (0.0, 0.0, -math.sin(half_angle), math.cos(half_angle))
    )


def test_transform_stamp_accepts_small_tf_lag() -> None:
    """A 1.2 ms lag is accepted only by a larger configured tolerance."""
    requested = Time(nanoseconds=1_000_000_000)
    available = TimeMsg(sec=0, nanosec=998_800_000)

    assert transform_stamp_is_usable(requested, available, 0.01)
    assert not transform_stamp_is_usable(requested, available, 0.001)


def test_transform_stamp_rejects_zero_timestamp() -> None:
    """An unavailable dynamic TF timestamp cannot be used as fallback."""
    requested = Time(nanoseconds=1_000_000_000)

    assert not transform_stamp_is_usable(requested, TimeMsg(), 0.01)
