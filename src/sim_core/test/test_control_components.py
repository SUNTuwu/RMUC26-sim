from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim_core.components.comp_chassis import ChassisComponent
from sim_core.components.comp_gimbal import GimbalComponent


class FakeTime:
    def __init__(self, nanoseconds: int) -> None:
        self.nanoseconds = nanoseconds

    def __sub__(self, other: "FakeTime") -> "FakeTime":
        return FakeTime(self.nanoseconds - other.nanoseconds)


def make_twist(chassis_yaw_rate: float, gimbal_yaw_rate: float):
    return SimpleNamespace(
        angular=SimpleNamespace(
            x=gimbal_yaw_rate,
            z=chassis_yaw_rate,
        )
    )


def make_chassis_twist(vx: float, vy: float):
    return SimpleNamespace(linear=SimpleNamespace(x=vx, y=vy))


def test_chassis_force_uses_base_link_coordinates() -> None:
    controller = ChassisComponent(
        timeout_sec=0.5,
        linear_accel_limit=0.0,
        velocity_p_gain=2.0,
        velocity_d_gain=0.0,
        max_force=0.0,
    )
    now = FakeTime(0)
    controller.update_from_twist(make_chassis_twist(1.0, 0.0), now)
    base_rot_mat = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    force_world = controller.compute_drive_force(
        now,
        dt=0.1,
        base_rot_mat=base_rot_mat,
        base_linear_velocity_world=np.zeros(3, dtype=np.float64),
    )

    assert force_world == pytest.approx((0.0, 2.0, 0.0))


def make_controller(
    *,
    timeout_sec: float = 0.5,
    chassis_accel_limit: float = 0.0,
    gimbal_accel_limit: float = 0.0,
) -> GimbalComponent:
    return GimbalComponent(
        timeout_sec,
        chassis_accel_limit,
        gimbal_accel_limit,
        2.0,
        0.5,
        5.0,
        3.0,
        0.2,
        4.0,
    )


def test_twist_angular_fields_follow_sim_cmd_vel_contract() -> None:
    controller = make_controller()

    controller.update_from_twist(
        SimpleNamespace(angular=SimpleNamespace(x=1.5, z=-2.5)),
        FakeTime(0),
    )

    assert controller.raw_gimbal_yaw_rate == pytest.approx(1.5)
    assert controller.raw_chassis_yaw_rate == pytest.approx(-2.5)


def test_target_rates_are_slew_limited_before_pd_control() -> None:
    controller = make_controller(chassis_accel_limit=2.0, gimbal_accel_limit=4.0)
    now = FakeTime(0)
    controller.update_from_twist(make_twist(5.0, -5.0), now)

    chassis_torque, gimbal_torque = controller.compute_drive_torques(
        now,
        0.1,
        chassis_yaw_rate=0.0,
        gimbal_yaw_rate=0.0,
    )

    assert chassis_torque == pytest.approx(0.4)
    assert gimbal_torque == pytest.approx(-1.2)


def test_pd_uses_measured_acceleration_and_clamps_torque() -> None:
    controller = make_controller()
    now = FakeTime(0)
    controller.update_from_twist(make_twist(4.0, -2.0), now)
    controller.compute_drive_torques(now, 0.1, 0.0, 0.0)

    chassis_torque, gimbal_torque = controller.compute_drive_torques(
        now,
        0.1,
        chassis_yaw_rate=2.0,
        gimbal_yaw_rate=-1.0,
    )

    assert chassis_torque == pytest.approx(-5.0)
    assert gimbal_torque == pytest.approx(-1.0)


def test_stale_command_brakes_both_axes_toward_zero() -> None:
    controller = make_controller(timeout_sec=0.5)
    command_time = FakeTime(0)
    controller.update_from_twist(make_twist(4.0, -2.0), command_time)

    chassis_torque, gimbal_torque = controller.compute_drive_torques(
        FakeTime(1_000_000_000),
        0.1,
        chassis_yaw_rate=1.0,
        gimbal_yaw_rate=-1.0,
    )

    assert chassis_torque == pytest.approx(-2.0)
    assert gimbal_torque == pytest.approx(3.0)


def test_runtime_never_assigns_qpos_or_qvel() -> None:
    runtime_path = Path(__file__).parents[1] / "src" / "runtime.py"
    tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
    forbidden_targets: list[str] = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            target_text = ast.unparse(target)
            if target_text.startswith(("self.data.qpos", "self.data.qvel")):
                forbidden_targets.append(target_text)

    assert forbidden_targets == []
