from __future__ import annotations

import numpy as np


def _slew_rate_limit_vector(
    current: np.ndarray,
    target: np.ndarray,
    max_accel: float,
    dt: float,
) -> np.ndarray:
    if max_accel <= 0.0 or dt <= 0.0:
        return target.copy()
    delta = target - current
    delta_norm = float(np.linalg.norm(delta))
    max_delta = max_accel * dt
    if delta_norm <= max_delta or delta_norm <= 1e-12:
        return target.copy()
    return current + delta / delta_norm * max_delta


def _clamp_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0.0:
        return vec.copy()
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm <= max_norm or vec_norm <= 1e-12:
        return vec.copy()
    return vec * (max_norm / vec_norm)


class ChassisComponent:
    def __init__(
        self,
        timeout_sec: float,
        linear_accel_limit: float,
        velocity_p_gain: float,
        velocity_d_gain: float,
        max_force: float,
    ) -> None:
        self.timeout_sec = max(float(timeout_sec), 0.0)
        self.linear_accel_limit = max(float(linear_accel_limit), 0.0)
        self.velocity_p_gain = max(float(velocity_p_gain), 0.0)
        self.velocity_d_gain = max(float(velocity_d_gain), 0.0)
        self.max_force = max(float(max_force), 0.0)
        self.raw_vx = 0.0
        self.raw_vy = 0.0
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.last_cmd_time = None
        self.last_local_velocity = np.zeros(2, dtype=np.float64)
        self.has_last_local_velocity = False

    def update_from_twist(self, msg, now) -> None:
        self.raw_vx = float(msg.linear.x)
        self.raw_vy = float(msg.linear.y)
        self.last_cmd_time = now

    def _sample_target_velocity(self, now, dt: float) -> np.ndarray:
        target = np.zeros(2, dtype=np.float64)
        if self.last_cmd_time is not None:
            age = (now - self.last_cmd_time).nanoseconds / 1e9
            if self.timeout_sec <= 0.0 or age <= self.timeout_sec:
                target[0] = self.raw_vx
                target[1] = self.raw_vy
        filtered = _slew_rate_limit_vector(
            np.array([self.filtered_vx, self.filtered_vy], dtype=np.float64),
            target,
            self.linear_accel_limit,
            dt,
        )
        self.filtered_vx = float(filtered[0])
        self.filtered_vy = float(filtered[1])
        return filtered

    def compute_drive_force(
        self,
        now,
        dt: float,
        gimbal_rot_mat: np.ndarray,
        base_linear_velocity_world: np.ndarray,
    ) -> np.ndarray:
        target_local_velocity = self._sample_target_velocity(now, dt)
        local_velocity = (
            np.asarray(gimbal_rot_mat, dtype=np.float64).T
            @ np.asarray(base_linear_velocity_world, dtype=np.float64)
        )[:2]
        local_accel = np.zeros(2, dtype=np.float64)
        if self.has_last_local_velocity and dt > 0.0:
            local_accel = (local_velocity - self.last_local_velocity) / dt
        self.last_local_velocity = local_velocity.copy()
        self.has_last_local_velocity = True
        velocity_error = target_local_velocity - local_velocity
        force_local = (
            velocity_error * self.velocity_p_gain - local_accel * self.velocity_d_gain
        )
        force_local = _clamp_vector_norm(force_local, self.max_force)
        force_world = np.asarray(gimbal_rot_mat, dtype=np.float64) @ np.array(
            [force_local[0], force_local[1], 0.0],
            dtype=np.float64,
        )
        return force_world
