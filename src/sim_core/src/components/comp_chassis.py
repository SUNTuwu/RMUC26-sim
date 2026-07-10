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


class ChassisComponent:
    def __init__(self, timeout_sec: float, linear_accel_limit: float) -> None:
        self.timeout_sec = max(float(timeout_sec), 0.0)
        self.linear_accel_limit = max(float(linear_accel_limit), 0.0)
        self.raw_vx = 0.0
        self.raw_vy = 0.0
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        self.last_cmd_time = None

    def update_from_twist(self, msg, now) -> None:
        self.raw_vx = float(msg.linear.x)
        self.raw_vy = float(msg.linear.y)
        self.last_cmd_time = now

    def sample(self, now, dt: float) -> tuple[float, float]:
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
        return self.filtered_vx, self.filtered_vy
