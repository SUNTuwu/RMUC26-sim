from __future__ import annotations


def _slew_rate_limit_scalar(current: float, target: float, max_accel: float, dt: float) -> float:
    if max_accel <= 0.0 or dt <= 0.0:
        return target
    max_delta = max_accel * dt
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target


class GimbalComponent:
    def __init__(
        self,
        timeout_sec: float,
        chassis_angular_accel_limit: float,
        gimbal_angular_accel_limit: float,
    ) -> None:
        self.timeout_sec = max(float(timeout_sec), 0.0)
        self.chassis_angular_accel_limit = max(float(chassis_angular_accel_limit), 0.0)
        self.gimbal_angular_accel_limit = max(float(gimbal_angular_accel_limit), 0.0)
        self.raw_chassis_yaw_rate = 0.0
        self.raw_gimbal_yaw_rate = 0.0
        self.filtered_chassis_yaw_rate = 0.0
        self.filtered_gimbal_yaw_rate = 0.0
        self.last_cmd_time = None

    def update_from_twist(self, msg, now) -> None:
        self.raw_chassis_yaw_rate = float(msg.angular.x)
        self.raw_gimbal_yaw_rate = float(msg.angular.z)
        self.last_cmd_time = now

    def sample(self, now, dt: float) -> tuple[float, float]:
        target_chassis_yaw_rate = 0.0
        target_gimbal_yaw_rate = 0.0
        if self.last_cmd_time is not None:
            age = (now - self.last_cmd_time).nanoseconds / 1e9
            if self.timeout_sec <= 0.0 or age <= self.timeout_sec:
                target_chassis_yaw_rate = self.raw_chassis_yaw_rate
                target_gimbal_yaw_rate = self.raw_gimbal_yaw_rate
        self.filtered_chassis_yaw_rate = _slew_rate_limit_scalar(
            self.filtered_chassis_yaw_rate,
            target_chassis_yaw_rate,
            self.chassis_angular_accel_limit,
            dt,
        )
        self.filtered_gimbal_yaw_rate = _slew_rate_limit_scalar(
            self.filtered_gimbal_yaw_rate,
            target_gimbal_yaw_rate,
            self.gimbal_angular_accel_limit,
            dt,
        )
        return self.filtered_chassis_yaw_rate, self.filtered_gimbal_yaw_rate
