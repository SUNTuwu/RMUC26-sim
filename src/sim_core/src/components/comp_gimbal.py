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


def _clamp_abs(value: float, max_abs: float) -> float:
    if max_abs <= 0.0:
        return value
    return max(-max_abs, min(max_abs, value))


class GimbalComponent:
    def __init__(
        self,
        timeout_sec: float,
        chassis_angular_accel_limit: float,
        gimbal_angular_accel_limit: float,
        chassis_velocity_p_gain: float,
        chassis_velocity_d_gain: float,
        chassis_max_torque: float,
        gimbal_velocity_p_gain: float,
        gimbal_velocity_d_gain: float,
        gimbal_max_torque: float,
    ) -> None:
        self.timeout_sec = max(float(timeout_sec), 0.0)
        self.chassis_angular_accel_limit = max(float(chassis_angular_accel_limit), 0.0)
        self.gimbal_angular_accel_limit = max(float(gimbal_angular_accel_limit), 0.0)
        self.chassis_velocity_p_gain = max(float(chassis_velocity_p_gain), 0.0)
        self.chassis_velocity_d_gain = max(float(chassis_velocity_d_gain), 0.0)
        self.chassis_max_torque = max(float(chassis_max_torque), 0.0)
        self.gimbal_velocity_p_gain = max(float(gimbal_velocity_p_gain), 0.0)
        self.gimbal_velocity_d_gain = max(float(gimbal_velocity_d_gain), 0.0)
        self.gimbal_max_torque = max(float(gimbal_max_torque), 0.0)
        self.raw_chassis_yaw_rate = 0.0
        self.raw_gimbal_yaw_rate = 0.0
        self.filtered_chassis_yaw_rate = 0.0
        self.filtered_gimbal_yaw_rate = 0.0
        self.last_cmd_time = None
        self.last_chassis_yaw_rate = 0.0
        self.last_gimbal_yaw_rate = 0.0
        self.has_last_yaw_rates = False

    def update_from_twist(self, msg, now) -> None:
        self.raw_gimbal_yaw_rate = float(msg.angular.x)
        self.raw_chassis_yaw_rate = float(msg.angular.z)
        self.last_cmd_time = now

    def _sample_target_rates(self, now, dt: float) -> tuple[float, float]:
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

    def compute_drive_torques(
        self,
        now,
        dt: float,
        chassis_yaw_rate: float,
        gimbal_yaw_rate: float,
    ) -> tuple[float, float]:
        target_chassis_yaw_rate, target_gimbal_yaw_rate = self._sample_target_rates(
            now,
            dt,
        )
        chassis_yaw_accel = 0.0
        gimbal_yaw_accel = 0.0
        if self.has_last_yaw_rates and dt > 0.0:
            chassis_yaw_accel = (chassis_yaw_rate - self.last_chassis_yaw_rate) / dt
            gimbal_yaw_accel = (gimbal_yaw_rate - self.last_gimbal_yaw_rate) / dt
        self.last_chassis_yaw_rate = float(chassis_yaw_rate)
        self.last_gimbal_yaw_rate = float(gimbal_yaw_rate)
        self.has_last_yaw_rates = True

        chassis_torque = (
            (target_chassis_yaw_rate - chassis_yaw_rate) * self.chassis_velocity_p_gain
            - chassis_yaw_accel * self.chassis_velocity_d_gain
        )
        gimbal_torque = (
            (target_gimbal_yaw_rate - gimbal_yaw_rate) * self.gimbal_velocity_p_gain
            - gimbal_yaw_accel * self.gimbal_velocity_d_gain
        )
        return (
            _clamp_abs(chassis_torque, self.chassis_max_torque),
            _clamp_abs(gimbal_torque, self.gimbal_max_torque),
        )
