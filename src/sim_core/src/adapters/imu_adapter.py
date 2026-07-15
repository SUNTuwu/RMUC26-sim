#!/usr/bin/env python3
"""Low-pass simulated IMU data before exposing Livox-compatible topics."""

from __future__ import annotations

import copy
import math

import rclpy
import sensor_msgs.msg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def _sim_imu_topic(ip_address: str) -> str:
    return f"/sim/imu_{ip_address.replace('.', '_')}"


def _livox_imu_topic(ip_address: str) -> str:
    return f"/livox/imu_{ip_address.replace('.', '_')}"


class ImuLowPassFilter:
    def __init__(self, cutoff_hz: float) -> None:
        if cutoff_hz <= 0.0:
            raise ValueError("cutoff_hz must be positive")
        self.cutoff_hz = float(cutoff_hz)
        self.last_stamp_ns: int | None = None
        self.angular_velocity: list[float] | None = None
        self.linear_acceleration: list[float] | None = None

    @staticmethod
    def _stamp_ns(msg: sensor_msgs.msg.Imu) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(
            msg.header.stamp.nanosec
        )

    @staticmethod
    def _read_vector(vector) -> list[float]:
        return [float(vector.x), float(vector.y), float(vector.z)]

    @staticmethod
    def _write_vector(vector, values: list[float]) -> None:
        vector.x, vector.y, vector.z = values

    def _reset(self, msg: sensor_msgs.msg.Imu, stamp_ns: int) -> sensor_msgs.msg.Imu:
        self.last_stamp_ns = stamp_ns
        self.angular_velocity = self._read_vector(msg.angular_velocity)
        self.linear_acceleration = self._read_vector(msg.linear_acceleration)
        return copy.deepcopy(msg)

    def filter(self, msg: sensor_msgs.msg.Imu) -> sensor_msgs.msg.Imu:
        stamp_ns = self._stamp_ns(msg)
        if self.last_stamp_ns is None or stamp_ns < self.last_stamp_ns:
            return self._reset(msg, stamp_ns)

        dt = (stamp_ns - self.last_stamp_ns) / 1_000_000_000.0
        self.last_stamp_ns = stamp_ns
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt)
        raw_angular_velocity = self._read_vector(msg.angular_velocity)
        raw_linear_acceleration = self._read_vector(msg.linear_acceleration)
        self.angular_velocity = [
            previous + alpha * (current - previous)
            for previous, current in zip(
                self.angular_velocity,
                raw_angular_velocity,
                strict=True,
            )
        ]
        self.linear_acceleration = [
            previous + alpha * (current - previous)
            for previous, current in zip(
                self.linear_acceleration,
                raw_linear_acceleration,
                strict=True,
            )
        ]

        filtered = copy.deepcopy(msg)
        self._write_vector(filtered.angular_velocity, self.angular_velocity)
        self._write_vector(filtered.linear_acceleration, self.linear_acceleration)
        return filtered


class ImuAdapter(Node):
    def __init__(self) -> None:
        super().__init__("imu_adapter")
        imu_ips = [
            str(value).strip()
            for value in self.declare_parameter(
                "imu_ips",
                ["192.168.10.4", "192.168.10.5"],
            ).value
            if str(value).strip()
        ]
        cutoff_hz = float(
            self.declare_parameter("imu_low_pass_cutoff_hz", 30.0).value
        )
        if not imu_ips:
            raise ValueError("imu_ips must contain at least one IP address")
        if cutoff_hz <= 0.0:
            raise ValueError("imu_low_pass_cutoff_hz must be positive")

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._imu_publishers = {}
        self._imu_filters = {}
        self._imu_subscriptions = []
        for ip_address in dict.fromkeys(imu_ips):
            raw_topic = _sim_imu_topic(ip_address)
            output_topic = _livox_imu_topic(ip_address)
            self._imu_publishers[ip_address] = self.create_publisher(
                sensor_msgs.msg.Imu,
                output_topic,
                sensor_qos,
            )
            self._imu_filters[ip_address] = ImuLowPassFilter(cutoff_hz)
            self._imu_subscriptions.append(
                self.create_subscription(
                    sensor_msgs.msg.Imu,
                    raw_topic,
                    lambda msg, ip=ip_address: self._imu_callback(ip, msg),
                    sensor_qos,
                )
            )
            self.get_logger().info(
                f"IMU low-pass: {raw_topic} -> {output_topic}, "
                f"cutoff={cutoff_hz:.2f} Hz"
            )

    def _imu_callback(self, ip_address: str, msg: sensor_msgs.msg.Imu) -> None:
        filtered = self._imu_filters[ip_address].filter(msg)
        self._imu_publishers[ip_address].publish(filtered)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
