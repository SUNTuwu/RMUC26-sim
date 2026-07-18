#!/usr/bin/env python3
"""Bridge Point-LIO gimbal odometry to chassis odometry for sim navigation."""

import copy
import math

import rclpy
from auto_aim_interfaces.msg import Cvmode, SentryGimbalCommand
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def quaternion_conjugate(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def quaternion_multiply(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def rotate_vector(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    q_vec = (v[0], v[1], v[2], 0.0)
    q_rotated = quaternion_multiply(
        quaternion_multiply(q, q_vec), quaternion_conjugate(q)
    )
    return (q_rotated[0], q_rotated[1], q_rotated[2])


def normalize_quaternion(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in q))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / norm for component in q)


def transform_child_pose_to_base(
    child_position: tuple[float, float, float],
    child_orientation: tuple[float, float, float, float],
    base_to_child_translation: tuple[float, float, float],
    base_to_child_rotation: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    """Convert an odom->child pose into an odom->base pose."""
    q_odom_child = normalize_quaternion(child_orientation)
    q_base_child = normalize_quaternion(base_to_child_rotation)
    q_child_base = quaternion_conjugate(q_base_child)

    child_to_base_translation = rotate_vector(
        q_child_base,
        tuple(-component for component in base_to_child_translation),
    )
    base_position_delta = rotate_vector(
        q_odom_child,
        child_to_base_translation,
    )
    base_position = tuple(
        child_position[index] + base_position_delta[index] for index in range(3)
    )
    q_odom_base = normalize_quaternion(
        quaternion_multiply(q_odom_child, q_child_base)
    )
    return base_position, q_odom_base


def transform_stamp_is_usable(
    requested_stamp: Time,
    available_stamp,
    tolerance_sec: float,
) -> bool:
    """Return whether a nearby TF sample is safe to use as a fallback."""
    requested_ns = requested_stamp.nanoseconds
    available_ns = (
        int(available_stamp.sec) * 1_000_000_000 + int(available_stamp.nanosec)
    )
    tolerance_ns = int(max(tolerance_sec, 0.0) * 1_000_000_000)
    return available_ns > 0 and abs(requested_ns - available_ns) <= tolerance_ns


class NavFeedbackAdapter(Node):
    """Publishes chassis odometry and external feedback topics for sim navigation."""

    def __init__(self) -> None:
        super().__init__("nav_feedback_adapter")

        self.declare_parameter("gimbal_odom_topic", "/gimbal_Odometry")
        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("cv_mode_topic", "/serial_driver/cv_mode")
        self.declare_parameter(
            "sentry_gimbal_command_topic",
            "/serial_driver/sentry_gimbal_command",
        )
        self.declare_parameter("gimbal_joint_name", "gimbal_to_base")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("armor_mode", 0)
        self.declare_parameter("bullet_speed", 28.0)
        self.declare_parameter("shoot_delay", 0.07)
        self.declare_parameter("command_timeout_sec", 0.5)
        self.declare_parameter("tf_latest_fallback_tolerance_sec", 0.01)

        self.gimbal_odom_topic = str(self.get_parameter("gimbal_odom_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.cv_mode_topic = str(self.get_parameter("cv_mode_topic").value)
        self.sentry_gimbal_command_topic = str(
            self.get_parameter("sentry_gimbal_command_topic").value
        )
        self.gimbal_joint_name = str(self.get_parameter("gimbal_joint_name").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.armor_mode = int(self.get_parameter("armor_mode").value)
        self.bullet_speed = float(self.get_parameter("bullet_speed").value)
        self.shoot_delay = float(self.get_parameter("shoot_delay").value)
        self.command_timeout_sec = max(
            float(self.get_parameter("command_timeout_sec").value),
            0.05,
        )
        self.tf_latest_fallback_tolerance_sec = max(
            float(
                self.get_parameter("tf_latest_fallback_tolerance_sec").value
            ),
            0.0,
        )
        publish_rate = max(float(self.get_parameter("publish_rate").value), 1.0)

        self.latest_gimbal_odom: Odometry | None = None
        self.gimbal_joint_vel = 0.0
        self.current_target_id = 255
        self.last_command_time = self.get_clock().now()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Odometry,
            self.gimbal_odom_topic,
            self.gimbal_odom_callback,
            10,
        )
        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            10,
        )
        self.create_subscription(
            SentryGimbalCommand,
            self.sentry_gimbal_command_topic,
            self.sentry_gimbal_command_callback,
            10,
        )

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.cv_mode_pub = self.create_publisher(Cvmode, self.cv_mode_topic, 10)
        self.create_timer(1.0 / publish_rate, self.publish_feedback)

        self.get_logger().info(
            "nav feedback adapter ready: "
            f"gimbal_odom={self.gimbal_odom_topic}, odom={self.odom_topic}, "
            f"cv_mode={self.cv_mode_topic}, "
            "tf_latest_fallback_tolerance_sec="
            f"{self.tf_latest_fallback_tolerance_sec:.3f}"
        )

    def gimbal_odom_callback(self, msg: Odometry) -> None:
        self.latest_gimbal_odom = msg

    def joint_state_callback(self, msg: JointState) -> None:
        try:
            index = msg.name.index(self.gimbal_joint_name)
        except ValueError:
            return

        if index < len(msg.velocity):
            self.gimbal_joint_vel = float(msg.velocity[index])

    def sentry_gimbal_command_callback(self, msg: SentryGimbalCommand) -> None:
        self.current_target_id = int(msg.target_id)
        self.last_command_time = self.get_clock().now()

    def command_is_recent(self) -> bool:
        age = (self.get_clock().now() - self.last_command_time).nanoseconds / 1e9
        return age <= self.command_timeout_sec

    def lookup_base_to_child(self, child_frame_id: str, source_stamp: Time):
        """Look up a mount transform with a bounded latest-TF fallback."""
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                child_frame_id,
                source_stamp,
            )
        except TransformException as exact_error:
            if self.tf_latest_fallback_tolerance_sec <= 0.0:
                raise

            latest = self.tf_buffer.lookup_transform(
                self.base_frame,
                child_frame_id,
                Time(),
            )
            if not transform_stamp_is_usable(
                source_stamp,
                latest.header.stamp,
                self.tf_latest_fallback_tolerance_sec,
            ):
                raise exact_error

            self.get_logger().debug(
                "using latest base-to-child TF within "
                f"{self.tf_latest_fallback_tolerance_sec * 1000.0:.1f} ms "
                "of the odometry stamp",
                throttle_duration_sec=1.0,
            )
            return latest

    def publish_chassis_odom(self) -> None:
        if self.latest_gimbal_odom is None:
            return

        source = copy.deepcopy(self.latest_gimbal_odom)
        child_frame_id = source.child_frame_id or "left_livox_frame"
        q_base_child = (0.0, 0.0, 0.0, 1.0)

        if child_frame_id != self.base_frame:
            try:
                base_to_child = self.lookup_base_to_child(
                    child_frame_id,
                    Time.from_msg(source.header.stamp),
                )
            except TransformException as ex:
                self.get_logger().warn(
                    "failed to query transform from "
                    f"{child_frame_id} to {self.base_frame}: {ex}",
                    throttle_duration_sec=1.0,
                )
                return

            tf_translation = base_to_child.transform.translation
            tf_rotation = base_to_child.transform.rotation
            q_base_child = normalize_quaternion(
                (tf_rotation.x, tf_rotation.y, tf_rotation.z, tf_rotation.w)
            )
            source_position = source.pose.pose.position
            source_orientation = source.pose.pose.orientation
            base_position, q_odom_base = transform_child_pose_to_base(
                (source_position.x, source_position.y, source_position.z),
                (
                    source_orientation.x,
                    source_orientation.y,
                    source_orientation.z,
                    source_orientation.w,
                ),
                (tf_translation.x, tf_translation.y, tf_translation.z),
                q_base_child,
            )
            source.pose.pose.position.x = base_position[0]
            source.pose.pose.position.y = base_position[1]
            source.pose.pose.position.z = base_position[2]
            source.pose.pose.orientation.x = q_odom_base[0]
            source.pose.pose.orientation.y = q_odom_base[1]
            source.pose.pose.orientation.z = q_odom_base[2]
            source.pose.pose.orientation.w = q_odom_base[3]

        source_linear = source.twist.twist.linear
        source_angular = source.twist.twist.angular
        linear_in_base = rotate_vector(
            q_base_child,
            (source_linear.x, source_linear.y, source_linear.z),
        )
        angular_in_base = rotate_vector(
            q_base_child,
            (source_angular.x, source_angular.y, source_angular.z),
        )

        chassis_odom = source
        chassis_odom.child_frame_id = self.base_frame
        chassis_odom.twist.twist.linear.x = linear_in_base[0]
        chassis_odom.twist.twist.linear.y = linear_in_base[1]
        chassis_odom.twist.twist.linear.z = linear_in_base[2]
        chassis_odom.twist.twist.angular.x = angular_in_base[0]
        chassis_odom.twist.twist.angular.y = angular_in_base[1]
        chassis_odom.twist.twist.angular.z = angular_in_base[2] + self.gimbal_joint_vel

        self.odom_pub.publish(chassis_odom)

    def publish_cv_mode(self) -> None:
        msg = Cvmode()
        msg.cur_cv_mode = int(clamp(self.armor_mode, 0, 255))
        msg.bullet_speed = float(self.bullet_speed)
        msg.shoot_delay = float(self.shoot_delay)
        msg.target_locked = (
            1 if self.command_is_recent() and self.current_target_id != 255 else 0
        )
        self.cv_mode_pub.publish(msg)

    def publish_feedback(self) -> None:
        self.publish_chassis_odom()
        self.publish_cv_mode()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavFeedbackAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
