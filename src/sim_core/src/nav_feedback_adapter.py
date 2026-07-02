#!/usr/bin/env python3
"""Bridge MuJoCo sim topics to the minimal feedback topics expected by external nav stacks."""

import math

import rclpy
from auto_aim_interfaces.msg import Cvmode, SentryGimbalCommand
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = yaw * 0.5
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


class NavFeedbackAdapter(Node):
    """Publishes external-style gimbal feedback topics from MuJoCo sim state."""

    def __init__(self) -> None:
        super().__init__("nav_feedback_adapter")

        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("gimbal_odom_topic", "/gimbal_Odometry")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("cv_mode_topic", "/serial_driver/cv_mode")
        self.declare_parameter(
            "sentry_gimbal_command_topic",
            "/serial_driver/sentry_gimbal_command",
        )
        self.declare_parameter("gimbal_joint_name", "gimbal_yaw_joint")
        self.declare_parameter("gimbal_link_frame", "main_gimbal_link")
        self.declare_parameter("gimbal_height", 0.25)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("armor_mode", 0)
        self.declare_parameter("bullet_speed", 28.0)
        self.declare_parameter("shoot_delay", 0.07)
        self.declare_parameter("command_timeout_sec", 0.5)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.gimbal_odom_topic = str(self.get_parameter("gimbal_odom_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.cv_mode_topic = str(self.get_parameter("cv_mode_topic").value)
        self.sentry_gimbal_command_topic = str(
            self.get_parameter("sentry_gimbal_command_topic").value
        )
        self.gimbal_joint_name = str(self.get_parameter("gimbal_joint_name").value)
        self.gimbal_link_frame = str(self.get_parameter("gimbal_link_frame").value)
        self.gimbal_height = float(self.get_parameter("gimbal_height").value)
        self.armor_mode = int(self.get_parameter("armor_mode").value)
        self.bullet_speed = float(self.get_parameter("bullet_speed").value)
        self.shoot_delay = float(self.get_parameter("shoot_delay").value)
        self.command_timeout_sec = max(
            float(self.get_parameter("command_timeout_sec").value),
            0.05,
        )
        publish_rate = max(float(self.get_parameter("publish_rate").value), 1.0)

        self.latest_odom: Odometry | None = None
        self.gimbal_joint_pos = 0.0
        self.gimbal_joint_vel = 0.0
        self.current_target_id = 255
        self.last_command_time = self.get_clock().now()

        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
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

        self.gimbal_odom_pub = self.create_publisher(Odometry, self.gimbal_odom_topic, 10)
        self.cv_mode_pub = self.create_publisher(Cvmode, self.cv_mode_topic, 10)
        self.create_timer(1.0 / publish_rate, self.publish_feedback)

        self.get_logger().info(
            "nav feedback adapter ready: "
            f"odom={self.odom_topic}, gimbal_odom={self.gimbal_odom_topic}, "
            f"cv_mode={self.cv_mode_topic}"
        )

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def joint_state_callback(self, msg: JointState) -> None:
        try:
            index = msg.name.index(self.gimbal_joint_name)
        except ValueError:
            return

        if index < len(msg.position):
            self.gimbal_joint_pos = float(msg.position[index])
        if index < len(msg.velocity):
            self.gimbal_joint_vel = float(msg.velocity[index])

    def sentry_gimbal_command_callback(self, msg: SentryGimbalCommand) -> None:
        self.current_target_id = int(msg.target_id)
        self.last_command_time = self.get_clock().now()

    def command_is_recent(self) -> bool:
        age = (self.get_clock().now() - self.last_command_time).nanoseconds / 1e9
        return age <= self.command_timeout_sec

    def publish_gimbal_odom(self) -> None:
        if self.latest_odom is None:
            return

        source = self.latest_odom
        base_orientation = source.pose.pose.orientation
        base_yaw = yaw_from_quaternion(
            base_orientation.x,
            base_orientation.y,
            base_orientation.z,
            base_orientation.w,
        )
        gimbal_yaw = math.atan2(
            math.sin(base_yaw + self.gimbal_joint_pos),
            math.cos(base_yaw + self.gimbal_joint_pos),
        )
        qx, qy, qz, qw = quaternion_from_yaw(gimbal_yaw)

        gimbal_odom = Odometry()
        gimbal_odom.header = source.header
        gimbal_odom.child_frame_id = self.gimbal_link_frame
        gimbal_odom.pose.pose.position.x = source.pose.pose.position.x
        gimbal_odom.pose.pose.position.y = source.pose.pose.position.y
        gimbal_odom.pose.pose.position.z = source.pose.pose.position.z + self.gimbal_height
        gimbal_odom.pose.pose.orientation.x = qx
        gimbal_odom.pose.pose.orientation.y = qy
        gimbal_odom.pose.pose.orientation.z = qz
        gimbal_odom.pose.pose.orientation.w = qw
        gimbal_odom.twist.twist.linear.x = source.twist.twist.linear.x
        gimbal_odom.twist.twist.linear.y = source.twist.twist.linear.y
        gimbal_odom.twist.twist.linear.z = source.twist.twist.linear.z
        gimbal_odom.twist.twist.angular.x = source.twist.twist.angular.x
        gimbal_odom.twist.twist.angular.y = source.twist.twist.angular.y
        gimbal_odom.twist.twist.angular.z = (
            source.twist.twist.angular.z + self.gimbal_joint_vel
        )

        self.gimbal_odom_pub.publish(gimbal_odom)

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
        self.publish_gimbal_odom()
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
