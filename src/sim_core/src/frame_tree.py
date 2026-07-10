from __future__ import annotations

import importlib
import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass


FRAME_BASE_LINK = "base_link"
FRAME_GIMBAL = "main_gimbal_link"
FRAME_LEFT_LIVOX = "left_livox_frame"
FRAME_RIGHT_LIVOX = "right_livox_frame"
FRAME_GIMBAL_ODOM = "main_gimbal_odom"
JOINT_GIMBAL_YAW = "gimbal_to_base"

DEFAULT_LEFT_LIVOX_POS = (0.0, 0.180765, 0.0)
DEFAULT_LEFT_LIVOX_RPY = (math.pi, 0.0, math.pi / 2.0)
DEFAULT_RIGHT_LIVOX_POS = (0.0, -0.180765, 0.0)
DEFAULT_RIGHT_LIVOX_RPY = (math.pi, 0.0, -math.pi / 2.0)
DEFAULT_BASE_LINK_POS = (0.0, 0.0, -0.34264)
DEFAULT_BASE_LINK_RPY = (0.0, 0.0, 0.0)
DEFAULT_GIMBAL_ODOM_POS = (0.0, 0.0, 0.0)
DEFAULT_GIMBAL_ODOM_RPY = (0.0, 0.0, 0.0)
DEFAULT_BASE_JOINT_AXIS = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class FramePose:
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotFrameTree:
    frames: dict[str, FramePose]
    base_joint_axis: tuple[float, float, float]

    def require_frame(self, frame_name: str) -> FramePose:
        if frame_name not in self.frames:
            raise KeyError(f"Robot frame not found: {frame_name}")
        return self.frames[frame_name]


def _urdf_rpy_to_quat_wxyz(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _make_frame_pose(
    pos: tuple[float, float, float], rpy: tuple[float, float, float]
) -> FramePose:
    return FramePose(
        pos=tuple(float(value) for value in pos),
        quat_wxyz=_urdf_rpy_to_quat_wxyz(*rpy),
    )


def default_robot_frame_tree() -> RobotFrameTree:
    return RobotFrameTree(
        frames={
            FRAME_GIMBAL_ODOM: _make_frame_pose(
                DEFAULT_GIMBAL_ODOM_POS,
                DEFAULT_GIMBAL_ODOM_RPY,
            ),
            FRAME_LEFT_LIVOX: _make_frame_pose(
                DEFAULT_LEFT_LIVOX_POS,
                DEFAULT_LEFT_LIVOX_RPY,
            ),
            FRAME_RIGHT_LIVOX: _make_frame_pose(
                DEFAULT_RIGHT_LIVOX_POS,
                DEFAULT_RIGHT_LIVOX_RPY,
            ),
            FRAME_BASE_LINK: _make_frame_pose(
                DEFAULT_BASE_LINK_POS,
                DEFAULT_BASE_LINK_RPY,
            ),
        },
        base_joint_axis=DEFAULT_BASE_JOINT_AXIS,
    )


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_origin_xyz_rpy(
    origin_elem,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xyz_text = origin_elem.attrib.get("xyz", "0 0 0") if origin_elem is not None else "0 0 0"
    rpy_text = origin_elem.attrib.get("rpy", "0 0 0") if origin_elem is not None else "0 0 0"
    xyz = tuple(float(value) for value in xyz_text.split())
    rpy = tuple(float(value) for value in rpy_text.split())
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError(f"Invalid origin xyz/rpy: xyz={xyz_text!r}, rpy={rpy_text!r}")
    return xyz, rpy


def _parse_axis_xyz(axis_elem) -> tuple[float, float, float]:
    axis_text = axis_elem.attrib.get("xyz", "0 0 1") if axis_elem is not None else "0 0 1"
    axis = tuple(float(value) for value in axis_text.split())
    if len(axis) != 3:
        raise ValueError(f"Invalid joint axis xyz: {axis_text!r}")
    return axis


def _expand_xacro_to_urdf_xml(xacro_path: str) -> str:
    try:
        xacro_module = importlib.import_module("xacro")
        return xacro_module.process_file(xacro_path).toxml()
    except Exception:
        result = subprocess.run(
            ["xacro", xacro_path],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


def _find_joint_spec(
    root: ET.Element, parent_link: str, child_link: str
) -> dict[str, object]:
    for joint_elem in root.iter():
        if _xml_local_name(joint_elem.tag) != "joint":
            continue
        parent_elem = None
        child_elem = None
        origin_elem = None
        axis_elem = None
        for sub_elem in joint_elem:
            sub_name = _xml_local_name(sub_elem.tag)
            if sub_name == "parent":
                parent_elem = sub_elem
            elif sub_name == "child":
                child_elem = sub_elem
            elif sub_name == "origin":
                origin_elem = sub_elem
            elif sub_name == "axis":
                axis_elem = sub_elem
        if parent_elem is None or child_elem is None:
            continue
        if (
            parent_elem.attrib.get("link") == parent_link
            and child_elem.attrib.get("link") == child_link
        ):
            xyz, rpy = _parse_origin_xyz_rpy(origin_elem)
            return {
                "pos": xyz,
                "quat_wxyz": _urdf_rpy_to_quat_wxyz(*rpy),
                "axis": _parse_axis_xyz(axis_elem),
            }
    raise KeyError(f"Joint origin not found for {parent_link} -> {child_link}")


def load_robot_frame_tree(robot_description_xacro_path: str) -> RobotFrameTree:
    if not robot_description_xacro_path:
        return default_robot_frame_tree()

    root = ET.fromstring(_expand_xacro_to_urdf_xml(robot_description_xacro_path))
    gimbal_odom_joint = _find_joint_spec(root, FRAME_GIMBAL, FRAME_GIMBAL_ODOM)
    left_joint = _find_joint_spec(root, FRAME_GIMBAL, FRAME_LEFT_LIVOX)
    right_joint = _find_joint_spec(root, FRAME_GIMBAL, FRAME_RIGHT_LIVOX)
    base_joint = _find_joint_spec(root, FRAME_GIMBAL, FRAME_BASE_LINK)
    return RobotFrameTree(
        frames={
            FRAME_GIMBAL_ODOM: FramePose(
                pos=tuple(gimbal_odom_joint["pos"]),
                quat_wxyz=tuple(gimbal_odom_joint["quat_wxyz"]),
            ),
            FRAME_LEFT_LIVOX: FramePose(
                pos=tuple(left_joint["pos"]),
                quat_wxyz=tuple(left_joint["quat_wxyz"]),
            ),
            FRAME_RIGHT_LIVOX: FramePose(
                pos=tuple(right_joint["pos"]),
                quat_wxyz=tuple(right_joint["quat_wxyz"]),
            ),
            FRAME_BASE_LINK: FramePose(
                pos=tuple(base_joint["pos"]),
                quat_wxyz=tuple(base_joint["quat_wxyz"]),
            ),
        },
        base_joint_axis=tuple(base_joint["axis"]),
    )
