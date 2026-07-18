from __future__ import annotations

import math
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.node import Node

from .frame_tree import (
    FRAME_BASE_LINK,
    FRAME_GIMBAL,
    FRAME_GIMBAL_ODOM,
    FRAME_LEFT_LIVOX,
    FRAME_RIGHT_LIVOX,
    JOINT_GIMBAL_YAW,
    RobotFrameTree,
)


RENDER_GEOM_GROUP = 1
LIDAR_TRACE_GEOM_GROUP = 0
LIDAR_DEBUG_GEOM_GROUP = 3
COLLISION_GEOM_GROUP = 2
ENV_CONTYPE = 1
ROBOT_CONTYPE = 2

DEFAULT_PHYSICS_INTEGRATOR = "implicitfast"
DEFAULT_CONTACT_SOLREF = (0.03, 1.2)
DEFAULT_CONTACT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
DEFAULT_GIMBAL_VISUAL_HALF_EXTENTS_XYZ = (0.11, 0.11, 0.04)
DEFAULT_GIMBAL_COLLISION_HALF_EXTENTS_XYZ = (0.17, 0.17, 0.09)
DEFAULT_GIMBAL_BODY_MASS = 1e-3
DEFAULT_GIMBAL_BODY_DIAGINERTIA_XYZ = (1e-6, 1e-6, 1e-6)
DEFAULT_BASE_BOTTOM_HEIGHT = 0.08
DEFAULT_BASE_TOP_HEIGHT = 0.22
DEFAULT_BASE_DIAMETER = 0.5
DEFAULT_BASE_BODY_MASS = 20.0
DEFAULT_ENABLE_BASE_BORDER = True
DEFAULT_BASE_BORDER_SIZE_XYZ = (0.5, 0.5, 0.1)
BASE_BORDER_TOP_OFFSET = 0.005
DEFAULT_WHEEL_SPACING_X = 0.4
DEFAULT_WHEEL_SPACING_Y = 0.4
DEFAULT_WHEEL_DIAMETER = 0.14
DEFAULT_WHEEL_CONTACT_SOLREF = (0.05, 1.2)
DEFAULT_WHEEL_CONTACT_SOLIMP = (0.8, 0.95, 0.003, 0.5, 2.0)
DEFAULT_BASE_DIRECTION_ARROW_LENGTH = 0.35
DEFAULT_GIMBAL_DIRECTION_ARROW_LENGTH = 0.25
DEFAULT_DIRECTION_ARROW_RADIUS = 0.01
DEFAULT_BASE_DIRECTION_ARROW_RGBA = (1.0, 0.5, 0.0, 1.0)
DEFAULT_GIMBAL_DIRECTION_ARROW_RGBA = (1.0, 0.05, 0.05, 1.0)
DEFAULT_LIVOX_VISUAL_RADIUS = 0.035
DEFAULT_LIVOX_VISUAL_HEIGHT = 0.05
DEFAULT_LIVOX_BODY_MASS = 0.265
DEFAULT_LIVOX_BODY_DIAGINERTIA_XYZ = (0.0001, 0.0001, 0.0001)
DEFAULT_FRAME_ORIGIN_DEBUG_RADIUS = 0.018
DEFAULT_GIMBAL_ORIGIN_DEBUG_RADIUS = 0.02
DEFAULT_LIVOX_ORIGIN_DEBUG_RADIUS = 0.015
DEFAULT_IMU_ORIGIN_DEBUG_RADIUS = 0.01
DEFAULT_LIDAR_SITE_RADIUS = 0.01
DEFAULT_IMU_SITE_RADIUS = 0.006
DEFAULT_IMU_ACCEL_DEBUG_RADIUS = 0.003
DEFAULT_IMU_ACCEL_DEBUG_SCALE = 0.12
DEFAULT_IMU_ACCEL_DEBUG_MIN_LENGTH = 0.002
DEFAULT_BASE_FORCE_DEBUG_RADIUS = 0.006
DEFAULT_BASE_FORCE_DEBUG_SCALE = 0.004
DEFAULT_BASE_FORCE_DEBUG_MIN_LENGTH = 0.03
DEFAULT_BOUNDARY_WALL_THICKNESS = 0.05
DEFAULT_BOUNDARY_WALL_HEIGHT = 1.2
DEFAULT_LIVOX_IMU_OFFSET_XYZ = (-0.011, -0.02329, 0.04412)
DEFAULT_LIVOX_IMU_RPY = (0.0, 0.0, 0.0)


def frame_resource(frame_name: str, suffix: str) -> str:
    return f"{frame_name}__{suffix}"


def livox_lidar_site_name(frame_name: str) -> str:
    return frame_resource(frame_name, "lidar_site")


def livox_imu_site_name(frame_name: str) -> str:
    return frame_resource(frame_name, "imu_site")


def livox_accel_sensor_name(frame_name: str) -> str:
    return frame_resource(frame_name, "imu_acc")


def livox_gyro_sensor_name(frame_name: str) -> str:
    return frame_resource(frame_name, "imu_gyro")


def livox_accel_debug_geom_name(frame_name: str) -> str:
    return frame_resource(frame_name, "imu_accel_debug")


def livox_accel_debug_body_name(frame_name: str) -> str:
    return frame_resource(frame_name, "imu_accel_debug_body")


def base_force_debug_geom_name() -> str:
    return frame_resource(FRAME_BASE_LINK, "force_debug")


def base_force_debug_body_name() -> str:
    return frame_resource(FRAME_BASE_LINK, "force_debug_body")


def _format_values(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _format_xyz(values: tuple[float, float, float]) -> str:
    return _format_values(values)


def _format_quat_wxyz(values: tuple[float, float, float, float]) -> str:
    return _format_values(values)


def _quat_conjugate_wxyz(
    quat_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w, x, y, z = quat_wxyz
    return (w, -x, -y, -z)


def _quat_multiply_wxyz(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rotate_vector_by_quat_wxyz(
    quat_wxyz: tuple[float, float, float, float],
    vec_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    vec_quat = (0.0, vec_xyz[0], vec_xyz[1], vec_xyz[2])
    rotated = _quat_multiply_wxyz(
        _quat_multiply_wxyz(quat_wxyz, vec_quat),
        _quat_conjugate_wxyz(quat_wxyz),
    )
    return (rotated[1], rotated[2], rotated[3])


def _invert_pose(
    pos_xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    quat_inv = _quat_conjugate_wxyz(quat_wxyz)
    pos_inv = _rotate_vector_by_quat_wxyz(
        quat_inv,
        (-pos_xyz[0], -pos_xyz[1], -pos_xyz[2]),
    )
    return pos_inv, quat_inv


def _coerce_vector3_param(raw_value, param_name: str) -> tuple[float, float, float]:
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 3:
        raise ValueError(f"{param_name} must be a 3-element list [x, y, z]")
    return tuple(float(value) for value in raw_value)


def _declare_vector3_param(
    node: Node, param_name: str, default_value: tuple[float, float, float]
) -> tuple[float, float, float]:
    return _coerce_vector3_param(
        node.declare_parameter(param_name, list(default_value)).value,
        param_name,
    )


def _declare_rgba_param(
    node: Node,
    param_name: str,
    default_value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    raw_value = node.declare_parameter(param_name, list(default_value)).value
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 4:
        raise ValueError(f"{param_name} must be a 4-element list [r, g, b, a]")
    values = tuple(float(value) for value in raw_value)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{param_name} values must be in the range [0, 1]")
    return values


def _declare_nonnegative_float_param(
    node: Node, param_name: str, default_value: float
) -> float:
    return max(float(node.declare_parameter(param_name, default_value).value), 0.0)


def _declare_positive_float_param(
    node: Node,
    param_name: str,
    default_value: float,
) -> float:
    value = float(node.declare_parameter(param_name, default_value).value)
    if value <= 0.0:
        raise ValueError(f"{param_name} must be positive")
    return value


def _declare_positive_vector3_param(
    node: Node,
    param_name: str,
    default_value: tuple[float, float, float],
) -> tuple[float, float, float]:
    values = _declare_vector3_param(node, param_name, default_value)
    if any(value <= 0.0 for value in values):
        raise ValueError(f"{param_name} values must be positive")
    return values


def _declare_float_vector_param(
    node: Node,
    param_name: str,
    default_value: tuple[float, ...],
) -> tuple[float, ...]:
    raw_value = node.declare_parameter(param_name, list(default_value)).value
    if (
        not isinstance(raw_value, (list, tuple))
        or len(raw_value) != len(default_value)
    ):
        raise ValueError(
            f"{param_name} must be a {len(default_value)}-element list"
        )
    return tuple(float(value) for value in raw_value)


def _declare_solref_param(
    node: Node,
    param_name: str,
    default_value: tuple[float, float],
) -> tuple[float, float]:
    values = _declare_float_vector_param(node, param_name, default_value)
    if any(value <= 0.0 for value in values):
        raise ValueError(
            f"{param_name} must use positive [time_constant, damping_ratio] values"
        )
    return values


def _declare_solimp_param(
    node: Node,
    param_name: str,
    default_value: tuple[float, float, float, float, float],
) -> tuple[float, float, float, float, float]:
    values = _declare_float_vector_param(node, param_name, default_value)
    d0, d_width, width, midpoint, power = values
    if not 0.0 <= d0 <= 1.0 or not 0.0 <= d_width <= 1.0:
        raise ValueError(f"{param_name} impedance values must be in [0, 1]")
    if width <= 0.0:
        raise ValueError(f"{param_name} width must be positive")
    if not 0.0 <= midpoint <= 1.0:
        raise ValueError(f"{param_name} midpoint must be in [0, 1]")
    if power < 1.0:
        raise ValueError(f"{param_name} power must be at least 1")
    return values


def load_physics_params(node: Node, physics_dt: float) -> dict[str, object]:
    integrator_key = str(
        node.declare_parameter(
            "physics_integrator",
            DEFAULT_PHYSICS_INTEGRATOR,
        ).value
    ).strip().lower()
    integrators = {
        "euler": "Euler",
        "rk4": "RK4",
        "implicit": "implicit",
        "implicitfast": "implicitfast",
    }
    if integrator_key not in integrators:
        raise ValueError(
            "physics_integrator must be one of: euler, rk4, implicit, implicitfast"
        )
    contact_solref = _declare_solref_param(
        node,
        "contact_solref",
        DEFAULT_CONTACT_SOLREF,
    )
    if contact_solref[0] < 2.0 * physics_dt:
        raise ValueError(
            "contact_solref time constant must be at least 2 * physics_dt"
        )
    return {
        "timestep": float(physics_dt),
        "integrator": integrators[integrator_key],
        "contact_solref": contact_solref,
        "contact_solimp": _declare_solimp_param(
            node,
            "contact_solimp",
            DEFAULT_CONTACT_SOLIMP,
        ),
    }


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


def load_scene_geometry_params(node: Node) -> dict[str, object]:
    livox_imu_offset_xyz = _declare_vector3_param(
        node,
        "livox_imu_offset_xyz",
        DEFAULT_LIVOX_IMU_OFFSET_XYZ,
    )
    livox_imu_rpy = _declare_vector3_param(
        node,
        "livox_imu_rpy",
        DEFAULT_LIVOX_IMU_RPY,
    )
    base_bottom_height = _declare_nonnegative_float_param(
        node,
        "base_bottom_height",
        DEFAULT_BASE_BOTTOM_HEIGHT,
    )
    base_top_height = _declare_positive_float_param(
        node,
        "base_top_height",
        DEFAULT_BASE_TOP_HEIGHT,
    )
    if base_top_height <= base_bottom_height:
        raise ValueError("base_top_height must be greater than base_bottom_height")
    base_border_size_xyz = _declare_positive_vector3_param(
        node,
        "base_border_size_xyz",
        DEFAULT_BASE_BORDER_SIZE_XYZ,
    )
    enable_base_border = bool(
        node.declare_parameter(
            "enable_base_border",
            DEFAULT_ENABLE_BASE_BORDER,
        ).value
    )
    if (
        enable_base_border
        and base_border_size_xyz[2]
        > base_top_height + BASE_BORDER_TOP_OFFSET - base_bottom_height
    ):
        raise ValueError(
            "base_border_size_xyz height must not extend below the base"
        )
    return {
        "gimbal_visual_half_extents_xyz": _declare_vector3_param(
            node,
            "gimbal_visual_half_extents_xyz",
            DEFAULT_GIMBAL_VISUAL_HALF_EXTENTS_XYZ,
        ),
        "gimbal_collision_half_extents_xyz": _declare_vector3_param(
            node,
            "gimbal_collision_half_extents_xyz",
            DEFAULT_GIMBAL_COLLISION_HALF_EXTENTS_XYZ,
        ),
        "gimbal_body_mass": _declare_nonnegative_float_param(
            node,
            "gimbal_body_mass",
            DEFAULT_GIMBAL_BODY_MASS,
        ),
        "gimbal_body_diaginertia_xyz": _declare_vector3_param(
            node,
            "gimbal_body_diaginertia_xyz",
            DEFAULT_GIMBAL_BODY_DIAGINERTIA_XYZ,
        ),
        "base_bottom_height": base_bottom_height,
        "base_top_height": base_top_height,
        "base_diameter": _declare_positive_float_param(
            node,
            "base_diameter",
            DEFAULT_BASE_DIAMETER,
        ),
        "base_body_mass": _declare_positive_float_param(
            node,
            "base_body_mass",
            DEFAULT_BASE_BODY_MASS,
        ),
        "enable_base_border": enable_base_border,
        "base_border_size_xyz": base_border_size_xyz,
        "wheel_spacing_x": _declare_positive_float_param(
            node,
            "wheel_spacing_x",
            DEFAULT_WHEEL_SPACING_X,
        ),
        "wheel_spacing_y": _declare_positive_float_param(
            node,
            "wheel_spacing_y",
            DEFAULT_WHEEL_SPACING_Y,
        ),
        "wheel_diameter": _declare_positive_float_param(
            node,
            "wheel_diameter",
            DEFAULT_WHEEL_DIAMETER,
        ),
        "wheel_contact_solref": _declare_solref_param(
            node,
            "wheel_contact_solref",
            DEFAULT_WHEEL_CONTACT_SOLREF,
        ),
        "wheel_contact_solimp": _declare_solimp_param(
            node,
            "wheel_contact_solimp",
            DEFAULT_WHEEL_CONTACT_SOLIMP,
        ),
        "base_direction_arrow_length": _declare_positive_float_param(
            node,
            "base_direction_arrow_length",
            DEFAULT_BASE_DIRECTION_ARROW_LENGTH,
        ),
        "gimbal_direction_arrow_length": _declare_positive_float_param(
            node,
            "gimbal_direction_arrow_length",
            DEFAULT_GIMBAL_DIRECTION_ARROW_LENGTH,
        ),
        "direction_arrow_radius": _declare_positive_float_param(
            node,
            "direction_arrow_radius",
            DEFAULT_DIRECTION_ARROW_RADIUS,
        ),
        "base_direction_arrow_rgba": _declare_rgba_param(
            node,
            "base_direction_arrow_rgba",
            DEFAULT_BASE_DIRECTION_ARROW_RGBA,
        ),
        "gimbal_direction_arrow_rgba": _declare_rgba_param(
            node,
            "gimbal_direction_arrow_rgba",
            DEFAULT_GIMBAL_DIRECTION_ARROW_RGBA,
        ),
        "livox_visual_radius": _declare_nonnegative_float_param(
            node,
            "livox_visual_radius",
            DEFAULT_LIVOX_VISUAL_RADIUS,
        ),
        "livox_visual_height": _declare_nonnegative_float_param(
            node,
            "livox_visual_height",
            DEFAULT_LIVOX_VISUAL_HEIGHT,
        ),
        "livox_body_mass": _declare_nonnegative_float_param(
            node,
            "livox_body_mass",
            DEFAULT_LIVOX_BODY_MASS,
        ),
        "livox_body_diaginertia_xyz": _declare_vector3_param(
            node,
            "livox_body_diaginertia_xyz",
            DEFAULT_LIVOX_BODY_DIAGINERTIA_XYZ,
        ),
        "frame_origin_debug_radius": _declare_nonnegative_float_param(
            node,
            "frame_origin_debug_radius",
            DEFAULT_FRAME_ORIGIN_DEBUG_RADIUS,
        ),
        "gimbal_origin_debug_radius": _declare_nonnegative_float_param(
            node,
            "gimbal_origin_debug_radius",
            DEFAULT_GIMBAL_ORIGIN_DEBUG_RADIUS,
        ),
        "livox_origin_debug_radius": _declare_nonnegative_float_param(
            node,
            "livox_origin_debug_radius",
            DEFAULT_LIVOX_ORIGIN_DEBUG_RADIUS,
        ),
        "imu_origin_debug_radius": _declare_nonnegative_float_param(
            node,
            "imu_origin_debug_radius",
            DEFAULT_IMU_ORIGIN_DEBUG_RADIUS,
        ),
        "lidar_site_radius": _declare_nonnegative_float_param(
            node,
            "lidar_site_radius",
            DEFAULT_LIDAR_SITE_RADIUS,
        ),
        "imu_site_radius": _declare_nonnegative_float_param(
            node,
            "imu_site_radius",
            DEFAULT_IMU_SITE_RADIUS,
        ),
        "imu_accel_debug_radius": _declare_nonnegative_float_param(
            node,
            "imu_accel_debug_radius",
            DEFAULT_IMU_ACCEL_DEBUG_RADIUS,
        ),
        "imu_accel_debug_scale": _declare_nonnegative_float_param(
            node,
            "imu_accel_debug_scale",
            DEFAULT_IMU_ACCEL_DEBUG_SCALE,
        ),
        "imu_accel_debug_min_length": _declare_nonnegative_float_param(
            node,
            "imu_accel_debug_min_length",
            DEFAULT_IMU_ACCEL_DEBUG_MIN_LENGTH,
        ),
        "base_force_debug_radius": _declare_nonnegative_float_param(
            node,
            "base_force_debug_radius",
            DEFAULT_BASE_FORCE_DEBUG_RADIUS,
        ),
        "base_force_debug_scale": _declare_nonnegative_float_param(
            node,
            "base_force_debug_scale",
            DEFAULT_BASE_FORCE_DEBUG_SCALE,
        ),
        "base_force_debug_min_length": _declare_nonnegative_float_param(
            node,
            "base_force_debug_min_length",
            DEFAULT_BASE_FORCE_DEBUG_MIN_LENGTH,
        ),
        "boundary_wall_thickness": _declare_nonnegative_float_param(
            node,
            "boundary_wall_thickness",
            DEFAULT_BOUNDARY_WALL_THICKNESS,
        ),
        "boundary_wall_height": _declare_nonnegative_float_param(
            node,
            "boundary_wall_height",
            DEFAULT_BOUNDARY_WALL_HEIGHT,
        ),
        "livox_imu_offset_xyz": livox_imu_offset_xyz,
        "livox_imu_rpy": livox_imu_rpy,
        "livox_imu_quat": _urdf_rpy_to_quat_wxyz(*livox_imu_rpy),
    }


def resolve_assets_dir() -> str:
    candidate_dirs: list[str] = []
    try:
        candidate_dirs.append(get_package_share_directory("sim_assets"))
    except PackageNotFoundError:
        pass

    this_file = os.path.abspath(__file__)
    candidate_dirs.extend(
        [
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(this_file))),
                "sim_assets",
            ),
            os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.dirname(this_file))
                        )
                    )
                ),
                "src",
                "sim_assets",
            ),
        ]
    )
    for candidate in candidate_dirs:
        mesh_file = os.path.join(candidate, "meshes", "mesh_view", "mesh_view.obj")
        if os.path.isfile(mesh_file):
            return candidate
    raise FileNotFoundError(
        "Could not resolve sim_assets directory containing meshes/mesh_view/mesh_view.obj"
    )


def _list_obj_files(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        name for name in os.listdir(directory) if name.lower().endswith(".obj")
    )


def _build_livox_body_xml(
    frame_name: str,
    pose,
    scene_geometry: dict[str, object],
) -> str:
    livox_visual_radius = scene_geometry["livox_visual_radius"]
    livox_visual_half_height = scene_geometry["livox_visual_height"] / 2.0
    livox_body_mass = scene_geometry["livox_body_mass"]
    livox_body_diaginertia_xyz = scene_geometry["livox_body_diaginertia_xyz"]
    livox_origin_debug_radius = scene_geometry["livox_origin_debug_radius"]
    imu_origin_debug_radius = scene_geometry["imu_origin_debug_radius"]
    lidar_site_radius = scene_geometry["lidar_site_radius"]
    imu_site_radius = scene_geometry["imu_site_radius"]
    livox_imu_offset_xyz = scene_geometry["livox_imu_offset_xyz"]
    livox_imu_quat = scene_geometry["livox_imu_quat"]
    display_name = frame_resource(frame_name, "display")
    imu_origin_name = frame_resource(frame_name, "imu_origin")
    return f"""
      <body name="{frame_name}" pos="{_format_xyz(pose.pos)}"
            quat="{_format_quat_wxyz(pose.quat_wxyz)}">
        <geom name="{frame_resource(frame_name, 'origin_debug')}" type="sphere"
              size="{livox_origin_debug_radius}"
              material="mat_frame_origin_debug" mass="0"
              contype="0" conaffinity="0" group="1"/>
        <geom name="{display_name}" type="cylinder"
              size="{_format_values((livox_visual_radius, livox_visual_half_height))}"
              material="mat_lidar" mass="0"
              contype="{ROBOT_CONTYPE}" conaffinity="{ENV_CONTYPE}" condim="1"
              friction="0 0 0" group="{RENDER_GEOM_GROUP}"/>
        <inertial pos="0 0 0" mass="{livox_body_mass}"
                  diaginertia="{_format_xyz(livox_body_diaginertia_xyz)}"/>
        <geom name="{imu_origin_name}" pos="{_format_xyz(livox_imu_offset_xyz)}"
              type="sphere" size="{imu_origin_debug_radius}"
              material="mat_imu_origin_debug" mass="0"
              contype="0" conaffinity="0" group="1"/>
        <site name="{livox_lidar_site_name(frame_name)}"
              type="sphere" size="{lidar_site_radius}" rgba="1 0 1 0.5"/>
        <site name="{livox_imu_site_name(frame_name)}"
              pos="{_format_xyz(livox_imu_offset_xyz)}"
              quat="{_format_quat_wxyz(livox_imu_quat)}"
              type="sphere" size="{imu_site_radius}" rgba="1 1 0 0.6"/>
      </body>"""


def _build_x_direction_arrow_xml(
    frame_name: str,
    length: float,
    radius: float,
    z_offset: float,
    material_name: str,
) -> str:
    head_base_x = length * 0.75
    head_half_width = length * 0.15
    tip = (length, 0.0, 0.0)
    return f"""
      <body name="{frame_resource(frame_name, 'x_direction_arrow')}"
            pos="0 0 {z_offset:.9g}">
        <geom name="{frame_resource(frame_name, 'x_direction_arrow_shaft')}"
              type="cylinder" size="{radius:.9g}"
              fromto="{_format_values((0.0, 0.0, 0.0, *tip))}"
              material="{material_name}" mass="0"
              contype="0" conaffinity="0" group="{RENDER_GEOM_GROUP}"/>
        <geom name="{frame_resource(frame_name, 'x_direction_arrow_left')}"
              type="cylinder" size="{radius:.9g}"
              fromto="{_format_values((head_base_x, head_half_width, 0.0, *tip))}"
              material="{material_name}" mass="0"
              contype="0" conaffinity="0" group="{RENDER_GEOM_GROUP}"/>
        <geom name="{frame_resource(frame_name, 'x_direction_arrow_right')}"
              type="cylinder" size="{radius:.9g}"
              fromto="{_format_values((head_base_x, -head_half_width, 0.0, *tip))}"
              material="{material_name}" mass="0"
              contype="0" conaffinity="0" group="{RENDER_GEOM_GROUP}"/>
      </body>"""


def build_scene_xml(
    meshdir: str,
    frame_tree: RobotFrameTree,
    robot_init_location: tuple[float, float, float],
    boundary_x_min: float,
    boundary_x_max: float,
    boundary_y_min: float,
    boundary_y_max: float,
    scene_geometry: dict[str, object],
    physics: dict[str, object],
    enable_left_livox: bool,
    enable_right_livox: bool,
) -> str:
    spawn_x, spawn_y, spawn_z = robot_init_location
    gimbal_odom_pose = frame_tree.require_frame(FRAME_GIMBAL_ODOM)
    base_link_pose = frame_tree.require_frame(FRAME_BASE_LINK)
    left_livox_pose = frame_tree.require_frame(FRAME_LEFT_LIVOX)
    right_livox_pose = frame_tree.require_frame(FRAME_RIGHT_LIVOX)
    base_joint_axis = frame_tree.base_joint_axis
    gimbal_relative_to_base_pos, gimbal_relative_to_base_quat = _invert_pose(
        base_link_pose.pos,
        base_link_pose.quat_wxyz,
    )
    base_spawn_pos = (
        spawn_x + base_link_pose.pos[0],
        spawn_y + base_link_pose.pos[1],
        spawn_z + base_link_pose.pos[2],
    )
    base_spawn_quat = base_link_pose.quat_wxyz
    collision_dir = os.path.join(meshdir, "mesh_collision_env")
    collision_mesh_files = _list_obj_files(collision_dir)
    view_scene_mesh_rel = "mesh_view/mesh_view.obj"
    lidar_scene_mesh_rel = "mesh_lidar/mesh_lidar.obj"

    if not os.path.isfile(os.path.join(meshdir, view_scene_mesh_rel)):
        raise FileNotFoundError(
            f"View scene mesh not found: {os.path.join(meshdir, view_scene_mesh_rel)}"
        )
    if not os.path.isfile(os.path.join(meshdir, lidar_scene_mesh_rel)):
        raise FileNotFoundError(
            f"LiDAR scene mesh not found: {os.path.join(meshdir, lidar_scene_mesh_rel)}"
        )
    if not collision_mesh_files:
        raise FileNotFoundError(
            f"No OBJ files found under {collision_dir}. Run scripts/convert_collision_fbx.py first."
        )

    collision_asset_xml = "\n".join(
        f'    <mesh name="env_collision_mesh_{i}" file="mesh_collision_env/{name}"/>'
        for i, name in enumerate(collision_mesh_files)
    )
    collision_geom_xml = "\n".join(
        f'      <geom name="env_collision_geom_{i}" type="mesh" mesh="env_collision_mesh_{i}" '
        f'material="mat_collision_debug" contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}" '
        f'condim="1" friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>'
        for i, _ in enumerate(collision_mesh_files)
    )

    boundary_wall_thickness = scene_geometry["boundary_wall_thickness"]
    boundary_wall_half_height = scene_geometry["boundary_wall_height"] / 2.0
    boundary_half_span_y = 0.5 * (boundary_y_max - boundary_y_min)
    boundary_half_span_x = 0.5 * (boundary_x_max - boundary_x_min)
    boundary_geom_xml = f"""
      <geom name="boundary_wall_pos_x" type="box"
            pos="{boundary_x_max + boundary_wall_thickness} 0 {boundary_wall_half_height}"
            size="{boundary_wall_thickness} {boundary_half_span_y} {boundary_wall_half_height}"
            rgba="0 0 0 0" contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}"
            condim="1" friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>
      <geom name="boundary_wall_neg_x" type="box"
            pos="{boundary_x_min - boundary_wall_thickness} 0 {boundary_wall_half_height}"
            size="{boundary_wall_thickness} {boundary_half_span_y} {boundary_wall_half_height}"
            rgba="0 0 0 0" contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}"
            condim="1" friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>
      <geom name="boundary_wall_pos_y" type="box"
            pos="0 {boundary_y_max + boundary_wall_thickness} {boundary_wall_half_height}"
            size="{boundary_half_span_x} {boundary_wall_thickness} {boundary_wall_half_height}"
            rgba="0 0 0 0" contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}"
            condim="1" friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>
      <geom name="boundary_wall_neg_y" type="box"
            pos="0 {boundary_y_min - boundary_wall_thickness} {boundary_wall_half_height}"
            size="{boundary_half_span_x} {boundary_wall_thickness} {boundary_wall_half_height}"
            rgba="0 0 0 0" contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}"
            condim="1" friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>"""

    livox_blocks: list[str] = []
    sensor_blocks: list[str] = []
    debug_blocks: list[str] = []
    debug_blocks.append(
        f"""
      <body name="{base_force_debug_body_name()}" mocap="true" pos="0 0 0" quat="1 0 0 0">
        <geom name="{base_force_debug_geom_name()}" type="cylinder"
              size="{_format_values((scene_geometry['base_force_debug_radius'], scene_geometry['base_force_debug_min_length'] / 2.0))}"
              rgba="0.1 1.0 0.2 0.9" mass="0"
              contype="0" conaffinity="0" group="1"/>
      </body>"""
    )
    for frame_name, enabled, pose in (
        (FRAME_LEFT_LIVOX, enable_left_livox, left_livox_pose),
        (FRAME_RIGHT_LIVOX, enable_right_livox, right_livox_pose),
    ):
        if not enabled:
            continue
        livox_blocks.append(_build_livox_body_xml(frame_name, pose, scene_geometry))
        debug_blocks.append(
            f"""
      <body name="{livox_accel_debug_body_name(frame_name)}" mocap="true" pos="0 0 0" quat="1 0 0 0">
        <geom name="{livox_accel_debug_geom_name(frame_name)}" type="cylinder"
              size="{_format_values((scene_geometry['imu_accel_debug_radius'], scene_geometry['imu_accel_debug_min_length'] / 2.0))}"
              material="mat_imu_origin_debug" mass="0"
              contype="0" conaffinity="0" group="1"/>
      </body>"""
        )
        sensor_blocks.append(
            f'    <accelerometer name="{livox_accel_sensor_name(frame_name)}" site="{livox_imu_site_name(frame_name)}"/>'
        )
        sensor_blocks.append(
            f'    <gyro name="{livox_gyro_sensor_name(frame_name)}" site="{livox_imu_site_name(frame_name)}"/>'
        )

    gimbal_visual_half_extents_xyz = scene_geometry["gimbal_visual_half_extents_xyz"]
    gimbal_collision_half_extents_xyz = scene_geometry["gimbal_collision_half_extents_xyz"]
    gimbal_body_mass = scene_geometry["gimbal_body_mass"]
    gimbal_body_diaginertia_xyz = scene_geometry["gimbal_body_diaginertia_xyz"]
    base_bottom_height = scene_geometry["base_bottom_height"]
    base_top_height = scene_geometry["base_top_height"]
    base_radius = scene_geometry["base_diameter"] / 2.0
    base_half_height = (base_top_height - base_bottom_height) / 2.0
    base_center_height = (base_top_height + base_bottom_height) / 2.0
    base_body_mass = scene_geometry["base_body_mass"]
    base_border_size_xyz = scene_geometry["base_border_size_xyz"]
    base_border_half_extents_xyz = tuple(
        value / 2.0 for value in base_border_size_xyz
    )
    wheel_radius = scene_geometry["wheel_diameter"] / 2.0
    wheel_half_spacing_x = scene_geometry["wheel_spacing_x"] / 2.0
    wheel_half_spacing_y = scene_geometry["wheel_spacing_y"] / 2.0
    wheel_solref = _format_values(scene_geometry["wheel_contact_solref"])
    wheel_solimp = _format_values(scene_geometry["wheel_contact_solimp"])
    if scene_geometry["wheel_contact_solref"][0] < 2.0 * physics["timestep"]:
        raise ValueError(
            "wheel_contact_solref time constant must be at least 2 * physics_dt"
        )
    wheel_geoms_xml = "".join(
        f"""
      <geom name="{frame_resource(FRAME_BASE_LINK, f'wheel_{name}')}" type="sphere"
            size="{wheel_radius:.9g}" pos="{x:.9g} {y:.9g} {wheel_radius:.9g}"
            material="mat_wheel" mass="0" priority="1"
            solref="{wheel_solref}" solimp="{wheel_solimp}"
            contype="{ROBOT_CONTYPE}" conaffinity="{ENV_CONTYPE}" condim="1"
            friction="0 0 0" group="{RENDER_GEOM_GROUP}"/>"""
        for name, x, y in (
            ("front_left", wheel_half_spacing_x, wheel_half_spacing_y),
            ("front_right", wheel_half_spacing_x, -wheel_half_spacing_y),
            ("rear_left", -wheel_half_spacing_x, wheel_half_spacing_y),
            ("rear_right", -wheel_half_spacing_x, -wheel_half_spacing_y),
        )
    )
    base_visual_top = base_top_height
    base_border_xml = ""
    if scene_geometry["enable_base_border"]:
        base_visual_top += BASE_BORDER_TOP_OFFSET
        base_border_center_z = base_visual_top - base_border_half_extents_xyz[2]
        base_border_xml = f"""
      <geom name="{frame_resource(FRAME_BASE_LINK, 'border')}" type="box"
            size="{_format_xyz(base_border_half_extents_xyz)}"
            pos="0 0 {base_border_center_z:.9g}"
            material="mat_chassis" mass="0"
            contype="{ROBOT_CONTYPE}" conaffinity="{ENV_CONTYPE}" condim="1"
            friction="0 0 0" group="1"/>"""
    direction_arrow_radius = scene_geometry["direction_arrow_radius"]
    base_direction_arrow_xml = _build_x_direction_arrow_xml(
        FRAME_BASE_LINK,
        scene_geometry["base_direction_arrow_length"],
        direction_arrow_radius,
        base_visual_top + direction_arrow_radius,
        "mat_base_direction_arrow",
    )
    gimbal_direction_arrow_xml = _build_x_direction_arrow_xml(
        FRAME_GIMBAL,
        scene_geometry["gimbal_direction_arrow_length"],
        direction_arrow_radius,
        direction_arrow_radius,
        "mat_gimbal_direction_arrow",
    )
    frame_origin_debug_radius = scene_geometry["frame_origin_debug_radius"]
    gimbal_origin_debug_radius = scene_geometry["gimbal_origin_debug_radius"]
    livox_blocks_xml = "".join(livox_blocks)
    sensor_blocks_xml = "\n".join(sensor_blocks)
    debug_blocks_xml = "".join(debug_blocks)

    return f"""<mujoco model="sentry_sim_node">
  <compiler angle="radian" meshdir="{meshdir}"/>
  <option timestep="{physics['timestep']}" gravity="0 0 -9.81"
          integrator="{physics['integrator']}"/>
  <default>
    <geom solref="{_format_values(physics['contact_solref'])}"
          solimp="{_format_values(physics['contact_solimp'])}"/>
  </default>
  <visual>
    <map force="0.1" zfar="200"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.1 0.1 0.2" width="512" height="512"/>
    <texture name="tex_grid" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"
             width="512" height="512" mark="edge" markrgb="0.5 0.5 0.5"/>
    <material name="mat_ground" texture="tex_grid" texrepeat="10 10" texuniform="true" reflectance="0.1"/>
    <material name="mat_chassis" rgba="0.4 0.4 0.5 1.0"/>
    <material name="mat_wheel" rgba="0.08 0.08 0.09 1.0"/>
    <material name="mat_gimbal" rgba="0.2 0.2 0.3 1.0"/>
    <material name="mat_lidar" rgba="0.8 0.2 0.2 1.0"/>
    <material name="mat_arena" rgba="0.5 0.5 0.6 1.0"/>
    <material name="mat_lidar_debug" rgba="0.0 0.8 1.0 0.35"/>
    <material name="mat_collision_debug" rgba="1.0 0.55 0.1 0.28"/>
    <material name="mat_frame_origin_debug" rgba="1.0 0.0 1.0 1.0"/>
    <material name="mat_imu_origin_debug" rgba="1.0 1.0 0.0 1.0"/>
    <material name="mat_base_direction_arrow" rgba="{_format_values(scene_geometry['base_direction_arrow_rgba'])}"/>
    <material name="mat_gimbal_direction_arrow" rgba="{_format_values(scene_geometry['gimbal_direction_arrow_rgba'])}"/>
    <mesh name="arena_view_mesh" file="{view_scene_mesh_rel}"/>
    <mesh name="lidar_detect_mesh" file="{lidar_scene_mesh_rel}"/>
{collision_asset_xml}
  </asset>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="50 50 0.1" material="mat_ground"
          contype="{ENV_CONTYPE}" conaffinity="{ROBOT_CONTYPE}" condim="1"
          friction="0 0 0" group="{COLLISION_GEOM_GROUP}"/>
    <light name="sun" directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"
           pos="5 5 10" dir="-0.5 -0.5 -1"/>
    <body name="arena" pos="0 0 0">
      <geom name="arena_view_geom" type="mesh" mesh="arena_view_mesh" material="mat_arena"
            contype="0" conaffinity="0" group="{RENDER_GEOM_GROUP}"/>
      <geom name="lidar_geom_0" type="mesh" mesh="lidar_detect_mesh"
            rgba="0 0 0 0" contype="0" conaffinity="0" group="{LIDAR_TRACE_GEOM_GROUP}"/>
      <geom name="lidar_debug_geom" type="mesh" mesh="lidar_detect_mesh"
            material="mat_lidar_debug" contype="0" conaffinity="0" group="{LIDAR_DEBUG_GEOM_GROUP}"/>
{collision_geom_xml}
{boundary_geom_xml}
    </body>{debug_blocks_xml}
    <body name="{FRAME_BASE_LINK}" pos="{_format_xyz(base_spawn_pos)}"
          quat="{_format_quat_wxyz(base_spawn_quat)}">
      <freejoint name="base_freejoint"/>
      <geom name="{frame_resource(FRAME_BASE_LINK, 'origin_debug')}" type="sphere"
            size="{frame_origin_debug_radius}"
            material="mat_frame_origin_debug" mass="0"
            contype="0" conaffinity="0" group="1"/>
      <geom name="{frame_resource(FRAME_BASE_LINK, 'body')}" type="cylinder"
            size="{_format_values((base_radius, base_half_height))}"
            pos="0 0 {base_center_height:.9g}"
            material="mat_chassis" mass="{base_body_mass}"
            contype="{ROBOT_CONTYPE}" conaffinity="{ENV_CONTYPE}" condim="1"
            friction="0 0 0" group="1"/>{base_border_xml}{wheel_geoms_xml}{base_direction_arrow_xml}
      <body name="{FRAME_GIMBAL}" pos="{_format_xyz(gimbal_relative_to_base_pos)}"
            quat="{_format_quat_wxyz(gimbal_relative_to_base_quat)}">
        <joint name="{JOINT_GIMBAL_YAW}" type="hinge"
               axis="{_format_xyz(base_joint_axis)}" damping="0"/>
        <inertial pos="0 0 0" mass="{gimbal_body_mass}"
                  diaginertia="{_format_xyz(gimbal_body_diaginertia_xyz)}"/>
      <geom name="{frame_resource(FRAME_GIMBAL, 'origin_debug')}" type="sphere"
            size="{gimbal_origin_debug_radius}"
            material="mat_frame_origin_debug" mass="0"
            contype="0" conaffinity="0" group="1"/>
      <geom name="{frame_resource(FRAME_GIMBAL, 'display')}" type="box"
            size="{_format_xyz(gimbal_visual_half_extents_xyz)}"
            pos="0 0 {-gimbal_visual_half_extents_xyz[2]:.9g}"
            material="mat_gimbal" mass="0"
            contype="0" conaffinity="0" group="1"/>
      <geom name="{frame_resource(FRAME_GIMBAL, 'collision')}" type="box"
            size="{_format_xyz(gimbal_collision_half_extents_xyz)}"
            pos="0 0 {-gimbal_collision_half_extents_xyz[2]:.9g}"
            rgba="0 0 0 0" mass="0"
            contype="{ROBOT_CONTYPE}" conaffinity="{ENV_CONTYPE}" condim="1"
            friction="0 0 0" group="1"/>{gimbal_direction_arrow_xml}
      <body name="{FRAME_GIMBAL_ODOM}" pos="{_format_xyz(gimbal_odom_pose.pos)}"
            quat="{_format_quat_wxyz(gimbal_odom_pose.quat_wxyz)}">
        <geom name="{frame_resource(FRAME_GIMBAL_ODOM, 'origin_debug')}" type="sphere"
              size="{frame_origin_debug_radius}"
              material="mat_frame_origin_debug" mass="0"
              contype="0" conaffinity="0" group="1"/>
      </body>{livox_blocks_xml}
      </body>
    </body>
  </worldbody>
  <sensor>
{sensor_blocks_xml}
    <framepos name="{frame_resource(FRAME_BASE_LINK, 'pos_sensor')}" objtype="body" objname="{FRAME_BASE_LINK}"/>
    <framequat name="{frame_resource(FRAME_BASE_LINK, 'quat_sensor')}" objtype="body" objname="{FRAME_BASE_LINK}"/>
  </sensor>
</mujoco>"""
