"""LiDAR bridge: convert MuJoCo ray-traced PointCloud2 data into Livox CustomMsg format.

This module mirrors the real Livox driver output so that downstream nodes
(e.g. point_lio) can consume the simulated LiDAR data exactly as if it came
from a real Livox mid360.
"""

import numpy as np

try:
    from livox_ros_driver2.msg import CustomMsg, CustomPoint
except ImportError:
    # Allow the module to be imported before livox_ros_driver2 is built;
    # the actual conversion functions will fail at runtime if the types
    # are not available yet.
    CustomMsg = None  # type: ignore
    CustomPoint = None  # type: ignore


MID360_SCAN_LINES = 4
LIVOX_VALID_TAG = 0x10
LIVOX_INVALID_TAG = 0x00
DEFAULT_SCAN_PERIOD_NS = int(1_000_000_000 / 10.0)
DEFAULT_RANGE_MAX_M = 30.0


def _stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _ns_to_stamp(stamp_type, timestamp_ns: int):
    stamp = stamp_type()
    safe_ns = max(int(timestamp_ns), 0)
    stamp.sec = safe_ns // 1_000_000_000
    stamp.nanosec = safe_ns % 1_000_000_000
    return stamp


def _compute_line_indices(ray_phi: np.ndarray, phi_bounds: tuple[float, float]) -> np.ndarray:
    if len(ray_phi) == 0:
        return np.zeros(0, dtype=np.uint8)

    # The Mid-360 scan pattern is emitted as four interleaved channels.
    # Preserve that ordering directly so `line` stays stable even when
    # many rays are invalid in the current frame.
    if len(ray_phi) >= MID360_SCAN_LINES and len(ray_phi) % MID360_SCAN_LINES == 0:
        first_group = ray_phi[:MID360_SCAN_LINES]
        group_line_indices = np.empty(MID360_SCAN_LINES, dtype=np.uint8)
        group_line_indices[np.argsort(first_group)] = np.arange(
            MID360_SCAN_LINES, dtype=np.uint8
        )
        return np.tile(group_line_indices, len(ray_phi) // MID360_SCAN_LINES)

    phi_min, phi_max = phi_bounds
    if phi_max <= phi_min:
        return np.zeros(len(ray_phi), dtype=np.uint8)

    line_centers = np.linspace(phi_min, phi_max, MID360_SCAN_LINES, dtype=np.float64)
    distance_to_line = np.abs(ray_phi[:, np.newaxis] - line_centers[np.newaxis, :])
    return np.argmin(distance_to_line, axis=1).astype(np.uint8)


def _compute_offset_times(num_points: int, scan_period_ns: int) -> np.ndarray:
    if num_points <= 1:
        return np.zeros(num_points, dtype=np.uint32)

    safe_period_ns = max(int(scan_period_ns), num_points)
    point_offsets = np.floor(
        np.arange(num_points, dtype=np.float64) * safe_period_ns / num_points
    )
    return point_offsets.astype(np.uint32)


def _synthesize_reflectivity(intensities: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    normalized_intensity = np.clip(intensities, 0.0, 1.0)
    normalized_range = np.clip(ranges / DEFAULT_RANGE_MAX_M, 0.0, 1.0)

    # MuJoCo rays do not provide Livox-like reflectivity, so use a deterministic
    # proxy that keeps nearby returns brighter and avoids an all-constant cloud.
    reflectivity = 0.55 * normalized_intensity + 0.45 * (1.0 - normalized_range)
    return np.clip(np.round(reflectivity * 255.0), 1, 255).astype(np.uint8)


def _pack_custom_point_array(
    points_local: np.ndarray,
    reflectivity: np.ndarray,
    line_indices: np.ndarray,
    offset_times: np.ndarray,
    tags: np.ndarray,
) -> object:
    """Convert raw MuJoCo ray results into a list of CustomPoint messages."""
    if CustomPoint is None:
        raise RuntimeError(
            "livox_ros_driver2.msg.CustomPoint is not available. "
            "Build livox_ros_driver2 first."
        )

    points = []
    for i in range(len(points_local)):
        pt = CustomPoint()
        pt.x = float(points_local[i, 0])
        pt.y = float(points_local[i, 1])
        pt.z = float(points_local[i, 2])
        pt.reflectivity = int(reflectivity[i])
        pt.tag = int(tags[i])
        pt.line = int(line_indices[i])
        pt.offset_time = int(offset_times[i])
        points.append(pt)

    return points


def ranges_to_custom_msg(
    ranges: np.ndarray,
    ray_dirs: np.ndarray,
    ray_theta: np.ndarray,
    ray_phi: np.ndarray,
    frame_id: str,
    stamp,
    lidar_id: int = 0,
    scan_period_ns: int = DEFAULT_SCAN_PERIOD_NS,
) -> object:
    """Convert MuJoCo LiDAR ranges into a Livox CustomMsg.

    Parameters
    ----------
    ranges : np.ndarray  shape (N,)
        Ray distances [m].
    ray_dirs : np.ndarray  shape (N, 3)
        Precomputed unit ray directions in LiDAR-local frame.
    ray_theta : np.ndarray  shape (N,)
        Horizontal scan angle [rad].
    ray_phi : np.ndarray  shape (N,)
        Vertical scan angle [rad].
    frame_id : str
        TF frame of the LiDAR sensor.
    stamp : builtin_interfaces.msg.Time
        ROS time stamp.
    lidar_id : int
        LiDAR device id (used by point_lio to distinguish left/right).

    Returns
    -------
    sim_interfaces.msg.CustomMsg
    """
    if CustomMsg is None:
        raise RuntimeError(
            "sim_interfaces.msg.CustomMsg is not available. "
            "Build sim_interfaces first."
        )

    publish_time_ns = _stamp_to_ns(stamp)
    msg = CustomMsg()
    msg.header.frame_id = frame_id
    msg.lidar_id = lidar_id

    if len(ranges) == 0:
        msg.header.stamp = stamp
        msg.timebase = publish_time_ns
        msg.point_num = 0
        msg.points = []
        return msg

    valid = (ranges > 0.1) & (ranges < 100.0)
    intensities = np.ones(len(ranges), dtype=np.float32) * 0.5
    line_indices = _compute_line_indices(
        ray_phi,
        (
            float(np.min(ray_phi)) if len(ray_phi) else -0.12,
            float(np.max(ray_phi)) if len(ray_phi) else 0.12,
        ),
    )
    offset_times = _compute_offset_times(len(ranges), scan_period_ns)

    # Keep the ROS header timestamp at the frame publish/end time. `point_lio`
    # uses `header.stamp` together with each point's `offset_time`; publishing
    # the scan start time here shifts the whole frame backwards.
    msg.header.stamp = stamp
    msg.timebase = publish_time_ns

    if not np.any(valid):
        msg.point_num = 0
        msg.points = []
        return msg

    valid_ranges = ranges[valid]
    points_local = ray_dirs[valid] * valid_ranges[:, np.newaxis]
    reflectivity = _synthesize_reflectivity(intensities[valid], valid_ranges)
    tags = np.full(len(valid_ranges), LIVOX_VALID_TAG, dtype=np.uint8)

    points = _pack_custom_point_array(
        points_local,
        reflectivity,
        line_indices[valid],
        offset_times[valid],
        tags,
    )
    msg.points = points
    msg.point_num = len(points)

    return msg
