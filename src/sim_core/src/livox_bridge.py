"""LiDAR bridge: convert MuJoCo ray-traced PointCloud2 data into Livox CustomMsg format.

This module mirrors the real Livox driver output so that downstream nodes
(e.g. point_lio) can consume the simulated LiDAR data exactly as if it came
from a real Livox mid360.
"""

import numpy as np
import sensor_msgs.msg

try:
    from livox_ros_driver2.msg import CustomMsg, CustomPoint
except ImportError:
    # Allow the module to be imported before livox_ros_driver2 is built;
    # the actual conversion functions will fail at runtime if the types
    # are not available yet.
    CustomMsg = None  # type: ignore
    CustomPoint = None  # type: ignore


MID360_SCAN_LINES = 4
LIVOX_TAG = 0x00
POINT_OFFSET_STEP_NS = 5_000
REFLECTIVITY_MAX = 40.0
REFLECTIVITY_DECAY_METERS = 2.0


def _stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _compute_line_indices(ray_phi: np.ndarray) -> np.ndarray:
    if len(ray_phi) == 0:
        return np.zeros(0, dtype=np.uint8)

    # Mid-360 是 4 线交织输出；直接从首组 4 个竖直角恢复 line 顺序，
    # 后续整帧按同一模式重复即可。
    if len(ray_phi) >= MID360_SCAN_LINES and len(ray_phi) % MID360_SCAN_LINES == 0:
        first_group = ray_phi[:MID360_SCAN_LINES]
        group_line_indices = np.empty(MID360_SCAN_LINES, dtype=np.uint8)
        group_line_indices[np.argsort(first_group)] = np.arange(
            MID360_SCAN_LINES, dtype=np.uint8
        )
        return np.tile(group_line_indices, len(ray_phi) // MID360_SCAN_LINES)

    line_centers = np.linspace(
        float(np.min(ray_phi)),
        float(np.max(ray_phi)),
        MID360_SCAN_LINES,
        dtype=np.float64,
    )
    distance_to_line = np.abs(ray_phi[:, np.newaxis] - line_centers[np.newaxis, :])
    return np.argmin(distance_to_line, axis=1).astype(np.uint8)


def _compute_offset_times(num_points: int) -> np.ndarray:
    if num_points <= 0:
        return np.zeros(num_points, dtype=np.uint32)

    # 对齐当前核查目标：相邻两点 offset_time 固定差 5000ns。
    point_offsets = np.arange(num_points, dtype=np.uint64) * np.uint64(POINT_OFFSET_STEP_NS)
    return np.minimum(point_offsets, np.uint64(np.iinfo(np.uint32).max)).astype(np.uint32)


def _synthesize_reflectivity(ranges: np.ndarray) -> np.ndarray:
    # MuJoCo 没有真实反射率，这里只生成一个低两位数的距离衰减代理值，
    # 让数值量级更接近当前实机录包。
    reflectivity = REFLECTIVITY_MAX * np.exp(
        -np.maximum(ranges, 0.0) / REFLECTIVITY_DECAY_METERS
    )
    return np.clip(np.round(reflectivity), 1, 255).astype(np.uint8)


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


def _build_pointcloud2_array(
    points_local: np.ndarray,
    reflectivity: np.ndarray,
    line_indices: np.ndarray,
    offset_times: np.ndarray,
    tags: np.ndarray,
    stamp_ns: int,
) -> np.ndarray:
    last_offset = int(offset_times[-1]) if len(offset_times) > 0 else 0
    point_timestamps = (
        stamp_ns - (last_offset - offset_times.astype(np.int64))
    ).astype(np.float64)

    cloud_dtype = np.dtype(
        {
            "names": ("x", "y", "z", "intensity", "tag", "line", "timestamp"),
            "formats": (
                np.float32,
                np.float32,
                np.float32,
                np.float32,
                np.uint8,
                np.uint8,
                np.float64,
            ),
            "offsets": (0, 4, 8, 12, 16, 17, 18),
            "itemsize": 26,
        }
    )

    cloud = np.zeros(len(points_local), dtype=cloud_dtype)
    cloud["x"] = points_local[:, 0].astype(np.float32)
    cloud["y"] = points_local[:, 1].astype(np.float32)
    cloud["z"] = points_local[:, 2].astype(np.float32)
    cloud["intensity"] = reflectivity.astype(np.float32)
    cloud["tag"] = tags.astype(np.uint8)
    cloud["line"] = line_indices.astype(np.uint8)
    cloud["timestamp"] = point_timestamps
    return cloud


def ranges_to_pointcloud2_msg(
    ranges: np.ndarray,
    ray_dirs: np.ndarray,
    ray_phi: np.ndarray,
    frame_id: str,
    stamp,
    points_local: np.ndarray | None = None,
) -> sensor_msgs.msg.PointCloud2:
    """把 MuJoCo 射线结果转换成 Livox PointXYZRTLT PointCloud2。"""
    stamp_ns = _stamp_to_ns(stamp)
    line_indices = _compute_line_indices(ray_phi)
    offset_times = _compute_offset_times(len(ranges))
    valid = (ranges > 0.1) & (ranges < 100.0)

    if points_local is None:
        all_points_local = ray_dirs * ranges[:, np.newaxis]
    else:
        all_points_local = np.asarray(points_local, dtype=np.float32).copy()

    if len(ranges) == 0:
        cloud = sensor_msgs.msg.PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = frame_id
        cloud.height = 1
        cloud.width = 0
        cloud.is_bigendian = False
        cloud.is_dense = True
        cloud.point_step = 26
        cloud.row_step = 0
        cloud.fields = [
            sensor_msgs.msg.PointField(name="x", offset=0, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
            sensor_msgs.msg.PointField(name="y", offset=4, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
            sensor_msgs.msg.PointField(name="z", offset=8, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
            sensor_msgs.msg.PointField(name="intensity", offset=12, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
            sensor_msgs.msg.PointField(name="tag", offset=16, datatype=sensor_msgs.msg.PointField.UINT8, count=1),
            sensor_msgs.msg.PointField(name="line", offset=17, datatype=sensor_msgs.msg.PointField.UINT8, count=1),
            sensor_msgs.msg.PointField(name="timestamp", offset=18, datatype=sensor_msgs.msg.PointField.FLOAT64, count=1),
        ]
        return cloud

    all_points_local = np.asarray(all_points_local, dtype=np.float32)
    reflectivity = np.zeros(len(ranges), dtype=np.uint8)
    tags = np.full(len(ranges), LIVOX_TAG, dtype=np.uint8)

    # PointCloud2 也保留整帧顺序和空回波占位，便于对齐实机包结构。
    all_points_local[~valid] = 0.0
    if np.any(valid):
        reflectivity[valid] = _synthesize_reflectivity(ranges[valid])

    cloud_array = _build_pointcloud2_array(
        all_points_local,
        reflectivity,
        line_indices,
        offset_times,
        tags,
        stamp_ns,
    )

    cloud = sensor_msgs.msg.PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = frame_id
    cloud.height = 1
    cloud.width = len(cloud_array)
    cloud.is_bigendian = False
    cloud.is_dense = True
    cloud.point_step = 26
    cloud.row_step = cloud.width * cloud.point_step
    cloud.fields = [
        sensor_msgs.msg.PointField(name="x", offset=0, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
        sensor_msgs.msg.PointField(name="y", offset=4, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
        sensor_msgs.msg.PointField(name="z", offset=8, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
        sensor_msgs.msg.PointField(name="intensity", offset=12, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1),
        sensor_msgs.msg.PointField(name="tag", offset=16, datatype=sensor_msgs.msg.PointField.UINT8, count=1),
        sensor_msgs.msg.PointField(name="line", offset=17, datatype=sensor_msgs.msg.PointField.UINT8, count=1),
        sensor_msgs.msg.PointField(name="timestamp", offset=18, datatype=sensor_msgs.msg.PointField.FLOAT64, count=1),
    ]
    cloud.data = cloud_array.tobytes()
    return cloud


def ranges_to_custom_msg(
    ranges: np.ndarray,
    ray_dirs: np.ndarray,
    ray_phi: np.ndarray,
    frame_id: str,
    stamp,
    lidar_id: int = 0,
    points_local: np.ndarray | None = None,
) -> object:
    """把 MuJoCo 射线结果转换成 Livox CustomMsg。"""
    if CustomMsg is None:
        raise RuntimeError(
            "livox_ros_driver2.msg.CustomMsg is not available. "
            "Build livox_ros_driver2 first."
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
    line_indices = _compute_line_indices(ray_phi)
    offset_times = _compute_offset_times(len(ranges))

    # 这里继续使用帧末时间；前面已经确认当前 timestamp 语义是正确的。
    msg.header.stamp = stamp
    msg.timebase = publish_time_ns

    if points_local is None:
        all_points_local = ray_dirs * ranges[:, np.newaxis]
    else:
        all_points_local = np.asarray(points_local, dtype=np.float32).copy()

    all_points_local = np.asarray(all_points_local, dtype=np.float32)
    reflectivity = np.zeros(len(ranges), dtype=np.uint8)
    tags = np.full(len(ranges), LIVOX_TAG, dtype=np.uint8)

    # 保留整帧发射顺序；空回波继续占位，避免破坏 line 和 offset_time。
    if np.any(valid):
        all_points_local[~valid] = 0.0
        reflectivity[valid] = _synthesize_reflectivity(ranges[valid])
    else:
        all_points_local[:] = 0.0

    points = _pack_custom_point_array(
        all_points_local,
        reflectivity,
        line_indices,
        offset_times,
        tags,
    )
    msg.points = points
    msg.point_num = len(points)

    return msg
