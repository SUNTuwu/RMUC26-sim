#!/usr/bin/env bash
set -euo pipefail

ROOT_WS="."
NAV_WS="./src/external/RM2026-sentry-ws"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_DISTRO_UPPER="${ROS_DISTRO^^}"
ROS_CXX_FLAGS="-DROS_${ROS_DISTRO_UPPER}"

reset_ros_env() {
  unset AMENT_PREFIX_PATH
  unset COLCON_PREFIX_PATH
  unset CMAKE_PREFIX_PATH
  unset LD_LIBRARY_PATH
  unset PYTHONPATH
  unset PKG_CONFIG_PATH
  unset ROS_PACKAGE_PATH
  unset ROS_ETC_DIR
  unset ROS_ROOT
}
NAV_PACKAGES=(
  auto_aim_interfaces
  customized_client_msgs
  rm_decision_interfaces
  serial_driver_ch343
  nav_serial_driver_ch343
  livox_ros_driver2
  pointcloud_preprocessor
  io_bringup
  dynamic_rog_map
  nav2_trapezoid_smoother
  pb_omni_pid_pursuit_controller
  mapping_bringup
  nav_bringup
  main_bringup
)

reset_ros_env

if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
else
  echo "ROS setup not found: /opt/ros/${ROS_DISTRO}/setup.bash" >&2
  exit 1
fi

cd "${NAV_WS}"

# # Ensure all selected packages are regenerated with Ninja instead of any stale generator.
# for pkg in "${NAV_PACKAGES[@]}"; do
#   rm -rf "${NAV_WS}/build/${pkg}" "${NAV_WS}/install/${pkg}"
# done

colcon build \
  --symlink-install \
  --executor sequential \
  --packages-select "${NAV_PACKAGES[@]}" \
  --cmake-args -G Ninja "-DCMAKE_CXX_FLAGS=${ROS_CXX_FLAGS}"
