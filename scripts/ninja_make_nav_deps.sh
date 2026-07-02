#!/usr/bin/env bash
set -euo pipefail

ROOT_WS="/home/somo/dev/sentry_sim"
NAV_WS="/home/somo/dev/sentry_sim/src/external/RM2026-sentry-ws"
ROS_CMAKE_DEFINE=""
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

if [[ -f "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
  set -u
fi

case "${ROS_DISTRO:-humble}" in
  jazzy)
    ROS_CMAKE_DEFINE="-DROS_JAZZY=1"
    ;;
  humble)
    ROS_CMAKE_DEFINE="-DROS_HUMBLE=1"
    ;;
esac

if [[ -f "${ROOT_WS}/install/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${ROOT_WS}/install/setup.bash"
  set -u
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
  --cmake-args -G Ninja "${ROS_CMAKE_DEFINE}"
