#!/usr/bin/env bash
set -euo pipefail

ROOT_WS="/home/somo/dev/sentry_sim"
NAV_WS="/home/somo/dev/sentry_sim/src/external/RM2026-sentry-ws"
IGNORED_EXTERNAL_PACKAGES=(
  auto_aim_interfaces
  customized_client_msgs
  rm_decision_interfaces
  serial_driver_ch343
  nav_serial_driver_ch343
  pb_omni_pid_pursuit_controller
  livox_ros_driver2
  pointcloud_preprocessor
  io_bringup
  dynamic_rog_map
  nav2_trapezoid_smoother
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

if [[ -f "${NAV_WS}/install/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${NAV_WS}/install/setup.bash"
  set -u
fi

cd "${ROOT_WS}"

colcon build \
  --symlink-install \
  --executor sequential \
  --packages-select \
    sim_assets \
    sim_description \
    sim_core \
    sim_bringup \
  --packages-ignore "${IGNORED_EXTERNAL_PACKAGES[@]}" \
  --cmake-args -G Ninja
