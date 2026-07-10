#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/.."
NAV_DIR="${ROOT_DIR}/src/external/RM2026-sentry-ws"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"

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

source_setup() {
  local setup_file="$1"
  set +u
  # shellcheck disable=SC1090
  source "${setup_file}"
  set -u
}

cleanup_nav_build() {
  local pkg
  for pkg in "${NAV_PACKAGES[@]}"; do
    rm -rf "${NAV_DIR}/build/${pkg}" "${NAV_DIR}/install/${pkg}"
  done
}

NAV_PACKAGES=(
  auto_aim_interfaces
  customized_client_msgs
  rm_decision_interfaces
  serial_driver_ch343
  nav_serial_driver_ch343
  livox_ros_driver2
  pointcloud_preprocessor
  point_lio
  io_bringup
  dynamic_rog_map
  nav2_trapezoid_smoother
  pb_omni_pid_pursuit_controller
  mapping_bringup
  nav_bringup
  main_bringup
)

############################### MAIN ####################################

reset_ros_env

if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  echo "ROS setup not found: /opt/ros/${ROS_DISTRO}/setup.bash" >&2
  exit 1
fi

source_setup "/opt/ros/${ROS_DISTRO}/setup.bash"

if [[ "${1:-}" == "--rebuild" ]]; then
  cleanup_nav_build
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--rebuild]" >&2
  exit 1
fi

cd "${NAV_DIR}"

colcon build \
  --symlink-install \
  --packages-select "${NAV_PACKAGES[@]}" \
  --cmake-args -G Ninja "-DCMAKE_CXX_FLAGS=-DROS_${ROS_DISTRO^^}"
