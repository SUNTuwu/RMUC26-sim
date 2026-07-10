#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/.."
NAV_DIR="${ROOT_DIR}/src/external/RM2026-sentry-ws"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"

SIM_PACKAGES=(
  sim_assets
  sim_core
  sim_bringup
)

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

cleanup_sim_build() {
  local pkg
  for pkg in "${SIM_PACKAGES[@]}"; do
    rm -rf "${ROOT_DIR}/build/${pkg}" "${ROOT_DIR}/install/${pkg}"
  done
}

############################### MAIN ####################################

reset_ros_env

if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  echo "ROS setup not found: /opt/ros/${ROS_DISTRO}/setup.bash" >&2
  exit 1
fi

source_setup "/opt/ros/${ROS_DISTRO}/setup.bash"

if [[ -f "${NAV_DIR}/install/setup.bash" ]]; then
  source_setup "${NAV_DIR}/install/setup.bash"
fi

if [[ "${1:-}" == "--rebuild" ]]; then
  cleanup_sim_build
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--rebuild]" >&2
  exit 1
fi

cd "${ROOT_DIR}"

colcon build \
  --symlink-install \
  --packages-select "${SIM_PACKAGES[@]}" \
  --cmake-args -G Ninja "-DCMAKE_CXX_FLAGS=-DROS_${ROS_DISTRO^^}"
