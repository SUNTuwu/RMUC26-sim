#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/.."

source_setup() {
  local setup_file="$1"
  set +u
  # shellcheck disable=SC1090
  source "${setup_file}"
  set -u
}

cd "${ROOT_DIR}"

export ROS_HOME="${ROS_HOME:-${ROOT_DIR}/.ros}"
mkdir -p "${ROS_HOME}/log"

source_setup "${ROOT_DIR}/.venv/bin/activate"
if [[ -f "${ROOT_DIR}/src/external/RM2026-sentry-ws/install/setup.bash" ]]; then
  source_setup "${ROOT_DIR}/src/external/RM2026-sentry-ws/install/setup.bash"
fi
source_setup "${ROOT_DIR}/install/setup.bash"

ros2 launch sim_bringup keyboard_control.launch.py
