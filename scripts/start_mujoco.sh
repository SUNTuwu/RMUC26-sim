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

resolve_enable_viewer() {
  if [[ -n "${ENABLE_VIEWER:-}" ]]; then
    printf '%s\n' "${ENABLE_VIEWER}"
    return
  fi

  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    printf 'false\n'
    return
  fi

  if command -v glxinfo >/dev/null 2>&1; then
    if ! glxinfo -B >/dev/null 2>&1; then
      echo "OpenGL context unavailable on current display; auto-disabling MuJoCo viewer." >&2
      printf 'false\n'
      return
    fi
  fi

  printf 'true\n'
}

cd "${ROOT_DIR}"

export ROS_HOME="${ROS_HOME:-${ROOT_DIR}/.ros}"
mkdir -p "${ROS_HOME}/log"

source_setup "${ROOT_DIR}/.venv/bin/activate"
if [[ -f "${ROOT_DIR}/src/external/RM2026-sentry-ws/install/setup.bash" ]]; then
  source_setup "${ROOT_DIR}/src/external/RM2026-sentry-ws/install/setup.bash"
fi
source_setup "${ROOT_DIR}/install/setup.bash"

export ROBOT_TYPE="${ROBOT_TYPE:-sim_sentry_fold}"

ros2 launch sim_bringup sim.launch.py \
  enable_viewer:="$(resolve_enable_viewer)"
