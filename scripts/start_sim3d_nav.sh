#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export ROS_HOME="${ROS_HOME:-$ROOT_DIR/.ros}"
mkdir -p "$ROS_HOME/log"

set +u
source .venv/bin/activate
source src/external/RM2026-sentry-ws/install/setup.bash
source install/setup.bash
set -u

ROBOT_TYPE=${ROBOT_TYPE:-sim_sentry_fold}
USE_NAV_RVIZ=${USE_NAV_RVIZ:-true}

ros2 launch sim_bringup sim3d_nav.launch.py \
  robot_type:=$ROBOT_TYPE \
  use_nav_rviz:=$USE_NAV_RVIZ \
  map_file:=none \
  localization:=none \
  segmentation:=none \
  lio:=pointlio
