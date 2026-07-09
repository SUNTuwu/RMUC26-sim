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

SIM_CONFIG_FILE="$ROOT_DIR/src/sim_bringup/config/sim_config_${ROBOT_TYPE}.yaml"
if [[ ! -f "$SIM_CONFIG_FILE" ]]; then
  SIM_CONFIG_FILE="$ROOT_DIR/src/sim_bringup/config/sim_config.yaml"
fi

ROBOT_DESCRIPTION_XACRO_PATH="$ROOT_DIR/src/external/RM2026-sentry-ws/src/main_bringup/urdf/${ROBOT_TYPE}.urdf.xacro"
if [[ ! -f "$ROBOT_DESCRIPTION_XACRO_PATH" ]]; then
  ALT_SIM_XACRO="$ROOT_DIR/src/sim_bringup/urdf/${ROBOT_TYPE}.urdf.xacro"
  if [[ -f "$ALT_SIM_XACRO" ]]; then
    ROBOT_DESCRIPTION_XACRO_PATH="$ALT_SIM_XACRO"
  else
    ROBOT_DESCRIPTION_XACRO_PATH="$ROOT_DIR/src/sim_bringup/urdf/sentry.urdf.xacro"
  fi
fi

ros2 launch sim_bringup sim3d_nav.launch.py \
  robot_type:=$ROBOT_TYPE \
  sim_config_file:=$SIM_CONFIG_FILE \
  robot_description_xacro_path:=$ROBOT_DESCRIPTION_XACRO_PATH \
  use_nav_rviz:=$USE_NAV_RVIZ \
  map_file:=none \
  localization:=none \
  segmentation:=none \
  lio:=dual
  