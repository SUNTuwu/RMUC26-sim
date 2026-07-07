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

ENABLE_VIEWER=${ENABLE_VIEWER:-true}

ros2 launch sim_bringup control_test.launch.py