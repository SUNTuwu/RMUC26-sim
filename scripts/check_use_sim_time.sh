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

echo "[check] /clock"
if timeout 3 ros2 topic echo /clock --once >/dev/null 2>&1; then
  echo "/clock -> ok"
else
  echo "/clock -> missing"
fi

echo
echo "[check] use_sim_time"
if [[ -z "$(ros2 node list)" ]]; then
  echo "no nodes found"
  exit 0
fi

ros2 node list | sort -u | while read -r node; do
  echo "[node] $node"
  case "$(timeout 2 ros2 param get "$node" use_sim_time 2>&1 || true)" in
    *"Boolean value is: True"*) echo "$node -> True" ;;
    *"Boolean value is: False"*) echo "$node -> False" ;;
    *"Timed out"*|*"timeout"*) echo "$node -> timeout" ;;
    *) echo "$node -> N/A" ;;
  esac
done
