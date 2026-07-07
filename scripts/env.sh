#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_FALLBACK_BIN:-python3}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python interpreter not found, tried: python3.12 and python3" >&2
  exit 1
fi

echo "[env.sh] root: ${ROOT_DIR}"
echo "[env.sh] python: $(command -v "${PYTHON_BIN}")"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[env.sh] creating ${VENV_DIR}"
else
  echo "[env.sh] reusing existing ${VENV_DIR}"
fi

"${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"

set +u
source "${VENV_DIR}/bin/activate"
set -u

python -m pip install --upgrade pip setuptools wheel

# Keep the recreated venv aligned with the current local-only packages in ./.venv.
python -m pip install \
  absl-py==2.4.0 \
  etils==1.14.0 \
  evdev==1.9.3 \
  fsspec==2026.6.0 \
  glfw==2.10.0 \
  mujoco==3.10.0 \
  mujoco_lidar==0.3.3 \
  numpy==1.26.4 \
  pynput==1.8.2 \
  PyOpenGL==3.1.10 \
  python-xlib==0.33

echo "[env.sh] .venv is ready"
echo "[env.sh] activate with: source .venv/bin/activate"
