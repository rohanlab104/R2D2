#!/usr/bin/env bash
# Start the headless FactoryMind simulation and serve the Three.js 3D viewer.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="python3"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

exec "${PYTHON}" -m factorymind.web_main
