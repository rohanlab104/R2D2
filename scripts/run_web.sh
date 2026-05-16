#!/usr/bin/env bash
# Launch FactoryMind R2D2 with the 3D web viewer instead of pygame.
#
# Open http://<gx10-ip>:8080 in any browser on the same network, or
# http://localhost:8080 if you are on the GX10 itself.
#
# Env vars (optional):
#   FACTORYMIND_WEB_HOST   default 0.0.0.0
#   FACTORYMIND_WEB_PORT   default 8080
#   USE_LOCAL_NIM=true GX10_IP=localhost AGENTS_USE_MOCK=false  (recommended)

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="python3"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

if [[ -f .env ]]; then
  set -a && source .env && set +a
fi

export FACTORYMIND_WEB_HOST="${FACTORYMIND_WEB_HOST:-0.0.0.0}"
export FACTORYMIND_WEB_PORT="${FACTORYMIND_WEB_PORT:-8080}"

exec "${PYTHON}" -m factorymind.web_main
