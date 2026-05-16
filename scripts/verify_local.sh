#!/usr/bin/env bash
# Verify local NIM endpoints. Run on the GX10 (or on a laptop with the tunnel up).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a && source .env && set +a
fi

export USE_LOCAL_NIM=true
export GX10_IP="${GX10_IP:-localhost}"

PYTHON="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

"${PYTHON}" -m factorymind.inference
