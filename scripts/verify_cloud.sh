#!/usr/bin/env bash
# Hour 0-1: confirm both Nemotron models work via NVIDIA cloud API.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "Copy .env.example -> .env and set NVIDIA_API_KEY from build.nvidia.com"
  exit 1
fi

# shellcheck disable=SC1091
set -a && source .env && set +a

export USE_LOCAL_NIM=false

echo "=== Leader (Nano 9B) ==="
python3 -m factorymind.inference --leader-only

echo ""
echo "=== Strategist (49B) ==="
python3 -m factorymind.inference --strategist-only

echo ""
echo "Post in team chat: cloud inference confirmed for both models"
