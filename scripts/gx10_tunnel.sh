#!/usr/bin/env bash
# Forward GX10 ports to your laptop:
#   - 8000 / 8001 : NIM endpoints (Nemotron Nano 9B, Llama-3.3 Nemotron Super 49B)
#   - 8080        : the 3D web viewer (factorymind.web_main)
#
# Once this is running on your laptop, open http://localhost:8080 in any
# browser to watch the simulation that's executing on the GX10.
#
# Usage:
#   export GX10_USER=your_username
#   export GX10_IP=128.114.x.x        # from event organizers
#   ./scripts/gx10_tunnel.sh
#
# In another terminal on your laptop you can also do API smoke tests:
#   USE_LOCAL_NIM=true GX10_IP=localhost python3 -m factorymind.inference
#
# Override the web port if you ran the GX10 with FACTORYMIND_WEB_PORT=...:
#   FACTORYMIND_WEB_PORT=9090 ./scripts/gx10_tunnel.sh

set -euo pipefail

: "${GX10_USER:?Set GX10_USER (your SSH username on the GX10)}"
: "${GX10_IP:?Set GX10_IP (from check-in / event organizers)}"

WEB_PORT="${FACTORYMIND_WEB_PORT:-8080}"

echo "Tunneling -> ${GX10_USER}@${GX10_IP}"
echo "  localhost:8000        -> GX10 NIM (leader/worker, Nemotron Nano 9B)"
echo "  localhost:8001        -> GX10 NIM (strategist, Llama-3.3 Nemotron Super 49B)"
echo "  localhost:${WEB_PORT}        -> 3D web viewer (open http://localhost:${WEB_PORT})"
echo "Press Ctrl+C to stop."
exec ssh -N \
  -L 8000:localhost:8000 \
  -L 8001:localhost:8001 \
  -L "${WEB_PORT}:localhost:${WEB_PORT}" \
  "${GX10_USER}@${GX10_IP}"
