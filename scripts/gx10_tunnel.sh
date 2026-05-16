#!/usr/bin/env bash
# Forward GX10 NIM ports to your laptop so USE_LOCAL_NIM=true GX10_IP=localhost works.
#
# Usage:
#   export GX10_USER=your_username
#   export GX10_IP=128.114.x.x      # from event organizers
#   ./scripts/gx10_tunnel.sh
#
# Keep this terminal open while hacking. In another terminal:
#   USE_LOCAL_NIM=true GX10_IP=localhost python3 -m factorymind.inference

set -euo pipefail

: "${GX10_USER:?Set GX10_USER (your SSH username on the GX10)}"
: "${GX10_IP:?Set GX10_IP (from check-in / event organizers)}"

echo "Tunneling localhost:8000 and :8001 -> ${GX10_USER}@${GX10_IP} ..."
echo "Press Ctrl+C to stop."
exec ssh -N \
  -L 8000:localhost:8000 \
  -L 8001:localhost:8001 \
  "${GX10_USER}@${GX10_IP}"
