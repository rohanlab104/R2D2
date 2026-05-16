#!/usr/bin/env bash
# Hour 18-20: block outbound cloud API while local NIM keeps working.
set -euo pipefail

echo "Blocking NVIDIA cloud API egress (requires sudo)..."
sudo iptables -A OUTPUT -d build.nvidia.com -j DROP
sudo iptables -A OUTPUT -d integrate.api.nvidia.com -j DROP
echo "Cloud blocked. Confirm local inference still works:"
echo "  USE_LOCAL_NIM=true GX10_IP=localhost python3 -m factorymind.inference"
echo "To undo: ./scripts/reset_demo.sh"
