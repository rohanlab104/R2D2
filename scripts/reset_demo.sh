#!/usr/bin/env bash
# One-command reset if NIM or the disconnect demo gets wedged. Run on the GX10.
set -euo pipefail

echo "Removing cloud block iptables rules (if present)..."
if command -v iptables >/dev/null 2>&1; then
  sudo iptables -D OUTPUT -d build.nvidia.com -j DROP 2>/dev/null || true
  sudo iptables -D OUTPUT -d integrate.api.nvidia.com -j DROP 2>/dev/null || true
else
  echo "iptables not found — skip"
fi

echo "Stopping NIM containers (by name)..."
for name in nim-nano-9b nim-super-49b; do
  if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
    docker rm -f "${name}" >/dev/null 2>&1 || true
    echo "  stopped ${name}"
  fi
done

echo
echo "Done. Restart NIM with:"
echo "  ./scripts/run_nim_nano.sh   # terminal 1"
echo "  ./scripts/run_nim_49b.sh    # terminal 2"
