#!/usr/bin/env bash
# Quick health check against a running NIM (GX10 or tunneled localhost).
# Usage: ./scripts/test_nim_curl.sh [host] [port] [model]
set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-8000}"
MODEL="${3:-nvidia/nvidia-nemotron-nano-9b-v2}"

URL="http://${HOST}:${PORT}/v1/chat/completions"

echo "GET http://${HOST}:${PORT}/v1/models"
curl -fsS "http://${HOST}:${PORT}/v1/models" | python3 -m json.tool || true
echo

echo "POST ${URL}"
curl -fsS "${URL}" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"say hi\"}], \"max_tokens\": 32}" \
  | python3 -m json.tool
