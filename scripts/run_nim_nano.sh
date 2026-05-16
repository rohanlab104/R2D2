#!/usr/bin/env bash
# Run Nemotron-Nano-9B NIM on the DGX Spark GX10 (host port 8000).
# Name is run_nim_nano.sh (nim, not num). Loads ../.env; NGC uses NVIDIA_API_KEY if unset.
# Run ON the GX10, in its own terminal. Leave it foregrounded.
#
# Verify the exact image tag for DGX Spark on build.nvidia.com → Deploy.
# First pull can take 20-40 minutes; subsequent runs are seconds.

set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${_repo_root}/.env" ]]; then
  set -a && source "${_repo_root}/.env" && set +a
else
  echo "No ${_repo_root}/.env — create it (cp .env.example .env) or export NGC_API_KEY." >&2
fi

# nvcr.io login uses NGC_API_KEY; it is usually identical to NVIDIA_API_KEY.
if [[ -z "${NGC_API_KEY:-}" && -n "${NVIDIA_API_KEY:-}" ]]; then
  NGC_API_KEY="${NVIDIA_API_KEY}"
fi

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is still empty after loading ${_repo_root}/.env" >&2
  echo "Fix: set NGC_API_KEY=... in .env, or paste the same value as NVIDIA_API_KEY." >&2
  echo "Check for typos (NGC_API_KEY), spaces around =, or editing .env.example instead of .env." >&2
  exit 1
fi

IMAGE="${NIM_LEADER_IMAGE:-nvcr.io/nim/nvidia/nemotron-nano-9b-v2:latest}"
PORT="${NIM_LEADER_PORT:-8000}"
NAME="${NIM_LEADER_NAME:-nim-nano-9b}"
CACHE_DIR="${NIM_CACHE_DIR:-${HOME}/.cache/nim}"

mkdir -p "${CACHE_DIR}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon not reachable. Ask an NVIDIA mentor." >&2
  exit 1
fi

echo "Logging in to nvcr.io (using NGC_API_KEY)..."
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo "Pulling ${IMAGE} (first run may take 20-40 min)..."
docker pull "${IMAGE}"

# Stop any previous instance with the same name.
docker rm -f "${NAME}" >/dev/null 2>&1 || true

echo "Starting ${NAME} on host port ${PORT} (container 8000)..."
exec docker run --rm \
  --name "${NAME}" \
  --gpus all \
  --shm-size=16g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e "NGC_API_KEY=${NGC_API_KEY}" \
  -v "${CACHE_DIR}:/opt/nim/.cache" \
  -p "${PORT}:8000" \
  "${IMAGE}"
