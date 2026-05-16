#!/usr/bin/env bash
# Run Llama-3.3 Nemotron Super 49B NIM on the DGX Spark GX10 (host port 8001).
# Run ON the GX10 in a SECOND terminal — start the pull early, the image is large.
#
# DGX Spark has 128 GB unified memory; running 49B alongside 9B is tight.
# If the 49B fails to start due to memory, stop the 9B briefly to confirm,
# then look for a quantized DGX-Spark-specific tag on build.nvidia.com.

set -euo pipefail

: "${NGC_API_KEY:?Set NGC_API_KEY (your build.nvidia.com / NGC API key)}"

IMAGE="${NIM_LEADER_IMAGE:-nvcr.io/nim/nvidia/llama-3_3-nemotron-super-49b-v1_5:latest}"
PORT="${NIM_LEADER_PORT:-8001}"
NAME="${NIM_LEADER_NAME:-nim-super-49b}"
CACHE_DIR="${NIM_CACHE_DIR:-${HOME}/.cache/nim}"

mkdir -p "${CACHE_DIR}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon not reachable. Ask an NVIDIA mentor." >&2
  exit 1
fi

echo "Logging in to nvcr.io (using NGC_API_KEY)..."
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo "Pulling ${IMAGE} (large download — be patient)..."
docker pull "${IMAGE}"

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
