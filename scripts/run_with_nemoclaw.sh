#!/usr/bin/env bash
# Run FactoryMind under a runtime policy.
#
# In order of preference:
#   1. nemoclaw   — the NVIDIA hackathon policy runner (if installed)
#   2. firejail   — an existing Linux sandbox (apt install firejail)
#   3. plain run  — no sandbox, but still emits a banner + log so the demo
#                   shows the policy file and intent.
#
# Either way you get logs/nemoclaw.log capturing what the policy would have
# enforced. That log is the artifact you screenshot for the demo.

set -euo pipefail

cd "$(dirname "$0")/.."

POLICY="${NEMOCLAW_POLICY:-scripts/nemoclaw_policy.yaml}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/nemoclaw.log"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${POLICY}" ]]; then
  echo "Policy file not found: ${POLICY}" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a && source .env && set +a
fi

PYTHON="python3"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

{
  echo "===================================================================="
  echo "FactoryMind R2D2 — agent policy run"
  echo "  policy : ${POLICY}"
  echo "  python : $(${PYTHON} -V 2>&1)"
  echo "  start  : $(stamp)"
  echo "===================================================================="
} | tee -a "${LOG_FILE}"

run_with_nemoclaw() {
  echo "[$(stamp)] launcher=nemoclaw" | tee -a "${LOG_FILE}"
  exec nemoclaw run --policy "${POLICY}" --log "${LOG_FILE}" -- \
    "${PYTHON}" -m factorymind.main
}

run_with_firejail() {
  echo "[$(stamp)] launcher=firejail (nemoclaw not found, using fallback sandbox)" \
    | tee -a "${LOG_FILE}"
  # Translate the policy intent into firejail flags.
  exec firejail \
    --quiet \
    --noprofile \
    --net=none \
    --whitelist="$(pwd)" \
    --read-only=/etc \
    --private-tmp \
    --caps.drop=all \
    --nonewprivs \
    --noroot \
    -- "${PYTHON}" -m factorymind.main 2>&1 | tee -a "${LOG_FILE}"
}

run_plain() {
  echo "[$(stamp)] launcher=plain (no sandbox available; intent is documented in policy)" \
    | tee -a "${LOG_FILE}"
  echo "[$(stamp)] policy_summary=$(grep -E '^(name|description):' ${POLICY} | tr '\n' ' ')" \
    | tee -a "${LOG_FILE}"
  exec "${PYTHON}" -m factorymind.main 2>&1 | tee -a "${LOG_FILE}"
}

if command -v nemoclaw >/dev/null 2>&1; then
  run_with_nemoclaw
elif command -v firejail >/dev/null 2>&1; then
  run_with_firejail
else
  echo "WARNING: neither 'nemoclaw' nor 'firejail' found on PATH." | tee -a "${LOG_FILE}"
  echo "         Install one for actual enforcement; running unsandboxed." | tee -a "${LOG_FILE}"
  run_plain
fi
