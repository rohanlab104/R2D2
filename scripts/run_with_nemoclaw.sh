#!/usr/bin/env bash
# Run FactoryMind under a runtime policy.
#
# IMPORTANT — Official NVIDIA NemoClaw has NO `nemoclaw run` subcommand.
# NemoClaw manages OpenClaw sandboxes via `nemoclaw onboard`, `nemoclaw list`,
# `nemoclaw <name> connect`, etc. To run an arbitrary host command inside an
# onboarded sandbox, use OpenShell:
#   openshell sandbox exec -n <sandbox-name> -- <cmd...>
#
# This script picks an execution mode:
#
#   1. OpenShell sandbox exec (optional)
#      Set FACTORYMIND_USE_OPEN_SHELL_EXEC=1 and sync your repo into the sandbox
#      (see README). Uses NEMOCLAW_SANDBOX_NAME or FACTORYMIND_SANDBOX (default
#      my-assistant) and FACTORYMIND_SANDBOX_DIR (default /sandbox/R2D2).
#
#   2. firejail — Linux sandbox approximating locked-down network + cwd (if installed)
#
#   3. plain — run Python on the host (FactoryMind still loads nemoclaw_policy.yaml
#      and writes ALLOW/DENY lines to logs/nemoclaw.log from main.py)
#
# Either way you get logs/nemoclaw.log capturing launcher intent + app policy lines.

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

FACTORYMIND_MODULE="${FACTORYMIND_MODULE:-factorymind.web_main}"
SANDBOX="${NEMOCLAW_SANDBOX_NAME:-${FACTORYMIND_SANDBOX:-my-assistant}}"
SANDBOX_DIR="${FACTORYMIND_SANDBOX_DIR:-/sandbox/R2D2}"
# Interpreter inside the sandbox (host .venv paths do not exist there).
SANDBOX_PYTHON="${FACTORYMIND_SANDBOX_PYTHON:-python3}"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

{
  echo "===================================================================="
  echo "FactoryMind R2D2 — agent policy run"
  echo "  policy : ${POLICY}"
  echo "  python : $(${PYTHON} -V 2>&1)"
  echo "  module : ${FACTORYMIND_MODULE}"
  echo "  start  : $(stamp)"
  echo "===================================================================="
} | tee -a "${LOG_FILE}"

run_openshell_sandbox_exec() {
  echo "[$(stamp)] launcher=openshell sandbox exec (sandbox=${SANDBOX} cwd=${SANDBOX_DIR} py=${SANDBOX_PYTHON})" | tee -a "${LOG_FILE}"
  # shellcheck disable=SC2086
  exec openshell sandbox exec -n "${SANDBOX}" -- \
    bash -lc "cd '${SANDBOX_DIR}' && exec '${SANDBOX_PYTHON}' -m ${FACTORYMIND_MODULE}" \
    2>&1 | tee -a "${LOG_FILE}"
}

run_with_firejail() {
  echo "[$(stamp)] launcher=firejail (approximate Linux sandbox; no GPU/network to host)" \
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
    -- "${PYTHON}" -m "${FACTORYMIND_MODULE}" 2>&1 | tee -a "${LOG_FILE}"
}

run_plain() {
  echo "[$(stamp)] launcher=plain (host Python; in-app policy still loads ${POLICY})" \
    | tee -a "${LOG_FILE}"
  echo "[$(stamp)] policy_summary=$(grep -E '^(name|description):' "${POLICY}" | tr '\n' ' ')" \
    | tee -a "${LOG_FILE}"
  exec "${PYTHON}" -m "${FACTORYMIND_MODULE}" 2>&1 | tee -a "${LOG_FILE}"
}

if [[ "${FACTORYMIND_USE_OPEN_SHELL_EXEC:-0}" == "1" ]]; then
  if ! command -v openshell >/dev/null 2>&1; then
    echo "FACTORYMIND_USE_OPEN_SHELL_EXEC=1 but 'openshell' not on PATH." >&2
    exit 1
  fi
  run_openshell_sandbox_exec
elif command -v firejail >/dev/null 2>&1; then
  if command -v nemoclaw >/dev/null 2>&1; then
    echo "[$(stamp)] NOTE: 'nemoclaw run' does not exist in official NemoClaw — using firejail fallback." \
      | tee -a "${LOG_FILE}"
    echo "       For real OpenShell policy, onboard with 'nemoclaw onboard', sync this repo into the sandbox," \
      | tee -a "${LOG_FILE}"
    echo "       then run: FACTORYMIND_USE_OPEN_SHELL_EXEC=1 ./scripts/run_with_nemoclaw.sh" \
      | tee -a "${LOG_FILE}"
  fi
  run_with_firejail
else
  if command -v nemoclaw >/dev/null 2>&1; then
    echo "[$(stamp)] NOTE: Official NemoClaw has no 'nemoclaw run'. Running on host with in-app policy." \
      | tee -a "${LOG_FILE}"
    echo "       Docs: https://docs.nvidia.com/nemoclaw/latest/reference/cli-selection-guide.html" \
      | tee -a "${LOG_FILE}"
    echo "       Optional: FACTORYMIND_USE_OPEN_SHELL_EXEC=1 + openshell sandbox exec (after uploading repo)." \
      | tee -a "${LOG_FILE}"
  else
    echo "WARNING: neither 'openshell' (optional exec mode), nor 'firejail' found on PATH." \
      | tee -a "${LOG_FILE}"
    echo "         Running unsandboxed on host; FactoryMind still logs policy to logs/nemoclaw.log." \
      | tee -a "${LOG_FILE}"
  fi
  run_plain
fi
