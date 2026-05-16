"""Nemotron inference abstraction.

Wraps the OpenAI-compatible NVIDIA API (or local NIM endpoints on the GX10)
so the rest of the codebase never touches HTTP directly.

Where does inference actually run?
----------------------------------
This module is just an HTTP client. The GPU work happens wherever the
configured URL points:

  - USE_LOCAL_NIM=false                 -> NVIDIA cloud (build.nvidia.com)
  - USE_LOCAL_NIM=true,  GX10_IP=...    -> NIM containers on the DGX Spark GX10

When you run on the GX10 itself, set GX10_IP=localhost. Traffic never leaves
the box; the Blackwell GPU handles everything.

Environment variables
---------------------
NVIDIA_API_KEY          : Required for cloud mode (integrate.api.nvidia.com).
USE_LOCAL_NIM           : "true" to use local NIM on GX10, "false" for cloud.
GX10_IP                 : "localhost" when running on GX10 (default), or the
                          GX10 IP / SSH-tunneled host from a remote machine.
NIM_LEADER_BASE_URL     : Override leader endpoint (default http://{GX10_IP}:8000/v1).
NIM_STRATEGIST_BASE_URL : Override strategist endpoint (default http://{GX10_IP}:8001/v1).
LOCAL_NIM_BASE_URL      : Shared local endpoint override for both roles.
LEADER_MODEL            : Override leader model id.
STRATEGIST_MODEL        : Override strategist model id.
NGC_API_KEY             : For docker login / NIM image pulls (often == NVIDIA_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
LEADER_MODEL = os.getenv("LEADER_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
STRATEGIST_MODEL = os.getenv(
    "STRATEGIST_MODEL",
    "nvidia/llama-3_3-nemotron-super-49b-v1_5",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
USE_LOCAL = os.getenv("USE_LOCAL_NIM", "false").lower() == "true"
GX10_IP = os.getenv("GX10_IP", "localhost").strip() or "localhost"
_CLOUD_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
_SHARED_LOCAL_URL = (
    os.getenv("LOCAL_NIM_BASE_URL")
    or os.getenv("NIM_BASE_URL")
    or ""
).rstrip("/")
_API_KEY = os.getenv("NVIDIA_API_KEY") or "no-key-needed-for-local"
_TIMEOUT_SECONDS = float(os.getenv("NEMOTRON_TIMEOUT_SECONDS", "3.0"))


def _leader_base_url() -> str:
    if not USE_LOCAL:
        return _CLOUD_URL
    return (
        os.getenv("NIM_LEADER_BASE_URL")
        or _SHARED_LOCAL_URL
        or f"http://{GX10_IP}:8000/v1"
    ).rstrip("/")


def _strategist_base_url() -> str:
    if not USE_LOCAL:
        return _CLOUD_URL
    return (
        os.getenv("NIM_STRATEGIST_BASE_URL")
        or _SHARED_LOCAL_URL
        or f"http://{GX10_IP}:8001/v1"
    ).rstrip("/")


def _make_client(base_url: str) -> OpenAI:
    # max_retries=0 so a dead endpoint surfaces in `_TIMEOUT_SECONDS` instead
    # of getting silently retried for ~30s while the worker thread holds the
    # pool slot. Callers (agents.py) fall back to mock decisions on failure.
    return OpenAI(
        base_url=base_url,
        api_key=_API_KEY,
        timeout=_TIMEOUT_SECONDS,
        max_retries=0,
    )


_leader_client = _make_client(_leader_base_url())
_strategist_client = _make_client(_strategist_base_url())


# ---------------------------------------------------------------------------
# Status / introspection
# ---------------------------------------------------------------------------

def describe_endpoints() -> str:
    """Human-readable summary of where inference will happen."""
    if USE_LOCAL:
        location = (
            "DGX Spark GX10 (this machine)" if GX10_IP == "localhost" else f"GX10 at {GX10_IP}"
        )
        return (
            f"Inference target: LOCAL NIM on {location}\n"
            f"  leader     -> {_leader_base_url()}  ({LEADER_MODEL})\n"
            f"  strategist -> {_strategist_base_url()}  ({STRATEGIST_MODEL})"
        )
    return (
        "Inference target: NVIDIA CLOUD (build.nvidia.com)\n"
        f"  leader     -> {_CLOUD_URL}  ({LEADER_MODEL})\n"
        f"  strategist -> {_CLOUD_URL}  ({STRATEGIST_MODEL})"
    )


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def ask_nemotron(
    prompt: str,
    model: str,
    max_tokens: int = 512,
    *,
    client: OpenAI | None = None,
) -> str:
    """Send a single-turn prompt to the specified Nemotron model.

    Returns the raw text content of the first choice. Raises on HTTP errors
    so callers can decide whether to fall back.
    """
    active = client or _leader_client
    response = active.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Role-specific wrappers
# ---------------------------------------------------------------------------

def ask_leader(prompt: str) -> str:
    """Query the fast leader model (Nemotron Nano 9B)."""
    return ask_nemotron(
        prompt,
        model=LEADER_MODEL,
        max_tokens=256,
        client=_leader_client,
    )


def ask_strategist(prompt: str) -> str:
    """Query the strategist model (Nemotron Super 49B)."""
    return ask_nemotron(
        prompt,
        model=STRATEGIST_MODEL,
        max_tokens=512,
        client=_strategist_client,
    )


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def safe_parse_json(text: str) -> dict | list | None:
    """Strip markdown code fences and parse the first JSON object/array.

    Returns the parsed object, or None if parsing fails.
    """
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _check_api_key() -> bool:
    """Validate that the configured mode has the credentials it needs."""
    if USE_LOCAL:
        return True
    key = os.getenv("NVIDIA_API_KEY", "")
    if not key or key == "your_key_here":
        print(
            "ERROR: cloud mode needs NVIDIA_API_KEY in .env (build.nvidia.com).\n"
            "       If you meant to use local NIM, set USE_LOCAL_NIM=true.",
            file=sys.stderr,
        )
        return False
    return True


def run_smoke(*, leader: bool = True, strategist: bool = True) -> int:
    """Verify leader and/or strategist endpoints. Returns exit code (0 = ok)."""
    print(describe_endpoints())
    print()

    if not _check_api_key():
        return 1

    errors = 0

    if leader:
        print(f"[leader] calling {_leader_base_url()} ...")
        try:
            result = ask_leader("Say hello in exactly five words.")
            print(f"  OK: {result.strip()[:200]}")
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            errors += 1

    if strategist:
        print(f"[strategist] calling {_strategist_base_url()} ...")
        try:
            result = ask_strategist(
                "Reply with one short sentence confirming you are online."
            )
            print(f"  OK: {result.strip()[:200]}")
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            errors += 1

    print()
    if errors:
        print(f"{errors} check(s) failed.", file=sys.stderr)
        return 1

    if leader and strategist:
        kind = "local NIM" if USE_LOCAL else "cloud inference"
        print(f"{kind} confirmed for both models")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Nemotron inference endpoints.")
    parser.add_argument("--leader-only", action="store_true")
    parser.add_argument("--strategist-only", action="store_true")
    args = parser.parse_args()

    run_leader = not args.strategist_only
    run_strategist = not args.leader_only
    raise SystemExit(run_smoke(leader=run_leader, strategist=run_strategist))
