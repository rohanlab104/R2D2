"""Nemotron inference abstraction.

Wraps the OpenAI-compatible NVIDIA API (or a local NIM endpoint) so the
rest of the codebase never touches HTTP directly.

Environment variables
---------------------
NVIDIA_API_KEY   : Required when USE_LOCAL_NIM is false.
USE_LOCAL_NIM    : Set to "true" to hit http://localhost:8000/v1 instead of
                   the NVIDIA cloud endpoint.
"""

from __future__ import annotations

import json
import os
import re

from openai import OpenAI

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
LEADER_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"
STRATEGIST_MODEL = "nvidia/llama-3_3-nemotron-super-49b-v1_5"

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------
USE_LOCAL = os.getenv("USE_LOCAL_NIM", "false").lower() == "true"

_BASE_URL = (
    "http://localhost:8000/v1"
    if USE_LOCAL
    else "https://integrate.api.nvidia.com/v1"
)

_API_KEY = os.getenv("NVIDIA_API_KEY", "no-key-needed-for-local")

_client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY)


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def ask_nemotron(prompt: str, model: str, max_tokens: int = 512) -> str:
    """Send a single-turn prompt to the specified Nemotron model.

    Returns the raw text content of the first choice.
    Raises on HTTP errors so callers can decide whether to fall back.
    """
    response = _client.chat.completions.create(
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
    """Query the fast leader model (Nemotron Nano 9B) with the given prompt."""
    return ask_nemotron(prompt, model=LEADER_MODEL, max_tokens=256)


def ask_strategist(prompt: str) -> str:
    """Query the large strategist model (Nemotron Super 49B) with the given prompt."""
    return ask_nemotron(prompt, model=STRATEGIST_MODEL, max_tokens=512)


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def safe_parse_json(text: str) -> dict | list | None:
    """Strip markdown code fences and attempt to parse the remaining JSON.

    Returns the parsed object, or None if parsing fails.
    """
    # Strip triple-backtick fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Extract the first {...} or [...] block
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

if __name__ == "__main__":
    print(f"Endpoint: {_BASE_URL}")
    print(f"Local NIM: {USE_LOCAL}")
    result = ask_leader("Say hello in 5 words.")
    print("Leader response:", result)
