"""Blackboard data structure and agent decision functions.

Person C owns this file.

The USE_MOCK flag (env var AGENTS_USE_MOCK=true) bypasses Nemotron calls and
returns hardcoded sensible responses so the rest of the team can run end-to-end
while Person A wires up inference.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Mock flag — set AGENTS_USE_MOCK=true to skip real LLM calls
# ---------------------------------------------------------------------------
USE_MOCK = os.getenv("AGENTS_USE_MOCK", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

LEADER_PROMPT_TEMPLATE = """\
You are a factory floor leader robot (id={robot_id}) managing workers.

Current world state summary:
- Open tasks: {open_tasks}
- Your position: {pos}
- Recent blackboard: {recent_messages}

Respond with valid JSON only (no markdown):
{{
  "claim_task_id": <int or null>,
  "destination": [x, y],
  "post_message": "<short message for blackboard>",
  "reasoning": "<one sentence>"
}}
"""

STRATEGIST_PROMPT_TEMPLATE = """\
You are a high-level factory strategist overseeing all robots.

World state:
- Layout: {layout}
- Stats: {stats}
- Open tasks: {open_tasks}
- Retrieved strategies: {retrieved_strategies}
- Recent blackboard: {recent_messages}

Respond with valid JSON only (no markdown):
{{
  "should_post": true,
  "directive": "<actionable one-sentence directive for all leaders>",
  "reasoning": "<two sentences explaining strategy>"
}}
"""


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------

class Blackboard:
    """Thread-safe-ish message board shared by all agents.

    Each message is a dict: {"from": int, "type": str, "content": str, "timestamp": float}
    """

    def __init__(self) -> None:
        self._messages: list[dict] = []

    def post(self, from_id: int, msg_type: str, content: str) -> None:
        """Append a new message to the board."""
        self._messages.append({
            "from": from_id,
            "type": msg_type,
            "content": content,
            "timestamp": time.time(),
        })

    def read_recent(self, n: int = 20) -> list[dict]:
        """Return the last n messages."""
        return self._messages[-n:]

    def read_by_type(self, msg_type: str) -> list[dict]:
        """Return all messages matching msg_type."""
        return [m for m in self._messages if m["type"] == msg_type]

    def clear(self) -> None:
        """Remove all messages from the board."""
        self._messages.clear()

    def to_list(self) -> list[dict]:
        """Return a shallow copy of all messages (used to update world_state)."""
        return list(self._messages)


# ---------------------------------------------------------------------------
# Agent decision functions
# ---------------------------------------------------------------------------

def leader_decide(robot: dict, world_state: dict, blackboard: Blackboard) -> dict:
    """Choose what a leader robot should do next.

    Returns a dict with keys:
        claim_task_id : int | None
        destination   : [x, y]
        post_message  : str
        reasoning     : str
    """
    if USE_MOCK:
        return _mock_leader_decide(robot, world_state)

    from factorymind import inference

    open_tasks = [t for t in world_state["tasks"] if t.get("status") == "open"]
    recent = blackboard.read_recent(5)
    prompt = LEADER_PROMPT_TEMPLATE.format(
        robot_id=robot["id"],
        open_tasks=open_tasks[:5],
        pos=robot["pos"],
        recent_messages=recent,
    )
    raw = inference.ask_leader(prompt)
    parsed = inference.safe_parse_json(raw)
    if parsed and isinstance(parsed, dict):
        return parsed
    # Fallback to mock if parse fails
    return _mock_leader_decide(robot, world_state)


def strategist_decide(
    world_state: dict,
    blackboard: Blackboard,
    retrieved_strategies: list[dict],
) -> dict:
    """High-level strategic directive for all leaders.

    Returns a dict with keys:
        should_post : bool
        directive   : str
        reasoning   : str
    """
    if USE_MOCK:
        return _mock_strategist_decide(world_state)

    from factorymind import inference

    open_tasks = [t for t in world_state["tasks"] if t.get("status") == "open"]
    recent = blackboard.read_recent(10)
    prompt = STRATEGIST_PROMPT_TEMPLATE.format(
        layout=world_state.get("layout"),
        stats=world_state.get("stats"),
        open_tasks=open_tasks[:10],
        retrieved_strategies=retrieved_strategies,
        recent_messages=recent,
    )
    raw = inference.ask_strategist(prompt)
    parsed = inference.safe_parse_json(raw)
    if parsed and isinstance(parsed, dict):
        return parsed
    return _mock_strategist_decide(world_state)


def worker_step(worker_robot: dict, world_state: dict, blackboard: Blackboard) -> list[int]:
    """Advance a worker one step toward its assigned destination (rule-based, no LLM).

    Returns the new [x, y] position.  The path is pre-computed by main.py's A*;
    this function just pops the next cell.
    """
    path = worker_robot.get("path", [])
    if path:
        return path[0]  # main.py will pop it after calling this
    return worker_robot["pos"]


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------

def _mock_leader_decide(robot: dict, world_state: dict) -> dict:
    """Return a sensible mock leader decision without calling any LLM."""
    open_tasks = [t for t in world_state["tasks"] if t.get("status") == "open"]
    claim_id: Optional[int] = None
    destination = robot["pos"]

    if open_tasks:
        task = random.choice(open_tasks)
        claim_id = task["id"]
        destination = task["pickup"]

    return {
        "claim_task_id": claim_id,
        "destination": destination,
        "post_message": f"Leader {robot['id']} claiming task {claim_id}",
        "reasoning": "Mock: picked random open task.",
    }


def _mock_strategist_decide(world_state: dict) -> dict:
    """Return a hardcoded reasonable strategist directive."""
    open_count = sum(1 for t in world_state["tasks"] if t.get("status") == "open")
    directive = (
        "Prioritise clearing the Parts queue — dispatch all idle workers to pickup."
        if open_count > 3
        else "Steady state: maintain round-robin task distribution across workstations."
    )
    return {
        "should_post": True,
        "directive": directive,
        "reasoning": (
            "Mock strategist: queue depth used as primary heuristic. "
            "Leaders should coordinate via CLAIM messages to avoid contention."
        ),
    }
