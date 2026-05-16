"""Blackboard data structure and agent decision functions.

Person C owns this file.

The USE_MOCK flag (env var AGENTS_USE_MOCK=true) bypasses Nemotron calls and
returns hardcoded sensible responses so the rest of the team can run end-to-end
while Person A wires up inference.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

from factorymind.state import BOTTLENECK, BOTTLENECK_BRIDGE, CLAIM, LEADER, STRATEGY

# ---------------------------------------------------------------------------
# Mock flag — set AGENTS_USE_MOCK=true to skip real LLM calls
# ---------------------------------------------------------------------------
def _use_mock() -> bool:
    """Read mock mode dynamically so REPL sessions can flip it after import."""
    return os.getenv("AGENTS_USE_MOCK", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

LEADER_PROMPT_TEMPLATE = """\
You are Leader R{robot_id}, a factory floor coordinator. Choose one useful next move.

Factory context:
- Layout: {layout}
- Workstations: {workstations}
- Your position: {pos}
- Open tasks: {open_tasks}
- Recent strategist directive: {recent_strategy}
- Recent blackboard: {recent_messages}

Rules:
- Claim at most one unassigned task. Do not claim a task already claimed or assigned.
- Prefer the task where your robot or nearby workers can reach the pickup fastest.
- In BOTTLENECK_BRIDGE, avoid sending unnecessary traffic through the bridge; obey recent BOTTLENECK or STRATEGY messages.
- If no good task exists, set claim_task_id to null and destination to your current position.
- Blackboard text must look intentional and specific, e.g. "Claiming Parts->QA task 7; pickup is 3 cells away."
- Do not invent task ids or workstation names.

Return valid JSON only, exactly this schema:
{{
  "claim_task_id": 7,
  "destination": [5, 5],
  "post_message": "Claiming Parts->QA task 7; pickup is 3 cells away.",
  "reasoning": "I am the nearest unassigned leader and the route does not conflict with recent claims."
}}

Think through conflicts in the reasoning field, then choose the safest coordinated action.
"""

STRATEGIST_PROMPT_TEMPLATE = """\
You are the factory strategist. Every 30 seconds, decide whether the team needs a visible coordination directive.

World state:
- Layout: {layout}
- Workstations: {workstations}
- Stats: {stats}
- Open tasks: {open_tasks}
- Robot positions: {robots}
- Retrieved strategies: {retrieved_strategies}
- Recent blackboard: {recent_messages}

Goals:
- Identify bottlenecks, idle capacity, duplicated claims, and route conflicts.
- Reference the current layout and workstation names.
- Use retrieved strategies only when they fit this layout.
- Be willing to say no change needed; do not post bloat.
- If BOTTLENECK_BRIDGE is congested, give a concrete routing directive for the next 30 seconds.

Return valid JSON only, exactly this schema:
{{
  "should_post": true,
  "directive": "Bottleneck at center bridge: route Assembly->QA traffic through the west approach for the next 30 seconds.",
  "reasoning": "Several active routes cross y=25 while open work remains on both sides. A temporary lane preference should reduce head-on congestion."
}}

Most important: post only when the directive changes behavior on screen; otherwise set should_post false.
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

    def read_recent_by_type(self, msg_type: str, seconds: float = 30.0) -> list[dict]:
        """Return recent messages of a type within the last ``seconds``."""
        cutoff = time.time() - seconds
        return [
            m for m in self._messages
            if m["type"] == msg_type and m.get("timestamp", 0.0) >= cutoff
        ]

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
    if _use_mock():
        return _mock_leader_decide(robot, world_state, blackboard)

    from factorymind import inference

    open_tasks = _candidate_tasks(world_state)
    recent = blackboard.read_recent(8)
    prompt = LEADER_PROMPT_TEMPLATE.format(
        robot_id=robot["id"],
        layout=world_state.get("layout"),
        workstations=_summarize_workstations(world_state),
        open_tasks=[_summarize_task(t, robot["pos"]) for t in open_tasks[:6]],
        pos=robot["pos"],
        recent_strategy=_latest_directive(blackboard),
        recent_messages=recent,
    )
    try:
        raw = inference.ask_leader(prompt)
        parsed = inference.safe_parse_json(raw)
    except Exception:
        parsed = None
    if parsed and isinstance(parsed, dict):
        return _normalize_leader_decision(parsed, robot, world_state)
    # Fallback to mock if parse fails
    return _mock_leader_decide(robot, world_state, blackboard)


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
    if _use_mock():
        return _mock_strategist_decide(world_state)

    from factorymind import inference

    open_tasks = _candidate_tasks(world_state, include_assigned=True)
    recent = blackboard.read_recent(12)
    prompt = STRATEGIST_PROMPT_TEMPLATE.format(
        layout=world_state.get("layout"),
        workstations=_summarize_workstations(world_state),
        stats=world_state.get("stats"),
        open_tasks=[_summarize_task(t) for t in open_tasks[:10]],
        robots=_summarize_robots(world_state),
        retrieved_strategies=retrieved_strategies,
        recent_messages=recent,
    )
    try:
        raw = inference.ask_strategist(prompt)
        parsed = inference.safe_parse_json(raw)
    except Exception:
        parsed = None
    if parsed and isinstance(parsed, dict):
        return _normalize_strategist_decision(parsed)
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


def choose_worker_task(worker_robot: dict, world_state: dict, blackboard: Blackboard) -> Optional[int]:
    """Choose an open task for an idle worker using blackboard-aware heuristics.

    Workers prefer the nearest unassigned pickup. If a leader has claimed a task
    but this worker is closer to pickup, the worker may assist by taking it over.
    Recent bottleneck messages penalize cross-bridge traffic.
    """
    if worker_robot.get("current_task") is not None or worker_robot.get("path"):
        return None

    tasks = _candidate_tasks(world_state, include_assigned=True)
    if not tasks:
        return None

    leaders = {r["id"]: r for r in world_state["robots"] if r.get("role") == LEADER}
    claimed_ids = _claimed_task_ids(blackboard)
    bottleneck_recent = bool(blackboard.read_recent_by_type(BOTTLENECK, seconds=35.0))

    best_task: Optional[dict] = None
    best_score = 10**9
    for task in tasks:
        assigned_to = task.get("assigned_to")
        worker_dist = _manhattan(worker_robot["pos"], task["pickup"])

        if assigned_to is not None:
            leader = leaders.get(assigned_to)
            if not leader:
                continue
            if worker_dist >= _manhattan(leader["pos"], task["pickup"]):
                continue

        score = worker_dist
        if task["id"] in claimed_ids:
            score -= 2
        if bottleneck_recent and _crosses_bridge(task):
            score += 12

        if score < best_score:
            best_task = task
            best_score = score

    return best_task["id"] if best_task else None


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------

def _mock_leader_decide(robot: dict, world_state: dict, blackboard: Optional[Blackboard] = None) -> dict:
    """Return a sensible mock leader decision without calling any LLM."""
    open_tasks = _candidate_tasks(world_state)
    claim_id: Optional[int] = None
    destination = robot["pos"]
    task: Optional[dict] = None

    if open_tasks:
        task = min(
            open_tasks,
            key=lambda t: (_manhattan(robot["pos"], t["pickup"]), t["id"]),
        )
        claim_id = task["id"]
        destination = task["pickup"]

    distance = _manhattan(robot["pos"], destination)
    message = (
        _format_claim_message(task, distance)
        if task
        else f"Leader {robot['id']} holding position; no unassigned task is ready."
    )
    return {
        "claim_task_id": claim_id,
        "destination": destination,
        "post_message": message,
        "reasoning": "Mock: chose the nearest unassigned pickup while avoiding duplicate claims.",
    }


def _mock_strategist_decide(world_state: dict) -> dict:
    """Return a hardcoded reasonable strategist directive."""
    open_count = sum(1 for t in world_state["tasks"] if t.get("status") == "open")
    layout = world_state.get("layout")
    if layout == BOTTLENECK_BRIDGE:
        directive = (
            "Bottleneck at center bridge: keep cross-wall trips sparse and let nearest workers assist claimed pickups."
        )
    elif open_count > 3:
        directive = (
            "Open floor surge: nearest idle workers should clear pickup queues before leaders take long routes."
        )
    else:
        directive = "No change needed: task flow is light, so keep nearest-pickup dispatch."
    return {
        "should_post": open_count > 0,
        "directive": directive,
        "reasoning": (
            "Mock strategist: layout and queue depth are the main heuristics. "
            "The directive is concise so the blackboard stays readable."
        ),
    }


# ---------------------------------------------------------------------------
# Shared heuristics and formatting
# ---------------------------------------------------------------------------

def _candidate_tasks(world_state: dict, include_assigned: bool = False) -> list[dict]:
    """Return open tasks, optionally including leader-assigned tasks for assist logic."""
    tasks = []
    for task in world_state.get("tasks", []):
        if task.get("status") != "open":
            continue
        if include_assigned or task.get("assigned_to") is None:
            tasks.append(task)
    return tasks


def _manhattan(a: list[int], b: list[int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _summarize_task(task: dict, from_pos: Optional[list[int]] = None) -> dict:
    summary = {
        "id": task.get("id"),
        "route": f"{task.get('pickup_name', '?')}->{task.get('delivery_name', '?')}",
        "pickup": task.get("pickup"),
        "delivery": task.get("delivery"),
        "assigned_to": task.get("assigned_to"),
    }
    if from_pos is not None and task.get("pickup"):
        summary["pickup_distance"] = _manhattan(from_pos, task["pickup"])
    return summary


def _summarize_workstations(world_state: dict) -> list[dict]:
    return [
        {"name": ws.get("name"), "pos": ws.get("pos")}
        for ws in world_state.get("workstations", [])
    ]


def _summarize_robots(world_state: dict) -> list[dict]:
    return [
        {
            "id": r.get("id"),
            "role": r.get("role"),
            "pos": r.get("pos"),
            "task": r.get("current_task"),
        }
        for r in world_state.get("robots", [])
    ]


def _latest_directive(blackboard: Blackboard) -> str:
    for msg in reversed(blackboard.read_recent(20)):
        if msg.get("type") in {STRATEGY, BOTTLENECK}:
            return str(msg.get("content", ""))
    return "None."


def _claimed_task_ids(blackboard: Blackboard) -> set[int]:
    ids: set[int] = set()
    for msg in blackboard.read_recent_by_type(CLAIM, seconds=30.0):
        content = str(msg.get("content", ""))
        for match in re.findall(r"(?:task\s*#?|task-)(\d+)", content, flags=re.IGNORECASE):
            ids.add(int(match))
    return ids


def _format_claim_message(task: dict, distance: int) -> str:
    route = f"{task.get('pickup_name', '?')}->{task.get('delivery_name', '?')}"
    return f"Claiming {route} task {task['id']}; pickup is {distance} cells away."


def _crosses_bridge(task: dict) -> bool:
    pickup = task.get("pickup") or [0, 0]
    delivery = task.get("delivery") or [0, 0]
    return (pickup[1] < 25 < delivery[1]) or (delivery[1] < 25 < pickup[1])


def _normalize_leader_decision(parsed: dict, robot: dict, world_state: dict) -> dict:
    """Validate model output and keep it inside the simulation contract."""
    open_ids = {t["id"] for t in _candidate_tasks(world_state)}
    claim_id = parsed.get("claim_task_id")
    if claim_id not in open_ids:
        claim_id = None

    task = next((t for t in world_state.get("tasks", []) if t.get("id") == claim_id), None)
    destination = parsed.get("destination")
    if task:
        destination = task["pickup"]
    elif not (
        isinstance(destination, list)
        and len(destination) == 2
        and all(isinstance(v, int) for v in destination)
    ):
        destination = robot["pos"]

    message = str(parsed.get("post_message", "")).strip()
    if task and ("task" not in message.lower() or str(task["id"]) not in message):
        message = _format_claim_message(task, _manhattan(robot["pos"], task["pickup"]))
    elif not message:
        message = f"Leader {robot['id']} holding position; no safe claim right now."

    return {
        "claim_task_id": claim_id,
        "destination": destination,
        "post_message": message[:180],
        "reasoning": str(parsed.get("reasoning", "Validated model decision."))[:220],
    }


def _normalize_strategist_decision(parsed: dict) -> dict:
    directive = str(parsed.get("directive", "")).strip()
    should_post = bool(parsed.get("should_post")) and bool(directive)
    return {
        "should_post": should_post,
        "directive": directive[:220],
        "reasoning": str(parsed.get("reasoning", ""))[:260],
    }
