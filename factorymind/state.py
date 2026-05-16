"""Shared world_state schema, constants, and factory layout definitions.

This is the SINGLE source of truth for the world_state dict structure.
Every other module must conform to the schema returned by create_initial_state().
"""

from __future__ import annotations
import random

# ---------------------------------------------------------------------------
# Message types posted to the blackboard
# ---------------------------------------------------------------------------
CLAIM = "CLAIM"
INTENT = "INTENT"
BOTTLENECK = "BOTTLENECK"
STRATEGY = "STRATEGY"
COMPLETE = "COMPLETE"

# ---------------------------------------------------------------------------
# Robot roles
# ---------------------------------------------------------------------------
LEADER = "LEADER"
WORKER = "WORKER"

# ---------------------------------------------------------------------------
# Layout identifiers
# ---------------------------------------------------------------------------
OPEN_FLOOR = "OPEN_FLOOR"
BOTTLENECK_BRIDGE = "BOTTLENECK_BRIDGE"

# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------
GRID_WIDTH = 50
GRID_HEIGHT = 50

# ---------------------------------------------------------------------------
# Workstation definitions per layout
# ---------------------------------------------------------------------------
_WORKSTATIONS_BASE = [
    {"name": "Parts",    "pos": [4,  4],  "color": [220, 50,  50]},
    {"name": "Assembly", "pos": [45, 4],  "color": [50,  80,  220]},
    {"name": "QA",       "pos": [45, 45], "color": [50,  180, 80]},
    {"name": "Shipping", "pos": [4,  45], "color": [220, 200, 50]},
]

# 8 spawn slots arranged in a 4×2 block at grid centre (22–25, 23–24)
_SPAWN_POSITIONS: list[list[int]] = [
    [22, 23], [23, 23], [24, 23], [25, 23],
    [22, 24], [23, 24], [24, 24], [25, 24],
]


def _make_workstations(layout: str) -> list[dict]:
    """Return a fresh copy of workstations for the given layout."""
    import copy
    stations = copy.deepcopy(_WORKSTATIONS_BASE)
    return stations


def _make_wall(layout: str) -> list[list[int]]:
    """Return a list of [x, y] grid cells that are blocked.

    For BOTTLENECK_BRIDGE the wall runs across y=25 with a gap from x=22 to x=28.
    For OPEN_FLOOR the wall is empty.
    """
    if layout != BOTTLENECK_BRIDGE:
        return []
    wall = []
    for x in range(GRID_WIDTH):
        if x < 22 or x > 28:
            wall.append([x, 25])
    return wall


def create_initial_state(
    layout: str,
    num_leaders: int = 2,
    num_workers: int = 3,
) -> dict:
    """Create and return a fresh world_state dict for the given layout.

    Schema
    ------
    {
        "robots":      [{"id": int, "role": str, "pos": [x, y],
                         "path": [], "current_task": None}, ...],
        "tasks":       [],
        "workstations":[{"name": str, "pos": [x, y], "color": [r, g, b]}, ...],
        "blackboard":  [],
        "layout":      str,
        "wall":        [[x, y], ...],
        "stats":       {"completed": 0, "elapsed": 0.0, "rate": 0.0},
        "connection_status": "online",
        "spawn_positions": [list(sp) for sp in _SPAWN_POSITIONS[:num_leaders + num_workers]],
        "tick":        0,
    }
    """
    robots: list[dict] = []
    robot_id = 0

    for _ in range(num_leaders):
        sp = _SPAWN_POSITIONS[robot_id % len(_SPAWN_POSITIONS)]
        robots.append({
            "id": robot_id,
            "role": LEADER,
            "pos": list(sp),
            "spawn_pos": list(sp),
            "path": [],
            "current_task": None,
        })
        robot_id += 1

    for _ in range(num_workers):
        sp = _SPAWN_POSITIONS[robot_id % len(_SPAWN_POSITIONS)]
        robots.append({
            "id": robot_id,
            "role": WORKER,
            "pos": list(sp),
            "spawn_pos": list(sp),
            "path": [],
            "current_task": None,
        })
        robot_id += 1

    return {
        "robots": robots,
        "tasks": [],
        "workstations": _make_workstations(layout),
        "blackboard": [],
        "layout": layout,
        "wall": _make_wall(layout),
        "stats": {"completed": 0, "elapsed": 0.0, "rate": 0.0},
        "connection_status": "online",
        "tick": 0,
    }


if __name__ == "__main__":
    state = create_initial_state(OPEN_FLOOR)
    print("OPEN_FLOOR state keys:", list(state.keys()))
    print("Robots:", len(state["robots"]))
    state2 = create_initial_state(BOTTLENECK_BRIDGE)
    print("BOTTLENECK wall cells:", len(state2["wall"]))
    print("state.py OK")
