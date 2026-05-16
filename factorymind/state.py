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
    {"name": "Parts",    "pos": [5,  5],  "color": [220, 50,  50]},
    {"name": "Assembly", "pos": [30, 5],  "color": [50,  80,  220]},
    {"name": "QA",       "pos": [30, 35], "color": [50,  180, 80]},
    {"name": "Shipping", "pos": [5,  35], "color": [220, 200, 50]},
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
    num_workers: int = 6,
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
        "tick":        0,
    }
    """
    robots: list[dict] = []
    robot_id = 0

    for _ in range(num_leaders):
        robots.append({
            "id": robot_id,
            "role": LEADER,
            "pos": [random.randint(10, 40), random.randint(10, 40)],
            "path": [],
            "current_task": None,
        })
        robot_id += 1

    for _ in range(num_workers):
        robots.append({
            "id": robot_id,
            "role": WORKER,
            "pos": [random.randint(10, 40), random.randint(10, 40)],
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
