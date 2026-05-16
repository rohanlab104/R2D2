"""Shared world_state schema, constants, and factory layout definitions.

This is the SINGLE source of truth for the world_state dict structure.
Every other module must conform to the schema returned by create_initial_state().
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Message types posted to the blackboard
# ---------------------------------------------------------------------------
CLAIM = "CLAIM"
INTENT = "INTENT"
BOTTLENECK = "BOTTLENECK"
STRATEGY = "STRATEGY"
COMPLETE = "COMPLETE"
# NemoClaw / OpenClaw runtime policy events (allow/deny + agent decisions).
POLICY = "POLICY"

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

STATION_QUOTA_NAMES: tuple[str, ...] = ("Parts", "Assembly", "QA", "Shipping")


def empty_station_quotas() -> dict[str, dict[str, int]]:
    """Per-station targets from chat ('5 tasks for Parts' counts toward Parts)."""
    return {name: {"target": 0, "completed": 0} for name in STATION_QUOTA_NAMES}

# Legacy centre cluster (fallback only).
_SPAWN_POSITIONS: list[list[int]] = [
    [22, 23], [23, 23], [24, 23], [25, 23],
    [22, 24], [23, 24], [24, 24], [25, 24],
]

# Workers start spread across the floor (one robot per cell).
_DISPERSED_WORKER_SPAWNS: list[list[int]] = [
    [11, 11], [17, 9], [9, 17], [14, 14],
    [37, 11], [41, 16], [34, 9], [39, 14],
    [37, 39], [41, 35], [34, 42], [39, 37],
    [11, 39], [16, 41], [9, 35], [14, 37],
    [24, 12], [24, 36], [12, 24], [38, 24],
]

# Leaders sit off the warehouse floor visually, acting like dispatch/control agents.
_LEADER_POSITIONS: list[list[int]] = [
    [2, 24], [2, 26],
]


def _blocked_cells(layout: str) -> set[tuple[int, int]]:
    """Grid cells robots must not spawn on (walls + workstation footprints)."""
    blocked: set[tuple[int, int]] = {tuple(c) for c in _make_wall(layout)}
    for ws in _WORKSTATIONS_BASE:
        wx, wy = ws["pos"]
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = wx + dx, wy + dy
                if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                    blocked.add((x, y))
    return blocked


def _worker_spawn_positions(count: int, layout: str) -> list[list[int]]:
    """Return ``count`` unique spawn cells spread around the factory floor."""
    if count <= 0:
        return []
    blocked = _blocked_cells(layout)
    spawns: list[list[int]] = []
    for cell in _DISPERSED_WORKER_SPAWNS:
        if tuple(cell) in blocked:
            continue
        spawns.append(list(cell))
        if len(spawns) >= count:
            return spawns
    # Fill any remainder on a coarse grid scan (still dispersed).
    for y in range(8, GRID_HEIGHT - 8, 4):
        for x in range(8, GRID_WIDTH - 8, 4):
            if (x, y) in blocked or [x, y] in spawns:
                continue
            spawns.append([x, y])
            if len(spawns) >= count:
                return spawns
    # Last resort: legacy centre slots.
    while len(spawns) < count:
        spawns.append(list(_SPAWN_POSITIONS[len(spawns) % len(_SPAWN_POSITIONS)]))
    return spawns[:count]


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
    num_leaders: int | None = None,
    num_workers: int | None = None,
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
    import os

    if num_leaders is None:
        num_leaders = int(os.getenv("NUM_LEADERS", "1"))
    if num_workers is None:
        num_workers = int(os.getenv("NUM_WORKERS", "12"))

    robots: list[dict] = []
    robot_id = 0
    worker_spawns = _worker_spawn_positions(num_workers, layout)

    for _ in range(num_leaders):
        sp = _LEADER_POSITIONS[robot_id % len(_LEADER_POSITIONS)]
        robots.append({
            "id": robot_id,
            "role": LEADER,
            "pos": list(sp),
            "spawn_pos": list(sp),
            "path": [],
            "current_task": None,
        })
        robot_id += 1

    for i in range(num_workers):
        sp = worker_spawns[i % len(worker_spawns)]
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
        # NEW: pending tasks the leader has not yet delegated. Populated by
        # natural-language "do 100 deliveries" requests; drained by the
        # fleet dispatcher in main.py / web_main.py one-by-one as workers
        # become idle.
        "task_queue": [],
        "workstations": _make_workstations(layout),
        "blackboard": [],
        "layout": layout,
        "wall": _make_wall(layout),
        "stats": {
            "completed": 0,
            "elapsed": 0.0,
            "rate": 0.0,
            # Total work the user requested in this run (queued + done).
            "queued_total": 0,
            # NemoClaw runtime accounting.
            "policy_allowed": 0,
            "policy_denied": 0,
        },
        "connection_status": "online",
        # When True the leader is the OpenClaw-style autonomous fleet manager.
        "fleet_mode": False,
        "fleet_dispatch_cursor": 0,
        "spawn_positions": [list(r["spawn_pos"]) for r in robots],
        "station_quotas": empty_station_quotas(),
        "tick": 0,
    }


if __name__ == "__main__":
    state = create_initial_state(OPEN_FLOOR)
    print("OPEN_FLOOR state keys:", list(state.keys()))
    print("Robots:", len(state["robots"]))
    state2 = create_initial_state(BOTTLENECK_BRIDGE)
    print("BOTTLENECK wall cells:", len(state2["wall"]))
    print("state.py OK")
