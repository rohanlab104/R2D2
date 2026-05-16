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
# Color-keyed factory layout.
# Pickup zones (left half) hold colored blocks; matching drop zones (right
# half) accept them. A worker carrying color X may only deliver at the same
# color's drop zone (enforced by NemoClaw policy in nemoclaw.py).
_WORKSTATIONS_BASE = [
    # Pickup stockpiles — left column
    {"name": "Red-Pickup",    "pos": [6,  9],  "color": [232, 64,  64],  "kind": "pickup"},
    {"name": "Blue-Pickup",   "pos": [6,  20], "color": [60,  120, 240], "kind": "pickup"},
    {"name": "Green-Pickup",  "pos": [6,  31], "color": [60,  200, 110], "kind": "pickup"},
    {"name": "Yellow-Pickup", "pos": [6,  42], "color": [240, 200, 60],  "kind": "pickup"},
    # Drop boxes — right column, color-matched to pickups
    {"name": "Red-Drop",      "pos": [44, 9],  "color": [232, 64,  64],  "kind": "delivery"},
    {"name": "Blue-Drop",     "pos": [44, 20], "color": [60,  120, 240], "kind": "delivery"},
    {"name": "Green-Drop",    "pos": [44, 31], "color": [60,  200, 110], "kind": "delivery"},
    {"name": "Yellow-Drop",   "pos": [44, 42], "color": [240, 200, 60],  "kind": "delivery"},
]

# Color names rotated through when generating tasks (each task is pickup of
# color X -> drop of color X).
COLOR_NAMES: tuple[str, ...] = ("Red", "Blue", "Green", "Yellow")

# Quota tracking is keyed on the color, not the legacy station names.
STATION_QUOTA_NAMES: tuple[str, ...] = COLOR_NAMES


def empty_station_quotas() -> dict[str, dict[str, int]]:
    """Per-station targets from chat ('5 tasks for Parts' counts toward Parts)."""
    return {name: {"target": 0, "completed": 0} for name in STATION_QUOTA_NAMES}

# Legacy centre cluster (fallback only).
_SPAWN_POSITIONS: list[list[int]] = [
    [22, 23], [23, 23], [24, 23], [25, 23],
    [22, 24], [23, 24], [24, 24], [25, 24],
]

# Workers start spread across the mid-floor (between pickup and drop columns).
_DISPERSED_WORKER_SPAWNS: list[list[int]] = [
    [16, 10], [22, 8],  [28, 10], [34, 8],
    [16, 18], [22, 16], [28, 18], [34, 16],
    [16, 26], [22, 24], [28, 26], [34, 24],
    [16, 34], [22, 32], [28, 34], [34, 32],
    [16, 42], [22, 40], [28, 42], [34, 40],
]

# Leader sits in a tower at bottom-center, observing the whole floor.
_LEADER_POSITIONS: list[list[int]] = [
    [25, 47], [25, 48],
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
