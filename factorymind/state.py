"""Shared FactoryMind world-state schema for the autonomous factory demo."""

from __future__ import annotations

import copy
import math

# Message / log event types
CLAIM = "CLAIM"
INTENT = "INTENT"
BOTTLENECK = "BOTTLENECK"
STRATEGY = "STRATEGY"
COMPLETE = "COMPLETE"

# New leader thought-process event types
OBSERVE = "OBSERVE"
ASSIGN = "ASSIGN"
RETHINK = "RETHINK"
WARNING = "WARNING"

# Robot roles
LEADER = "LEADER"
WORKER = "WORKER"

# Layout identifiers retained for compatibility with older modules/tests.
OPEN_FLOOR = "OPEN_FLOOR"
BOTTLENECK_BRIDGE = "BOTTLENECK_BRIDGE"

GRID_WIDTH = 50
GRID_HEIGHT = 50

PACKAGE_COLORS: list[dict] = [
    {"name": "Red", "rgb": [232, 72, 72]},
    {"name": "Blue", "rgb": [68, 130, 235]},
    {"name": "Green", "rgb": [72, 210, 118]},
    {"name": "Yellow", "rgb": [235, 205, 70]},
    {"name": "Magenta", "rgb": [214, 78, 235]},
    {"name": "Cyan", "rgb": [0, 212, 255]},
]

_INITIAL_FACTORIES = [
    {"name": "Factory-A", "pos": [5, 5], "color": "Red"},
    {"name": "Factory-B", "pos": [5, 21], "color": "Blue"},
    {"name": "Factory-C", "pos": [5, 37], "color": "Green"},
]

_INITIAL_DROPBOXES = [
    {"name": "DropBox-Red", "pos": [40, 8], "color": "Red"},
    {"name": "DropBox-Blue", "pos": [42, 24], "color": "Blue"},
    {"name": "DropBox-Green", "pos": [40, 40], "color": "Green"},
]

_INITIAL_WORKERS = [[22, 43], [25, 43], [28, 43]]
_LEADER_POS = [25, 47]


def _initial_shelf_walls() -> list[list[int]]:
    """Three shelf racks, each rotated 45 deg in-plane around its own center.

    Each rack is a 11x3 rectangle (half_w=5.5, half_h=1.5). We rasterize the
    rotated shape by inverse-rotating every nearby grid cell and keeping the
    ones that fall inside the original axis-aligned rectangle.
    """
    cells: set[tuple[int, int]] = set()
    # (center_x, center_y, half_w, half_h) for each rack.
    racks = [
        (18.0, 17.0, 5.5, 1.5),
        (31.0, 25.0, 5.5, 1.5),
        (23.0, 34.0, 5.5, 1.5),
    ]
    # Inverse rotation by -45 deg.
    angle = math.radians(-45.0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for cx, cy, hw, hh in racks:
        # The rotated rectangle fits inside a (sqrt(2)*half) bounding box.
        bound = int(math.ceil(max(hw, hh) * math.sqrt(2.0))) + 1
        for x in range(int(cx) - bound, int(cx) + bound + 1):
            for y in range(int(cy) - bound, int(cy) + bound + 1):
                if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
                    continue
                dx = x - cx
                dy = y - cy
                # Apply the inverse rotation to test against the original AABB.
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                if abs(rx) <= hw + 0.4 and abs(ry) <= hh + 0.4:
                    cells.add((x, y))
    return [[x, y] for x, y in sorted(cells)]


def color_rgb(color_name: str) -> list[int]:
    """Return RGB for a configured package color."""
    for color in PACKAGE_COLORS:
        if color["name"] == color_name:
            return list(color["rgb"])
    return [180, 180, 180]


def next_color_name(index: int) -> str:
    return PACKAGE_COLORS[index % len(PACKAGE_COLORS)]["name"]


def make_factory(
    name: str,
    pos: list[int],
    color: str,
    factory_id: int,
    rotation_y: float = 0.0,
) -> dict:
    """Create a factory with a right-facing conveyor and terminal drop pad."""
    x, y = pos
    belt_dir = [1, 0]
    belt_length = 5
    belt_start = [x + 3, y + 1]
    drop_pad = [belt_start[0] + belt_dir[0] * belt_length, belt_start[1]]
    return {
        "id": factory_id,
        "name": name,
        "pos": list(pos),
        "size": [3, 3],
        "color": color,
        "color_rgb": color_rgb(color),
        "rotation_y": rotation_y,
        "belt_dir": belt_dir,
        "belt_length": belt_length,
        "belt_start": belt_start,
        "drop_pad": drop_pad,
        "produce_every": 6.5,
        "produce_timer": 1.4 + (factory_id * 1.35),
        "conveyor": [],
        "pad_packages": [],
    }


def make_dropbox(
    name: str,
    pos: list[int],
    color: str,
    dropbox_id: int,
    rotation_y: float = 0.0,
) -> dict:
    return {
        "id": dropbox_id,
        "name": name,
        "pos": list(pos),
        "color": color,
        "color_rgb": color_rgb(color),
        "rotation_y": rotation_y,
        "delivered": 0,
    }


def make_worker(worker_id: int, pos: list[int]) -> dict:
    return {
        "id": worker_id,
        "name": f"R{worker_id}D{worker_id}",
        "role": WORKER,
        "pos": list(pos),
        "spawn_pos": list(pos),
        "path": [],
        "current_task": None,
        "status": "idle",
        "carrying": None,
        "target_factory_id": None,
        "target_dropbox_id": None,
        "task_started_at": None,
        "last_llm_intent": None,
        "last_llm_reason": None,
        "llm_decisions": 0,
    }


def _initial_factories() -> list[dict]:
    return [
        make_factory(item["name"], item["pos"], item["color"], index)
        for index, item in enumerate(_INITIAL_FACTORIES)
    ]


def _initial_dropboxes() -> list[dict]:
    return [
        make_dropbox(item["name"], item["pos"], item["color"], index)
        for index, item in enumerate(_INITIAL_DROPBOXES)
    ]


def create_initial_state(
    layout: str = OPEN_FLOOR,
    num_leaders: int = 1,
    num_workers: int = 3,
) -> dict:
    """Create a fresh autonomous-factory world state.

    The single leader is stationary in a bottom-center command tower. Workers
    move packages from factory conveyor pads to same-color drop boxes.
    """
    factories = _initial_factories()
    dropboxes = _initial_dropboxes()
    workers = [
        make_worker(index + 1, _INITIAL_WORKERS[index % len(_INITIAL_WORKERS)])
        for index in range(num_workers)
    ]
    leader = {
        "id": 0,
        "name": "Leader",
        "role": LEADER,
        "pos": list(_LEADER_POS),
        "spawn_pos": list(_LEADER_POS),
        "path": [],
        "current_task": None,
        "status": "observing",
        "thinking": False,
    }
    return {
        "layout": layout,
        "wall": _initial_shelf_walls(),
        "leader": leader,
        "robots": [leader] + workers,
        "workers": workers,
        "factories": factories,
        "dropboxes": dropboxes,
        # Compatibility alias for older helper code; render/main now use factories.
        "workstations": copy.deepcopy(factories),
        "packages": [],
        "tasks": [],
        "blackboard": [],
        "thought_log": [],
        "policy_log": [],
        "policy_loaded": False,
        "policy_path": "",
        "policy_status": "not loaded",
        "connection_status": "online",
        "running": False,
        "speed_multiplier": 1,
        "tick": 0,
        "next_package_id": 0,
        "next_task_id": 0,
        "next_factory_id": len(factories),
        "next_dropbox_id": len(dropboxes),
        "next_worker_id": len(workers) + 1,
        "builder_mode": "wall",
        "leader_thinking": False,
        "inference_target": "",
        "stats": {
            "completed": 0,
            "elapsed": 0.0,
            "rate": 0.0,
            "last_route_time": None,
            "avg_route_time": None,
        },
    }


if __name__ == "__main__":
    state = create_initial_state()
    print("FactoryMind state keys:", list(state.keys()))
    print("Factories:", len(state["factories"]))
    print("Workers:", len(state["workers"]))
