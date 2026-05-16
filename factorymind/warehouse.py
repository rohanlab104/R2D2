"""Autonomous warehouse simulation backend.

This module is intentionally graphics-free. It owns the digital-twin state,
conveyor spawning, task lifecycle, leader assignment, worker behavior, metrics,
and demo controls. A future pygame/Three.js layer can poll this state directly.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

from factorymind.pathfinding import a_star, route_cost
from factorymind.state import GRID_HEIGHT, GRID_WIDTH, OPEN_FLOOR

ITEM_TYPES = ("electronics", "food", "tools", "medical")
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
TYPE_PRIORITY = {
    "medical": "high",
    "electronics": "medium",
    "tools": "medium",
    "food": "low",
}


def create_warehouse_state(
    *,
    layout: str = OPEN_FLOOR,
    num_workers: int = 4,
    running: bool = True,
) -> dict[str, Any]:
    """Create a clean JSON-shaped autonomous warehouse state."""
    workers = []
    spawn_positions = [[8, 20], [8, 23], [8, 26], [8, 29], [10, 22], [10, 27]]
    for index in range(num_workers):
        pos = spawn_positions[index % len(spawn_positions)]
        workers.append({
            "id": f"W-{index + 1}",
            "position": _pos(pos),
            "status": "idle",
            "assignedTaskId": None,
            "carryingItemId": None,
            "path": [],
            "spawnPosition": _pos(pos),
        })

    dropoff_zones = [
        {"id": "box-a", "label": "Box A", "acceptsType": "electronics", "position": _pos([46, 10])},
        {"id": "box-b", "label": "Box B", "acceptsType": "food", "position": _pos([46, 20])},
        {"id": "box-c", "label": "Box C", "acceptsType": "tools", "position": _pos([46, 30])},
        {"id": "box-d", "label": "Box D", "acceptsType": "medical", "position": _pos([46, 40])},
    ]

    conveyors = [
        {
            "id": "conv-north",
            "spawnPosition": _pos([2, 12]),
            "spawnIntervalMs": 2200,
            "enabled": True,
            "itemTypes": ["electronics", "tools"],
            "nextSpawnAtMs": 500,
        },
        {
            "id": "conv-mid",
            "spawnPosition": _pos([2, 25]),
            "spawnIntervalMs": 3000,
            "enabled": True,
            "itemTypes": ["food", "medical"],
            "nextSpawnAtMs": 1000,
        },
        {
            "id": "conv-south",
            "spawnPosition": _pos([2, 38]),
            "spawnIntervalMs": 3800,
            "enabled": True,
            "itemTypes": ["tools", "electronics", "food"],
            "nextSpawnAtMs": 1500,
        },
    ]

    state = {
        "workers": workers,
        "items": [],
        "tasks": [],
        "conveyors": conveyors,
        "dropoffZones": dropoff_zones,
        "wall": [],
        "layout": layout,
        "running": running,
        "simTimeMs": 0,
        "metrics": {
            "completedTasks": 0,
            "pendingTasks": 0,
            "activeWorkers": num_workers,
            "averageDeliveryTime": 0.0,
            "throughput": 0.0,
        },
        "events": [],
        "counters": {"item": 0, "task": 0},
    }
    log_event(state, "Warehouse simulation initialized.", "info")
    return state


def step_warehouse(state: dict[str, Any], dt_ms: int = 100) -> dict[str, Any]:
    """Advance the autonomous warehouse simulation by one tick."""
    if not state.get("running", True):
        update_metrics(state)
        return state

    state["simTimeMs"] = int(state.get("simTimeMs", 0)) + max(0, int(dt_ms))
    spawn_due_items(state)
    recover_disabled_workers(state)
    leader_assign_tasks(state)
    advance_workers(state)
    update_metrics(state)
    return state


def apply_demo_control(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Apply demo/debug controls without coupling to any UI framework."""
    kind = action.get("type")
    if kind == "start":
        state["running"] = True
        log_event(state, "Simulation started.", "info")
    elif kind == "pause":
        state["running"] = False
        log_event(state, "Simulation paused.", "warning")
    elif kind == "toggle_pause":
        state["running"] = not state.get("running", True)
        log_event(state, "Simulation started." if state["running"] else "Simulation paused.", "info")
    elif kind == "spike":
        trigger_order_spike(state, int(action.get("count", 8) or 8))
    elif kind == "disable_worker":
        disable_worker(state, str(action.get("workerId", "")))
    elif kind == "enable_worker":
        enable_worker(state, str(action.get("workerId", "")))
    elif kind == "wall_add":
        cell = action.get("cell")
        if _is_cell(cell) and cell not in state.get("wall", []):
            state.setdefault("wall", []).append(list(cell))
            log_event(state, f"Route blocked at {cell}; workers will reroute.", "warning")
            reroute_active_workers(state)
    elif kind == "wall_remove":
        cell = action.get("cell")
        if _is_cell(cell) and list(cell) in state.get("wall", []):
            state["wall"].remove(list(cell))
            log_event(state, f"Route block removed at {cell}.", "info")
            reroute_active_workers(state)
    elif kind == "reset":
        fresh = create_warehouse_state(running=state.get("running", True))
        state.clear()
        state.update(fresh)
    return state


def spawn_due_items(state: dict[str, Any]) -> None:
    """Spawn conveyor items whose scheduled time has arrived."""
    now = int(state.get("simTimeMs", 0))
    for conveyor in state.get("conveyors", []):
        if not conveyor.get("enabled", True):
            continue
        interval = int(conveyor.get("spawnIntervalMs", 3000))
        while now >= int(conveyor.get("nextSpawnAtMs", 0)):
            spawn_item_from_conveyor(state, conveyor)
            conveyor["nextSpawnAtMs"] = int(conveyor.get("nextSpawnAtMs", 0)) + interval


def spawn_item_from_conveyor(state: dict[str, Any], conveyor: dict[str, Any]) -> dict[str, Any]:
    """Create an item and its pending task from a conveyor."""
    counters = state.setdefault("counters", {"item": 0, "task": 0})
    counters["item"] += 1
    item_id = f"I-{counters['item']}"
    item_types = conveyor.get("itemTypes") or list(ITEM_TYPES)
    item_type = item_types[(counters["item"] - 1) % len(item_types)]
    priority = TYPE_PRIORITY.get(item_type, "medium")
    zone = dropoff_for_type(state, item_type)
    item = {
        "id": item_id,
        "type": item_type,
        "priority": priority,
        "position": copy.deepcopy(conveyor["spawnPosition"]),
        "destinationZoneId": zone["id"],
        "status": "waiting",
        "createdAtMs": state.get("simTimeMs", 0),
    }
    state.setdefault("items", []).append(item)
    task = create_task_for_item(state, item, zone)
    log_event(
        state,
        f"Conveyor {conveyor['id']} spawned item {item_id} ({item_type}); created {task['id']}.",
        "info",
    )
    return item


def create_task_for_item(
    state: dict[str, Any],
    item: dict[str, Any],
    zone: dict[str, Any],
) -> dict[str, Any]:
    """Create a pending transport task for an item."""
    counters = state.setdefault("counters", {"item": 0, "task": 0})
    counters["task"] += 1
    task = {
        "id": f"T-{counters['task']}",
        "itemId": item["id"],
        "assignedWorkerId": None,
        "pickup": copy.deepcopy(item["position"]),
        "dropoff": copy.deepcopy(zone["position"]),
        "destinationZoneId": zone["id"],
        "priority": item.get("priority", "medium"),
        "status": "pending",
        "createdAtMs": state.get("simTimeMs", 0),
        "assignedAtMs": None,
        "pickedUpAtMs": None,
        "deliveredAtMs": None,
        "completedAtMs": None,
    }
    state.setdefault("tasks", []).append(task)
    return task


def leader_assign_tasks(state: dict[str, Any]) -> None:
    """Assign pending tasks to idle workers using priority and route cost."""
    wall_set = _wall_set(state)
    pending = sorted(
        [t for t in state.get("tasks", []) if t.get("status") == "pending"],
        key=lambda t: (PRIORITY_ORDER.get(t.get("priority", "medium"), 1), t.get("createdAtMs", 0)),
    )
    idle_workers = [
        w for w in state.get("workers", [])
        if w.get("status") == "idle"
    ]

    for task in pending:
        if not idle_workers:
            break
        best = min(
            idle_workers,
            key=lambda w: (
                route_cost(_list_pos(w["position"]), _list_pos(task["pickup"]), wall_set)
                + route_cost(_list_pos(task["pickup"]), _list_pos(task["dropoff"]), wall_set),
                w["id"],
            ),
        )
        pickup_path = a_star(_list_pos(best["position"]), _list_pos(task["pickup"]), wall_set)
        if not pickup_path and _list_pos(best["position"]) != _list_pos(task["pickup"]):
            log_event(state, f"Leader could not route {best['id']} to {task['id']}; task remains pending.", "warning")
            continue
        assign_task(state, best, task, pickup_path)
        idle_workers.remove(best)


def assign_task(
    state: dict[str, Any],
    worker: dict[str, Any],
    task: dict[str, Any],
    path: list[list[int]],
) -> None:
    """Assign a task to a worker and log the leader decision."""
    task["status"] = "assigned"
    task["assignedWorkerId"] = worker["id"]
    task["assignedAtMs"] = state.get("simTimeMs", 0)
    item = item_by_id(state, task["itemId"])
    if item:
        item["status"] = "assigned"
    worker["status"] = "moving_to_pickup"
    worker["assignedTaskId"] = task["id"]
    worker["path"] = [_pos(p) for p in path]
    log_event(state, f"Leader assigned {worker['id']} to Task {task['id']}.", "info")


def advance_workers(state: dict[str, Any]) -> None:
    """Advance each worker one path cell and process pickup/dropoff transitions."""
    for worker in state.get("workers", []):
        if worker.get("status") == "disabled":
            continue
        if worker.get("path"):
            worker["position"] = worker["path"].pop(0)
        carried = item_by_id(state, worker.get("carryingItemId"))
        if carried:
            carried["position"] = copy.deepcopy(worker["position"])
        task = task_by_id(state, worker.get("assignedTaskId"))
        if not task:
            if worker.get("status") != "disabled":
                worker["status"] = "idle"
            continue
        if worker.get("status") == "moving_to_pickup":
            process_pickup_arrival(state, worker, task)
        elif worker.get("status") == "moving_to_dropoff":
            process_dropoff_arrival(state, worker, task)


def process_pickup_arrival(
    state: dict[str, Any],
    worker: dict[str, Any],
    task: dict[str, Any],
) -> None:
    """Pick up an item once a worker reaches the pickup cell."""
    if _list_pos(worker["position"]) != _list_pos(task["pickup"]):
        if not worker.get("path"):
            reroute_worker(state, worker, _list_pos(task["pickup"]))
        return

    item = item_by_id(state, task["itemId"])
    if item:
        item["status"] = "picked_up"
        item["position"] = copy.deepcopy(worker["position"])
    task["status"] = "picked_up"
    task["pickedUpAtMs"] = state.get("simTimeMs", 0)
    worker["status"] = "moving_to_dropoff"
    worker["carryingItemId"] = task["itemId"]
    worker["path"] = [_pos(p) for p in a_star(_list_pos(worker["position"]), _list_pos(task["dropoff"]), _wall_set(state))]
    log_event(state, f"{worker['id']} picked up item {task['itemId']} for {task['id']}.", "info")
    if not worker["path"] and _list_pos(worker["position"]) != _list_pos(task["dropoff"]):
        log_event(state, f"Route blocked, {worker['id']} cannot reach dropoff for {task['id']}.", "warning")


def process_dropoff_arrival(
    state: dict[str, Any],
    worker: dict[str, Any],
    task: dict[str, Any],
) -> None:
    """Complete a task once a worker reaches the dropoff cell."""
    if _list_pos(worker["position"]) != _list_pos(task["dropoff"]):
        if not worker.get("path"):
            reroute_worker(state, worker, _list_pos(task["dropoff"]))
        item = item_by_id(state, task["itemId"])
        if item:
            item["position"] = copy.deepcopy(worker["position"])
        return

    item = item_by_id(state, task["itemId"])
    if item:
        item["status"] = "delivered"
        item["position"] = copy.deepcopy(task["dropoff"])
    now = state.get("simTimeMs", 0)
    task["status"] = "completed"
    task["deliveredAtMs"] = now
    task["completedAtMs"] = now
    worker["status"] = "idle"
    worker["assignedTaskId"] = None
    worker["carryingItemId"] = None
    worker["path"] = []
    zone = zone_by_id(state, task.get("destinationZoneId"))
    label = zone.get("label", "output zone") if zone else "output zone"
    log_event(state, f"{worker['id']} delivered item {task['itemId']} to {label}.", "success")


def recover_disabled_workers(state: dict[str, Any]) -> None:
    """Return tasks owned by disabled workers to the pending queue."""
    for worker in state.get("workers", []):
        if worker.get("status") != "disabled":
            continue
        task_id = worker.get("assignedTaskId")
        task = task_by_id(state, task_id)
        if not task or task.get("status") in {"completed", "delivered"}:
            continue
        item = item_by_id(state, task["itemId"])
        if worker.get("carryingItemId") == task["itemId"]:
            task["pickup"] = copy.deepcopy(worker["position"])
            if item:
                item["position"] = copy.deepcopy(worker["position"])
                item["status"] = "waiting"
        elif item:
            item["status"] = "waiting"
        task["status"] = "pending"
        task["assignedWorkerId"] = None
        task["assignedAtMs"] = None
        worker["assignedTaskId"] = None
        worker["carryingItemId"] = None
        worker["path"] = []
        log_event(state, f"Leader reassigned {task_id} after {worker['id']} became disabled.", "warning")


def disable_worker(state: dict[str, Any], worker_id: str) -> bool:
    """Disable a worker and let recovery reclaim its task next tick."""
    worker = worker_by_id(state, worker_id)
    if not worker or worker.get("status") == "disabled":
        return False
    worker["status"] = "disabled"
    worker["path"] = []
    log_event(state, f"{worker_id} disabled.", "error")
    recover_disabled_workers(state)
    update_metrics(state)
    return True


def enable_worker(state: dict[str, Any], worker_id: str) -> bool:
    """Re-enable a disabled worker."""
    worker = worker_by_id(state, worker_id)
    if not worker or worker.get("status") != "disabled":
        return False
    worker["status"] = "idle"
    worker["assignedTaskId"] = None
    worker["carryingItemId"] = None
    worker["path"] = []
    log_event(state, f"{worker_id} re-enabled.", "success")
    update_metrics(state)
    return True


def trigger_order_spike(state: dict[str, Any], count: int = 8) -> None:
    """Spawn a burst of items immediately across conveyors."""
    conveyors = [c for c in state.get("conveyors", []) if c.get("enabled", True)]
    if not conveyors:
        return
    for index in range(max(1, count)):
        spawn_item_from_conveyor(state, conveyors[index % len(conveyors)])
    log_event(state, f"Order spike triggered: {count} new item(s).", "warning")


def reroute_active_workers(state: dict[str, Any]) -> None:
    """Replan active workers after obstacle changes."""
    for worker in state.get("workers", []):
        if worker.get("status") in {"moving_to_pickup", "moving_to_dropoff"}:
            task = task_by_id(state, worker.get("assignedTaskId"))
            if not task:
                continue
            goal = task["pickup"] if worker["status"] == "moving_to_pickup" else task["dropoff"]
            reroute_worker(state, worker, _list_pos(goal))


def reroute_worker(state: dict[str, Any], worker: dict[str, Any], goal: list[int]) -> None:
    """Recompute a worker path to a target cell."""
    path = a_star(_list_pos(worker["position"]), goal, _wall_set(state))
    if path or _list_pos(worker["position"]) == goal:
        worker["path"] = [_pos(p) for p in path]
        log_event(state, f"{worker['id']} rerouting to {goal}.", "info")
    else:
        log_event(state, f"Route blocked, {worker['id']} cannot reach {goal}.", "warning")


def update_metrics(state: dict[str, Any]) -> None:
    """Update lightweight metrics for the UI/event panel."""
    tasks = state.get("tasks", [])
    completed = [t for t in tasks if t.get("status") == "completed"]
    pending = [t for t in tasks if t.get("status") == "pending"]
    active_workers = [w for w in state.get("workers", []) if w.get("status") != "disabled"]
    durations = [
        (t["completedAtMs"] - t["createdAtMs"]) / 1000.0
        for t in completed
        if t.get("completedAtMs") is not None and t.get("createdAtMs") is not None
    ]
    elapsed_minutes = max(float(state.get("simTimeMs", 0)) / 60000.0, 1e-9)
    state["metrics"] = {
        "completedTasks": len(completed),
        "pendingTasks": len(pending),
        "activeWorkers": len(active_workers),
        "averageDeliveryTime": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "throughput": round(len(completed) / elapsed_minutes, 2),
    }


def log_event(state: dict[str, Any], message: str, event_type: str = "info") -> None:
    """Append a timestamped event and cap the log size."""
    state.setdefault("events", []).append({
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simTimeMs": state.get("simTimeMs", 0),
        "message": message,
        "type": event_type,
    })
    state["events"] = state["events"][-200:]


def dropoff_for_type(state: dict[str, Any], item_type: str) -> dict[str, Any]:
    """Find the output zone for an item type."""
    for zone in state.get("dropoffZones", []):
        if zone.get("acceptsType") == item_type:
            return zone
    raise ValueError(f"no dropoff zone accepts item type {item_type!r}")


def worker_by_id(state: dict[str, Any], worker_id: str | None) -> dict[str, Any] | None:
    return next((w for w in state.get("workers", []) if w.get("id") == worker_id), None)


def item_by_id(state: dict[str, Any], item_id: str | None) -> dict[str, Any] | None:
    return next((i for i in state.get("items", []) if i.get("id") == item_id), None)


def task_by_id(state: dict[str, Any], task_id: str | None) -> dict[str, Any] | None:
    return next((t for t in state.get("tasks", []) if t.get("id") == task_id), None)


def zone_by_id(state: dict[str, Any], zone_id: str | None) -> dict[str, Any] | None:
    return next((z for z in state.get("dropoffZones", []) if z.get("id") == zone_id), None)


def public_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly snapshot without internal-only counters."""
    snapshot = copy.deepcopy(state)
    snapshot.pop("counters", None)
    return snapshot


def viewer_state(state: dict[str, Any], *, speed_multiplier: int = 1) -> dict[str, Any]:
    """Return warehouse state plus compatibility fields for the current 3D UI."""
    snapshot = public_state(state)
    snapshot["robots"] = [
        {
            "id": worker["id"],
            "role": "WORKER",
            "pos": _list_pos(worker["position"]),
            "path": [_list_pos(p) for p in worker.get("path", [])],
            "current_task": worker.get("assignedTaskId"),
            "status": worker.get("status"),
            "carrying_item": worker.get("carryingItemId"),
        }
        for worker in state.get("workers", [])
    ]
    snapshot["workstations"] = _viewer_workstations(state)
    snapshot["tasks"] = [
        {
            "id": task["id"],
            "status": _viewer_task_status(task.get("status", "pending")),
            "pickup": _list_pos(task["pickup"]),
            "delivery": _list_pos(task["dropoff"]),
            "pickup_name": task.get("itemId"),
            "delivery_name": task.get("destinationZoneId"),
            "assigned_to": task.get("assignedWorkerId"),
            "priority": task.get("priority"),
            "itemId": task.get("itemId"),
        }
        for task in state.get("tasks", [])
        if task.get("status") != "completed"
    ]
    snapshot["blackboard"] = [
        {
            "from": -1,
            "type": event.get("type", "info").upper(),
            "content": event.get("message", ""),
            "timestamp": event.get("simTimeMs", 0) / 1000.0,
        }
        for event in state.get("events", [])
    ]
    metrics = state.get("metrics", {})
    snapshot["stats"] = {
        "completed": metrics.get("completedTasks", 0),
        "elapsed": round(state.get("simTimeMs", 0) / 1000.0, 1),
        "rate": metrics.get("throughput", 0.0),
        "avg_route_time": metrics.get("averageDeliveryTime", 0.0),
        "pending": metrics.get("pendingTasks", 0),
        "active_workers": metrics.get("activeWorkers", 0),
    }
    snapshot["connection_status"] = "online" if state.get("running", True) else "offline"
    snapshot["speed_multiplier"] = speed_multiplier
    snapshot["tick"] = int(state.get("simTimeMs", 0) // 100)
    return snapshot


def _viewer_workstations(state: dict[str, Any]) -> list[dict[str, Any]]:
    stations: list[dict[str, Any]] = []
    for conveyor in state.get("conveyors", []):
        stations.append({
            "name": f"Intake {conveyor['id'].replace('conv-', '').title()}",
            "pos": _list_pos(conveyor["spawnPosition"]),
            "color": [0, 212, 255],
            "kind": "conveyor",
        })
    zone_colors = {
        "electronics": [80, 128, 255],
        "food": [255, 204, 68],
        "tools": [255, 140, 0],
        "medical": [0, 255, 136],
    }
    for zone in state.get("dropoffZones", []):
        stations.append({
            "name": zone.get("label", zone.get("id")),
            "pos": _list_pos(zone["position"]),
            "color": zone_colors.get(zone.get("acceptsType"), [220, 220, 220]),
            "kind": "dropoff",
        })
    return stations


def _viewer_task_status(status: str) -> str:
    if status in {"delivered", "completed"}:
        return "done"
    if status == "picked_up":
        return "in_transit"
    return "open"


def _wall_set(state: dict[str, Any]) -> set[tuple[int, int]]:
    return {tuple(cell) for cell in state.get("wall", [])}


def _pos(cell: list[int] | tuple[int, int] | dict[str, int]) -> dict[str, int]:
    if isinstance(cell, dict):
        return {"x": int(cell["x"]), "y": int(cell["y"])}
    return {"x": int(cell[0]), "y": int(cell[1])}


def _list_pos(pos: dict[str, int] | list[int]) -> list[int]:
    if isinstance(pos, dict):
        return [int(pos["x"]), int(pos["y"])]
    return [int(pos[0]), int(pos[1])]


def _is_cell(cell: Any) -> bool:
    return (
        isinstance(cell, list)
        and len(cell) == 2
        and all(isinstance(v, int) for v in cell)
        and 0 <= cell[0] < GRID_WIDTH
        and 0 <= cell[1] < GRID_HEIGHT
    )
