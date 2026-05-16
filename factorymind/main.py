"""Main simulation loop for FactoryMind R2D2.

Person D owns this file.

Run with mock agents (recommended for initial testing):
    AGENTS_USE_MOCK=true python -m factorymind.main

Run with real Nemotron:
    NVIDIA_API_KEY=<key> python -m factorymind.main
"""

from __future__ import annotations

import heapq
import importlib
import copy
import os
import re
import socket
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

import pygame

from factorymind import state as S
from factorymind import render as R
from factorymind import memory as M
from factorymind import nemoclaw
from factorymind.agents import (
    Blackboard,
    choose_worker_task,
    interpret_user_tasks,
    leader_decide,
    strategist_decide,
    worker_decide,
)
from factorymind.state import (
    LEADER, WORKER, OPEN_FLOOR, BOTTLENECK_BRIDGE,
    GRID_WIDTH, GRID_HEIGHT,
    STATION_QUOTA_NAMES, empty_station_quotas,
)

# ---------------------------------------------------------------------------
# Timing constants (seconds)
# ---------------------------------------------------------------------------
FPS = 60
LEADER_TICK_INTERVAL = 1.5
WORKER_TICK_INTERVAL = 2.5  # workers think slightly less often than leaders
STRATEGIST_TICK_INTERVAL = 20.0
MOVE_TICK_INTERVAL = 0.05  # how often robots advance one step
SPEED_OPTIONS = (1, 2, 3, 4)

# When FLEET_MASTER_ONLY is on (default), the leader/master is the sole dispatcher:
# workers never self-claim tasks; they only execute master-assigned routes.
_FLEET_MASTER_ONLY = os.getenv("FLEET_MASTER_ONLY", "true").lower() == "true"
_ALL_AGENTS_LLM = os.getenv("ALL_AGENTS_LLM", "false").lower() == "true"
_COOPERATIVE_PATHING = os.getenv(
    "COOPERATIVE_PATHING",
    "false" if os.getenv("FLEET_MASTER_ONLY", "true").lower() == "true" else "true",
).lower() == "true"


def _master_controls_fleet(world_state: dict) -> bool:
    """True when the OpenClaw leader owns the task queue (not worker autonomy)."""
    return bool(world_state.get("fleet_mode") or world_state.get("task_queue"))

# ---------------------------------------------------------------------------
# Background inference workers
#
# LLM calls (cloud or local NIM) take 0.3-30s per request. Running them on
# the main thread freezes pygame's event loop and the OS marks the window
# "Not Responding". We dispatch decisions to a thread pool and apply them
# whenever they're ready; the main loop keeps rendering at 60 FPS.
# Pool size = leaders + workers + strategist + a little headroom.
# ---------------------------------------------------------------------------
_decision_executor = ThreadPoolExecutor(max_workers=12, thread_name_prefix="agent")
_pending_leader: dict[int, Future] = {}
_pending_worker: dict[int, Future] = {}
_pending_strategist: Optional[Future] = None
_sim_started: bool = False  # set True on bootstrap; sim runs continuously when auto-start is on

# ---------------------------------------------------------------------------
# A* pathfinding
# ---------------------------------------------------------------------------

def a_star(start: list[int], goal: list[int], wall_set: set) -> list[list[int]]:
    """Return a list of [x, y] cells from start (exclusive) to goal (inclusive).

    Uses Manhattan distance heuristic. Returns empty list if no path found.
    """
    sx, sy = start
    gx, gy = goal

    if [sx, sy] == [gx, gy]:
        return []

    open_heap: list[tuple[int, int, int, list]] = []
    heapq.heappush(open_heap, (0, 0, id((sx, sy)), [sx, sy]))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
    g_score: dict[tuple[int, int], int] = {(sx, sy): 0}

    def h(x: int, y: int) -> int:
        return abs(x - gx) + abs(y - gy)

    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        cx, cy = current

        if [cx, cy] == [gx, gy]:
            # Reconstruct
            path: list[list[int]] = []
            node: tuple[int, int] | None = (cx, cy)
            while node is not None:
                path.append(list(node))
                node = came_from[node]
            path.reverse()
            return path[1:]  # exclude start

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                continue
            if (nx, ny) in wall_set:
                continue
            ng = g + 1
            if ng < g_score.get((nx, ny), 10**9):
                g_score[(nx, ny)] = ng
                priority = ng + h(nx, ny)
                came_from[(nx, ny)] = (cx, cy)
                heapq.heappush(open_heap, (priority, ng, id((nx, ny)), [nx, ny]))

    return []  # no path


def _cooperative_a_star(
    start: list[int],
    goal: list[int],
    wall_set: set,
    reservations: dict[str, dict[int, set]],
    *,
    max_steps: int = 220,
) -> list[list[int]]:
    """Plan a path in x/y/time, respecting other robots' reserved cells.

    Robots may wait in place. This gives each agent an independent route plan
    that accounts for the paths already committed by its neighbors.
    """
    if start == goal:
        return []

    sx, sy = start
    gx, gy = goal
    start_state = (sx, sy, 0)
    open_heap: list[tuple[int, int, int, tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (abs(sx - gx) + abs(sy - gy), 0, 0, start_state))
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_state: None}
    g_score: dict[tuple[int, int, int], int] = {start_state: 0}

    cells_by_time = reservations.get("cells", {})
    edges_by_time = reservations.get("edges", {})
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

    while open_heap:
        _, g, _, state = heapq.heappop(open_heap)
        cx, cy, t = state
        if [cx, cy] == goal:
            path: list[list[int]] = []
            node: tuple[int, int, int] | None = state
            while node is not None:
                path.append([node[0], node[1]])
                node = came_from[node]
            path.reverse()
            return path[1:]
        if t >= max_steps:
            continue

        for dx, dy in moves:
            nx, ny, nt = cx + dx, cy + dy, t + 1
            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                continue
            if (nx, ny) in wall_set:
                continue
            if (nx, ny) in cells_by_time.get(nt, set()):
                continue
            if ((nx, ny), (cx, cy)) in edges_by_time.get(nt, set()):
                continue
            next_state = (nx, ny, nt)
            ng = g + 1
            if ng >= g_score.get(next_state, 10**9):
                continue
            g_score[next_state] = ng
            came_from[next_state] = state
            priority = ng + abs(nx - gx) + abs(ny - gy)
            # Waiting is useful, but moving wins ties so robots do not loiter.
            wait_bias = 1 if (nx, ny) == (cx, cy) else 0
            heapq.heappush(open_heap, (priority + wait_bias, ng, id(next_state), next_state))

    return []


def _reservation_table(world_state: dict, *, exclude_robot_id: int | None = None) -> dict[str, dict[int, set]]:
    """Build time-indexed cell and edge reservations from active robot paths."""
    cells: dict[int, set[tuple[int, int]]] = {}
    edges: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {}
    hold_steps = 3 if _FLEET_MASTER_ONLY else 8
    for robot in world_state.get("robots", []):
        if robot.get("id") == exclude_robot_id:
            continue
        if not robot.get("path"):
            continue
        positions = [list(robot.get("pos", [0, 0]))] + [list(p) for p in robot.get("path", [])]
        for t, pos in enumerate(positions):
            cells.setdefault(t, set()).add(tuple(pos))
            if t > 0:
                edges.setdefault(t, set()).add((tuple(positions[t - 1]), tuple(pos)))
        if positions:
            last = tuple(positions[-1])
            for t in range(len(positions), len(positions) + hold_steps):
                cells.setdefault(t, set()).add(last)
    return {"cells": cells, "edges": edges}


def _pathfinding_walls(
    robot: dict,
    goal: list[int],
    world_state: dict,
) -> set[tuple[int, int]]:
    """Walls plus other robots' tiles (fleet mode avoids head-on gridlock)."""
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    if _FLEET_MASTER_ONLY:
        for other in world_state.get("robots", []):
            if other.get("id") == robot.get("id"):
                continue
            wall_set.add(tuple(other.get("pos", [])))
    goal_cell = tuple(goal)
    start_cell = tuple(robot.get("pos", []))
    wall_set.discard(goal_cell)
    wall_set.discard(start_cell)
    return wall_set


def _planned_path_to(
    robot: dict,
    goal: list[int],
    world_state: dict,
    *,
    ignore_reservations: bool = False,
) -> list[list[int]]:
    """Return a route that accounts for walls and, by default, other robots."""
    wall_set = _pathfinding_walls(robot, goal, world_state)
    if _COOPERATIVE_PATHING and not ignore_reservations and not _FLEET_MASTER_ONLY:
        reservations = _reservation_table(world_state, exclude_robot_id=robot.get("id"))
        path = _cooperative_a_star(robot["pos"], goal, wall_set, reservations)
        if path:
            return path
    return a_star(robot["pos"], goal, wall_set)


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

_task_counter = 0


def _create_task(
    world_state: dict,
    pickup_ws: dict,
    delivery_ws: dict,
    *,
    source: str,
) -> dict:
    """Create a concrete pickup-to-delivery task from selected workstations."""
    global _task_counter
    task = {
        "id": _task_counter,
        "status": "open",
        "pickup": list(pickup_ws["pos"]),
        "delivery": list(delivery_ws["pos"]),
        "pickup_name": pickup_ws["name"],
        "delivery_name": delivery_ws["name"],
        "assigned_to": None,
        "source": source,
    }
    _task_counter += 1
    return task


# ---------------------------------------------------------------------------
# Robot movement helpers
# ---------------------------------------------------------------------------

def _is_workstation_cell(pos: list[int], world_state: dict) -> bool:
    """True when ``pos`` matches any workstation tile in the current layout."""
    for ws in world_state.get("workstations", []):
        if list(ws.get("pos", [])) == list(pos):
            return True
    return False


def _cell_occupied_by_other(
    cell: tuple[int, int],
    world_state: dict,
    *,
    exclude_robot_id: int,
) -> bool:
    for robot in world_state.get("robots", []):
        if robot.get("id") == exclude_robot_id:
            continue
        if tuple(robot.get("pos", [])) == cell:
            return True
    return False


def _robot_blocks_traffic(robot: dict) -> bool:
    """Whether this robot should deny another robot the cell it stands on."""
    if robot.get("role") == LEADER:
        return False
    if robot.get("current_task") is not None:
        return True
    return bool(robot.get("path"))


def _relocate_off_station_immediate(robot: dict, world_state: dict) -> bool:
    """Step off a station tile via a one-cell path (no position snap / teleport)."""
    pos = list(robot.get("pos", []))
    if not _is_workstation_cell(pos, world_state):
        return False
    if robot.get("path"):
        return False
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    spawn = tuple(robot.get("spawn_pos", pos))
    occupied = {
        tuple(r.get("pos", []))
        for r in world_state.get("robots", [])
        if r.get("id") != robot.get("id")
    }
    candidates: list[tuple[int, list[int]]] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
        nx, ny = pos[0] + dx, pos[1] + dy
        if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
            continue
        cell = (nx, ny)
        if cell in wall_set or cell in occupied:
            continue
        if _is_workstation_cell([nx, ny], world_state):
            continue
        dist = abs(nx - spawn[0]) + abs(ny - spawn[1])
        candidates.append((dist, [nx, ny]))
    if not candidates:
        return False
    candidates.sort(key=lambda item: item[0])
    robot["path"] = [candidates[0][1]]
    robot["yield_streak"] = 0
    return True


def _fleet_unstick_worker(robot: dict, world_state: dict) -> None:
    """Re-plan or walk toward spawn — never snap position (avoids visual teleports)."""
    robot["stall_ticks"] = 0
    robot["path"] = []
    _replenish_task_path(robot, world_state)
    if robot.get("path"):
        return
    spawn = robot.get("spawn_pos")
    if spawn and list(robot.get("pos", [])) != list(spawn):
        robot["path"] = _planned_path_to(
            robot, list(spawn), world_state, ignore_reservations=True,
        )


def _step_aside_from_workstation(robot: dict, world_state: dict) -> bool:
    """Move off a station tile; immediate relocate in master fleet mode."""
    if _FLEET_MASTER_ONLY:
        return _relocate_off_station_immediate(robot, world_state)
    pos = list(robot.get("pos", []))
    if not _is_workstation_cell(pos, world_state):
        return False
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    spawn = tuple(robot.get("spawn_pos", pos))
    occupied = {
        tuple(r.get("pos", []))
        for r in world_state.get("robots", [])
        if r.get("id") != robot.get("id")
    }
    candidates: list[tuple[int, list[int]]] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
        nx, ny = pos[0] + dx, pos[1] + dy
        if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
            continue
        cell = (nx, ny)
        if cell in wall_set or cell in occupied:
            continue
        if _is_workstation_cell([nx, ny], world_state):
            continue
        dist = abs(nx - spawn[0]) + abs(ny - spawn[1])
        candidates.append((dist, [nx, ny]))
    if not candidates:
        return False
    candidates.sort(key=lambda item: item[0])
    robot["path"] = [candidates[0][1]]
    robot["yield_streak"] = 0
    return True


def _replenish_task_path(robot: dict, world_state: dict) -> None:
    """Re-plan when a worker has a task but lost its route (avoids permanent stall)."""
    task_id = robot.get("current_task")
    if task_id is None or robot.get("path"):
        return
    task = next((t for t in world_state.get("tasks", []) if t["id"] == task_id), None)
    if task is None or task.get("status") == "done":
        robot["current_task"] = None
        return
    pos = list(robot.get("pos", []))
    if task["status"] == "open":
        if pos == list(task["pickup"]):
            task["status"] = "in_transit"
            goal = task["delivery"]
        else:
            goal = task["pickup"]
    elif task["status"] == "in_transit":
        if pos == list(task["delivery"]):
            return
        goal = task["delivery"]
    else:
        return
    robot["path"] = _planned_path_to(robot, list(goal), world_state)


def _park_robot_at_spawn(robot: dict, world_state: dict) -> None:
    """Route an idle robot away from a workstation tile back to its spawn slot.

    Without this, a worker that finishes at e.g. Shipping sits on the
    delivery cell forever; the next worker arriving at Shipping is then
    blocked by the move-coordinator's "cell occupied by a non-moving robot"
    check and the fleet deadlocks at the tail of the queue. Returning home
    keeps workstation tiles free for whoever is inbound next.
    """
    spawn = robot.get("spawn_pos")
    if not spawn:
        return
    if list(robot.get("pos", [])) == list(spawn):
        return
    # Use plain (non-cooperative) A* so an idle worker can't itself deadlock
    # against the reservations of others; this is a low-priority repositioning.
    robot["path"] = _planned_path_to(
        robot, list(spawn), world_state, ignore_reservations=True
    )


def _complete_task_delivery(
    robot: dict,
    task: dict,
    world_state: dict,
    blackboard: "Blackboard",
) -> None:
    """Mark a task finished when the robot is already at the delivery cell."""
    route_time = round(time.time() - task.get("started_at", time.time()), 2)
    task["status"] = "done"
    robot["current_task"] = None
    robot.pop("master_assigned", None)

    world_state["stats"]["completed"] += 1
    world_state["stats"]["last_route_time"] = route_time
    count = world_state["stats"]["completed"]
    prev_avg = world_state["stats"].get("avg_route_time", route_time)
    world_state["stats"]["avg_route_time"] = round(
        (prev_avg * (count - 1) + route_time) / count, 2
    )

    blackboard.post(
        robot["id"], "COMPLETE",
        f"task-{task['id']} {task.get('pickup_name','?')}→{task.get('delivery_name','?')} done",
        route_time=route_time,
    )

    robot["path"] = []
    robot["yield_streak"] = 0
    pos = robot["pos"]
    # Free the delivery tile immediately so the next pickup can proceed.
    if _is_workstation_cell(pos, world_state):
        if not _relocate_off_station_immediate(robot, world_state):
            if (
                not _master_controls_fleet(world_state)
                and not world_state.get("task_queue")
            ):
                _park_robot_at_spawn(robot, world_state)
    _fleet_dispatch_step(world_state, blackboard)


def _reconcile_stationary_task_progress(
    robot: dict,
    world_state: dict,
    blackboard: "Blackboard",
) -> None:
    """When A* returns an empty path (already at goal), advance task state without moving.

    Otherwise robots assigned e.g. Shipping→Parts while still standing on
    Shipping never get a movement tick and hold the fleet deadlocked.
    """
    if robot.get("path"):
        return
    task_id = robot.get("current_task")
    if task_id is None:
        return
    task = next((t for t in world_state["tasks"] if t["id"] == task_id), None)
    if task is None:
        return
    pos = list(robot["pos"])
    if task["status"] == "open" and pos == list(task["pickup"]):
        task["status"] = "in_transit"
        _apply_station_pickup_quota(task, world_state)
        robot["path"] = _planned_path_to(robot, task["delivery"], world_state)
    if task["status"] == "in_transit" and pos == list(task["delivery"]):
        _complete_task_delivery(robot, task, world_state, blackboard)


def _advance_robot(robot: dict, world_state: dict, blackboard: "Blackboard") -> None:
    """Move robot one cell along its pre-computed path, completing tasks as needed."""
    path = robot.get("path", [])
    if not path:
        return

    robot["pos"] = path.pop(0)

    # Check if robot reached its current task destination
    task_id = robot.get("current_task")
    if task_id is None:
        return

    task = next((t for t in world_state["tasks"] if t["id"] == task_id), None)
    if task is None:
        return

    pos = robot["pos"]
    # If at pickup and task not yet in transit
    if list(pos) == list(task["pickup"]) and task["status"] == "open":
        task["status"] = "in_transit"
        _apply_station_pickup_quota(task, world_state)
        robot["path"] = _planned_path_to(robot, task["delivery"], world_state)

    # If at delivery — task done, then stop until the next user task.
    elif list(pos) == list(task["delivery"]) and task["status"] == "in_transit":
        _complete_task_delivery(robot, task, world_state, blackboard)


def _workers_available_for_dispatch(world_state: dict) -> list[dict]:
    """Workers ready for a new master assignment (path/step-aside must not block)."""
    return [
        r for r in world_state.get("robots", [])
        if r.get("role") == WORKER and r.get("current_task") is None
    ]


def _advance_robots_fleet(world_state: dict, blackboard: "Blackboard") -> None:
    """Master fleet movement: ordered steps, no yield deadlock at stations."""
    workers = sorted(
        [r for r in world_state.get("robots", []) if r.get("role") == WORKER],
        key=lambda r: r["id"],
    )
    for robot in workers:
        _replenish_task_path(robot, world_state)
        if robot.get("current_task") is None and _is_workstation_cell(
            robot.get("pos", []), world_state,
        ):
            _relocate_off_station_immediate(robot, world_state)
        _reconcile_stationary_task_progress(robot, world_state, blackboard)

    occupied = {tuple(r["pos"]) for r in world_state.get("robots", [])}
    for robot in workers:
        path = robot.get("path") or []
        if not path:
            robot["stall_ticks"] = 0
            continue
        nxt = tuple(path[0])
        cur = tuple(robot["pos"])
        if nxt != cur and nxt in occupied:
            robot["stall_ticks"] = int(robot.get("stall_ticks", 0)) + 1
            if robot["stall_ticks"] >= 24:
                _fleet_unstick_worker(robot, world_state)
            continue
        occupied.discard(cur)
        before = tuple(robot["pos"])
        _advance_robot(robot, world_state, blackboard)
        after = tuple(robot["pos"])
        if before == after:
            robot["stall_ticks"] = int(robot.get("stall_ticks", 0)) + 1
            if robot["stall_ticks"] >= 24:
                _fleet_unstick_worker(robot, world_state)
        else:
            robot["stall_ticks"] = 0
        occupied.add(after)


def _advance_robots(world_state: dict, blackboard: "Blackboard") -> None:
    """Route to fleet or coordinated movement depending on mode."""
    if _FLEET_MASTER_ONLY and _master_controls_fleet(world_state):
        _advance_robots_fleet(world_state, blackboard)
    else:
        _advance_robots_coordinated(world_state, blackboard)


def _advance_robots_coordinated(world_state: dict, blackboard: "Blackboard") -> None:
    """Move robots one step with simple collision avoidance.

    Idle workers on station tiles step aside; stuck workers re-plan. Yield ties
    rotate by tick so the same robot does not block forever.
    """
    robots = world_state.get("robots", [])
    tick = int(world_state.get("tick", 0))

    for robot in robots:
        _replenish_task_path(robot, world_state)
        if (
            robot.get("role") == WORKER
            and robot.get("current_task") is None
            and not robot.get("path")
            and _is_workstation_cell(robot.get("pos", []), world_state)
        ):
            _step_aside_from_workstation(robot, world_state)
        _reconcile_stationary_task_progress(robot, world_state, blackboard)

    current = {r["id"]: tuple(r["pos"]) for r in robots}
    proposals = {
        r["id"]: tuple(r["path"][0])
        for r in robots
        if r.get("path")
    }
    if not proposals:
        return

    denied: set[int] = set()

    by_cell: dict[tuple[int, int], list[int]] = {}
    for rid, cell in proposals.items():
        by_cell.setdefault(cell, []).append(rid)
    for _cell, ids in by_cell.items():
        if len(ids) <= 1:
            continue
        winner = min(ids, key=lambda rid: (rid + tick) % 1000)
        denied.update(rid for rid in ids if rid != winner)

    for rid, target in proposals.items():
        if rid in denied:
            continue
        for other_id, other_target in proposals.items():
            if other_id == rid or other_id in denied:
                continue
            if target == current.get(other_id) and other_target == current.get(rid):
                denied.add(max(rid, other_id))

    moving_away = set(proposals.keys()) - denied
    for rid, target in proposals.items():
        if rid in denied:
            continue
        blocker = next(
            (
                other
                for other in robots
                if other["id"] != rid
                and tuple(other["pos"]) == target
                and other["id"] not in moving_away
                and _robot_blocks_traffic(other)
            ),
            None,
        )
        if blocker is not None:
            denied.add(rid)

    now = time.time()
    for robot in robots:
        rid = robot["id"]
        if rid in denied:
            streak = int(robot.get("yield_streak", 0)) + 1
            robot["yield_streak"] = streak
            if streak >= 4:
                if robot.get("current_task") is not None:
                    _replenish_task_path(robot, world_state)
                elif _is_workstation_cell(robot.get("pos", []), world_state):
                    _step_aside_from_workstation(robot, world_state)
                robot["yield_streak"] = 0
            elif now - robot.get("last_yield_post", 0.0) > 2.5:
                blackboard.post(
                    rid,
                    "INTENT",
                    f"R{rid} yielding one step to avoid a route conflict.",
                )
                robot["last_yield_post"] = now
            continue
        robot["yield_streak"] = 0
        if rid in proposals:
            _advance_robot(robot, world_state, blackboard)


def _assign_task_to_robot(robot: dict, task: dict, world_state: dict) -> None:
    """Assign a task to a robot and compute its initial path to the pickup."""
    previous_assignee = task.get("assigned_to")
    if previous_assignee is not None and previous_assignee != robot["id"]:
        previous_robot = next(
            (r for r in world_state["robots"] if r["id"] == previous_assignee),
            None,
        )
        if previous_robot and previous_robot.get("current_task") == task["id"]:
            previous_robot["current_task"] = None
            previous_robot["path"] = []
    task["status"] = "open"  # will become claimed
    task["assigned_to"] = robot["id"]
    task["started_at"] = time.time()
    robot["current_task"] = task["id"]
    robot.pop("master_assigned", None)
    robot["path"] = _planned_path_to(robot, task["pickup"], world_state)
    if (
        not robot["path"]
        and list(robot["pos"]) == list(task["pickup"])
        and task["status"] == "open"
    ):
        task["status"] = "in_transit"
        robot["path"] = _planned_path_to(robot, task["delivery"], world_state)
    if not robot["path"] and list(robot["pos"]) != list(task["pickup"]):
        robot["path"] = _planned_path_to(robot, task["pickup"], world_state)


def _route_cost(start: list[int], goal: list[int], wall_set: set) -> int:
    """Return shortest-path cell count, or a large cost if unreachable."""
    if start == goal:
        return 0
    path = a_star(start, goal, wall_set)
    return len(path) if path else 10**6


def _traffic_penalty_for_path(robot: dict, path: list[list[int]], world_state: dict | None) -> int:
    """Score route overlap with other agents' current commitments."""
    if not world_state or not path:
        return 0

    penalty = 0
    occupied_now = {
        tuple(r.get("pos", []))
        for r in world_state.get("robots", [])
        if r.get("id") != robot.get("id")
    }
    active_path_cells: set[tuple[int, int]] = set()
    for other in world_state.get("robots", []):
        if other.get("id") == robot.get("id"):
            continue
        active_path_cells.update(tuple(p) for p in other.get("path", []))

    for index, cell in enumerate(path):
        tcell = tuple(cell)
        if tcell in occupied_now:
            penalty += 8
        if tcell in active_path_cells:
            penalty += 3
        if index > 0 and cell == path[index - 1]:
            penalty += 1

    if world_state.get("layout") == BOTTLENECK_BRIDGE:
        bridge_cells = sum(1 for x, y in path if y == 25 and 22 <= x <= 28)
        penalty += bridge_cells * 2
    return penalty


def _estimate_task_cost(
    robot: dict,
    task: dict,
    wall_set: set,
    world_state: dict | None = None,
) -> int:
    """Estimate route length with congestion and existing robot paths included."""
    if world_state is not None:
        to_pickup_path = _planned_path_to(robot, task["pickup"], world_state)
        to_pickup = len(to_pickup_path) if to_pickup_path or robot["pos"] == task["pickup"] else 10**6
    else:
        to_pickup_path = []
        to_pickup = _route_cost(robot["pos"], task["pickup"], wall_set)
    to_delivery = _route_cost(task["pickup"], task["delivery"], wall_set)
    if to_pickup >= 10**6 or to_delivery >= 10**6:
        return 10**6
    route_stub = to_pickup_path + a_star(task["pickup"], task["delivery"], wall_set)
    return to_pickup + to_delivery + _traffic_penalty_for_path(robot, route_stub, world_state)


def _delegate_task_to_nearest_worker(
    leader: dict,
    task: dict,
    world_state: dict,
    blackboard: Blackboard,
) -> bool:
    """Let a leader choose the task while the nearest idle worker executes it."""
    idle_workers = [
        r for r in world_state.get("robots", [])
        if r.get("role") == WORKER and r.get("current_task") is None and not r.get("path")
    ]
    if not idle_workers:
        return False

    worker = min(
        idle_workers,
        key=lambda r: (
            _estimate_task_cost(r, task, {tuple(c) for c in world_state.get("wall", [])}, world_state),
            r["id"],
        ),
    )
    _assign_task_to_robot(worker, task, world_state)
    task["delegated_by"] = leader["id"]
    route = f"{task.get('pickup_name', '?')}->{task.get('delivery_name', '?')}"
    distance = _estimate_task_cost(
        worker,
        task,
        {tuple(c) for c in world_state.get("wall", [])},
        world_state,
    )
    blackboard.post(
        leader["id"],
        "INTENT",
        f"Delegating {route} task {task['id']} to W{worker['id']}; estimated route cost {distance} steps.",
    )
    print(
        f"[leader {leader['id']}] delegated task {task['id']} ({route}) to worker {worker['id']}",
        flush=True,
    )
    return True


def _dispatch_batch_to_workers(
    tasks: list[dict],
    world_state: dict,
    blackboard: Blackboard,
) -> int:
    """Dispatch a user-created batch across idle workers at the same time."""
    leaders = [r for r in world_state.get("robots", []) if r.get("role") == LEADER]
    idle_workers = [
        r for r in world_state.get("robots", [])
        if r.get("role") == WORKER and r.get("current_task") is None and not r.get("path")
    ]
    if not leaders or not idle_workers:
        return 0

    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    assignments: list[tuple[dict, dict, int]] = []
    remaining_tasks = list(tasks)
    remaining_workers = list(idle_workers)

    while remaining_tasks and remaining_workers:
        best_pair: tuple[dict, dict, int] | None = None
        for task in remaining_tasks:
            for worker in remaining_workers:
                cost = _estimate_task_cost(worker, task, wall_set, world_state)
                if cost >= 10**6:
                    continue
                if best_pair is None or (cost, worker["id"], task["id"]) < (
                    best_pair[2],
                    best_pair[1]["id"],
                    best_pair[0]["id"],
                ):
                    best_pair = (task, worker, cost)
        if best_pair is None:
            break
        task, worker, cost = best_pair
        assignments.append(best_pair)
        remaining_tasks.remove(task)
        remaining_workers.remove(worker)

    if not assignments:
        return 0

    avg_cost = round(sum(cost for _, _, cost in assignments) / len(assignments), 1)
    blackboard.post(
        -1,
        "STRATEGY",
        f"Planner precomputed {len(assignments)} worker route(s); average estimated completion cost {avg_cost} steps.",
    )
    print(
        f"[planner] precomputed {len(assignments)} assignment(s), avg estimated cost={avg_cost} steps",
        flush=True,
    )

    dispatched = 0
    for task, worker, cost in assignments:
        leader = leaders[dispatched % len(leaders)]
        _assign_task_to_robot(worker, task, world_state)
        task["delegated_by"] = leader["id"]
        task["estimated_route_cost"] = cost
        route = f"{task.get('pickup_name', '?')}->{task.get('delivery_name', '?')}"
        blackboard.post(
            worker["id"],
            "INTENT",
            f"W{worker['id']} bid {cost} steps for {route} task {task['id']}; accepting assignment.",
        )
        blackboard.post(
            leader["id"],
            "CLAIM",
            f"Batch dispatch: {route} task {task['id']} assigned to W{worker['id']} by lowest route cost ({cost} steps).",
        )
        print(
            f"[leader {leader['id']}] batch dispatch task {task['id']} ({route}) to worker {worker['id']}",
            flush=True,
        )
        dispatched += 1
    return dispatched


# ---------------------------------------------------------------------------
# Autonomous fleet dispatcher
#
# When the user enqueues a batch ("do 100 deliveries"), the leader runs an
# autonomous loop that delegates one task at a time to the closest idle
# worker. Each delegation is gated by NemoClaw — so the leader is, quite
# literally, an OpenClaw-based autonomous agent.
# ---------------------------------------------------------------------------

_FLEET_ROUTE_FLOW = ("Parts", "Assembly", "QA", "Shipping")


def _enqueue_fleet_tasks(
    world_state: dict,
    blackboard: Blackboard,
    routes: list[tuple[dict, dict]],
    *,
    source: str = "user",
    announce: bool = True,
) -> int:
    """Append `routes` to the leader's task queue; returns count enqueued."""
    if not routes:
        return 0
    queue = world_state.setdefault("task_queue", [])
    for pickup_ws, delivery_ws in routes:
        task = _create_task(world_state, pickup_ws, delivery_ws, source=source)
        task["status"] = "queued"
        task["priority"] = 100
        queue.append(task)
    stats = world_state.setdefault("stats", {})
    stats["queued_total"] = stats.get("queued_total", 0) + len(routes)
    world_state["fleet_mode"] = True
    if announce:
        blackboard.post(
            -1,
            "STRATEGY",
            f"OpenClaw leader queued {len(routes)} task(s); fleet dispatch active "
            f"(queue={len(queue)}, completed={stats.get('completed', 0)}).",
        )
    return len(routes)


def _top_up_fleet_queue(world_state: dict, blackboard: Blackboard) -> None:
    """Keep the task queue above a low-water mark so the factory never idles."""
    if world_state.get("manual_control"):
        return
    queue = world_state.setdefault("task_queue", [])
    num_workers = max(
        1,
        sum(1 for r in world_state.get("robots", []) if r.get("role") == WORKER),
    )
    low_water = int(
        os.getenv("FACTORYMIND_FLEET_LOW_WATER", str(max(num_workers * 3, 24)))
    )
    batch = int(
        os.getenv("FACTORYMIND_FLEET_REFILL_BATCH", str(max(num_workers * 4, 32)))
    )
    if len(queue) >= low_water:
        return
    routes = _generate_fleet_routes(world_state, batch)
    if not routes:
        return
    _enqueue_fleet_tasks(world_state, blackboard, routes, source="auto", announce=False)


def _interleave_station_route_groups(
    requests: list[tuple[dict, dict, int]],
) -> list[tuple[dict, dict]]:
    """Round-robin across station groups so the queue mixes legs (spread workload).

    Without this, ``5x Parts->Assembly`` then ``2x Assembly->QA`` runs every
    Parts leg before any Assembly pickup — looks like one station at a time.
    """
    buckets: list[list[tuple[dict, dict]]] = [
        [(pickup_ws, delivery_ws)] * count
        for pickup_ws, delivery_ws, count in requests
        if count > 0
    ]
    routes: list[tuple[dict, dict]] = []
    while True:
        progressed = False
        for bucket in buckets:
            if bucket:
                routes.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    return routes


def _apply_station_pickup_quota(task: dict, world_state: dict) -> None:
    """Count 'N tasks for Station' when the worker reaches that station (pickup)."""
    if task.get("_pickup_quota_applied"):
        return
    pickup_name = str(task.get("pickup_name", ""))
    quotas = world_state.get("station_quotas") or {}
    if pickup_name in quotas:
        quotas[pickup_name]["completed"] = quotas[pickup_name].get("completed", 0) + 1
    task["_pickup_quota_applied"] = True


def _generate_fleet_routes(
    world_state: dict,
    count: int,
    *,
    pickup_ws: dict | None = None,
    delivery_ws: dict | None = None,
) -> list[tuple[dict, dict]]:
    """Return a list of (pickup, delivery) routes for the leader to queue.

    If both endpoints are given, every route is identical — the leader just
    needs N workers to make that same trip. Otherwise we cycle through the
    natural Parts->Assembly->QA->Shipping flow so all stations stay busy.
    """
    stations = {ws["name"]: ws for ws in world_state.get("workstations", [])}
    if pickup_ws and delivery_ws:
        return [(pickup_ws, delivery_ws)] * max(0, count)

    flow = [stations[name] for name in _FLEET_ROUTE_FLOW if name in stations]
    if len(flow) < 2:
        # Should never happen with the default layout, but be defensive.
        return []

    routes: list[tuple[dict, dict]] = []
    for i in range(max(0, count)):
        src = flow[i % len(flow)]
        dst = flow[(i + 1) % len(flow)]
        routes.append((src, dst))
    return routes


def _fleet_pick_worker_nearest(
    task: dict,
    idle_workers: list[dict],
    wall_set: set,
    world_state: dict,
) -> tuple[dict | None, int]:
    """Pick the idle worker closest to this task's pickup (spread + less convoy)."""
    if not idle_workers:
        return None, 10**6
    best_worker: dict | None = None
    best_cost = 10**6
    for worker in sorted(idle_workers, key=lambda r: r["id"]):
        cost = _estimate_task_cost(worker, task, wall_set, world_state)
        if cost >= 10**6:
            continue
        if cost < best_cost or (cost == best_cost and best_worker is not None and worker["id"] < best_worker["id"]):
            best_cost = cost
            best_worker = worker
    if best_worker is None:
        return None, 10**6
    return best_worker, best_cost


def _fleet_dispatch_step(world_state: dict, blackboard: Blackboard) -> int:
    """One tick of the autonomous fleet loop.

    Drains queued tasks in order, assigning each to the idle worker nearest
    that task's pickup (deterministic tie-break by robot id). Returns the number
    of tasks dispatched this tick.
    """
    queue: list[dict] = world_state.get("task_queue", []) or []
    if not queue:
        return 0

    leaders = [r for r in world_state.get("robots", []) if r.get("role") == LEADER]
    if not leaders:
        return 0
    leader = leaders[0]

    idle_workers = _workers_available_for_dispatch(world_state)
    if not idle_workers:
        return 0

    engine = nemoclaw.get_engine()
    if engine is not None:
        tick_decision = engine.check_action(
            "leader",
            "fleet_dispatch_tick",
            extra={"queue_len": len(queue), "idle_workers": len(idle_workers)},
        )
        try:
            engine.enforce_decision(tick_decision)
        except nemoclaw.PolicyDenied:
            return 0

    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    dispatched = 0

    while queue and idle_workers:
        task = queue[0]
        worker, cost = _fleet_pick_worker_nearest(
            task, idle_workers, wall_set, world_state,
        )
        if worker is None or cost >= 10**6:
            # No reachable worker for this task right now — don't pop, retry next tick.
            break

        # NemoClaw gate per delegation.
        if engine is not None:
            decision = engine.check_action(
                "leader",
                "delegate_task",
                target=f"task-{task['id']} {task.get('pickup_name','?')}->{task.get('delivery_name','?')} -> W{worker['id']}",
                extra={"cost": cost, "queue_remaining": len(queue) - 1},
            )
            try:
                engine.enforce_decision(decision)
            except nemoclaw.PolicyDenied:
                queue.pop(0)
                blackboard.post(
                    leader["id"],
                    "POLICY",
                    f"OpenClaw leader: NemoClaw denied delegate_task {task['id']} -> W{worker['id']}; dropped.",
                )
                continue

        # Move task from queue to active list and assign.
        queue.pop(0)
        idle_workers.remove(worker)
        task["status"] = "open"
        if task not in world_state.setdefault("tasks", []):
            world_state["tasks"].append(task)
        _assign_task_to_robot(worker, task, world_state)
        task["delegated_by"] = leader["id"]
        task["master_order"] = True
        task["estimated_route_cost"] = cost
        worker["master_assigned"] = task["id"]

        route = f"{task.get('pickup_name', '?')}->{task.get('delivery_name', '?')}"
        blackboard.post(
            leader["id"],
            "CLAIM",
            f"MASTER -> W{worker['id']}: task {task['id']} {route} "
            f"(queue #{dispatched + 1}, {len(queue)} left).",
        )
        blackboard.post(
            worker["id"],
            "INTENT",
            f"W{worker['id']} accepting fleet assignment {route} task {task['id']} (cost={cost}).",
        )
        print(
            f"[fleet leader {leader['id']}] dispatched task {task['id']} ({route}) -> "
            f"worker {worker['id']} cost={cost} queue_remaining={len(queue)}",
            flush=True,
        )
        dispatched += 1

    if dispatched and not queue:
        blackboard.post(
            leader["id"],
            "STRATEGY",
            "OpenClaw leader: fleet queue drained; awaiting completions before idling.",
        )
    return dispatched


# ---------------------------------------------------------------------------
# Demo / integration helpers
# ---------------------------------------------------------------------------

def _station_for_user_command(text: str, world_state: dict) -> dict | None:
    """Return the workstation named in a simple human command, if any."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    wanted = next((name for word, name in _station_aliases().items() if word in words), None)
    if wanted is None:
        return None
    return _station_by_name(wanted, world_state)


def _station_aliases() -> dict[str, str]:
    return {
        "parts": "Parts",
        "part": "Parts",
        "assembly": "Assembly",
        "assemble": "Assembly",
        "assembled": "Assembly",
        "qa": "QA",
        "quality": "QA",
        "check": "QA",
        "shipping": "Shipping",
        "ship": "Shipping",
        "shipment": "Shipping",
    }


def _station_by_name(name: str, world_state: dict) -> dict | None:
    return next(
        (ws for ws in world_state.get("workstations", []) if ws.get("name") == name),
        None,
    )


def _default_downstream_station(pickup_name: str, world_state: dict) -> dict | None:
    """Return the normal warehouse flow target for a vague command."""
    flow = {
        "Parts": "Assembly",
        "Assembly": "QA",
        "QA": "Shipping",
        "Shipping": "Parts",
    }
    target = flow.get(pickup_name)
    return _station_by_name(target, world_state) if target else None


def _station_mentions_in_order(text: str, world_state: dict) -> list[dict]:
    """Return distinct workstation mentions in the order the user typed them."""
    lower = text.lower()
    mentions: list[tuple[int, str]] = []
    for alias, canonical in _station_aliases().items():
        match = re.search(rf"\b{re.escape(alias)}\b", lower)
        if match:
            mentions.append((match.start(), canonical))
    ordered: list[dict] = []
    seen: set[str] = set()
    for _, name in sorted(mentions):
        if name in seen:
            continue
        station = _station_by_name(name, world_state)
        if station:
            ordered.append(station)
            seen.add(name)
    return ordered


def _station_mentions_with_positions(text: str, world_state: dict) -> list[tuple[int, int, dict]]:
    """Return workstation mentions as (start, end, station), preserving repeats."""
    lower = text.lower()
    mentions: list[tuple[int, int, str]] = []
    for alias, canonical in _station_aliases().items():
        for match in re.finditer(rf"\b{re.escape(alias)}\b", lower):
            mentions.append((match.start(), match.end(), canonical))

    ordered: list[tuple[int, int, dict]] = []
    seen_spans: set[tuple[int, int, str]] = set()
    for start, end, name in sorted(mentions):
        key = (start, end, name)
        if key in seen_spans:
            continue
        station = _station_by_name(name, world_state)
        if station:
            ordered.append((start, end, station))
            seen_spans.add(key)
    return ordered


_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}


def _count_requested_in_text(text: str, default: int = 1) -> int | str:
    """Return a small worker count, or 'all'/'rest' for relative requests."""
    lower = text.lower()
    if re.search(r"\b(rest|remaining|others)\b", lower):
        return "rest"
    if re.search(r"\b(all|everyone|everybody)\b", lower):
        return "all"
    for word, count in _COUNT_WORDS.items():
        if re.search(rf"\b{word}\b", lower):
            return count
    # NEW: support fleet-scale numbers (e.g. "do 100 deliveries").
    number = re.search(r"\b(\d{1,4})\b", lower)
    return int(number.group(1)) if number else default


_FLEET_TASK_COUNT_LIMIT = 500


def _requested_task_count(text: str) -> int:
    count = _count_requested_in_text(text)
    if not isinstance(count, int):
        return 1
    return max(1, min(count, _FLEET_TASK_COUNT_LIMIT))


def _count_from_token(token: str) -> int:
    """Parse a numeric or word count from a regex capture group."""
    token = (token or "").strip().lower()
    if not token:
        return 1
    if token.isdigit():
        return max(1, min(int(token), _FLEET_TASK_COUNT_LIMIT))
    return max(1, min(_COUNT_WORDS.get(token, 1), _FLEET_TASK_COUNT_LIMIT))


# Matches: "do 5 tasks for Parts", "5 for assembly", "2 deliveries from QA"
_STATION_QUOTA_CLAUSE_RE = re.compile(
    r"(?:\bdo\s+)?"
    r"(?P<count>\d{1,4}|one|two|three|four|five|six|seven|eight)\s*"
    r"(?:tasks?|deliveries?|jobs?|trips?|runs?)?\s*"
    r"(?:for|at|from|on)\s+"
    r"(?P<alias>parts?|part|assembly|assemble\w*|qa|quality|shipping|ship)\b",
    re.IGNORECASE,
)


def _parse_station_quota_clause(
    clause: str,
    world_state: dict,
) -> tuple[dict, dict, int] | None:
    """Parse 'N tasks for Station' when delivery is the default downstream stop."""
    if _is_go_and_stop_command(clause):
        return None
    match = _STATION_QUOTA_CLAUSE_RE.search(clause)
    if not match:
        return None
    canonical = _station_aliases().get(match.group("alias").lower())
    if not canonical:
        return None
    pickup = _station_by_name(canonical, world_state)
    delivery = _default_downstream_station(canonical, world_state) if pickup else None
    if not pickup or not delivery:
        return None
    count = _count_from_token(match.group("count"))
    return pickup, delivery, count


def _parse_station_quota_requests(
    text: str,
    world_state: dict,
) -> list[tuple[dict, dict, int]]:
    """Parse multi-station counts like '5 tasks for Parts and 2 for Assembly'."""
    if _is_go_and_stop_command(text):
        return []
    requests: list[tuple[dict, dict, int]] = []
    for match in _STATION_QUOTA_CLAUSE_RE.finditer(text):
        clause = match.group(0)
        parsed = _parse_station_quota_clause(clause, world_state)
        if parsed:
            requests.append(parsed)
    return requests


def _reset_station_quota_targets(world_state: dict) -> None:
    """Clear user-requested per-station targets (completed counts are kept until reset)."""
    world_state["station_quotas"] = empty_station_quotas()


def _apply_station_quota_targets(
    world_state: dict,
    requests: list[tuple[dict, dict, int]],
) -> None:
    """Record how many pickups the user asked for at each station."""
    quotas = world_state.setdefault("station_quotas", empty_station_quotas())
    for pickup_ws, _delivery_ws, count in requests:
        name = pickup_ws.get("name", "")
        if name not in quotas:
            continue
        quotas[name]["target"] = quotas[name].get("target", 0) + count


def _station_quota_in_flight(world_state: dict, station_name: str) -> int:
    """Tasks still queued or active with this pickup station."""
    total = 0
    for task in world_state.get("task_queue", []) or []:
        if task.get("pickup_name") == station_name and task.get("status") != "done":
            total += 1
    for task in world_state.get("tasks", []) or []:
        if task.get("pickup_name") == station_name and task.get("status") in (
            "open", "in_transit", "queued",
        ):
            total += 1
    return total


def _parse_user_task_request(text: str, world_state: dict) -> tuple[dict, dict, int] | None:
    """Parse commands like 'move Parts to Assembly' into a real task route."""
    if _is_go_and_stop_command(text):
        return None
    stations = _station_mentions_in_order(text, world_state)
    if len(stations) < 2:
        return None

    lower = text.lower()
    has_task_verb = bool(
        re.search(r"\b(move|send|route|deliver|carry|transport|take|bring|ship|build)\b", lower)
    )
    has_route_word = bool(re.search(r"\b(to|from|into|toward|->)\b", lower))
    if not (has_task_verb or has_route_word):
        return None

    # Prefer explicit "from X to Y" if present; otherwise use first mention -> second mention.
    aliases = "|".join(re.escape(alias) for alias in _station_aliases())
    explicit = re.search(
        rf"\bfrom\s+(?P<src>{aliases})\b.*\bto\s+(?P<dst>{aliases})\b",
        lower,
    )
    if explicit:
        src_name = _station_aliases()[explicit.group("src")]
        dst_name = _station_aliases()[explicit.group("dst")]
        pickup = _station_by_name(src_name, world_state)
        delivery = _station_by_name(dst_name, world_state)
    else:
        pickup, delivery = stations[0], stations[1]

    if not pickup or not delivery or pickup["name"] == delivery["name"]:
        return None
    return pickup, delivery, _requested_task_count(text)


def _parse_user_task_requests(
    text: str,
    world_state: dict,
    blackboard: "Blackboard | None" = None,
) -> list[tuple[dict, dict, int]]:
    """Parse one or more route clauses from a user prompt.

    LLM interpretation is tried first (USE_LLM_TASK_INTERPRETER, default on).
    Regex parsing is only a fallback when the interpreter is off or unavailable.
    """
    if _is_go_and_stop_command(text):
        return []

    use_llm = os.getenv("USE_LLM_TASK_INTERPRETER", "true").lower() == "true"
    if use_llm and not _use_mock_agents():
        interpreted = _interpret_user_task_requests(text, world_state, blackboard)
        if interpreted:
            return interpreted

    return _parse_user_task_requests_regex(text, world_state)


def _use_mock_agents() -> bool:
    return os.getenv("AGENTS_USE_MOCK", "false").lower() == "true"


def _parse_user_task_requests_regex(
    text: str, world_state: dict,
) -> list[tuple[dict, dict, int]]:
    """Regex fallback when the LLM task interpreter is off or failed."""
    if _is_go_and_stop_command(text):
        return []

    quota_requests = _parse_station_quota_requests(text, world_state)
    if quota_requests:
        return quota_requests

    worker_count = max(1, sum(1 for r in world_state.get("robots", []) if r.get("role") == WORKER))
    remaining = worker_count
    requests: list[tuple[dict, dict, int]] = []
    clauses = [
        clause.strip()
        for clause in re.split(r"\b(?:and|then|plus)\b|[,;]", text, flags=re.IGNORECASE)
        if clause.strip()
    ]
    for clause in clauses:
        parsed = _parse_user_task_request(clause, world_state)
        if parsed is None:
            parsed = _parse_station_quota_clause(clause, world_state)
        if parsed is not None:
            pickup, delivery, count = parsed
            if re.search(r"\b(rest|remaining|others)\b", clause, flags=re.IGNORECASE):
                count = max(1, remaining)
            elif re.search(r"\b(all|everyone|everybody)\b", clause, flags=re.IGNORECASE):
                count = worker_count
            # Fleet mode: allow large queue requests (e.g. "100 deliveries");
            # only clamp small "5 workers"-style asks to the worker count.
            if count <= worker_count:
                count = max(1, count)
            else:
                count = min(count, _FLEET_TASK_COUNT_LIMIT)
            remaining = max(0, remaining - min(count, remaining))
            requests.append((pickup, delivery, count))

    if requests:
        return requests

    parsed = _parse_user_task_request(text, world_state)
    if parsed is not None:
        return [parsed]

    return []


def _interpret_user_task_requests(
    text: str,
    world_state: dict,
    blackboard: "Blackboard | None" = None,
) -> list[tuple[dict, dict, int]]:
    """Ask the interpreter model for structured mission groups (exact counts)."""
    parsed = interpret_user_tasks(text, world_state)
    if not parsed:
        return []

    worker_count = max(1, sum(1 for r in world_state.get("robots", []) if r.get("role") == WORKER))
    idle_worker_count = sum(
        1
        for r in world_state.get("robots", [])
        if r.get("role") == WORKER and r.get("current_task") is None and not r.get("path")
    )
    station_names = {ws.get("name"): ws for ws in world_state.get("workstations", [])}
    remaining = worker_count
    requests: list[tuple[dict, dict, int]] = []
    confidence = _coerce_float(parsed.get("confidence"), 0.0)
    if parsed.get("needs_clarification") and confidence < 0.85:
        question = str(parsed.get("clarifying_question", "")).strip()
        msg = question or "Could not interpret command — try: do 5 tasks for Parts and 2 for Assembly"
        print(f"[task-interpreter] clarification needed: {msg}", flush=True)
        if blackboard is not None:
            blackboard.post(-1, "STRATEGY", f"Task interpreter: {msg}")
        return []

    for item in parsed.get("tasks", []):
        if not isinstance(item, dict):
            continue
        pickup = station_names.get(str(item.get("pickup", "")).strip())
        delivery_name = str(item.get("delivery", "")).strip()
        delivery = station_names.get(delivery_name)
        if pickup and not delivery:
            delivery = _default_downstream_station(pickup["name"], world_state)
        if not pickup or not delivery or pickup["name"] == delivery["name"]:
            continue

        raw_count = item.get("count", 1)
        if isinstance(raw_count, str) and raw_count.lower() in {
            "all_idle_workers",
            "all_workers",
            "everyone",
            "all",
        }:
            count = idle_worker_count or worker_count
        elif isinstance(raw_count, str) and raw_count.lower() == "rest":
            count = remaining
        else:
            try:
                count = int(float(raw_count))
            except (TypeError, ValueError):
                count = 1
            count = max(1, min(count, _FLEET_TASK_COUNT_LIMIT))
        remaining = max(0, remaining - min(count, remaining))
        requests.append((pickup, delivery, count))

    if requests:
        summary = "; ".join(
            f"{cnt}x {pu['name']}->{de['name']}" for pu, de, cnt in requests
        )
        strategy = str(parsed.get("strategy", "fifo_exact_counts"))
        assumptions = parsed.get("assumptions", [])
        print(
            f"[task-interpreter] LLM plan: {summary}; "
            f"confidence={confidence:.2f}; strategy={strategy}; assumptions={assumptions}",
            flush=True,
        )
        if blackboard is not None:
            blackboard.post(
                -1,
                "STRATEGY",
                f"Master understood: {summary} (interleaved in the queue, exact counts).",
            )
    return requests


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _create_user_task(world_state: dict, pickup_ws: dict, delivery_ws: dict) -> dict:
    """Create a high-priority task from a user command."""
    task = _create_task(world_state, pickup_ws, delivery_ws, source="user")
    task["priority"] = 100
    return task


def _parse_bulk_fleet_request(text: str, world_state: dict) -> int:
    """Detect 'do 100 tasks' / '50 deliveries' style fleet commands.

    Returns the requested task count, or 0 if the text does not look like a
    bulk fleet request. Used as a fallback when no specific stations were
    named — we then auto-generate a Parts->Assembly->QA->Shipping cycle.
    """
    lower = text.lower()
    has_bulk_word = bool(
        re.search(
            r"\b(task|tasks|delivery|deliveries|run|runs|trip|trips|job|jobs|drop|drops|order|orders)\b",
            lower,
        )
    )
    if not has_bulk_word:
        return 0
    number = re.search(r"\b(\d{1,4})\b", lower)
    if not number:
        return 0
    return max(1, min(int(number.group(1)), _FLEET_TASK_COUNT_LIMIT))


def _apply_user_task_request(
    text: str,
    world_state: dict,
    blackboard: Blackboard,
) -> bool:
    """Convert a natural-language task request into queued fleet work.

    Tasks are pushed onto ``world_state["task_queue"]`` in order. The master
    leader drains the queue with deterministic round-robin worker assignment.
    """
    requests = _parse_user_task_requests(text, world_state, blackboard)
    bulk_count = 0
    if not requests:
        # Only use cycle fallback for explicit bulk with no station names (not vague chat).
        bulk_count = _parse_bulk_fleet_request(text, world_state)
        if bulk_count <= 0:
            if os.getenv("USE_LLM_TASK_INTERPRETER", "true").lower() == "true":
                blackboard.post(
                    -1,
                    "STRATEGY",
                    "Could not interpret that command. Example: "
                    "'do 5 tasks for Parts, and 2 for Assembly'.",
                )
            return False

    _cancel_pending_decisions()

    # Drop any in-flight assignments; queued work will replace them.
    for robot in world_state.get("robots", []):
        robot["current_task"] = None
        robot["path"] = []

    for task in world_state.get("tasks", []):
        if task.get("status") != "done":
            task["status"] = "cancelled"
            task["assigned_to"] = None

    # Reset the queue for this command (additive feels surprising when the
    # user just said "do 100 deliveries"). A "+100" prefix can be wired up
    # later if additive enqueue is preferred.
    world_state["task_queue"] = []
    world_state["fleet_dispatch_cursor"] = 0
    world_state["fleet_mode"] = True
    _reset_station_quota_targets(world_state)

    routes: list[tuple[dict, dict]] = []
    route_summaries: list[str] = []

    if requests:
        _apply_station_quota_targets(world_state, requests)
        routes = _interleave_station_route_groups(requests)
        for pickup_ws, delivery_ws, count in requests:
            route_summaries.append(f"{count}x {pickup_ws['name']}->{delivery_ws['name']}")
    else:
        # Bulk fleet request: rotate through the whole flow.
        routes = _generate_fleet_routes(world_state, bulk_count)
        route_summaries.append(f"{bulk_count}x cycle({'->'.join(_FLEET_ROUTE_FLOW)})")

    enqueued = _enqueue_fleet_tasks(world_state, blackboard, routes, source="user")
    dispatched = _fleet_dispatch_step(world_state, blackboard)
    world_state["manual_control"] = False
    summary = "; ".join(route_summaries)
    blackboard.post(
        -1,
        "STRATEGY",
        (
            f"OpenClaw fleet accepted: {summary}; interleaved queue; queued={enqueued}, "
            f"first wave dispatched={dispatched}, queue_remaining="
            f"{len(world_state.get('task_queue', []))}."
        ),
    )
    print(
        f"[command] queued {enqueued} fleet task(s) ({summary}); "
        f"first dispatch={dispatched}; "
        f"queue_remaining={len(world_state.get('task_queue', []))}",
        flush=True,
    )
    return True


def _is_go_and_stop_command(text: str) -> bool:
    """Detect direct movement commands that should override autonomous work."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    has_motion = bool(words & {"go", "move", "send", "park", "route"})
    has_stop = bool(words & {"stop", "hold", "wait", "park"})
    return has_motion and has_stop


def _station_hold_targets(station_pos: list[int], world_state: dict, count: int) -> list[list[int]]:
    """Return nearby legal cells so robots visibly gather around a station."""
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    sx, sy = station_pos
    offsets = [
        (0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (2, 0), (0, 2), (-2, 0), (0, -2),
    ]
    targets: list[list[int]] = []
    for dx, dy in offsets:
        cell = [sx + dx, sy + dy]
        if not (0 <= cell[0] < GRID_WIDTH and 0 <= cell[1] < GRID_HEIGHT):
            continue
        if tuple(cell) in wall_set:
            continue
        targets.append(cell)
        if len(targets) >= count:
            break
    return targets or [station_pos]


def _parse_worker_hold_requests(text: str, world_state: dict) -> list[tuple[dict, int]]:
    """Parse commands like 'two workers to Shipping, two workers to Parts'."""
    lower = text.lower()
    words = set(re.findall(r"[a-z0-9]+", lower))
    has_motion = bool(words & {"go", "move", "send", "park", "route", "hold", "wait"})
    if not has_motion:
        return []

    mentions = _station_mentions_with_positions(text, world_state)
    if not mentions:
        return []

    worker_count = max(1, sum(1 for r in world_state.get("robots", []) if r.get("role") == WORKER))
    has_worker_scope = bool(words & {"worker", "workers", "robot", "robots", "bot", "bots", "r2d2"})
    has_count = bool(re.search(r"\b(?:[1-8]|one|two|three|four|five|six|seven|eight|all|everyone|everybody|rest|remaining|others)\b", lower))
    has_hold_word = bool(words & {"park", "hold", "wait"})
    if not (has_worker_scope or has_count or has_hold_word):
        return []

    requests: list[tuple[dict, int]] = []
    remaining = worker_count
    previous_end = 0
    for start, end, station in mentions:
        chunk = text[previous_end:start]
        raw_count = _count_requested_in_text(chunk, default=1)
        if raw_count == "all":
            count = remaining
        elif raw_count == "rest":
            count = remaining
        else:
            count = int(raw_count)
        count = max(0, min(count, remaining))
        if count:
            requests.append((station, count))
            remaining -= count
        previous_end = end
        if remaining <= 0:
            break

    return requests


def _apply_worker_hold_requests(
    requests: list[tuple[dict, int]],
    world_state: dict,
    blackboard: Blackboard,
) -> bool:
    """Route workers to requested station hold positions with global nearest matching."""
    if not requests:
        return False

    _cancel_pending_decisions()
    workers = [r for r in world_state.get("robots", []) if r.get("role") == WORKER]
    if not workers:
        return False

    for robot in workers:
        robot["current_task"] = None
        robot["path"] = []

    for task in world_state.get("tasks", []):
        if task.get("status") != "done":
            task["status"] = "cancelled"
            task["assigned_to"] = None

    targets: list[tuple[dict, list[int]]] = []
    for station, count in requests:
        for target in _station_hold_targets(list(station["pos"]), world_state, count):
            targets.append((station, target))
            if sum(1 for s, _ in targets if s["name"] == station["name"]) >= count:
                break

    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    unassigned_workers = list(workers)
    assignments: list[tuple[dict, dict, list[int], int]] = []
    for station, target in targets:
        if not unassigned_workers:
            break
        worker = min(
            unassigned_workers,
            key=lambda r: (_route_cost(r["pos"], target, wall_set), r["id"]),
        )
        cost = _route_cost(worker["pos"], target, wall_set)
        if cost >= 10**6:
            continue
        assignments.append((worker, station, target, cost))
        unassigned_workers.remove(worker)

    for worker, station, target, _ in assignments:
        worker["path"] = _planned_path_to(worker, target, world_state)
        worker["hold_target"] = target
        worker["hold_station"] = station["name"]

    world_state["manual_control"] = True
    summary_counts: dict[str, int] = {}
    for _, station, _, _ in assignments:
        summary_counts[station["name"]] = summary_counts.get(station["name"], 0) + 1
    summary = "; ".join(f"{count} worker(s) to {name}" for name, count in summary_counts.items())
    blackboard.post(
        -1,
        "STRATEGY",
        f"Manual split command: {summary}.",
    )
    print(
        f"[command] split hold: {summary}",
        flush=True,
    )
    return bool(assignments)


def _apply_go_to_station_and_stop(
    text: str,
    world_state: dict,
    blackboard: Blackboard,
) -> bool:
    """Handle hard commands like 'go to Parts and stop' without LLM routing."""
    hold_requests = _parse_worker_hold_requests(text, world_state)
    if len(hold_requests) > 1 or (
        hold_requests and re.search(r"\b(worker|workers|robot|robots|bot|bots)\b", text, flags=re.IGNORECASE)
    ):
        return _apply_worker_hold_requests(hold_requests, world_state, blackboard)

    station = _station_for_user_command(text, world_state)
    if station is None or not _is_go_and_stop_command(text):
        return False

    _cancel_pending_decisions()
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    robots = world_state.get("robots", [])
    targets = _station_hold_targets(list(station["pos"]), world_state, len(robots))
    for index, robot in enumerate(robots):
        robot["current_task"] = None
        robot["path"] = a_star(robot["pos"], targets[index % len(targets)], wall_set)

    for task in world_state.get("tasks", []):
        if task.get("status") != "done":
            task["status"] = "cancelled"
            task["assigned_to"] = None

    world_state["manual_control"] = True
    blackboard.post(
        -1,
        "STRATEGY",
        f"Manual command: all robots moving to {station['name']} and holding there.",
    )
    print(
        f"[command] hard stop: routing {len(robots)} robots to {station['name']} "
        f"hold positions near {station['pos']}",
        flush=True,
    )
    return True

def _reset_demo_state(layout: str, speed_multiplier: int, connection_status: str) -> dict:
    """Create a fresh run while preserving demo controls."""
    global _task_counter
    _task_counter = 0
    world_state = S.create_initial_state(layout)
    world_state["speed_multiplier"] = speed_multiplier
    world_state["connection_status"] = connection_status
    world_state["manual_control"] = False
    return world_state


def _bootstrap_demo_run(world_state: dict, blackboard: Blackboard) -> None:
    """Start the sim and keep a backlog of cycle tasks (optional AUTO_START off)."""
    global _sim_started
    _sim_started = True
    world_state["fleet_mode"] = True
    if os.getenv("FACTORYMIND_AUTO_START", "true").lower() in ("false", "0", "no"):
        blackboard.post(
            -1,
            "STRATEGY",
            "Autonomous fleet off (FACTORYMIND_AUTO_START=false). Toggle env to run.",
        )
        return
    _top_up_fleet_queue(world_state, blackboard)
    num_workers = max(
        1,
        sum(1 for r in world_state.get("robots", []) if r.get("role") == WORKER),
    )
    for _ in range(num_workers + 4):
        if _fleet_dispatch_step(world_state, blackboard) == 0:
            break
    blackboard.post(
        -1,
        "STRATEGY",
        "Autonomous fleet running — use Builder to add/remove walls anytime.",
    )


def _cycle_speed(speed_multiplier: int) -> int:
    """Return the next demo speed in the 1x -> 2x -> 4x cycle."""
    index = SPEED_OPTIONS.index(speed_multiplier)
    return SPEED_OPTIONS[(index + 1) % len(SPEED_OPTIONS)]


def _set_inference_mode(use_local_nim: bool) -> None:
    """Rebuild inference clients so disconnect mode cannot call cloud endpoints."""
    os.environ["USE_LOCAL_NIM"] = "true" if use_local_nim else "false"
    os.environ.setdefault("GX10_IP", "localhost")
    if "factorymind.inference" in sys.modules:
        import factorymind.inference as inference
        importlib.reload(inference)


def _local_nim_available(timeout: float = 0.25) -> bool:
    """Return True when the configured local NIM ports accept TCP connections."""
    gx10_ip = os.getenv("GX10_IP", "localhost").strip() or "localhost"
    shared_url = (os.getenv("LOCAL_NIM_BASE_URL") or os.getenv("NIM_BASE_URL") or "").strip()
    leader_url = os.getenv("NIM_LEADER_BASE_URL") or shared_url or f"http://{gx10_ip}:8000/v1"
    strategist_url = (
        os.getenv("NIM_STRATEGIST_BASE_URL")
        or shared_url
        or f"http://{gx10_ip}:8001/v1"
    )
    return all(_endpoint_accepts_tcp(url, timeout) for url in {leader_url, strategist_url})


def _endpoint_accepts_tcp(url: str, timeout: float) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _is_duplicate_directive(blackboard: Blackboard, directive: str) -> bool:
    """Avoid filling the blackboard and memory with identical repeated directives."""
    recent = blackboard.read_recent(6)
    return any(str(msg.get("content", "")) == directive for msg in recent)


def _strategy_outcome_summary(world_state: dict, reasoning: str) -> str:
    """Store the live context that produced a strategist directive."""
    stats = world_state.get("stats", {})
    open_tasks = sum(1 for t in world_state["tasks"] if t.get("status") == "open")
    in_transit = sum(1 for t in world_state["tasks"] if t.get("status") == "in_transit")
    return (
        f"Recorded during live run: completed={stats.get('completed', 0)}, "
        f"rate={stats.get('rate', 0)} tasks/min, open={open_tasks}, "
        f"in_transit={in_transit}. Strategist reasoning: {reasoning[:180]}"
    )


# ---------------------------------------------------------------------------
# Decision application (called from main thread once a future completes)
# ---------------------------------------------------------------------------

def _apply_leader_decision(
    leader: dict,
    decision: dict,
    world_state: dict,
    blackboard: Blackboard,
) -> None:
    """Mutate world_state and blackboard based on a completed leader decision."""
    from factorymind.state import CLAIM

    claimed_id = decision.get("claim_task_id")
    fleet_mode = bool(world_state.get("fleet_mode") or world_state.get("task_queue"))
    if claimed_id is not None and not fleet_mode:
        # Outside fleet mode the leader may still grab a single task itself
        # (legacy behavior for "take Parts to QA" with no queue). In fleet
        # mode the OpenClaw dispatcher owns assignment and the leader stays
        # parked at its dispatch position to think.
        task = next(
            (t for t in world_state["tasks"]
             if t["id"] == claimed_id and t["status"] == "open"
             and t.get("assigned_to") is None),
            None,
        )
        if task:
            _assign_task_to_robot(leader, task, world_state)
            _delegate_task_to_nearest_worker(leader, task, world_state, blackboard)

    msg = decision.get("post_message", "")
    if msg:
        blackboard.post(leader["id"], CLAIM, msg)
        print(f"[leader {leader['id']}] {msg}", flush=True)
    reasoning = str(decision.get("reasoning", "")).strip()
    if reasoning:
        blackboard.post(leader["id"], "REASONING", reasoning[:140])
        print(f"[leader {leader['id']} reasoning] {reasoning[:180]}", flush=True)


def _apply_worker_decision(
    worker: dict,
    decision: dict,
    world_state: dict,
    blackboard: Blackboard,
) -> None:
    """Mutate state based on a per-worker LLM decision (each ball is an agent)."""
    from factorymind.state import CLAIM, INTENT

    # Master dispatch mode: workers only follow assignments from the fleet queue.
    if _FLEET_MASTER_ONLY and _master_controls_fleet(world_state):
        return

    claimed_id = decision.get("claim_task_id")
    if claimed_id is not None:
        task = next(
            (t for t in world_state["tasks"]
             if t["id"] == claimed_id and t["status"] == "open"),
            None,
        )
        if task:
            assigned_to = task.get("assigned_to")
            if assigned_to in (None, worker["id"]):
                _assign_task_to_robot(worker, task, world_state)
            else:
                leader = next(
                    (r for r in world_state["robots"] if r["id"] == assigned_to),
                    None,
                )
                worker_dist = abs(worker["pos"][0] - task["pickup"][0]) + \
                              abs(worker["pos"][1] - task["pickup"][1])
                leader_dist = (
                    abs(leader["pos"][0] - task["pickup"][0]) +
                    abs(leader["pos"][1] - task["pickup"][1])
                    if leader else 10**9
                )
                if worker_dist < leader_dist:
                    _assign_task_to_robot(worker, task, world_state)

    msg = decision.get("post_message", "")
    if msg:
        blackboard.post(worker["id"], INTENT, msg)
        print(f"[worker {worker['id']}] {msg}", flush=True)
    reasoning = str(decision.get("reasoning", "")).strip()
    if reasoning:
        blackboard.post(worker["id"], "REASONING", reasoning[:140])


def _assign_idle_workers(world_state: dict, blackboard: Blackboard) -> None:
    """Rule-based worker dispatch — disabled when the master owns the fleet queue."""
    if _FLEET_MASTER_ONLY and _master_controls_fleet(world_state):
        return
    idle_workers = [
        r for r in world_state["robots"]
        if r["role"] == WORKER and r.get("current_task") is None and not r.get("path")
    ]
    for worker in idle_workers:
        task_id = choose_worker_task(worker, world_state, blackboard)
        if task_id is None:
            continue
        task = next(
            (t for t in world_state["tasks"]
             if t["id"] == task_id and t["status"] == "open"),
            None,
        )
        if task:
            _assign_task_to_robot(worker, task, world_state)


def _apply_strategist_decision(
    decision: dict,
    world_state: dict,
    blackboard: Blackboard,
) -> None:
    """Post strategist directive and persist it for future runs."""
    from factorymind.state import BOTTLENECK, STRATEGY

    directive = decision.get("directive", "").strip()
    if not (decision.get("should_post") and directive):
        return
    if _is_duplicate_directive(blackboard, directive):
        return
    msg_type = BOTTLENECK if "bottleneck" in directive.lower() else STRATEGY
    blackboard.post(-1, msg_type, directive)
    print(f"[strategist/{msg_type}] {directive}", flush=True)
    reasoning_text = str(decision.get("reasoning", "")).strip()
    if reasoning_text:
        blackboard.post(-1, "REASONING", reasoning_text[:140])
        print(f"[strategist reasoning] {reasoning_text[:180]}", flush=True)
    M.record_strategy(
        world_state["layout"],
        directive,
        _strategy_outcome_summary(world_state, str(decision.get("reasoning", ""))),
    )


def _drain_completed_decisions(world_state: dict, blackboard: Blackboard) -> None:
    """Apply any leader/worker/strategist futures that finished since the last frame."""
    global _pending_strategist

    robot_lookup = {r["id"]: r for r in world_state["robots"]}

    for rid in list(_pending_leader.keys()):
        fut = _pending_leader[rid]
        if not fut.done():
            continue
        del _pending_leader[rid]
        leader = robot_lookup.get(rid)
        if leader is None or leader.get("role") != LEADER:
            continue
        try:
            decision = fut.result()
        except Exception as exc:
            print(f"[main] leader {rid} decision failed: {exc}", file=sys.stderr)
            continue
        _apply_leader_decision(leader, decision, world_state, blackboard)

    for rid in list(_pending_worker.keys()):
        fut = _pending_worker[rid]
        if not fut.done():
            continue
        del _pending_worker[rid]
        worker = robot_lookup.get(rid)
        if worker is None or worker.get("role") != WORKER:
            continue
        try:
            decision = fut.result()
        except Exception as exc:
            print(f"[main] worker {rid} decision failed: {exc}", file=sys.stderr)
            continue
        _apply_worker_decision(worker, decision, world_state, blackboard)

    if _pending_strategist is not None and _pending_strategist.done():
        fut = _pending_strategist
        _pending_strategist = None
        try:
            decision = fut.result()
        except Exception as exc:
            print(f"[main] strategist decision failed: {exc}", file=sys.stderr)
        else:
            _apply_strategist_decision(decision, world_state, blackboard)


def _cancel_pending_decisions() -> None:
    """Drop in-flight futures (e.g. on layout reset) so stale results don't apply."""
    global _pending_strategist
    for fut in _pending_leader.values():
        fut.cancel()
    _pending_leader.clear()
    for fut in _pending_worker.values():
        fut.cancel()
    _pending_worker.clear()
    if _pending_strategist is not None:
        _pending_strategist.cancel()
        _pending_strategist = None


def _pending_decision_count() -> int:
    return (
        len(_pending_leader)
        + len(_pending_worker)
        + (1 if _pending_strategist is not None else 0)
    )


def _wait_for_pending_decisions(
    world_state: dict,
    blackboard: Blackboard,
    *,
    max_wait: float,
) -> None:
    """Let in-flight model calls finish during precompute before advancing far."""
    deadline = time.time() + max_wait
    while _pending_decision_count() and time.time() < deadline:
        _drain_completed_decisions(world_state, blackboard)
        if _pending_decision_count():
            time.sleep(0.05)
    _drain_completed_decisions(world_state, blackboard)


def _snapshot_world_state(world_state: dict, blackboard: Blackboard) -> dict:
    """Return an immutable-ish snapshot for replay."""
    world_state["blackboard"] = blackboard.to_list()
    return copy.deepcopy(world_state)


def _precompute_timeline(duration: float) -> list[dict]:
    """Run the factory headlessly first, recording snapshots for later replay."""
    global _pending_strategist

    _cancel_pending_decisions()
    layout = os.getenv("LAYOUT", OPEN_FLOOR)
    speed_multiplier = 1
    world_state = _reset_demo_state(layout, speed_multiplier, "online")
    blackboard = Blackboard()

    snapshot_interval = float(os.getenv("FACTORYMIND_REPLAY_SNAPSHOT_INTERVAL", "0.2"))
    precompute_dt = float(os.getenv("FACTORYMIND_PRECOMPUTE_DT", "0.1"))
    wait_default = float(os.getenv("NEMOTRON_TIMEOUT_SECONDS", "60.0")) + 1.0
    decision_wait = float(os.getenv("FACTORYMIND_PRECOMPUTE_WAIT_SECONDS", str(wait_default)))

    sim_elapsed = 0.0
    last_leader_tick = 0.0
    last_worker_tick = 0.0
    last_strategist_tick = 0.0
    last_move_tick = 0.0
    next_snapshot = 0.0
    timeline: list[dict] = []

    print(
        f"[precompute] computing {duration:.1f}s timeline "
        f"(snapshot every {snapshot_interval:.1f}s, ALL_AGENTS_LLM={_ALL_AGENTS_LLM})",
        flush=True,
    )

    while sim_elapsed <= duration:
        now = sim_elapsed
        _drain_completed_decisions(world_state, blackboard)

        dispatched = False
        if now - last_leader_tick >= LEADER_TICK_INTERVAL:
            for leader in [r for r in world_state["robots"] if r["role"] == LEADER]:
                if leader["id"] not in _pending_leader:
                    _pending_leader[leader["id"]] = _decision_executor.submit(
                        leader_decide, leader, world_state, blackboard,
                    )
                    dispatched = True
            last_leader_tick = now

        if now - last_worker_tick >= WORKER_TICK_INTERVAL:
            if _ALL_AGENTS_LLM:
                for worker in [r for r in world_state["robots"] if r["role"] == WORKER]:
                    if worker["id"] in _pending_worker:
                        continue
                    if worker.get("current_task") is not None:
                        continue
                    _pending_worker[worker["id"]] = _decision_executor.submit(
                        worker_decide, worker, world_state, blackboard,
                    )
                    dispatched = True
            else:
                _assign_idle_workers(world_state, blackboard)
            last_worker_tick = now

        if (
            now - last_strategist_tick >= STRATEGIST_TICK_INTERVAL
            and _pending_strategist is None
        ):
            try:
                state_description = M.describe_world_state(world_state)
                retrieved = M.retrieve_strategies(
                    world_state["layout"],
                    state_description=state_description,
                )
            except Exception as exc:
                print(f"[precompute] strategy retrieval failed: {exc}", file=sys.stderr, flush=True)
                retrieved = []
            _pending_strategist = _decision_executor.submit(
                strategist_decide, world_state, blackboard, retrieved,
            )
            dispatched = True
            last_strategist_tick = now

        if dispatched:
            _wait_for_pending_decisions(
                world_state,
                blackboard,
                max_wait=decision_wait,
            )

        if now - last_move_tick >= MOVE_TICK_INTERVAL:
            move_steps = min(int((now - last_move_tick) // MOVE_TICK_INTERVAL), 4)
            for _ in range(max(1, move_steps)):
                _advance_robots(world_state, blackboard)
            last_move_tick += max(1, move_steps) * MOVE_TICK_INTERVAL

        world_state["speed_multiplier"] = speed_multiplier
        world_state["stats"]["elapsed"] = round(sim_elapsed, 1)
        completed = world_state["stats"]["completed"]
        world_state["stats"]["rate"] = (
            round(completed / sim_elapsed * 60, 1) if sim_elapsed > 0 else 0.0
        )
        world_state["tick"] += 1

        if sim_elapsed >= next_snapshot:
            timeline.append(_snapshot_world_state(world_state, blackboard))
            next_snapshot += snapshot_interval

        if int(sim_elapsed * 10) % 50 == 0:
            print(
                f"[precompute] sim_t={sim_elapsed:.1f}s snapshots={len(timeline)} "
                f"completed={world_state['stats']['completed']} "
                f"messages={len(blackboard.to_list())}",
                flush=True,
            )
        sim_elapsed = round(sim_elapsed + precompute_dt, 4)

    _wait_for_pending_decisions(world_state, blackboard, max_wait=decision_wait)
    timeline.append(_snapshot_world_state(world_state, blackboard))
    _cancel_pending_decisions()
    print(f"[precompute] ready: {len(timeline)} replay frames", flush=True)
    return timeline


def _replay_timeline(timeline: list[dict]) -> None:
    """Render a precomputed timeline without any model calls during playback."""
    if not timeline:
        print("[replay] no frames to replay", file=sys.stderr, flush=True)
        return

    print("[replay] initializing pygame display...", flush=True)
    screen = R.init_display()
    clock = pygame.time.Clock()
    replay_fps = float(os.getenv("FACTORYMIND_REPLAY_FPS", "12"))
    seconds_per_frame = 1.0 / replay_fps if replay_fps > 0 else 0.0
    frame_index = 0
    last_step = time.time()
    running = True
    print(
        f"[replay] playing {len(timeline)} frames at {replay_fps:g} fps; "
        "no LLM calls happen during replay",
        flush=True,
    )

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        now = time.time()
        if seconds_per_frame == 0.0 or now - last_step >= seconds_per_frame:
            frame_index = min(frame_index + 1, len(timeline) - 1)
            last_step = now

        R.render(screen, timeline[frame_index])
        if frame_index >= len(timeline) - 1:
            # Keep the final state visible for judges until they close it.
            pass
        clock.tick(FPS)

    pygame.quit()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _print_startup_banner() -> None:
    """Print a clear start-of-run banner so freezes are easy to diagnose."""
    print("=" * 72, flush=True)
    print("FactoryMind R2D2 — starting simulation", flush=True)
    try:
        from factorymind import inference
        print(inference.describe_endpoints(), flush=True)
    except Exception as exc:
        print(f"  (could not describe inference endpoints: {exc})", flush=True)
    print(
        f"  AGENTS_USE_MOCK={os.getenv('AGENTS_USE_MOCK', 'false')}  "
        f"ALL_AGENTS_LLM={os.getenv('ALL_AGENTS_LLM', 'true')}  "
        f"USE_LLM_TASK_INTERPRETER={os.getenv('USE_LLM_TASK_INTERPRETER', 'true')}  "
        f"COOPERATIVE_PATHING={os.getenv('COOPERATIVE_PATHING', 'true')}  "
        f"USE_LOCAL_NIM={os.getenv('USE_LOCAL_NIM', 'true')}  "
        f"GX10_IP={os.getenv('GX10_IP', 'localhost')}",
        flush=True,
    )
    print("  thread pool: 12 workers (leaders/strategist use async LLM futures)", flush=True)
    print("=" * 72, flush=True)


def _auto_fallback_to_mock_if_endpoints_dead() -> None:
    """If neither cloud nor local NIM is reachable, force AGENTS_USE_MOCK=true.

    This keeps the demo moving even when the inference endpoints are gone, and
    means a misconfigured `.env` never produces a frozen window.
    """
    if os.getenv("AGENTS_USE_MOCK", "false").lower() == "true":
        return  # already in mock mode, nothing to check

    use_local = os.getenv("USE_LOCAL_NIM", "true").lower() == "true"
    if use_local:
        if _local_nim_available(timeout=0.5):
            return
        reason = "local NIM (port 8000/8001) not reachable"
    else:
        # Cloud mode: a missing API key is the usual cause of silent hangs.
        key = os.getenv("NVIDIA_API_KEY", "").strip()
        if key and key != "your_key_here":
            return
        reason = "NVIDIA_API_KEY not set"

    print(
        f"[main] {reason}; falling back to AGENTS_USE_MOCK=true so the demo runs.",
        flush=True,
    )
    os.environ["AGENTS_USE_MOCK"] = "true"


def main() -> None:
    """Run the FactoryMind simulation."""
    global _pending_strategist, _sim_started

    _print_startup_banner()
    _auto_fallback_to_mock_if_endpoints_dead()

    precompute_seconds = float(os.getenv("FACTORYMIND_PRECOMPUTE_SECONDS", "0") or 0)
    if precompute_seconds > 0:
        timeline = _precompute_timeline(precompute_seconds)
        _replay_timeline(timeline)
        _decision_executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(0)

    # --- Init ---
    layout = os.getenv("LAYOUT", OPEN_FLOOR)
    speed_multiplier = 1
    initial_use_local = os.getenv("USE_LOCAL_NIM", "true").lower() == "true"
    initial_use_mock = os.getenv("AGENTS_USE_MOCK", "false").lower() == "true"
    max_runtime = float(os.getenv("FACTORYMIND_MAX_RUNTIME", "0") or 0)
    world_state = _reset_demo_state(layout, speed_multiplier, "online")
    blackboard = Blackboard()
    engine = nemoclaw.activate(blackboard=blackboard, world_state=world_state)
    blackboard.post(-1, "POLICY", f"OpenClaw runtime active: {engine.describe()}")
    blackboard.post(
        -1,
        "STRATEGY",
        "FactoryMind ready — autonomous fleet runs continuously. "
        "Use Builder to add walls; set FACTORYMIND_AUTO_START=false to pause.",
    )
    _bootstrap_demo_run(world_state, blackboard)
    manual_control = False

    print("[main] initializing pygame display...", flush=True)
    screen = R.init_display()
    print("[main] pygame ready, entering main loop", flush=True)
    clock = pygame.time.Clock()

    # Timers — start at -(interval) so first tick fires immediately on sim start
    sim_elapsed          = 0.0
    last_frame_time      = time.time()
    run_start_time       = last_frame_time
    last_leader_tick     = -LEADER_TICK_INTERVAL
    last_worker_tick     = -WORKER_TICK_INTERVAL
    last_strategist_tick = 0.0
    last_move_tick       = -MOVE_TICK_INTERVAL
    last_heartbeat       = time.time()
    heartbeat_seconds    = float(os.getenv("FACTORYMIND_HEARTBEAT_SECONDS", "10"))
    frames_since_beat    = 0

    running = True
    while running:
        wall_now = time.time()
        real_delta = wall_now - last_frame_time
        last_frame_time = wall_now

        if _sim_started:
            sim_elapsed += real_delta * speed_multiplier
        now = sim_elapsed

        # --- Heartbeat (proves the main loop is alive) ---
        frames_since_beat += 1
        if heartbeat_seconds > 0 and wall_now - last_heartbeat >= heartbeat_seconds:
            in_flight = (
                len(_pending_leader)
                + len(_pending_worker)
                + (1 if _pending_strategist else 0)
            )
            print(
                f"[main] alive  fps={frames_since_beat / (wall_now - last_heartbeat):.0f}  "
                f"sim_t={sim_elapsed:.1f}s  in_flight={in_flight}  "
                f"completed={world_state['stats'].get('completed', 0)}  started={_sim_started}",
                flush=True,
            )
            last_heartbeat = wall_now
            frames_since_beat = 0

        # --- Pygame events (use handle_event for all input) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            result = R.handle_event(event, world_state)
            if result is None:
                continue

            if result == "disconnect":
                from factorymind.state import STRATEGY
                _cancel_pending_decisions()
                if world_state["connection_status"] == "online":
                    _set_inference_mode(True)
                    world_state["connection_status"] = "offline"
                    local_ready = _local_nim_available()
                    os.environ["AGENTS_USE_MOCK"] = (
                        "true" if initial_use_mock or not local_ready else "false"
                    )
                    fallback = (
                        "local endpoints are reachable."
                        if local_ready
                        else "local endpoints are down, so mock fallback is active."
                    )
                    blackboard.post(
                        -1, STRATEGY,
                        f"Cloud disconnect: forcing local DGX Spark NIM; {fallback}",
                    )
                else:
                    _set_inference_mode(initial_use_local)
                    os.environ["AGENTS_USE_MOCK"] = "true" if initial_use_mock else "false"
                    world_state["connection_status"] = "online"
                    blackboard.post(
                        -1, STRATEGY,
                        "Demo reconnect: inference mode restored to startup configuration.",
                    )

            elif result == "reset":
                _cancel_pending_decisions()
                world_state = _reset_demo_state(OPEN_FLOOR, speed_multiplier, world_state["connection_status"])
                blackboard.clear()
                # Re-bind NemoClaw to the fresh world_state so policy stats
                # accrue against the active simulation.
                nemoclaw.activate(blackboard=blackboard, world_state=world_state)
                _bootstrap_demo_run(world_state, blackboard)
                manual_control = False
                sim_elapsed = 0.0
                last_leader_tick = -LEADER_TICK_INTERVAL
                last_worker_tick = -WORKER_TICK_INTERVAL
                last_strategist_tick = 0.0
                last_move_tick = -MOVE_TICK_INTERVAL

            elif result == "speedup":
                speed_multiplier = _cycle_speed(speed_multiplier)
                world_state["speed_multiplier"] = speed_multiplier

            elif result in ("mode_cursor", "mode_builder"):
                pass  # render.py manages _mode state

            elif isinstance(result, dict):
                action_type = result.get("type")
                if action_type == "wall_add":
                    cell = result["cell"]
                    wall_list = world_state.get("wall", [])
                    if cell not in wall_list:
                        wall_list.append(cell)
                        world_state["wall"] = wall_list
                        world_state["layout"] = "CUSTOM"
                elif action_type == "wall_remove":
                    cell = result["cell"]
                    wall_list = list(world_state.get("wall", []))
                    if cell in wall_list:
                        wall_list.remove(cell)
                        world_state["wall"] = wall_list

        # --- Drain completed LLM futures (always, to flush startup strategist) ---
        _drain_completed_decisions(world_state, blackboard)

        # --- Sim logic: only runs while started ---
        if _sim_started:
            if not manual_control:
                master_only = _FLEET_MASTER_ONLY and _master_controls_fleet(world_state)
                if not master_only:
                    # --- Leader ticks (dispatch async LLM calls) ---
                    if now - last_leader_tick >= LEADER_TICK_INTERVAL:
                        for leader in [r for r in world_state["robots"] if r["role"] == LEADER]:
                            if leader["id"] in _pending_leader:
                                continue
                            _pending_leader[leader["id"]] = _decision_executor.submit(
                                leader_decide, leader, world_state, blackboard,
                            )
                        last_leader_tick = now

                    # --- Worker ticks (only when master is NOT controlling the fleet) ---
                    if now - last_worker_tick >= WORKER_TICK_INTERVAL:
                        if _ALL_AGENTS_LLM:
                            workers = [r for r in world_state["robots"] if r["role"] == WORKER]
                            for worker in workers:
                                if worker["id"] in _pending_worker:
                                    continue
                                if worker.get("current_task") is not None:
                                    continue
                                _pending_worker[worker["id"]] = _decision_executor.submit(
                                    worker_decide, worker, world_state, blackboard,
                                )
                        else:
                            _assign_idle_workers(world_state, blackboard)
                        last_worker_tick = now

                    if (
                        now - last_strategist_tick >= STRATEGIST_TICK_INTERVAL
                        and _pending_strategist is None
                    ):
                        try:
                            state_description = M.describe_world_state(world_state)
                            retrieved = M.retrieve_strategies(
                                world_state["layout"],
                                state_description=state_description,
                            )
                        except Exception as exc:
                            print(f"[main] strategy retrieval failed: {exc}", file=sys.stderr, flush=True)
                            retrieved = []
                        _pending_strategist = _decision_executor.submit(
                            strategist_decide, world_state, blackboard, retrieved,
                        )
                        last_strategist_tick = now

            # --- Master fleet dispatch (deterministic; runs every tick while queued) ---
            if world_state.get("task_queue"):
                _fleet_dispatch_step(world_state, blackboard)

            # Keep backlog full while building / playing (not in manual go-to-station mode).
            if not manual_control:
                _top_up_fleet_queue(world_state, blackboard)

            # --- Movement tick ---
            if now - last_move_tick >= MOVE_TICK_INTERVAL:
                move_steps = min(int((now - last_move_tick) // MOVE_TICK_INTERVAL), 4)
                for _ in range(max(1, move_steps)):
                    _advance_robots(world_state, blackboard)
                last_move_tick += max(1, move_steps) * MOVE_TICK_INTERVAL
                if world_state.get("task_queue"):
                    _fleet_dispatch_step(world_state, blackboard)

            # --- Stats ---
            elapsed = sim_elapsed
            world_state["stats"]["elapsed"] = round(elapsed, 1)
            completed = world_state["stats"]["completed"]
            world_state["stats"]["rate"] = round(completed / elapsed * 60, 1) if elapsed > 0 else 0.0

        # --- Always sync blackboard + speed + tick, then render ---
        world_state["blackboard"] = blackboard.to_list()
        world_state["speed_multiplier"] = speed_multiplier
        world_state["tick"] += 1

        try:
            R.render(screen, world_state)
        except Exception as exc:
            print(f"[main] render failed: {exc}", file=sys.stderr, flush=True)
        if max_runtime and (
            sim_elapsed >= max_runtime
            or wall_now - run_start_time >= max_runtime
        ):
            running = False
        clock.tick(FPS)

    _cancel_pending_decisions()
    _decision_executor.shutdown(wait=False, cancel_futures=True)
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
