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

# Defaults favor the full agentic demo: every idle worker gets its own model
# decision, while path execution stays deterministic and locally verifiable.
_ALL_AGENTS_LLM = os.getenv("ALL_AGENTS_LLM", "true").lower() == "true"
_COOPERATIVE_PATHING = os.getenv("COOPERATIVE_PATHING", "true").lower() == "true"

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
_sim_started: bool = False  # True once user sends first chat prompt

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
    for robot in world_state.get("robots", []):
        if robot.get("id") == exclude_robot_id:
            continue
        positions = [list(robot.get("pos", [0, 0]))] + [list(p) for p in robot.get("path", [])]
        for t, pos in enumerate(positions):
            cells.setdefault(t, set()).add(tuple(pos))
            if t > 0:
                edges.setdefault(t, set()).add((tuple(positions[t - 1]), tuple(pos)))
        if positions:
            last = tuple(positions[-1])
            for t in range(len(positions), len(positions) + 12):
                cells.setdefault(t, set()).add(last)
    return {"cells": cells, "edges": edges}


def _planned_path_to(
    robot: dict,
    goal: list[int],
    world_state: dict,
    *,
    ignore_reservations: bool = False,
) -> list[list[int]]:
    """Return a route that accounts for walls and, by default, other robots."""
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    if _COOPERATIVE_PATHING and not ignore_reservations:
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
    if pos == task["pickup"] and task["status"] == "open":
        task["status"] = "in_transit"
        robot["path"] = _planned_path_to(robot, task["delivery"], world_state)

    # If at delivery — task done, then stop until the next user task.
    elif pos == task["delivery"] and task["status"] == "in_transit":
        route_time = round(time.time() - task.get("started_at", time.time()), 2)
        task["status"] = "done"
        robot["current_task"] = None

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


def _advance_robots_coordinated(world_state: dict, blackboard: "Blackboard") -> None:
    """Move robots one step with simple agent reservations to avoid collisions.

    Each robot proposes its next cell from its own path. If two robots want the
    same cell, or two robots would swap cells head-on, one waits and posts a
    brief blackboard message. This keeps path execution distributed while still
    making coordination visible.
    """
    robots = world_state.get("robots", [])
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
    for cell, ids in by_cell.items():
        if len(ids) <= 1:
            continue
        winner = min(ids)
        denied.update(rid for rid in ids if rid != winner)

    for rid, target in proposals.items():
        if rid in denied:
            continue
        for other_id, other_target in proposals.items():
            if other_id <= rid or other_id in denied:
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
            ),
            None,
        )
        if blocker is not None:
            denied.add(rid)

    now = time.time()
    for robot in robots:
        rid = robot["id"]
        if rid in denied:
            if now - robot.get("last_yield_post", 0.0) > 1.0:
                blackboard.post(
                    rid,
                    "INTENT",
                    f"R{rid} yielding one step to avoid a route conflict.",
                )
                robot["last_yield_post"] = now
            continue
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


def _requested_task_count(text: str) -> int:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    word_counts = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    for word, count in word_counts.items():
        if word in words:
            return count
    number = re.search(r"\b([1-5])\b", text)
    return int(number.group(1)) if number else 1


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


def _parse_user_task_requests(text: str, world_state: dict) -> list[tuple[dict, dict, int]]:
    """Parse one or more route clauses from a user prompt.

    Examples:
    - "deliver 4 Parts to QA"
    - "move 2 Assembly to Shipping and 2 Parts to QA"
    """
    if _is_go_and_stop_command(text):
        return []

    if os.getenv("USE_LLM_TASK_INTERPRETER", "true").lower() == "true":
        interpreted = _interpret_user_task_requests(text, world_state)
        if interpreted:
            return interpreted

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
        if parsed is not None:
            pickup, delivery, count = parsed
            if re.search(r"\b(rest|remaining|others)\b", clause, flags=re.IGNORECASE):
                count = max(1, remaining)
            elif re.search(r"\b(all|everyone|everybody)\b", clause, flags=re.IGNORECASE):
                count = worker_count
            count = max(1, min(count, worker_count))
            remaining = max(0, remaining - count)
            requests.append((pickup, delivery, count))

    if requests:
        return requests

    parsed = _parse_user_task_request(text, world_state)
    if parsed is not None:
        return [parsed]

    inferred = _infer_vague_task_requests(text, world_state)
    if inferred:
        return inferred

    return []


def _interpret_user_task_requests(text: str, world_state: dict) -> list[tuple[dict, dict, int]]:
    """Ask the interpreter model for structured mission groups."""
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
    if parsed.get("needs_clarification") and confidence < 0.7:
        question = str(parsed.get("clarifying_question", "")).strip()
        print(
            f"[task-interpreter] clarification needed: {question or 'no safe warehouse action inferred'}",
            flush=True,
        )
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
                count = int(raw_count)
            except (TypeError, ValueError):
                count = 1
        count = max(1, min(count, worker_count))
        remaining = max(0, remaining - count)
        requests.append((pickup, delivery, count))

    if requests:
        strategy = str(parsed.get("strategy", "minimize_average_completion_time"))
        constraints = parsed.get("constraints", [])
        assumptions = parsed.get("assumptions", [])
        print(
            f"[task-interpreter] parsed {len(requests)} route group(s); "
            f"intent={parsed.get('intent', 'create_delivery_tasks')}; "
            f"confidence={confidence:.2f}; strategy={strategy}; "
            f"constraints={constraints}; assumptions={assumptions}",
            flush=True,
        )
    return requests


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_vague_task_requests(text: str, world_state: dict) -> list[tuple[dict, dict, int]]:
    """Local fallback for vague commands when the LLM interpreter is unavailable."""
    lower = text.lower()
    actionish = bool(
        re.search(
            r"\b(clear|empty|backlog|catch\s*up|speed|faster|optimi[sz]e|rebalance|get\s+.*moving|busy|priority)\b",
            lower,
        )
    )
    if not actionish:
        return []

    worker_count = max(1, sum(1 for r in world_state.get("robots", []) if r.get("role") == WORKER))
    idle_count = sum(
        1
        for r in world_state.get("robots", [])
        if r.get("role") == WORKER and r.get("current_task") is None and not r.get("path")
    )
    count = idle_count or worker_count
    stations = _station_mentions_in_order(text, world_state)

    if len(stations) >= 2:
        pickup, delivery = stations[0], stations[1]
    elif len(stations) == 1:
        pickup = stations[0]
        delivery = _default_downstream_station(pickup["name"], world_state)
    else:
        pickup = _highest_backlog_station(world_state) or _station_by_name("Parts", world_state)
        delivery = _default_downstream_station(pickup["name"], world_state) if pickup else None

    if not pickup or not delivery or pickup["name"] == delivery["name"]:
        return []

    print(
        f"[task-interpreter] local vague fallback inferred {count}x "
        f"{pickup['name']}->{delivery['name']}",
        flush=True,
    )
    return [(pickup, delivery, count)]


def _highest_backlog_station(world_state: dict) -> dict | None:
    """Return the station with the most open pickup work, if any."""
    counts: dict[str, int] = {}
    for task in world_state.get("tasks", []):
        if task.get("status") == "open":
            name = str(task.get("pickup_name", ""))
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    name, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return _station_by_name(name, world_state) if count > 0 else None


def _create_user_task(world_state: dict, pickup_ws: dict, delivery_ws: dict) -> dict:
    """Create a high-priority task from a user command."""
    task = _create_task(world_state, pickup_ws, delivery_ws, source="user")
    task["priority"] = 100
    return task


def _apply_user_task_request(
    text: str,
    world_state: dict,
    blackboard: Blackboard,
) -> bool:
    """Convert a natural-language task request into structured work robots can execute."""
    requests = _parse_user_task_requests(text, world_state)
    if not requests:
        return False

    _cancel_pending_decisions()

    for robot in world_state.get("robots", []):
        robot["current_task"] = None
        robot["path"] = []

    for task in world_state.get("tasks", []):
        if task.get("status") != "done":
            task["status"] = "cancelled"
            task["assigned_to"] = None

    new_tasks: list[dict] = []
    route_summaries: list[str] = []
    for pickup_ws, delivery_ws, count in requests:
        new_tasks.extend(
            _create_user_task(world_state, pickup_ws, delivery_ws)
            for _ in range(count)
        )
        route_summaries.append(f"{count}x {pickup_ws['name']}->{delivery_ws['name']}")

    world_state["tasks"].extend(new_tasks)
    dispatched = _dispatch_batch_to_workers(new_tasks, world_state, blackboard)
    world_state["manual_control"] = False
    summary = "; ".join(route_summaries)
    blackboard.post(
        -1,
        "STRATEGY",
        (
            f"User task accepted: {summary}; "
            f"{dispatched} worker(s) dispatched now."
        ),
    )
    print(
        f"[command] created {len(new_tasks)} user task(s): {summary}",
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


def _apply_go_to_station_and_stop(
    text: str,
    world_state: dict,
    blackboard: Blackboard,
) -> bool:
    """Handle hard commands like 'go to Parts and stop' without LLM routing."""
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
    if claimed_id is not None:
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
                # Take over from a leader only if we're closer to the pickup.
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
    """Rule-based worker dispatch — runs every leader tick when ALL_AGENTS_LLM=false."""
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
                _advance_robots_coordinated(world_state, blackboard)
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
    blackboard.post(
        -1,
        "STRATEGY",
        "FactoryMind ready — type a delivery like 'take Parts to QA' to create leader-owned work.",
    )
    _sim_started = False
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
                blackboard.post(
                    -1,
                    "STRATEGY",
                    "Simulation reset — type a delivery like 'take Parts to QA' to begin.",
                )
                _sim_started = False
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
                elif action_type == "user_prompt":
                    user_text = result["text"]
                    _sim_started = True
                    blackboard.post(-1, "USER", user_text)
                    if _apply_go_to_station_and_stop(user_text, world_state, blackboard):
                        manual_control = True
                        last_leader_tick = now
                        last_worker_tick = now
                        last_strategist_tick = now
                        continue
                    if _apply_user_task_request(user_text, world_state, blackboard):
                        manual_control = False
                        last_leader_tick = now - LEADER_TICK_INTERVAL
                        last_worker_tick = now - WORKER_TICK_INTERVAL
                        last_strategist_tick = now
                        continue
                    manual_control = False
                    # Trigger immediate strategist response to user command
                    if _pending_strategist is None:
                        try:
                            retrieved = M.retrieve_strategies(
                                world_state["layout"],
                                state_description=M.describe_world_state(world_state),
                            )
                        except Exception:
                            retrieved = []
                        _pending_strategist = _decision_executor.submit(
                            strategist_decide, world_state, blackboard, retrieved, user_text,
                        )
                        last_strategist_tick = now

        # --- Drain completed LLM futures (always, to flush startup strategist) ---
        _drain_completed_decisions(world_state, blackboard)

        # --- Sim logic: only runs while started ---
        if _sim_started:
            if not manual_control:
                # --- Leader ticks (dispatch async LLM calls) ---
                if now - last_leader_tick >= LEADER_TICK_INTERVAL:
                    for leader in [r for r in world_state["robots"] if r["role"] == LEADER]:
                        if leader["id"] in _pending_leader:
                            continue  # previous decision still in flight; don't pile up
                        _pending_leader[leader["id"]] = _decision_executor.submit(
                            leader_decide, leader, world_state, blackboard,
                        )
                    last_leader_tick = now

                # --- Worker ticks: each ball is its own LLM-driven agent ---
                if now - last_worker_tick >= WORKER_TICK_INTERVAL:
                    if _ALL_AGENTS_LLM:
                        workers = [r for r in world_state["robots"] if r["role"] == WORKER]
                        for worker in workers:
                            if worker["id"] in _pending_worker:
                                continue
                            if worker.get("current_task") is not None:
                                continue  # busy executing a claim — let it finish
                            _pending_worker[worker["id"]] = _decision_executor.submit(
                                worker_decide, worker, world_state, blackboard,
                            )
                    else:
                        _assign_idle_workers(world_state, blackboard)
                    last_worker_tick = now

                # --- Strategist tick (one in flight at a time) ---
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

            # --- Movement tick ---
            if now - last_move_tick >= MOVE_TICK_INTERVAL:
                move_steps = min(int((now - last_move_tick) // MOVE_TICK_INTERVAL), 4)
                for _ in range(max(1, move_steps)):
                    _advance_robots_coordinated(world_state, blackboard)
                last_move_tick += max(1, move_steps) * MOVE_TICK_INTERVAL

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
