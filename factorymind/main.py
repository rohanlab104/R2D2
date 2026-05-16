"""Main simulation loop for FactoryMind R2D2.

Person D owns this file.

Run with mock agents (recommended for initial testing):
    AGENTS_USE_MOCK=true python -m factorymind.main

Run with real Nemotron:
    NVIDIA_API_KEY=<key> python -m factorymind.main
"""

from __future__ import annotations

import heapq
import os
import random
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import pygame

from factorymind import state as S
from factorymind import render as R
from factorymind import memory as M
from factorymind.agents import (
    Blackboard,
    choose_worker_task,
    leader_decide,
    strategist_decide,
    worker_step,
)
from factorymind.state import (
    LEADER, WORKER, OPEN_FLOOR, BOTTLENECK_BRIDGE,
    GRID_WIDTH, GRID_HEIGHT,
)

# ---------------------------------------------------------------------------
# Timing constants (seconds)
# ---------------------------------------------------------------------------
FPS = 60
LEADER_TICK_INTERVAL = 3.0
STRATEGIST_TICK_INTERVAL = 30.0
TASK_SPAWN_INTERVAL = 2.0
MOVE_TICK_INTERVAL = 0.1  # how often robots advance one step

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


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

_task_counter = 0

def _spawn_task(world_state: dict) -> dict:
    """Create a random pickup→delivery task between two different workstations."""
    global _task_counter
    stations = world_state["workstations"]
    pickup_ws, delivery_ws = random.sample(stations, 2)
    task = {
        "id": _task_counter,
        "status": "open",
        "pickup": list(pickup_ws["pos"]),
        "delivery": list(delivery_ws["pos"]),
        "pickup_name": pickup_ws["name"],
        "delivery_name": delivery_ws["name"],
        "assigned_to": None,
    }
    _task_counter += 1
    return task


# ---------------------------------------------------------------------------
# Robot movement helpers
# ---------------------------------------------------------------------------

def _advance_robot(robot: dict, world_state: dict) -> None:
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
        robot["path"] = a_star(
            pos, task["delivery"],
            {tuple(c) for c in world_state.get("wall", [])}
        )

    # If at delivery
    elif pos == task["delivery"] and task["status"] == "in_transit":
        task["status"] = "done"
        robot["current_task"] = None
        robot["path"] = []
        world_state["stats"]["completed"] += 1


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
    robot["current_task"] = task["id"]
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    robot["path"] = a_star(robot["pos"], task["pickup"], wall_set)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the FactoryMind simulation."""
    # --- Init ---
    layout = os.getenv("LAYOUT", OPEN_FLOOR)
    world_state = S.create_initial_state(layout)
    blackboard = Blackboard()

    screen = R.init_display()
    clock = pygame.time.Clock()

    # Timers
    last_leader_tick     = time.time()
    last_strategist_tick = time.time()
    last_task_spawn      = time.time()
    last_move_tick       = time.time()
    sim_start            = time.time()

    running = True
    while running:
        now = time.time()

        # --- Pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                action = R.get_button_click(event.pos)
                if action == "disconnect":
                    world_state["connection_status"] = (
                        "online"
                        if world_state["connection_status"] == "offline"
                        else "offline"
                    )
                elif action == "layout_open":
                    world_state = S.create_initial_state(OPEN_FLOOR)
                    blackboard.clear()
                elif action == "layout_bottleneck":
                    world_state = S.create_initial_state(BOTTLENECK_BRIDGE)
                    blackboard.clear()

        # --- Spawn new tasks ---
        if now - last_task_spawn >= TASK_SPAWN_INTERVAL:
            world_state["tasks"].append(_spawn_task(world_state))
            last_task_spawn = now

        # --- Leader ticks ---
        if now - last_leader_tick >= LEADER_TICK_INTERVAL:
            leaders = [r for r in world_state["robots"] if r["role"] == LEADER]
            for leader in leaders:
                decision = leader_decide(leader, world_state, blackboard)

                # Claim a task
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

                # Post to blackboard
                msg = decision.get("post_message", "")
                if msg:
                    from factorymind.state import CLAIM
                    blackboard.post(leader["id"], CLAIM, msg)

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

            last_leader_tick = now

        # --- Strategist tick ---
        if now - last_strategist_tick >= STRATEGIST_TICK_INTERVAL:
            retrieved = M.retrieve_strategies(world_state["layout"])
            decision = strategist_decide(world_state, blackboard, retrieved)
            if decision.get("should_post"):
                from factorymind.state import BOTTLENECK, STRATEGY
                directive = decision.get("directive", "")
                msg_type = BOTTLENECK if "bottleneck" in directive.lower() else STRATEGY
                blackboard.post(-1, msg_type, directive)
            last_strategist_tick = now

        # --- Movement tick ---
        if now - last_move_tick >= MOVE_TICK_INTERVAL:
            for robot in world_state["robots"]:
                if robot["role"] == WORKER:
                    next_pos = worker_step(robot, world_state, blackboard)
                    if robot.get("path") and next_pos != robot["path"][0]:
                        robot["path"].insert(0, next_pos)
                _advance_robot(robot, world_state)
            last_move_tick = now

        # --- Update shared state from blackboard ---
        world_state["blackboard"] = blackboard.to_list()

        # --- Stats ---
        elapsed = now - sim_start
        world_state["stats"]["elapsed"] = round(elapsed, 1)
        completed = world_state["stats"]["completed"]
        world_state["stats"]["rate"] = round(completed / elapsed * 60, 1) if elapsed > 0 else 0.0
        world_state["tick"] += 1

        # --- Render ---
        R.render(screen, world_state)
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
