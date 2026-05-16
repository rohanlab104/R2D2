"""End-to-end smoke test: runs 50 simulation ticks with mock agents.

Asserts that at least one task gets completed and the blackboard has messages.
"""

from __future__ import annotations

import os
import sys

# Ensure mock mode is on regardless of environment
os.environ["AGENTS_USE_MOCK"] = "true"

import pytest

from factorymind.state import create_initial_state, OPEN_FLOOR, LEADER, CLAIM
from factorymind.agents import Blackboard, leader_decide, strategist_decide


def _a_star_simple(start: list[int], goal: list[int]) -> list[list[int]]:
    """Minimal BFS path (no walls) for use inside the test."""
    from collections import deque
    if start == goal:
        return []
    visited = {tuple(start)}
    queue: deque = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        x, y = pos
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            new_path = path + [[nx, ny]]
            if [nx, ny] == goal:
                return new_path
            visited.add((nx, ny))
            queue.append(([nx, ny], new_path))
    return []


def _spawn_task(world_state: dict, task_id: int) -> dict:
    import random
    stations = world_state["workstations"]
    pickup_ws, delivery_ws = random.sample(stations, 2)
    return {
        "id": task_id,
        "status": "open",
        "pickup": list(pickup_ws["pos"]),
        "delivery": list(delivery_ws["pos"]),
        "pickup_name": pickup_ws["name"],
        "delivery_name": delivery_ws["name"],
        "assigned_to": None,
    }


def _advance_robot(robot: dict, world_state: dict) -> None:
    """Move robot one step, completing tasks when destination reached."""
    path = robot.get("path", [])
    if not path:
        return
    robot["pos"] = path.pop(0)

    task_id = robot.get("current_task")
    if task_id is None:
        return
    task = next((t for t in world_state["tasks"] if t["id"] == task_id), None)
    if task is None:
        return

    pos = robot["pos"]
    if pos == task["pickup"] and task["status"] == "open":
        task["status"] = "in_transit"
        robot["path"] = _a_star_simple(pos, task["delivery"])
    elif pos == task["delivery"] and task["status"] == "in_transit":
        task["status"] = "done"
        robot["current_task"] = None
        robot["path"] = []
        world_state["stats"]["completed"] += 1


def test_smoke_50_ticks() -> None:
    """Run 50 ticks and verify at least one task completes and BB has messages."""
    world_state = create_initial_state(OPEN_FLOOR, num_leaders=2, num_workers=4)
    blackboard = Blackboard()
    task_counter = 0

    for tick in range(50):
        # Spawn a task every 5 ticks
        if tick % 5 == 0:
            task = _spawn_task(world_state, task_counter)
            world_state["tasks"].append(task)
            task_counter += 1

        # Leader decisions every 3 ticks
        if tick % 3 == 0:
            for robot in world_state["robots"]:
                if robot["role"] == LEADER:
                    decision = leader_decide(robot, world_state, blackboard)

                    claimed_id = decision.get("claim_task_id")
                    if claimed_id is not None:
                        task = next(
                            (t for t in world_state["tasks"]
                             if t["id"] == claimed_id and t["status"] == "open"
                             and t.get("assigned_to") is None),
                            None,
                        )
                        if task:
                            task["assigned_to"] = robot["id"]
                            robot["current_task"] = task["id"]
                            robot["path"] = _a_star_simple(robot["pos"], task["pickup"])

                    msg = decision.get("post_message", "")
                    if msg:
                        blackboard.post(robot["id"], CLAIM, msg)

                    # Assign idle workers too
                    open_tasks = [
                        t for t in world_state["tasks"]
                        if t["status"] == "open" and t.get("assigned_to") is None
                    ]
                    idle_workers = [
                        r for r in world_state["robots"]
                        if r["role"] != LEADER and r.get("current_task") is None and not r.get("path")
                    ]
                    for worker, task in zip(idle_workers, open_tasks):
                        task["assigned_to"] = worker["id"]
                        worker["current_task"] = task["id"]
                        worker["path"] = _a_star_simple(worker["pos"], task["pickup"])

        # Move all robots several steps per tick so tasks complete within 50 ticks
        for _ in range(8):
            for robot in world_state["robots"]:
                _advance_robot(robot, world_state)

        world_state["tick"] = tick
        world_state["blackboard"] = blackboard.to_list()

    completed = world_state["stats"]["completed"]
    bb_messages = blackboard.to_list()

    assert completed >= 1, f"Expected at least 1 completed task, got {completed}"
    assert len(bb_messages) >= 1, "Expected at least 1 blackboard message"
