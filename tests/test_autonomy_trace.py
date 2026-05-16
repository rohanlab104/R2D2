"""Autonomy trace and ETA speedup checks."""

from __future__ import annotations

import os

os.environ["AGENTS_USE_MOCK"] = "true"
os.environ["USE_LLM_TASK_INTERPRETER"] = "false"

from factorymind import main as sim
from factorymind.agents import Blackboard
from factorymind.state import OPEN_FLOOR, WORKER, create_initial_state


def test_batch_eta_metrics_show_team_speedup() -> None:
    world_state = create_initial_state(OPEN_FLOOR)
    parts = sim._station_by_name("Parts", world_state)
    qa = sim._station_by_name("QA", world_state)
    assert parts and qa

    tasks = [
        {
            "id": i,
            "pickup": list(parts["pos"]),
            "delivery": list(qa["pos"]),
            "priority": 100,
        }
        for i in range(4)
    ]
    workers = [r for r in world_state["robots"] if r["role"] == WORKER]

    metrics = sim._batch_eta_metrics(tasks, workers, set())

    assert metrics["solo_steps"] is not None
    assert metrics["team_steps"] is not None
    assert metrics["solo_steps"] > metrics["team_steps"]
    assert metrics["speedup"] > 1.0


def test_user_task_request_posts_autonomy_evidence() -> None:
    world_state = create_initial_state(OPEN_FLOOR)
    blackboard = Blackboard()

    assert sim._apply_user_task_request(
        "deliver 4 Parts to QA as fast as possible",
        world_state,
        blackboard,
    )

    stats = world_state["stats"]
    assert stats["eta_solo_steps"] > stats["eta_team_steps"]
    assert stats["eta_speedup"] > 1.0

    stages = [item["stage"] for item in world_state["autonomy"]["trace"]]
    assert "OBSERVE" in stages
    assert "PLAN" in stages
    assert "ACT" in stages

    message_types = {msg["type"] for msg in blackboard.to_list()}
    assert "TOOL" in message_types
    assert "FOUND" in message_types
