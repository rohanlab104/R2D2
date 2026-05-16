"""Smoke tests for the autonomous factory simulation."""

from __future__ import annotations

from factorymind.agents import Blackboard
from factorymind.main import (
    _apply_leader_plan,
    _deterministic_assignments,
    _make_package,
    _move_workers,
)
from factorymind.state import create_initial_state


def test_autonomous_package_delivery_smoke() -> None:
    world_state = create_initial_state()
    blackboard = Blackboard()
    factory = world_state["factories"][0]

    package = _make_package(world_state, factory)
    package["status"] = "pad"
    package["progress"] = factory["belt_length"]
    factory["pad_packages"].append(package)

    assignments = _deterministic_assignments(world_state)
    assert assignments, "Expected leader planner to assign an idle worker"

    _apply_leader_plan(
        world_state,
        {"assignments": assignments, "thought": "test dispatch", "warning": ""},
        blackboard,
        elapsed=0.0,
    )

    elapsed = 0.0
    for _ in range(160):
        elapsed += 0.1
        _move_workers(world_state, elapsed, blackboard)
        if world_state["stats"]["completed"]:
            break

    assert world_state["stats"]["completed"] == 1
    assert world_state["dropboxes"][0]["delivered"] == 1
    assert any(msg["type"] == "COMPLETE" for msg in world_state["thought_log"])


def test_initial_autonomous_layout_counts() -> None:
    world_state = create_initial_state()

    assert len(world_state["factories"]) == 3
    assert len(world_state["dropboxes"]) == 3
    assert len(world_state["workers"]) == 3
    assert world_state["leader"]["role"] == "LEADER"
