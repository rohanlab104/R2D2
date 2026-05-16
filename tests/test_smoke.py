"""Smoke tests for the autonomous factory simulation."""

from __future__ import annotations

import math

from factorymind.agents import Blackboard
from factorymind.main import (
    _apply_leader_plan,
    _deterministic_assignments,
    _handle_builder_action,
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


def test_duplicate_dropbox_uses_closest_matching_bin() -> None:
    world_state = create_initial_state()
    factory = world_state["factories"][0]
    _handle_builder_action(world_state, {"type": "place_dropbox", "cell": [14, 6], "color": factory["color"]})

    package = _make_package(world_state, factory)
    package["status"] = "pad"
    package["progress"] = factory["belt_length"]
    factory["pad_packages"].append(package)

    assignments = _deterministic_assignments(world_state)
    assert assignments
    assert assignments[0]["dropbox_id"] == world_state["dropboxes"][-1]["id"]


def test_builder_stores_rotated_dropbox() -> None:
    world_state = create_initial_state()

    _handle_builder_action(
        world_state,
        {"type": "place_dropbox", "cell": [14, 6], "color": "Red", "rotation_y": math.pi / 2},
    )

    assert math.isclose(world_state["dropboxes"][-1]["rotation_y"], math.pi / 2)


def test_worker_replans_around_traffic_blocker() -> None:
    world_state = create_initial_state()
    blackboard = Blackboard()
    worker = world_state["workers"][0]
    blocker = world_state["workers"][1]

    worker["pos"] = [10, 10]
    worker["status"] = "to_dropbox"
    worker["target_dropbox_id"] = world_state["dropboxes"][0]["id"]
    worker["path"] = [[11, 10], [12, 10]]
    blocker["pos"] = [11, 10]

    for step in range(8):
        _move_workers(world_state, float(step), blackboard)

    assert worker["path"]
    assert worker["path"][0] != [11, 10]
    assert any("rerouting around occupied cells" in event["content"] for event in world_state["thought_log"])
