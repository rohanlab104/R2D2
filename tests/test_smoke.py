"""Smoke tests for the autonomous factory simulation."""

from __future__ import annotations

import math
import time

from factorymind import main as M
from factorymind.agents import Blackboard
from factorymind.main import (
    _apply_leader_plan,
    _drain_worker_llm_decisions,
    _deterministic_assignments,
    _handle_builder_action,
    _make_package,
    _move_workers,
)
from factorymind.state import create_initial_state
from factorymind.web_main import _state_from_uploaded_layout


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


def test_builder_remove_asset_removes_sim_objects() -> None:
    world_state = create_initial_state()

    shelf_cell = list(world_state["wall"][0])
    _handle_builder_action(world_state, {"type": "remove_asset", "cell": shelf_cell})
    assert shelf_cell not in world_state["wall"]

    worker = world_state["workers"][0]
    _handle_builder_action(world_state, {"type": "remove_asset", "cell": list(worker["pos"])})
    assert all(item["id"] != worker["id"] for item in world_state["workers"])
    assert all(item["id"] != worker["id"] for item in world_state["robots"] if item.get("role") == "WORKER")

    factory = world_state["factories"][0]
    _handle_builder_action(world_state, {"type": "remove_asset", "cell": list(factory["pos"])})
    assert all(item["id"] != factory["id"] for item in world_state["factories"])

    dropbox = world_state["dropboxes"][0]
    _handle_builder_action(world_state, {"type": "remove_asset", "cell": list(dropbox["pos"])})
    assert all(item["id"] != dropbox["id"] for item in world_state["dropboxes"])


def test_uploaded_generic_bins_match_nearest_conveyor_color() -> None:
    action = {
        "type": "apply_layout",
        "worker_count": 3,
        "order_volume": 6,
        "layout": {
            "id": "shifted-bin-order",
            "name": "Shifted Bin Order",
            "grid": [["floor"]],
            "layout": {
                "shelves": [],
                "conveyors": [
                    {"id": "red", "x": 5, "y": 5, "packageColor": "Red", "cells": [[5, 5]]},
                    {"id": "blue", "x": 5, "y": 25, "packageColor": "Blue", "cells": [[5, 25]]},
                    {"id": "green", "x": 5, "y": 45, "packageColor": "Green", "cells": [[5, 45]]},
                ],
                "bins": [
                    {"id": "near-blue", "x": 40, "y": 25, "cells": [[40, 25]]},
                    {"id": "near-green", "x": 40, "y": 45, "cells": [[40, 45]]},
                    {"id": "near-red", "x": 40, "y": 5, "cells": [[40, 5]]},
                ],
                "chargers": [],
                "obstacles": [],
                "workerSpawns": [{"id": "spawn", "x": 24, "y": 42, "cells": [[24, 42]]}],
            },
        },
    }

    world_state = _state_from_uploaded_layout(action, 1, "online", "test", False)

    assert [box["color"] for box in world_state["dropboxes"]] == ["Blue", "Green", "Red"]


def test_uploaded_typed_bins_keep_exact_legend_colors() -> None:
    colors = ["Red", "Blue", "Green", "Yellow", "Magenta", "Cyan"]
    action = {
        "type": "apply_layout",
        "worker_count": 3,
        "order_volume": 6,
        "layout": {
            "id": "typed-legend",
            "name": "Typed Legend",
            "grid": [["floor"]],
            "layout": {
                "shelves": [],
                "conveyors": [
                    {
                        "id": f"conveyor-{color.lower()}",
                        "x": 3,
                        "y": 3 + index * 7,
                        "tile": f"conveyor_{color.lower()}",
                        "packageColor": color,
                        "cells": [[3, 3 + index * 7]],
                    }
                    for index, color in enumerate(colors)
                ],
                "bins": [
                    {
                        "id": f"bin-{color.lower()}",
                        "x": 40,
                        "y": 3 + index * 7,
                        "tile": f"bin_{color.lower()}",
                        "packageColor": color,
                        "acceptsType": color,
                        "cells": [[40, 3 + index * 7]],
                    }
                    for index, color in enumerate(colors)
                ],
                "chargers": [],
                "obstacles": [],
                "workerSpawns": [{"id": "spawn", "x": 24, "y": 42, "cells": [[24, 42]]}],
            },
        },
    }

    world_state = _state_from_uploaded_layout(action, 1, "online", "test", False)

    assert [factory["color"] for factory in world_state["factories"]] == colors
    assert [box["color"] for box in world_state["dropboxes"]] == colors

    blue_factory = world_state["factories"][1]
    package = _make_package(world_state, blue_factory)
    package["status"] = "pad"
    package["progress"] = blue_factory["belt_length"]
    blue_factory["pad_packages"].append(package)
    assignments = _deterministic_assignments(world_state)
    assert assignments
    target_box = next(box for box in world_state["dropboxes"] if box["id"] == assignments[0]["dropbox_id"])
    assert package["color"] == "Blue"
    assert target_box["color"] == "Blue"


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


def test_worker_llm_runs_on_task_assignment(monkeypatch) -> None:
    world_state = create_initial_state()
    world_state["llm_ready"] = True
    blackboard = Blackboard()
    factory = world_state["factories"][0]

    package = _make_package(world_state, factory)
    package["status"] = "pad"
    package["progress"] = factory["belt_length"]
    factory["pad_packages"].append(package)

    calls = []

    def fake_worker(prompt: str) -> str:
        calls.append(prompt)
        return '{"intent":"accept_task","reason":"route is clear and package matches bin","confidence":0.91}'

    monkeypatch.setattr("factorymind.inference.ask_worker", fake_worker)
    _apply_leader_plan(
        world_state,
        {"assignments": _deterministic_assignments(world_state), "thought": "", "warning": ""},
        blackboard,
        elapsed=0.0,
    )

    worker = world_state["workers"][0]
    for _ in range(25):
        _drain_worker_llm_decisions(world_state, blackboard, elapsed=0.1)
        if worker.get("last_llm_intent"):
            break
        time.sleep(0.01)

    try:
        assert calls
        assert worker["last_llm_intent"] == "accept_task"
        assert worker["llm_decisions"] == 1
        assert any(event["type"] == "INTENT" and "LLM intent=accept_task" in event["content"] for event in world_state["thought_log"])
    finally:
        M._cancel_pending_decisions()
