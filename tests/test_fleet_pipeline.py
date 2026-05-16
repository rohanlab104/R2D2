"""Regression tests for the OpenClaw / NemoClaw autonomous fleet pipeline.

These tests exercise the path the live demo depends on:

1. The leader is wrapped in NemoClaw, every ``delegate_task`` is gated by the
   policy engine, and allow/deny counts roll up into ``world_state.stats``.
2. Natural-language commands like ``"do 100 deliveries"`` parse into the
   right autonomous task count, and the queue is populated correctly.
3. The autonomous fleet dispatcher hands tasks to the closest idle worker
   (telling each worker what to do), advances them along their A* paths,
   and increments the ``completed`` counter as each task finishes.

No real LLM is called — ``AGENTS_USE_MOCK=true`` is forced on import.
"""

from __future__ import annotations

import os

os.environ["AGENTS_USE_MOCK"] = "true"

import pytest

from factorymind import main as M
from factorymind import nemoclaw
from factorymind.agents import Blackboard
from factorymind.state import LEADER, OPEN_FLOOR, WORKER, create_initial_state


@pytest.fixture(autouse=True)
def _fresh_engine():
    """Each test starts with a brand-new global NemoClaw engine."""
    nemoclaw.reset_for_tests()
    M._task_counter = 0
    yield
    nemoclaw.reset_for_tests()


@pytest.fixture
def world() -> tuple[dict, Blackboard]:
    world_state = create_initial_state(OPEN_FLOOR, num_leaders=1, num_workers=4)
    blackboard = Blackboard()
    nemoclaw.activate(blackboard=blackboard, world_state=world_state)
    return world_state, blackboard


# ---------------------------------------------------------------------------
# NemoClaw guardrails
# ---------------------------------------------------------------------------

def test_nemoclaw_allows_delegate_and_denies_exec_shell(world):
    world_state, _ = world
    engine = nemoclaw.get_engine()
    assert engine is not None, "policy engine must be active for the fleet demo"

    allow = engine.check_action("leader", "delegate_task")
    deny = engine.check_action("leader", "exec_shell")

    assert allow.allowed is True
    assert deny.allowed is False
    snap = engine.stats_snapshot()
    assert snap["allow"] >= 1
    assert snap["deny"] >= 1
    assert world_state["stats"]["policy_allowed"] >= 1
    assert world_state["stats"]["policy_denied"] >= 1


def test_nemoclaw_workers_cannot_delegate(world):
    engine = nemoclaw.get_engine()
    decision = engine.check_action("worker", "delegate_task")
    assert decision.allowed is False, (
        "workers must not be able to assign work to other agents; only the "
        "OpenClaw leader is the autonomous fleet manager."
    )


def test_nemoclaw_network_only_allows_local_nim(world):
    engine = nemoclaw.get_engine()
    assert engine.check_network("http://127.0.0.1:8000/v1").allowed is True
    assert engine.check_network("http://localhost:8001/v1").allowed is True
    # build.nvidia.com (or any other host) must NOT be reachable — the
    # disconnect-from-cloud demo depends on this.
    assert engine.check_network("https://build.nvidia.com/v1").allowed is False


# ---------------------------------------------------------------------------
# Task-count parsing — the "autonomous task counts" the user types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("do 100 deliveries", 100),
        ("run 50 trips", 50),
        ("kick off 250 jobs", 250),
        ("please drop 7 orders", 7),
        # Guard the cap so a typo doesn't queue a million tasks.
        ("do 9999 deliveries", 500),
    ],
)
def test_bulk_fleet_request_parses_count(text, expected):
    world_state = create_initial_state(OPEN_FLOOR)
    assert M._parse_bulk_fleet_request(text, world_state) == expected


def test_bulk_fleet_request_ignores_non_fleet_text():
    world_state = create_initial_state(OPEN_FLOOR)
    # No bulk keyword -> not a fleet request.
    assert M._parse_bulk_fleet_request("go to Parts", world_state) == 0
    # No number -> not a fleet request.
    assert M._parse_bulk_fleet_request("do some deliveries", world_state) == 0


def test_count_words_and_numbers_both_work():
    assert M._count_requested_in_text("two workers to Shipping") == 2
    assert M._count_requested_in_text("send three") == 3
    assert M._count_requested_in_text("queue 137 trips") == 137
    assert M._count_requested_in_text("everyone go") == "all"
    assert M._count_requested_in_text("send the rest") == "rest"


# ---------------------------------------------------------------------------
# Queueing — "do N deliveries" must populate the queue with N entries
# ---------------------------------------------------------------------------

def test_apply_user_task_request_queues_full_fleet(world):
    world_state, blackboard = world
    ok = M._apply_user_task_request("do 50 deliveries", world_state, blackboard)

    assert ok is True
    assert world_state["fleet_mode"] is True
    # After the very first dispatch the queue should contain 50 - num_workers
    # entries (each idle worker grabs one task immediately).
    num_workers = sum(1 for r in world_state["robots"] if r["role"] == WORKER)
    assert len(world_state["task_queue"]) == 50 - num_workers
    assert world_state["stats"]["queued_total"] == 50
    # The dispatched tasks live in world_state["tasks"] and are owned by workers.
    dispatched = [t for t in world_state["tasks"] if t.get("assigned_to") is not None]
    assert len(dispatched) == num_workers
    for task in dispatched:
        assigned_robot = next(
            r for r in world_state["robots"] if r["id"] == task["assigned_to"]
        )
        assert assigned_robot["role"] == WORKER, (
            "leader delegates work — only workers should execute tasks in fleet mode"
        )


def test_apply_user_task_request_with_route_clamps_to_limit(world):
    world_state, blackboard = world
    M._apply_user_task_request("send 800 Parts to Assembly", world_state, blackboard)
    # 800 should be clamped to the safety cap of 500.
    assert world_state["stats"]["queued_total"] == 500


# ---------------------------------------------------------------------------
# Autonomous loop — leader commands workers, queue drains, tasks complete
# ---------------------------------------------------------------------------

def test_autonomous_fleet_drains_and_completes(world):
    world_state, blackboard = world

    requested = 12
    M._apply_user_task_request(f"do {requested} deliveries", world_state, blackboard)
    assert world_state["stats"]["queued_total"] == requested

    # Spin the autonomous loop until everything finishes or we time out.
    # ``_fleet_dispatch_step`` runs each iteration so newly-idle workers
    # pick up the next queued task without human intervention.
    max_iterations = 6_000
    for _ in range(max_iterations):
        if (
            not world_state.get("task_queue")
            and world_state["stats"]["completed"] >= requested
        ):
            break
        M._fleet_dispatch_step(world_state, blackboard)
        M._advance_robots(world_state, blackboard)
    else:
        pytest.fail(
            f"fleet did not drain after {max_iterations} ticks; "
            f"completed={world_state['stats']['completed']}, "
            f"queue={len(world_state['task_queue'])}"
        )

    assert world_state["stats"]["completed"] == requested, (
        "every queued task must complete autonomously"
    )
    assert world_state["task_queue"] == [], "queue must end empty"

    # NemoClaw must have approved at least one delegate_task per completed
    # task. The leader gates the fleet_dispatch_tick too, so the allow count
    # is strictly larger than `requested`.
    engine = nemoclaw.get_engine()
    snap = engine.stats_snapshot()
    assert snap["allow"] >= requested
    assert snap["deny"] == 0, (
        "no denial expected with the default policy; leaders, workers, and "
        "strategist all stay within their allow lists"
    )
    assert world_state["stats"]["policy_allowed"] >= requested


def test_leader_does_not_execute_tasks_directly(world):
    """The leader is a dispatcher; only workers should ever carry a task."""
    world_state, blackboard = world
    M._apply_user_task_request("do 6 deliveries", world_state, blackboard)

    # Run a handful of dispatcher + movement ticks.
    for _ in range(400):
        M._fleet_dispatch_step(world_state, blackboard)
        M._advance_robots(world_state, blackboard)
        if not world_state["task_queue"] and all(
            not r.get("path") for r in world_state["robots"]
        ):
            break

    leaders = [r for r in world_state["robots"] if r["role"] == LEADER]
    for leader in leaders:
        assert leader["current_task"] is None, (
            "leader must remain at its dispatch position — workers do the runs"
        )


def test_fleet_dispatch_records_claim_and_intent_messages(world):
    """The leader must visibly tell each worker what to do."""
    world_state, blackboard = world
    M._apply_user_task_request("do 4 deliveries", world_state, blackboard)

    claims = [m for m in blackboard.to_list() if m["type"] == "CLAIM"]
    intents = [m for m in blackboard.to_list() if m["type"] == "INTENT"]
    # First dispatch wave should produce one CLAIM per assigned task and one
    # INTENT acknowledgement per worker accepting the assignment.
    assert len(claims) >= 1
    assert len(intents) >= 1
    assert any(
        kw in m["content"].lower() for m in claims for kw in ("delegating", "master ->")
    ), claims
    assert any("accepting" in m["content"].lower() for m in intents), intents
