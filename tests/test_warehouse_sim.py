from __future__ import annotations

from factorymind.warehouse import (
    apply_demo_control,
    create_warehouse_state,
    disable_worker,
    step_warehouse,
)


def _run_steps(state: dict, steps: int, dt_ms: int = 100) -> None:
    for _ in range(steps):
        step_warehouse(state, dt_ms)


def test_autonomous_warehouse_completes_tasks() -> None:
    state = create_warehouse_state()

    _run_steps(state, 900, dt_ms=100)

    assert state["items"], "conveyors should spawn items automatically"
    assert state["tasks"], "spawned items should create tasks"
    assert state["metrics"]["completedTasks"] >= 1
    assert any("Leader assigned" in event["message"] for event in state["events"])
    assert any("delivered item" in event["message"] for event in state["events"])


def test_order_spike_queues_tasks_when_workers_are_busy() -> None:
    state = create_warehouse_state()

    apply_demo_control(state, {"type": "spike", "count": 10})
    step_warehouse(state, 100)

    assert len(state["items"]) == 10
    assert len(state["tasks"]) == 10
    assert state["metrics"]["pendingTasks"] >= 6
    assert sum(1 for w in state["workers"] if w["status"] != "idle") == 4


def test_disabled_worker_task_recovers_to_pending() -> None:
    state = create_warehouse_state()

    apply_demo_control(state, {"type": "spike", "count": 1})
    step_warehouse(state, 100)
    assigned = next(w for w in state["workers"] if w["assignedTaskId"])
    task_id = assigned["assignedTaskId"]

    assert disable_worker(state, assigned["id"])

    task = next(t for t in state["tasks"] if t["id"] == task_id)
    assert task["status"] == "pending"
    assert task["assignedWorkerId"] is None
    assert any("disabled" in event["message"] for event in state["events"])
    assert any("reassigned" in event["message"] for event in state["events"])
