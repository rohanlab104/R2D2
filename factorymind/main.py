"""Autonomous FactoryMind master/slave simulation loop."""

from __future__ import annotations

import copy
import heapq
import json
import os
import socket
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from factorymind import state as S
from factorymind.agents import Blackboard
from factorymind.state import GRID_HEIGHT, GRID_WIDTH, OPEN_FLOOR

FPS = 60
MOVE_TICK_INTERVAL = 0.12
LEADER_TICK_INTERVAL = 1.25
FACTORY_TICK_INTERVAL = 0.05
CONVEYOR_SPEED = 0.45
MAX_FACTORY_BACKLOG = 4
MAX_CONVEYOR_PACKAGES = 2
SPEED_OPTIONS = (1, 2, 4)
TRAFFIC_REPLAN_TICKS = 8

_decision_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="openclaw")
_pending_leader: Future | None = None
_pending_worker_llm: dict[tuple[int, str, int], Future] = {}
_task_counter = 0

POLICY_PATH = Path("scripts/nemoclaw_policy.yaml")
POLICY_LOG_PATH = Path("logs/nemoclaw.log")


def a_star(start: list[int], goal: list[int], wall_set: set[tuple[int, int]]) -> list[list[int]]:
    """Return a path from start exclusive to goal inclusive using 8-way movement."""
    if start == goal:
        return []
    sx, sy = start
    gx, gy = goal
    open_heap: list[tuple[int, int, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0, 0, 0, (sx, sy)))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
    g_score: dict[tuple[int, int], int] = {(sx, sy): 0}

    def h(x: int, y: int) -> int:
        dx, dy = abs(x - gx), abs(y - gy)
        return 10 * max(dx, dy) + 4 * min(dx, dy)

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        cx, cy = current
        if [cx, cy] == goal:
            path: list[list[int]] = []
            node: tuple[int, int] | None = current
            while node is not None:
                path.append([node[0], node[1]])
                node = came_from[node]
            path.reverse()
            return path[1:]
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                continue
            if (nx, ny) in wall_set:
                continue
            if dx != 0 and dy != 0 and (cx + dx, cy) in wall_set and (cx, cy + dy) in wall_set:
                continue
            step = 14 if dx != 0 and dy != 0 else 10
            ng = g + step
            if ng >= g_score.get((nx, ny), 10**9):
                continue
            g_score[(nx, ny)] = ng
            came_from[(nx, ny)] = current
            heapq.heappush(open_heap, (ng + h(nx, ny), ng, id((nx, ny)), (nx, ny)))
    return []


def _route_len(start: list[int], goal: list[int], wall_set: set[tuple[int, int]]) -> int:
    if start == goal:
        return 0
    path = a_star(start, goal, wall_set)
    return len(path) if path else 10**6


def _set_inference_mode(use_local_nim: bool) -> None:
    os.environ["USE_LOCAL_NIM"] = "true" if use_local_nim else "false"
    if "factorymind.inference" in sys.modules:
        import importlib
        import factorymind.inference as inference
        importlib.reload(inference)


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


def _local_nim_available(timeout: float = 0.2) -> bool:
    gx10_ip = os.getenv("GX10_IP", "localhost").strip() or "localhost"
    shared = (os.getenv("LOCAL_NIM_BASE_URL") or os.getenv("NIM_BASE_URL") or "").strip()
    leader_url = os.getenv("NIM_LEADER_BASE_URL") or shared or f"http://{gx10_ip}:8001/v1"
    worker_url = os.getenv("NIM_WORKER_BASE_URL") or shared or f"http://{gx10_ip}:8000/v1"
    return all(_endpoint_accepts_tcp(url, timeout) for url in {leader_url, worker_url})


def _describe_inference_target() -> str:
    try:
        from factorymind import inference
        return inference.describe_endpoints()
    except Exception as exc:
        return f"Inference target unavailable: {exc}"


def _print_startup_banner() -> str:
    print("=" * 72, flush=True)
    print("FactoryMind 3D - autonomous master/slave factory simulation", flush=True)
    desc = _describe_inference_target()
    print(desc, flush=True)
    print(
        f"  USE_LOCAL_NIM={os.getenv('USE_LOCAL_NIM', 'true')}  "
        f"GX10_IP={os.getenv('GX10_IP', 'localhost')}  "
        f"policy={POLICY_PATH}",
        flush=True,
    )
    print("=" * 72, flush=True)
    return desc


def _load_nemoclaw_policy(world_state: dict) -> None:
    POLICY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if POLICY_PATH.exists():
        text = POLICY_PATH.read_text(encoding="utf-8")
        world_state["policy_loaded"] = True
        world_state["policy_path"] = str(POLICY_PATH)
        world_state["policy_status"] = "loaded"
        _policy_log(world_state, "system", "load_policy", True, f"{POLICY_PATH} ({len(text)} bytes)")
    else:
        world_state["policy_loaded"] = False
        world_state["policy_status"] = "missing"
        _policy_log(world_state, "system", "load_policy", False, f"{POLICY_PATH} missing")


def _policy_log(world_state: dict, agent: str, action: str, allowed: bool, detail: str) -> None:
    entry = {
        "timestamp": time.time(),
        "agent": agent,
        "action": action,
        "allowed": allowed,
        "detail": detail,
    }
    world_state.setdefault("policy_log", []).append(entry)
    if len(world_state["policy_log"]) > 200:
        del world_state["policy_log"][:-200]
    POLICY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    verdict = "ALLOW" if allowed else "DENY"
    with POLICY_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{entry['timestamp']:.3f} {verdict} {agent} {action}: {detail}\n")


def _thought(world_state: dict, msg_type: str, agent: str, content: str,
             elapsed: float, blackboard: Blackboard | None = None) -> None:
    entry = {
        "timestamp": time.time(),
        "elapsed": round(elapsed, 2),
        "type": msg_type,
        "agent": agent,
        "content": content,
    }
    world_state.setdefault("thought_log", []).append(entry)
    if len(world_state["thought_log"]) > 220:
        del world_state["thought_log"][:-220]
    if blackboard is not None:
        blackboard.post(-1 if agent == "Leader" else 0, msg_type, content, elapsed=elapsed)


def _reset_demo_state(speed_multiplier: int, connection_status: str, inference_target: str) -> dict:
    global _task_counter
    _task_counter = 0
    world_state = S.create_initial_state(OPEN_FLOOR)
    world_state["speed_multiplier"] = speed_multiplier
    world_state["connection_status"] = connection_status
    world_state["inference_target"] = inference_target
    _load_nemoclaw_policy(world_state)
    return world_state


def _cycle_speed(speed_multiplier: int) -> int:
    index = SPEED_OPTIONS.index(speed_multiplier)
    return SPEED_OPTIONS[(index + 1) % len(SPEED_OPTIONS)]


def _make_package(world_state: dict, factory: dict) -> dict:
    package_id = world_state["next_package_id"]
    world_state["next_package_id"] += 1
    package = {
        "id": package_id,
        "factory_id": factory["id"],
        "color": factory["color"],
        "color_rgb": list(factory["color_rgb"]),
        "status": "conveyor",
        "progress": 0.0,
        "reserved_by": None,
    }
    world_state.setdefault("packages", []).append(package)
    return package


def _update_factories(world_state: dict, dt: float, elapsed: float, blackboard: Blackboard) -> None:
    for factory in world_state.get("factories", []):
        remaining = []
        for package in factory.get("conveyor", []):
            package["progress"] = float(package.get("progress", 0.0)) + CONVEYOR_SPEED * dt
            if package["progress"] >= float(factory.get("belt_length", 5)):
                package["status"] = "pad"
                package["progress"] = float(factory.get("belt_length", 5))
                factory.setdefault("pad_packages", []).append(package)
            else:
                remaining.append(package)
        factory["conveyor"] = remaining
        backlog = len(factory.get("conveyor", [])) + len(factory.get("pad_packages", []))
        if backlog >= MAX_FACTORY_BACKLOG or len(factory.get("conveyor", [])) >= MAX_CONVEYOR_PACKAGES:
            factory["produce_timer"] = max(float(factory.get("produce_timer", 0.0)), 0.65)
            continue
        factory["produce_timer"] = float(factory.get("produce_timer", 0.0)) - dt
        while (
            factory["produce_timer"] <= 0.0
            and backlog < MAX_FACTORY_BACKLOG
            and len(factory.get("conveyor", [])) < MAX_CONVEYOR_PACKAGES
        ):
            package = _make_package(world_state, factory)
            factory.setdefault("conveyor", []).append(package)
            backlog += 1
            factory["produce_timer"] += float(factory.get("produce_every", 6.5))

    if int(elapsed * 2) % 6 == 0 and world_state.get("_last_observe_bucket") != int(elapsed * 2):
        world_state["_last_observe_bucket"] = int(elapsed * 2)
        counts = ", ".join(
            f"{factory['name']}={len(factory.get('pad_packages', []))}"
            for factory in world_state.get("factories", [])
        )
        _thought(world_state, S.OBSERVE, "Leader", f"Pad inventory: {counts}", elapsed, blackboard)


def _leader_plan_prompt(world_state: dict) -> str:
    factories = [
        {
            "id": f["id"],
            "name": f["name"],
            "color": f["color"],
            "drop_pad": f["drop_pad"],
            "packages_waiting": len([p for p in f.get("pad_packages", []) if p.get("reserved_by") is None]),
        }
        for f in world_state.get("factories", [])
    ]
    workers = [
        {
            "id": w["id"],
            "name": w["name"],
            "pos": w["pos"],
            "status": w["status"],
            "carrying": (w.get("carrying") or {}).get("color"),
        }
        for w in world_state.get("workers", [])
    ]
    dropboxes = [
        {"id": b["id"], "name": b["name"], "color": b["color"], "pos": b["pos"]}
        for b in world_state.get("dropboxes", [])
    ]
    policy_status = world_state.get("policy_status", "unknown")
    return f"""You are the FactoryMind leader running inside OpenClaw/NemoClaw.
Observe the entire floor and assign idle workers to packages waiting on conveyor pads.

NemoClaw guardrails are active: policy_status={policy_status}.
- You may post directives to workers.
- You cannot directly move workers or modify factories.
- Workers can pick up only from conveyor pads and drop only at matching-color drop boxes.
- Use only idle workers.
- Assign each worker and each package at most once.
- Prefer the nearest reachable same-color drop box; the runtime will deny unsafe directives.
- Do not assign disabled, busy, carrying, unreachable, or mismatched-color routes.

Factories: {factories}
Workers: {workers}
Drop boxes: {dropboxes}

Return JSON only:
{{"assignments":[{{"worker_id":1,"factory_id":0,"reason":"closest idle worker to red pad"}}],
 "thought":"brief visible leader reasoning"}}
"""


def _leader_decide_autonomous(world_state: dict) -> dict:
    """Run leader planning. LLM output is validated; deterministic planning keeps the loop live."""
    assignments = _deterministic_assignments(world_state)
    thought = "Nearest idle workers matched to waiting packages and same-color drop boxes."
    llm_ready = world_state.get("llm_ready", False)
    if llm_ready and os.getenv("FACTORYMIND_DISABLE_LLM", "false").lower() != "true":
        try:
            from factorymind import inference
            raw = inference.ask_leader(_leader_plan_prompt(world_state))
            parsed = inference.safe_parse_json(raw)
            if isinstance(parsed, dict):
                thought = str(parsed.get("thought") or thought)
                proposed = parsed.get("assignments", [])
                validated = _validate_llm_assignments(world_state, proposed)
                if validated:
                    assignments = validated
        except Exception as exc:
            return {
                "assignments": assignments,
                "thought": thought,
                "warning": f"Leader LLM unavailable; deterministic policy planner used ({exc}).",
            }
    return {"assignments": assignments, "thought": thought, "warning": ""}


def _worker_llm_enabled(world_state: dict) -> bool:
    return (
        bool(world_state.get("llm_ready"))
        and os.getenv("FACTORYMIND_DISABLE_LLM", "false").lower() != "true"
        and os.getenv("FACTORYMIND_DISABLE_WORKER_LLM", "false").lower() != "true"
    )


def _worker_llm_prompt(snapshot: dict) -> str:
    return f"""You are an autonomous warehouse worker bot running on the local GX10 worker model.
Choose a safe high-level intent for your next action. Deterministic pathfinding and NemoClaw policy execute the actual movement.

Guardrails:
- Never request direct coordinate teleporting.
- Never pick up except at your assigned conveyor pad.
- Never deliver except to your assigned same-color drop bin.
- If blocked, choose wait or reroute only.
- Return compact JSON only.

Worker decision context:
{json.dumps(snapshot, sort_keys=True)}

Return JSON only:
{{"intent":"accept_task|reroute|wait|deliver|complete","reason":"short operational reason","confidence":0.0}}
"""


def _worker_decide_llm(snapshot: dict) -> dict:
    from factorymind import inference

    raw = inference.ask_worker(_worker_llm_prompt(snapshot))
    parsed = inference.safe_parse_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError("worker LLM did not return a JSON object")
    allowed = {"accept_task", "reroute", "wait", "deliver", "complete"}
    intent = str(parsed.get("intent") or snapshot.get("default_intent") or "wait").strip().lower()
    if intent not in allowed:
        intent = str(snapshot.get("default_intent") or "wait")
    reason = str(parsed.get("reason") or "Worker local model produced an intent.").strip()
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "worker_id": snapshot["worker_id"],
        "worker_name": snapshot["worker_name"],
        "event": snapshot["event"],
        "task_id": snapshot.get("task_id"),
        "intent": intent,
        "reason": reason[:160],
        "confidence": max(0.0, min(confidence, 1.0)),
    }


def _submit_worker_llm_decision(world_state: dict, worker: dict, event: str, payload: dict) -> None:
    if not _worker_llm_enabled(world_state):
        return
    task_id = int(worker.get("current_task") if worker.get("current_task") is not None else -1)
    key = (int(worker["id"]), event, task_id)
    if key in _pending_worker_llm:
        return
    snapshot = {
        "worker_id": worker["id"],
        "worker_name": worker.get("name", f"Worker-{worker['id']}"),
        "event": event,
        "default_intent": payload.get("default_intent", "wait"),
        "task_id": worker.get("current_task"),
        "status": worker.get("status"),
        "pos": worker.get("pos"),
        "target_factory_id": worker.get("target_factory_id"),
        "target_dropbox_id": worker.get("target_dropbox_id"),
        "carrying_color": (worker.get("carrying") or {}).get("color"),
        "path_remaining": len(worker.get("path") or []),
        "payload": payload,
    }
    _pending_worker_llm[key] = _decision_executor.submit(_worker_decide_llm, snapshot)
    worker["llm_pending"] = event
    _policy_log(world_state, worker.get("name", "Worker"), "worker_llm_submit", True, f"{event} -> local worker model")


def _drain_worker_llm_decisions(world_state: dict, blackboard: Blackboard, elapsed: float) -> None:
    for key, future in list(_pending_worker_llm.items()):
        if not future.done():
            continue
        _pending_worker_llm.pop(key, None)
        try:
            decision = future.result()
        except Exception as exc:
            worker = _worker_by_id(world_state, key[0])
            if worker:
                worker["llm_pending"] = None
                worker["last_llm_error"] = str(exc)[:120]
                _thought(world_state, S.WARNING, worker["name"], f"Worker LLM unavailable; deterministic behavior continues ({exc}).", elapsed, blackboard)
                _policy_log(world_state, worker["name"], "worker_llm_result", False, str(exc)[:180])
            continue
        worker = _worker_by_id(world_state, int(decision["worker_id"]))
        if not worker:
            continue
        worker["llm_pending"] = None
        worker["last_llm_intent"] = decision["intent"]
        worker["last_llm_reason"] = decision["reason"]
        worker["llm_decisions"] = int(worker.get("llm_decisions", 0)) + 1
        _thought(
            world_state,
            S.INTENT,
            worker.get("name", decision["worker_name"]),
            f"LLM intent={decision['intent']} ({decision['reason']})",
            elapsed,
            blackboard,
        )
        _policy_log(
            world_state,
            worker.get("name", decision["worker_name"]),
            "worker_llm_result",
            True,
            f"{decision['event']} -> {decision['intent']} confidence={decision['confidence']:.2f}",
        )


def _validate_llm_assignments(world_state: dict, proposed: object) -> list[dict]:
    if not isinstance(proposed, list):
        return []
    idle_ids = {
        w["id"] for w in world_state.get("workers", [])
        if w.get("status") == "idle" and not w.get("path") and not w.get("carrying")
    }
    factory_ids = {f["id"] for f in world_state.get("factories", [])}
    assignments = []
    used_workers: set[int] = set()
    used_packages: set[int] = set()
    for item in proposed:
        if not isinstance(item, dict):
            continue
        worker_id = item.get("worker_id")
        factory_id = item.get("factory_id")
        if worker_id not in idle_ids or worker_id in used_workers or factory_id not in factory_ids:
            continue
        factory = _factory_by_id(world_state, factory_id)
        package = _first_unreserved_package(factory) if factory else None
        dropbox = _nearest_dropbox_for_color(
            world_state,
            factory["color"],
            factory["drop_pad"],
            {tuple(c) for c in world_state.get("wall", [])},
        ) if factory else None
        if not factory or not package or package["id"] in used_packages or not dropbox:
            continue
        if not _nemoclaw_allows_leader_assignment(world_state, _worker_by_id(world_state, worker_id), factory, package, dropbox, "llm_validate"):
            continue
        assignments.append({
            "worker_id": worker_id,
            "factory_id": factory_id,
            "package_id": package["id"],
            "dropbox_id": dropbox["id"],
            "reason": str(item.get("reason") or "LLM selected this worker/package pair."),
        })
        used_workers.add(worker_id)
        used_packages.add(package["id"])
    return assignments


def _deterministic_assignments(world_state: dict) -> list[dict]:
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    idle_workers = [
        w for w in world_state.get("workers", [])
        if w.get("status") == "idle" and not w.get("path") and not w.get("carrying")
    ]
    available = []
    for factory in world_state.get("factories", []):
        matching_dropboxes = _dropboxes_for_color(world_state, factory["color"])
        if not matching_dropboxes:
            continue
        for package in factory.get("pad_packages", []):
            if package.get("reserved_by") is None:
                for dropbox in matching_dropboxes:
                    available.append((factory, package, dropbox))
    assignments = []
    used_packages: set[int] = set()
    for worker in idle_workers:
        best = None
        best_cost = 10**9
        for factory, package, dropbox in available:
            if package["id"] in used_packages:
                continue
            to_pickup = _route_len(worker["pos"], factory["drop_pad"], wall_set)
            to_drop = _route_len(factory["drop_pad"], dropbox["pos"], wall_set)
            cost = to_pickup + to_drop
            if cost < best_cost:
                best = (factory, package, dropbox, to_pickup, cost)
                best_cost = cost
        if best is None or best_cost >= 10**6:
            continue
        factory, package, dropbox, to_pickup, cost = best
        used_packages.add(package["id"])
        assignments.append({
            "worker_id": worker["id"],
            "factory_id": factory["id"],
            "package_id": package["id"],
            "dropbox_id": dropbox["id"],
            "reason": f"closest route: {to_pickup} cells to {factory['name']}, {cost} total",
        })
    return assignments


def _apply_leader_plan(world_state: dict, plan: dict, blackboard: Blackboard, elapsed: float) -> None:
    world_state["leader_thinking"] = False
    leader = world_state.get("leader", {})
    leader["thinking"] = False
    if plan.get("warning"):
        _thought(world_state, S.WARNING, "Leader", str(plan["warning"])[:110], elapsed, blackboard)
    if plan.get("thought"):
        _thought(world_state, S.RETHINK, "Leader", str(plan["thought"])[:110], elapsed, blackboard)
    assignments = plan.get("assignments") or []
    if not assignments:
        _thought(world_state, S.OBSERVE, "Leader", "No idle worker/package pair needs dispatch right now.", elapsed, blackboard)
        return
    for item in assignments:
        if _assign_worker_to_package(world_state, item, elapsed, blackboard):
            _policy_log(
                world_state,
                "Leader",
                "post_directive",
                True,
                f"Worker-{item['worker_id']} -> factory {item['factory_id']} package {item['package_id']}",
            )


def _assign_worker_to_package(world_state: dict, assignment: dict,
                              elapsed: float, blackboard: Blackboard) -> bool:
    worker = _worker_by_id(world_state, assignment.get("worker_id"))
    factory = _factory_by_id(world_state, assignment.get("factory_id"))
    dropbox = _dropbox_by_id(world_state, assignment.get("dropbox_id"))
    if not worker or not factory:
        return False
    if worker.get("status") != "idle" or worker.get("carrying") or worker.get("path"):
        return False
    package = next(
        (p for p in factory.get("pad_packages", []) if p.get("id") == assignment.get("package_id")),
        None,
    )
    if not package or package.get("reserved_by") is not None:
        return False
    if dropbox and dropbox.get("color") != package.get("color"):
        _policy_log(world_state, worker["name"], "assign_dropbox", False, "drop box color mismatch")
        return False
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    best_dropbox = _nearest_dropbox_for_color(world_state, package["color"], factory["drop_pad"], wall_set)
    if best_dropbox:
        dropbox = best_dropbox
    if not dropbox:
        return False
    if not _nemoclaw_allows_leader_assignment(world_state, worker, factory, package, dropbox, "post_directive"):
        return False
    path = a_star(worker["pos"], factory["drop_pad"], wall_set)
    if worker["pos"] != factory["drop_pad"] and not path:
        _thought(world_state, S.WARNING, "Leader", f"{worker['name']} cannot reach {factory['name']} pad.", elapsed, blackboard)
        return False
    drop_route_len = _route_len(factory["drop_pad"], dropbox["pos"], wall_set)
    if factory["drop_pad"] != dropbox["pos"] and drop_route_len >= 10**6:
        _thought(world_state, S.WARNING, "Leader", f"{dropbox['name']} is unreachable from {factory['name']} pad.", elapsed, blackboard)
        return False
    global _task_counter
    task_id = _task_counter
    _task_counter += 1
    task = {
        "id": task_id,
        "worker_id": worker["id"],
        "package_id": package["id"],
        "factory_id": factory["id"],
        "dropbox_id": dropbox["id"],
        "color": package["color"],
        "status": "assigned",
        "started_elapsed": elapsed,
    }
    world_state.setdefault("tasks", []).append(task)
    package["reserved_by"] = worker["id"]
    worker["current_task"] = task_id
    worker["status"] = "to_pickup"
    worker["target_factory_id"] = factory["id"]
    worker["target_dropbox_id"] = dropbox["id"]
    worker["task_started_at"] = elapsed
    worker["path"] = path
    content = f"{worker['name']} -> {factory['name']} ({assignment.get('reason')}); then {dropbox['name']}"
    _thought(world_state, S.ASSIGN, "Leader", content[:120], elapsed, blackboard)
    _submit_worker_llm_decision(
        world_state,
        worker,
        "task_accept",
        {
            "default_intent": "accept_task",
            "package_color": package["color"],
            "pickup": factory["drop_pad"],
            "dropoff": dropbox["pos"],
            "route_cells_to_pickup": len(path),
            "leader_reason": str(assignment.get("reason") or ""),
        },
    )
    return True


def _move_workers(world_state: dict, elapsed: float, blackboard: Blackboard) -> None:
    occupied = {
        tuple(w["pos"]): w["id"]
        for w in world_state.get("workers", [])
        if w.get("status") != "disabled"
    }
    for worker in world_state.get("workers", []):
        if worker.get("status") == "disabled":
            continue
        if not worker.get("path"):
            _worker_transition(world_state, worker, elapsed, blackboard)
            continue
        next_pos = worker["path"][0]
        if not _policy_worker_can_move(world_state, worker, next_pos):
            worker["path"] = []
            _replan_worker(world_state, worker, elapsed, blackboard)
            continue
        blocker_id = occupied.get(tuple(next_pos))
        if blocker_id is not None and blocker_id != worker["id"]:
            worker["traffic_wait_ticks"] = int(worker.get("traffic_wait_ticks", 0)) + 1
            worker["blocked_by_worker_id"] = blocker_id
            if worker["traffic_wait_ticks"] >= TRAFFIC_REPLAN_TICKS:
                worker["traffic_wait_ticks"] = 0
                blockers = _dynamic_blockers(world_state, worker)
                _submit_worker_llm_decision(
                    world_state,
                    worker,
                    "traffic_conflict",
                    {
                        "default_intent": "reroute",
                        "blocked_by_worker_id": blocker_id,
                        "blocked_cell": next_pos,
                    },
                )
                if _replan_worker(world_state, worker, elapsed, blackboard, blockers):
                    _thought(
                        world_state,
                        S.RETHINK,
                        worker["name"],
                        f"Traffic conflict with Worker-{blocker_id}; rerouting around occupied cells.",
                        elapsed,
                        blackboard,
                    )
                    _policy_log(world_state, worker["name"], "reroute_traffic", True, f"blocked by Worker-{blocker_id}")
            continue
        worker["traffic_wait_ticks"] = 0
        worker["blocked_by_worker_id"] = None
        old_pos = tuple(worker["pos"])
        new_pos = tuple(worker["path"].pop(0))
        worker["pos"] = [new_pos[0], new_pos[1]]
        occupied.pop(old_pos, None)
        occupied[new_pos] = worker["id"]
        _worker_transition(world_state, worker, elapsed, blackboard)


def _dynamic_blockers(world_state: dict, worker: dict) -> set[tuple[int, int]]:
    return {
        tuple(other["pos"])
        for other in world_state.get("workers", [])
        if other.get("id") != worker.get("id") and other.get("status") != "disabled"
    }


def _replan_worker(
    world_state: dict,
    worker: dict,
    elapsed: float,
    blackboard: Blackboard,
    dynamic_blockers: set[tuple[int, int]] | None = None,
) -> bool:
    base_wall_set = {tuple(c) for c in world_state.get("wall", [])}
    wall_set = base_wall_set | (dynamic_blockers or set())
    target: list[int] | None = None
    if worker.get("status") == "to_pickup":
        factory = _factory_by_id(world_state, worker.get("target_factory_id"))
        target = factory.get("drop_pad") if factory else None
    elif worker.get("status") == "to_dropbox":
        dropbox = _dropbox_by_id(world_state, worker.get("target_dropbox_id"))
        target = dropbox.get("pos") if dropbox else None

    if target:
        worker["path"] = a_star(worker["pos"], target, wall_set)
        if worker["pos"] == target or worker["path"]:
            return True
        if dynamic_blockers:
            worker["path"] = a_star(worker["pos"], target, base_wall_set)
            if worker["pos"] == target or worker["path"]:
                _thought(world_state, S.WARNING, worker["name"], "Traffic route boxed in; waiting for aisle to clear.", elapsed, blackboard)
                return False

    if worker.get("carrying"):
        _thought(world_state, S.WARNING, worker["name"], "Route blocked while carrying; holding package.", elapsed, blackboard)
        return False

    factory = _factory_by_id(world_state, worker.get("target_factory_id"))
    if factory:
        for package in factory.get("pad_packages", []):
            if package.get("reserved_by") == worker["id"]:
                package["reserved_by"] = None
                break
    task = _task_by_id(world_state, worker.get("current_task"))
    if task:
        task["status"] = "blocked"
    worker["current_task"] = None
    worker["target_factory_id"] = None
    worker["target_dropbox_id"] = None
    worker["status"] = "idle"
    _thought(world_state, S.WARNING, worker["name"], "Route blocked; assignment released.", elapsed, blackboard)
    return False


def _worker_transition(world_state: dict, worker: dict, elapsed: float, blackboard: Blackboard) -> None:
    if worker.get("status") == "to_pickup":
        factory = _factory_by_id(world_state, worker.get("target_factory_id"))
        if factory and worker["pos"] == factory["drop_pad"]:
            _pickup_package(world_state, worker, factory, elapsed, blackboard)
    elif worker.get("status") == "to_dropbox":
        dropbox = _dropbox_by_id(world_state, worker.get("target_dropbox_id"))
        if dropbox and worker["pos"] == dropbox["pos"]:
            _drop_package(world_state, worker, dropbox, elapsed, blackboard)


def _pickup_package(world_state: dict, worker: dict, factory: dict,
                    elapsed: float, blackboard: Blackboard) -> None:
    package = next(
        (p for p in factory.get("pad_packages", []) if p.get("reserved_by") == worker["id"]),
        None,
    )
    if not package:
        worker["status"] = "idle"
        worker["current_task"] = None
        return
    allowed = worker["pos"] == factory["drop_pad"]
    _policy_log(world_state, worker["name"], "pickup_package", allowed, f"{factory['name']} pad")
    if not allowed:
        return
    factory["pad_packages"].remove(package)
    package["status"] = "carried"
    worker["carrying"] = package
    worker["status"] = "to_dropbox"
    dropbox = _dropbox_by_id(world_state, worker.get("target_dropbox_id"))
    if dropbox:
        worker["path"] = a_star(worker["pos"], dropbox["pos"], {tuple(c) for c in world_state.get("wall", [])})
        _submit_worker_llm_decision(
            world_state,
            worker,
            "delivery_start",
            {
                "default_intent": "deliver",
                "package_color": package["color"],
                "dropbox": dropbox["name"],
                "route_cells_to_dropoff": len(worker.get("path") or []),
            },
        )
        if worker["pos"] != dropbox["pos"] and not worker["path"]:
            package["status"] = "pad"
            package["reserved_by"] = None
            task = _task_by_id(world_state, worker.get("current_task"))
            if task:
                task["status"] = "blocked"
            worker["carrying"] = None
            worker["current_task"] = None
            worker["target_factory_id"] = None
            worker["target_dropbox_id"] = None
            worker["status"] = "idle"
            factory.setdefault("pad_packages", []).append(package)
            _thought(world_state, S.WARNING, worker["name"], f"No route to {dropbox['name']}; package returned to pad.", elapsed, blackboard)


def _drop_package(world_state: dict, worker: dict, dropbox: dict,
                  elapsed: float, blackboard: Blackboard) -> None:
    package = worker.get("carrying")
    if not package:
        worker["status"] = "idle"
        return
    allowed = package.get("color") == dropbox.get("color")
    _policy_log(world_state, worker["name"], "drop_package", allowed, f"{dropbox['name']} color={dropbox['color']}")
    if not allowed:
        _thought(world_state, S.WARNING, worker["name"], f"Blocked mismatched drop at {dropbox['name']}.", elapsed, blackboard)
        return
    route_time = round(elapsed - float(worker.get("task_started_at") or elapsed), 2)
    package["status"] = "delivered"
    package["delivered_to"] = dropbox["id"]
    dropbox["delivered"] = int(dropbox.get("delivered", 0)) + 1
    task = _task_by_id(world_state, worker.get("current_task"))
    if task:
        task["status"] = "done"
        task["route_time"] = route_time
    stats = world_state["stats"]
    stats["completed"] += 1
    stats["last_route_time"] = route_time
    count = stats["completed"]
    prev_avg = stats.get("avg_route_time") or route_time
    stats["avg_route_time"] = round((prev_avg * (count - 1) + route_time) / count, 2)
    _thought(
        world_state,
        S.COMPLETE,
        worker["name"],
        f"delivered {package['color']} package to {dropbox['name']} [{route_time:.2f}s]",
        elapsed,
        blackboard,
    )
    _submit_worker_llm_decision(
        world_state,
        worker,
        "task_complete",
        {
            "default_intent": "complete",
            "package_color": package["color"],
            "dropbox": dropbox["name"],
            "route_time": route_time,
        },
    )
    worker["carrying"] = None
    worker["current_task"] = None
    worker["target_factory_id"] = None
    worker["target_dropbox_id"] = None
    worker["task_started_at"] = None
    worker["status"] = "idle"
    worker["path"] = []


def _policy_worker_can_move(world_state: dict, worker: dict, cell: list[int]) -> bool:
    allowed = 0 <= cell[0] < GRID_WIDTH and 0 <= cell[1] < GRID_HEIGHT
    if not allowed:
        _policy_log(world_state, worker["name"], "move", False, f"outside simulation area: {cell}")
        return False
    if tuple(cell) in {tuple(wall) for wall in world_state.get("wall", [])}:
        _policy_log(world_state, worker["name"], "move", False, f"blocked by wall: {cell}")
        return False
    return allowed


def _factory_by_id(world_state: dict, factory_id: int | None) -> dict | None:
    return next((f for f in world_state.get("factories", []) if f.get("id") == factory_id), None)


def _dropbox_by_id(world_state: dict, dropbox_id: int | None) -> dict | None:
    return next((b for b in world_state.get("dropboxes", []) if b.get("id") == dropbox_id), None)


def _worker_by_id(world_state: dict, worker_id: int | None) -> dict | None:
    return next((w for w in world_state.get("workers", []) if w.get("id") == worker_id), None)


def _task_by_id(world_state: dict, task_id: int | None) -> dict | None:
    return next((t for t in world_state.get("tasks", []) if t.get("id") == task_id), None)


def _dropboxes_for_color(world_state: dict, color: str) -> list[dict]:
    return [b for b in world_state.get("dropboxes", []) if b.get("color") == color]


def _dropbox_for_color(world_state: dict, color: str) -> dict | None:
    return next(iter(_dropboxes_for_color(world_state, color)), None)


def _nearest_dropbox_for_color(
    world_state: dict,
    color: str,
    pickup: list[int],
    wall_set: set[tuple[int, int]] | None = None,
) -> dict | None:
    walls = wall_set if wall_set is not None else {tuple(c) for c in world_state.get("wall", [])}
    best: tuple[int, dict] | None = None
    for dropbox in _dropboxes_for_color(world_state, color):
        cost = _route_len(pickup, dropbox["pos"], walls)
        if cost >= 10**6:
            continue
        if best is None or cost < best[0]:
            best = (cost, dropbox)
    return best[1] if best else None


def _first_unreserved_package(factory: dict) -> dict | None:
    return next((p for p in factory.get("pad_packages", []) if p.get("reserved_by") is None), None)


def _nemoclaw_allows_leader_assignment(
    world_state: dict,
    worker: dict | None,
    factory: dict | None,
    package: dict | None,
    dropbox: dict | None,
    action: str,
) -> bool:
    def deny(detail: str) -> bool:
        _policy_log(world_state, "NemoClaw", action, False, detail)
        return False

    if not world_state.get("policy_loaded"):
        _policy_log(world_state, "NemoClaw", action, True, "policy file missing; runtime safety checks still enforced")
    if not worker or not factory or not package or not dropbox:
        return deny("directive references missing worker, factory, package, or dropbox")
    if worker.get("status") != "idle" or worker.get("path") or worker.get("carrying"):
        return deny(f"{worker.get('name', 'worker')} is not idle")
    if package.get("reserved_by") not in (None, worker.get("id")):
        return deny(f"package {package.get('id')} is already reserved")
    if package not in factory.get("pad_packages", []):
        return deny(f"package {package.get('id')} is not waiting on {factory.get('name')} pad")
    if package.get("color") != dropbox.get("color"):
        return deny(f"package color {package.get('color')} cannot go to {dropbox.get('name')} color {dropbox.get('color')}")
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    if worker.get("pos") != factory.get("drop_pad") and _route_len(worker["pos"], factory["drop_pad"], wall_set) >= 10**6:
        return deny(f"{worker.get('name')} cannot reach {factory.get('name')} pickup pad")
    if factory.get("drop_pad") != dropbox.get("pos") and _route_len(factory["drop_pad"], dropbox["pos"], wall_set) >= 10**6:
        return deny(f"{dropbox.get('name')} is unreachable from {factory.get('name')} pickup pad")
    _policy_log(
        world_state,
        "NemoClaw",
        action,
        True,
        f"{worker.get('name')} may move package {package.get('id')} {package.get('color')} -> {dropbox.get('name')}",
    )
    return True


def _cell_occupied_by_asset(world_state: dict, cell: list[int]) -> bool:
    x, y = cell
    if [x, y] in world_state.get("wall", []):
        return True
    if any(w.get("pos") == [x, y] for w in world_state.get("workers", [])):
        return True
    for factory in world_state.get("factories", []):
        fx, fy = factory.get("pos", [0, 0])
        width = 3 + int(factory.get("belt_length", 5))
        if fx <= x < fx + width and fy <= y < fy + 3:
            return True
    for dropbox in world_state.get("dropboxes", []):
        bx, by = dropbox.get("pos", [0, 0])
        width, height = _rotated_footprint(3, 2, float(dropbox.get("rotation_y") or 0.0))
        if bx <= x < bx + width and by <= y < by + height:
            return True
    return False


def _rotated_footprint(width: int, height: int, rotation_y: float) -> tuple[int, int]:
    quarter_turns = round(rotation_y / (3.141592653589793 / 2)) % 2
    return (height, width) if quarter_turns else (width, height)


def _area_is_clear(world_state: dict, cell: list[int], width: int, height: int) -> bool:
    x0, y0 = cell
    if x0 < 0 or y0 < 0 or x0 + width > GRID_WIDTH or y0 + height > GRID_HEIGHT:
        return False
    return not any(
        _cell_occupied_by_asset(world_state, [x, y])
        for x in range(x0, x0 + width)
        for y in range(y0, y0 + height)
    )


def _place_factory(world_state: dict, cell: list[int], color: str, rotation_y: float = 0.0) -> None:
    if not _area_is_clear(world_state, cell, 8, 3):
        return
    factory_id = world_state["next_factory_id"]
    world_state["next_factory_id"] += 1
    suffix = chr(ord("A") + factory_id)
    factory = S.make_factory(f"Factory-{suffix}", cell, color, factory_id, rotation_y)
    world_state["factories"].append(factory)
    world_state["workstations"] = copy.deepcopy(world_state["factories"])


def _place_dropbox(world_state: dict, cell: list[int], color: str, rotation_y: float = 0.0) -> None:
    width, height = _rotated_footprint(3, 2, rotation_y)
    if not _area_is_clear(world_state, cell, width, height):
        return
    box_id = world_state["next_dropbox_id"]
    world_state["next_dropbox_id"] += 1
    world_state["dropboxes"].append(S.make_dropbox(f"DropBox-{color}", cell, color, box_id, rotation_y))


def _place_worker(world_state: dict, cell: list[int]) -> None:
    if not _area_is_clear(world_state, cell, 1, 1):
        return
    worker_id = world_state["next_worker_id"]
    world_state["next_worker_id"] += 1
    worker = S.make_worker(worker_id, cell)
    world_state["workers"].append(worker)
    world_state["robots"].append(worker)


def _remove_asset(world_state: dict, cell: list[int]) -> None:
    worker = next((w for w in world_state.get("workers", []) if w.get("pos") == cell), None)
    if worker:
        _release_removed_worker(world_state, worker, return_package=True)
        world_state["workers"] = [w for w in world_state.get("workers", []) if w.get("id") != worker.get("id")]
        world_state["robots"] = [world_state.get("leader")] + world_state["workers"]
        return

    factory = _factory_at_cell(world_state, cell)
    if factory:
        _remove_factory(world_state, factory)
        return

    dropbox = _dropbox_at_cell(world_state, cell)
    if dropbox:
        _remove_dropbox(world_state, dropbox)
        return

    if cell in world_state.get("wall", []):
        world_state["wall"].remove(cell)


def _factory_at_cell(world_state: dict, cell: list[int]) -> dict | None:
    x, y = cell
    for factory in world_state.get("factories", []):
        fx, fy = factory.get("pos", [0, 0])
        width = 3 + int(factory.get("belt_length", 5))
        if fx <= x < fx + width and fy <= y < fy + 3:
            return factory
    return None


def _dropbox_at_cell(world_state: dict, cell: list[int]) -> dict | None:
    x, y = cell
    for dropbox in world_state.get("dropboxes", []):
        bx, by = dropbox.get("pos", [0, 0])
        width, height = _rotated_footprint(3, 2, float(dropbox.get("rotation_y") or 0.0))
        if bx <= x < bx + width and by <= y < by + height:
            return dropbox
    return None


def _remove_factory(world_state: dict, factory: dict) -> None:
    factory_id = factory.get("id")
    for worker in world_state.get("workers", []):
        if worker.get("target_factory_id") == factory_id or (worker.get("carrying") or {}).get("factory_id") == factory_id:
            _release_removed_worker(world_state, worker, return_package=False)
    for task in world_state.get("tasks", []):
        if task.get("factory_id") == factory_id and task.get("status") not in {"done", "blocked"}:
            task["status"] = "blocked"
    world_state["packages"] = [
        package for package in world_state.get("packages", [])
        if package.get("factory_id") != factory_id
    ]
    world_state["factories"] = [f for f in world_state.get("factories", []) if f.get("id") != factory_id]
    world_state["workstations"] = copy.deepcopy(world_state["factories"])


def _remove_dropbox(world_state: dict, dropbox: dict) -> None:
    dropbox_id = dropbox.get("id")
    for worker in world_state.get("workers", []):
        if worker.get("target_dropbox_id") == dropbox_id:
            _release_removed_worker(world_state, worker, return_package=True)
    for task in world_state.get("tasks", []):
        if task.get("dropbox_id") == dropbox_id and task.get("status") not in {"done", "blocked"}:
            task["status"] = "blocked"
    world_state["dropboxes"] = [b for b in world_state.get("dropboxes", []) if b.get("id") != dropbox_id]


def _release_removed_worker(world_state: dict, worker: dict, return_package: bool) -> None:
    carrying = worker.get("carrying")
    if carrying and return_package:
        factory = _factory_by_id(world_state, carrying.get("factory_id"))
        if factory:
            carrying["status"] = "pad"
            carrying["reserved_by"] = None
            factory.setdefault("pad_packages", []).append(carrying)
    for factory in world_state.get("factories", []):
        for package in factory.get("pad_packages", []):
            if package.get("reserved_by") == worker.get("id"):
                package["reserved_by"] = None
    task = _task_by_id(world_state, worker.get("current_task"))
    if task and task.get("status") not in {"done", "blocked"}:
        task["status"] = "blocked"
    worker["carrying"] = None
    worker["current_task"] = None
    worker["target_factory_id"] = None
    worker["target_dropbox_id"] = None
    worker["task_started_at"] = None
    worker["status"] = "idle"
    worker["path"] = []


def _handle_builder_action(world_state: dict, action: dict) -> None:
    action_type = action.get("type")
    if action_type == "wall_add":
        cell = action["cell"]
        if cell not in world_state["wall"] and not _cell_occupied_by_asset(world_state, cell):
            world_state["wall"].append(cell)
    elif action_type == "wall_remove":
        cell = action["cell"]
        if cell in world_state["wall"]:
            world_state["wall"].remove(cell)
    elif action_type == "place_factory":
        color = action.get("color") or S.next_color_name(len(world_state.get("factories", [])))
        _place_factory(world_state, action["cell"], color, _rotation_y_from_action(action))
    elif action_type == "place_dropbox":
        color = action.get("color") or _next_dropbox_color(world_state)
        _place_dropbox(world_state, action["cell"], color, _rotation_y_from_action(action))
    elif action_type == "place_worker":
        _place_worker(world_state, action["cell"])
    elif action_type == "remove_asset":
        _remove_asset(world_state, action["cell"])


def _next_dropbox_color(world_state: dict) -> str:
    factory_colors = [factory.get("color") for factory in world_state.get("factories", [])]
    box_colors = [box.get("color") for box in world_state.get("dropboxes", [])]
    for color in factory_colors:
        if color not in box_colors:
            return str(color)
    if factory_colors:
        return str(factory_colors[len(box_colors) % len(factory_colors)])
    return S.next_color_name(len(box_colors))


def _rotation_y_from_action(action: dict) -> float:
    try:
        return float(action.get("rotation_y", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _update_stats(world_state: dict, elapsed: float) -> None:
    stats = world_state["stats"]
    stats["elapsed"] = round(elapsed, 1)
    completed = stats.get("completed", 0)
    stats["rate"] = round(completed / elapsed * 60, 1) if elapsed > 0 else 0.0


def run_pygame() -> None:
    global _pending_leader
    import pygame
    from factorymind import render as R

    inference_target = _print_startup_banner()
    speed_multiplier = 1
    world_state = _reset_demo_state(speed_multiplier, "online", inference_target)
    blackboard = Blackboard()
    world_state["llm_ready"] = _local_nim_available() if os.getenv("USE_LOCAL_NIM", "true").lower() == "true" else True
    if not world_state["llm_ready"]:
        _thought(world_state, S.WARNING, "Leader", "Local NIM is not reachable; deterministic policy planner is active.", 0.0, blackboard)
    _thought(world_state, S.OBSERVE, "Leader", "FactoryMind ready. Press START to begin autonomous dispatch.", 0.0, blackboard)

    screen = R.init_display()
    clock = pygame.time.Clock()
    max_runtime = float(os.getenv("FACTORYMIND_MAX_RUNTIME", "0") or 0)
    last_frame_time = time.time()
    run_started_wall = last_frame_time
    sim_elapsed = 0.0
    last_move_tick = -MOVE_TICK_INTERVAL
    last_factory_tick = -FACTORY_TICK_INTERVAL
    last_leader_tick = -LEADER_TICK_INTERVAL
    running = True
    initial_use_local = os.getenv("USE_LOCAL_NIM", "true").lower() == "true"

    while running:
        wall_now = time.time()
        real_delta = wall_now - last_frame_time
        last_frame_time = wall_now
        if world_state.get("running"):
            sim_elapsed += real_delta * speed_multiplier
        now = sim_elapsed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            result = R.handle_event(event, world_state)
            if result is None:
                continue
            if result == "start":
                world_state["running"] = True
                _thought(world_state, S.OBSERVE, "Leader", "Autonomous loop started.", now, blackboard)
            elif result == "stop":
                world_state["running"] = False
                _thought(world_state, S.RETHINK, "Leader", "Simulation paused in place.", now, blackboard)
            elif result == "reset":
                _cancel_pending_decisions()
                world_state = _reset_demo_state(speed_multiplier, world_state["connection_status"], inference_target)
                world_state["llm_ready"] = _local_nim_available() if initial_use_local else True
                blackboard.clear()
                sim_elapsed = 0.0
                last_move_tick = -MOVE_TICK_INTERVAL
                last_factory_tick = -FACTORY_TICK_INTERVAL
                last_leader_tick = -LEADER_TICK_INTERVAL
                _thought(world_state, S.OBSERVE, "Leader", "State reset to 1 leader, 3 workers, 3 factories, 3 drop boxes.", 0.0, blackboard)
            elif result == "speed_toggle":
                speed_multiplier = _cycle_speed(speed_multiplier)
                world_state["speed_multiplier"] = speed_multiplier
            elif result == "disconnect":
                _set_inference_mode(True)
                inference_target = _describe_inference_target()
                world_state["inference_target"] = inference_target
                world_state["connection_status"] = "offline"
                world_state["llm_ready"] = _local_nim_available()
                _thought(world_state, S.WARNING, "Leader", "Cloud disconnected; GX10 local policy path active.", now, blackboard)
            elif result == "reconnect":
                _set_inference_mode(initial_use_local)
                inference_target = _describe_inference_target()
                world_state["inference_target"] = inference_target
                world_state["connection_status"] = "online"
                world_state["llm_ready"] = _local_nim_available() if initial_use_local else True
                _thought(world_state, S.OBSERVE, "Leader", "Cloud/demo inference mode restored.", now, blackboard)
            elif result in ("mode_cursor", "mode_builder"):
                pass
            elif isinstance(result, str) and result.startswith("builder_"):
                world_state["builder_mode"] = result.replace("builder_", "")
            elif isinstance(result, dict):
                _handle_builder_action(world_state, result)

        if _pending_leader is not None and _pending_leader.done():
            future = _pending_leader
            _pending_leader = None
            try:
                plan = future.result()
            except Exception as exc:
                plan = {
                    "assignments": _deterministic_assignments(copy.deepcopy(world_state)),
                    "thought": "",
                    "warning": f"Leader planner failed; deterministic policy planner used ({exc}).",
                }
            _apply_leader_plan(world_state, plan, blackboard, now)
        _drain_worker_llm_decisions(world_state, blackboard, now)

        if world_state.get("running"):
            if now - last_factory_tick >= FACTORY_TICK_INTERVAL:
                steps = max(1, min(int((now - last_factory_tick) // FACTORY_TICK_INTERVAL), 4))
                for _ in range(steps):
                    _update_factories(world_state, FACTORY_TICK_INTERVAL, now, blackboard)
                last_factory_tick += steps * FACTORY_TICK_INTERVAL

            if now - last_leader_tick >= LEADER_TICK_INTERVAL and _pending_leader is None:
                snapshot = copy.deepcopy(world_state)
                world_state["leader_thinking"] = True
                world_state["leader"]["thinking"] = True
                _pending_leader = _decision_executor.submit(_leader_decide_autonomous, snapshot)
                last_leader_tick = now

            if now - last_move_tick >= MOVE_TICK_INTERVAL:
                steps = max(1, min(int((now - last_move_tick) // MOVE_TICK_INTERVAL), 4))
                for _ in range(steps):
                    _move_workers(world_state, now, blackboard)
                last_move_tick += steps * MOVE_TICK_INTERVAL

        _update_stats(world_state, now)
        world_state["blackboard"] = blackboard.to_list()
        world_state["speed_multiplier"] = speed_multiplier
        world_state["tick"] += 1

        try:
            R.render(screen, world_state)
        except Exception as exc:
            print(f"[main] render failed: {exc}", file=sys.stderr, flush=True)
        if max_runtime and (now >= max_runtime or wall_now - run_started_wall >= max_runtime):
            running = False
        clock.tick(FPS)

    _cancel_pending_decisions()
    _decision_executor.shutdown(wait=False, cancel_futures=True)
    pygame.quit()
    sys.exit(0)


def _cancel_pending_decisions() -> None:
    global _pending_leader
    if _pending_leader is not None:
        _pending_leader.cancel()
        _pending_leader = None
    for future in _pending_worker_llm.values():
        future.cancel()
    _pending_worker_llm.clear()


def main() -> None:
    from factorymind.web_main import run
    run()


if __name__ == "__main__":
    main()
