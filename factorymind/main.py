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
import random
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
    leader_decide,
    strategist_decide,
    worker_decide,
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
LEADER_TICK_INTERVAL = 1.5
WORKER_TICK_INTERVAL = 2.5  # workers think slightly less often than leaders
STRATEGIST_TICK_INTERVAL = 20.0
TASK_SPAWN_INTERVAL = 1.2
MOVE_TICK_INTERVAL = 0.05  # how often robots advance one step
TARGET_ACTIVE_TASKS = 5
SPEED_OPTIONS = (1, 2, 4)

# Set ALL_AGENTS_LLM=false to fall back to the rule-based worker dispatcher
# (cheaper, no per-worker NIM traffic). Default is true so every ball is its
# own LLM-driven agent.
_ALL_AGENTS_LLM = os.getenv("ALL_AGENTS_LLM", "true").lower() == "true"

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
    """Create a random pickup-to-delivery task between two different workstations."""
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
# Demo / integration helpers
# ---------------------------------------------------------------------------

def _active_task_count(world_state: dict) -> int:
    """Return tasks still visible as work in the demo."""
    return sum(1 for t in world_state["tasks"] if t.get("status") != "done")


def _reset_demo_state(layout: str, speed_multiplier: int, connection_status: str) -> dict:
    """Create a fresh run while preserving demo controls."""
    global _task_counter
    _task_counter = 0
    world_state = S.create_initial_state(layout)
    world_state["speed_multiplier"] = speed_multiplier
    world_state["connection_status"] = connection_status
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

    msg = decision.get("post_message", "")
    if msg:
        blackboard.post(leader["id"], CLAIM, msg)


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
    wait_default = float(os.getenv("NEMOTRON_TIMEOUT_SECONDS", "3.0")) + 1.0
    decision_wait = float(os.getenv("FACTORYMIND_PRECOMPUTE_WAIT_SECONDS", str(wait_default)))

    sim_elapsed = 0.0
    last_leader_tick = 0.0
    last_worker_tick = 0.0
    last_strategist_tick = 0.0
    last_task_spawn = 0.0
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

        if now - last_task_spawn >= TASK_SPAWN_INTERVAL:
            if _active_task_count(world_state) < TARGET_ACTIVE_TASKS:
                world_state["tasks"].append(_spawn_task(world_state))
            last_task_spawn = now

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
                for robot in world_state["robots"]:
                    if robot["role"] == WORKER:
                        next_pos = worker_step(robot, world_state, blackboard)
                        if robot.get("path") and next_pos != robot["path"][0]:
                            robot["path"].insert(0, next_pos)
                    _advance_robot(robot, world_state)
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
        f"USE_LOCAL_NIM={os.getenv('USE_LOCAL_NIM', 'false')}  "
        f"GX10_IP={os.getenv('GX10_IP', 'localhost')}",
        flush=True,
    )
    print("  thread pool: 12 workers (each ball has its own LLM future)", flush=True)
    print("=" * 72, flush=True)


def _auto_fallback_to_mock_if_endpoints_dead() -> None:
    """If neither cloud nor local NIM is reachable, force AGENTS_USE_MOCK=true.

    This keeps the demo moving even when the inference endpoints are gone, and
    means a misconfigured `.env` never produces a frozen window.
    """
    if os.getenv("AGENTS_USE_MOCK", "false").lower() == "true":
        return  # already in mock mode, nothing to check

    use_local = os.getenv("USE_LOCAL_NIM", "false").lower() == "true"
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
    global _pending_strategist  # we both read and write it below

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
    initial_use_local = os.getenv("USE_LOCAL_NIM", "false").lower() == "true"
    initial_use_mock = os.getenv("AGENTS_USE_MOCK", "false").lower() == "true"
    max_runtime = float(os.getenv("FACTORYMIND_MAX_RUNTIME", "0") or 0)
    world_state = _reset_demo_state(layout, speed_multiplier, "online")
    blackboard = Blackboard()

    print("[main] initializing pygame display...", flush=True)
    screen = R.init_display()
    print("[main] pygame ready, entering main loop", flush=True)
    clock = pygame.time.Clock()

    # Timers
    sim_elapsed          = 0.0
    last_frame_time      = time.time()
    last_leader_tick     = 0.0
    last_worker_tick     = 0.0
    last_strategist_tick = 0.0
    last_task_spawn      = 0.0
    last_move_tick       = 0.0
    last_heartbeat       = time.time()
    frames_since_beat    = 0

    running = True
    while running:
        wall_now = time.time()
        real_delta = wall_now - last_frame_time
        last_frame_time = wall_now
        sim_elapsed += real_delta * speed_multiplier
        now = sim_elapsed

        # --- Heartbeat (proves the main loop is alive) ---
        frames_since_beat += 1
        if wall_now - last_heartbeat >= 5.0:
            in_flight = (
                len(_pending_leader)
                + len(_pending_worker)
                + (1 if _pending_strategist else 0)
            )
            print(
                f"[main] alive  fps={frames_since_beat / (wall_now - last_heartbeat):.0f}  "
                f"sim_t={sim_elapsed:.1f}s  in_flight_decisions={in_flight}  "
                f"completed={world_state['stats'].get('completed', 0)}",
                flush=True,
            )
            last_heartbeat = wall_now
            frames_since_beat = 0

        # --- Pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                action = R.get_button_click(event.pos)
                if action == "disconnect":
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
                            -1,
                            STRATEGY,
                            (
                                "Cloud disconnect: forcing local DGX Spark NIM; "
                                f"{fallback}"
                            ),
                        )
                    else:
                        _set_inference_mode(initial_use_local)
                        os.environ["AGENTS_USE_MOCK"] = "true" if initial_use_mock else "false"
                        world_state["connection_status"] = "online"
                        blackboard.post(
                            -1,
                            STRATEGY,
                            "Demo reconnect: inference mode restored to startup configuration.",
                        )
                elif action == "layout_open":
                    _cancel_pending_decisions()
                    world_state = _reset_demo_state(
                        OPEN_FLOOR,
                        speed_multiplier,
                        world_state["connection_status"],
                    )
                    blackboard.clear()
                    sim_elapsed = 0.0
                    last_leader_tick = last_worker_tick = last_strategist_tick = last_task_spawn = last_move_tick = 0.0
                elif action == "layout_bottleneck":
                    _cancel_pending_decisions()
                    world_state = _reset_demo_state(
                        BOTTLENECK_BRIDGE,
                        speed_multiplier,
                        world_state["connection_status"],
                    )
                    blackboard.clear()
                    sim_elapsed = 0.0
                    last_leader_tick = last_worker_tick = last_strategist_tick = last_task_spawn = last_move_tick = 0.0
                elif action == "reset":
                    _cancel_pending_decisions()
                    world_state = _reset_demo_state(
                        world_state["layout"],
                        speed_multiplier,
                        world_state["connection_status"],
                    )
                    blackboard.clear()
                    sim_elapsed = 0.0
                    last_leader_tick = last_worker_tick = last_strategist_tick = last_task_spawn = last_move_tick = 0.0
                elif action == "speedup":
                    speed_multiplier = _cycle_speed(speed_multiplier)
                    world_state["speed_multiplier"] = speed_multiplier

        # --- Apply any LLM decisions that finished since last frame ---
        _drain_completed_decisions(world_state, blackboard)

        # --- Spawn new tasks ---
        if now - last_task_spawn >= TASK_SPAWN_INTERVAL:
            if _active_task_count(world_state) < TARGET_ACTIVE_TASKS:
                world_state["tasks"].append(_spawn_task(world_state))
            last_task_spawn = now

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
                for robot in world_state["robots"]:
                    if robot["role"] == WORKER:
                        next_pos = worker_step(robot, world_state, blackboard)
                        if robot.get("path") and next_pos != robot["path"][0]:
                            robot["path"].insert(0, next_pos)
                    _advance_robot(robot, world_state)
            last_move_tick += max(1, move_steps) * MOVE_TICK_INTERVAL

        # --- Update shared state from blackboard ---
        world_state["blackboard"] = blackboard.to_list()
        world_state["speed_multiplier"] = speed_multiplier

        # --- Stats ---
        elapsed = sim_elapsed
        world_state["stats"]["elapsed"] = round(elapsed, 1)
        completed = world_state["stats"]["completed"]
        world_state["stats"]["rate"] = round(completed / elapsed * 60, 1) if elapsed > 0 else 0.0
        world_state["tick"] += 1

        # --- Render ---
        try:
            R.render(screen, world_state)
        except Exception as exc:
            print(f"[main] render failed: {exc}", file=sys.stderr, flush=True)
        if max_runtime and sim_elapsed >= max_runtime:
            running = False
        clock.tick(FPS)

    _cancel_pending_decisions()
    _decision_executor.shutdown(wait=False, cancel_futures=True)
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
