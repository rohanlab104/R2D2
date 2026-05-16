"""Headless web entry point — same simulation as ``factorymind.main``, but
rendered in 3D in a browser instead of a pygame window.

One command:

    python -m factorymind.web_main

Then open ``http://<gx10-ip>:8080`` (or ``http://localhost:8080`` if you're
on the GX10 itself). Every helper here is reused from ``factorymind.main``;
the only thing that changes is rendering and input.
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# Importing factorymind.main pulls in pygame, but we never call init_display()
# so no window or video driver is required.
from factorymind import main as M
from factorymind import memory as MEM
from factorymind import nemoclaw
from factorymind import web_server
from factorymind.agents import (
    Blackboard,
    leader_decide,
    strategist_decide,
    worker_decide,
)
from factorymind.state import (
    LEADER, OPEN_FLOOR, STRATEGY, WORKER,
)


# ---------------------------------------------------------------------------
# Action handlers (mirror the pygame button/event handlers in main.py)
# ---------------------------------------------------------------------------

def _handle_user_prompt(
    text: str,
    world_state: dict,
    blackboard: Blackboard,
    *,
    now: float,
    timers: dict,
) -> bool:
    """Apply a chat prompt the same way main.py does. Returns True if the
    request started new work (so the caller can flip ``manual_control``)."""
    user_text = text.strip()
    if not user_text:
        return False
    M._sim_started = True
    blackboard.post(-1, "USER", user_text)

    if M._apply_go_to_station_and_stop(user_text, world_state, blackboard):
        timers["last_leader_tick"] = now
        timers["last_worker_tick"] = now
        timers["last_strategist_tick"] = now
        return True

    if M._apply_user_task_request(user_text, world_state, blackboard):
        timers["last_leader_tick"] = now - M.LEADER_TICK_INTERVAL
        timers["last_worker_tick"] = now - M.WORKER_TICK_INTERVAL
        timers["last_strategist_tick"] = now
        return True

    # Free-form prompt — kick the strategist if it's not already busy.
    if M._pending_strategist is None:
        try:
            retrieved = MEM.retrieve_strategies(
                world_state["layout"],
                state_description=MEM.describe_world_state(world_state),
            )
        except Exception:
            retrieved = []
        M._pending_strategist = M._decision_executor.submit(
            strategist_decide, world_state, blackboard, retrieved, user_text,
        )
        timers["last_strategist_tick"] = now
    return False


def _handle_disconnect(
    world_state: dict,
    blackboard: Blackboard,
    *,
    initial_use_local: bool,
    initial_use_mock: bool,
) -> None:
    M._cancel_pending_decisions()
    if world_state["connection_status"] == "online":
        M._set_inference_mode(True)
        world_state["connection_status"] = "offline"
        local_ready = M._local_nim_available()
        os.environ["AGENTS_USE_MOCK"] = (
            "true" if initial_use_mock or not local_ready else "false"
        )
        fallback = (
            "local endpoints are reachable."
            if local_ready
            else "local endpoints are down, so mock fallback is active."
        )
        blackboard.post(
            -1, STRATEGY,
            f"Cloud disconnect: forcing local DGX Spark NIM; {fallback}",
        )
    else:
        M._set_inference_mode(initial_use_local)
        os.environ["AGENTS_USE_MOCK"] = "true" if initial_use_mock else "false"
        world_state["connection_status"] = "online"
        blackboard.post(
            -1, STRATEGY,
            "Demo reconnect: inference mode restored to startup configuration.",
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    M._print_startup_banner()
    M._auto_fallback_to_mock_if_endpoints_dead()

    layout = os.getenv("LAYOUT", OPEN_FLOOR)
    speed_multiplier = 1
    initial_use_local = os.getenv("USE_LOCAL_NIM", "true").lower() == "true"
    initial_use_mock = os.getenv("AGENTS_USE_MOCK", "false").lower() == "true"
    max_runtime = float(os.getenv("FACTORYMIND_MAX_RUNTIME", "0") or 0)

    host = os.getenv("FACTORYMIND_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("FACTORYMIND_WEB_PORT", "8080"))
    target_fps = float(os.getenv("FACTORYMIND_WEB_FPS", "60"))
    publish_dt = float(os.getenv("FACTORYMIND_WEB_PUBLISH_DT", "0.1"))

    world_state = M._reset_demo_state(layout, speed_multiplier, "online")
    blackboard = Blackboard()
    engine = nemoclaw.activate(blackboard=blackboard, world_state=world_state)
    blackboard.post(-1, "POLICY", f"OpenClaw runtime active: {engine.describe()}")
    blackboard.post(
        -1, "STRATEGY",
        "FactoryMind 3D ready — workers auto-dispatch on load; chat can add more tasks.",
    )
    M._bootstrap_demo_run(world_state, blackboard)
    manual_control = False

    url = web_server.serve_in_background(host=host, port=port)
    print("=" * 72, flush=True)
    print(f"[web] 3D viewer ready -> {url}", flush=True)
    if host in ("0.0.0.0", "::", ""):
        print("      (any device on this network can connect; "
              "use the GX10's IP from your laptop)", flush=True)
    print("=" * 72, flush=True)

    timers = {
        "last_leader_tick": -M.LEADER_TICK_INTERVAL,
        "last_worker_tick": -M.WORKER_TICK_INTERVAL,
        "last_strategist_tick": 0.0,
        "last_move_tick": -M.MOVE_TICK_INTERVAL,
    }
    sim_elapsed = 0.0
    run_start = time.time()
    last_frame = run_start
    last_heartbeat = run_start
    heartbeat_seconds = float(os.getenv("FACTORYMIND_HEARTBEAT_SECONDS", "10"))
    frames_since_beat = 0
    last_publish = 0.0
    frame_dt = 1.0 / max(target_fps, 1.0)

    try:
        while True:
            wall_now = time.time()
            real_delta = wall_now - last_frame
            last_frame = wall_now

            if M._sim_started:
                sim_elapsed += real_delta * speed_multiplier
            now = sim_elapsed

            # ---- heartbeat ------------------------------------------------
            frames_since_beat += 1
            if heartbeat_seconds > 0 and wall_now - last_heartbeat >= heartbeat_seconds:
                in_flight = (
                    len(M._pending_leader)
                    + len(M._pending_worker)
                    + (1 if M._pending_strategist else 0)
                )
                fps = frames_since_beat / max(wall_now - last_heartbeat, 1e-6)
                print(
                    f"[web] alive  fps={fps:.0f}  sim_t={sim_elapsed:.1f}s  "
                    f"in_flight={in_flight}  "
                    f"completed={world_state['stats'].get('completed', 0)}  "
                    f"started={M._sim_started}",
                    flush=True,
                )
                last_heartbeat = wall_now
                frames_since_beat = 0

            # ---- pull actions from the browser ----------------------------
            for action in web_server.drain_actions():
                kind = action.get("type")
                if kind == "user_prompt":
                    _handle_user_prompt(
                        str(action.get("text", "")),
                        world_state,
                        blackboard,
                        now=now,
                        timers=timers,
                    )
                    # Both go-to-station and task-request helpers set
                    # world_state["manual_control"] correctly.
                    manual_control = bool(world_state.get("manual_control", False))
                elif kind == "reset":
                    M._cancel_pending_decisions()
                    world_state = M._reset_demo_state(
                        OPEN_FLOOR, speed_multiplier, world_state["connection_status"]
                    )
                    blackboard.clear()
                    nemoclaw.activate(blackboard=blackboard, world_state=world_state)
                    M._bootstrap_demo_run(world_state, blackboard)
                    manual_control = False
                    sim_elapsed = 0.0
                    timers["last_leader_tick"] = -M.LEADER_TICK_INTERVAL
                    timers["last_worker_tick"] = -M.WORKER_TICK_INTERVAL
                    timers["last_strategist_tick"] = 0.0
                    timers["last_move_tick"] = -M.MOVE_TICK_INTERVAL
                elif kind == "speedup":
                    speed_multiplier = M._cycle_speed(speed_multiplier)
                    world_state["speed_multiplier"] = speed_multiplier
                elif kind == "disconnect":
                    _handle_disconnect(
                        world_state, blackboard,
                        initial_use_local=initial_use_local,
                        initial_use_mock=initial_use_mock,
                    )
                elif kind == "wall_add":
                    cell = action.get("cell")
                    if isinstance(cell, list) and len(cell) == 2:
                        wall_list = world_state.get("wall", [])
                        if cell not in wall_list:
                            wall_list.append(cell)
                            world_state["wall"] = wall_list
                            world_state["layout"] = "CUSTOM"
                elif kind == "wall_remove":
                    cell = action.get("cell")
                    if isinstance(cell, list) and len(cell) == 2:
                        wall_list = list(world_state.get("wall", []))
                        if cell in wall_list:
                            wall_list.remove(cell)
                            world_state["wall"] = wall_list
                # Unknown kinds are ignored — nothing to do.

            # ---- always drain in-flight LLM decisions ---------------------
            M._drain_completed_decisions(world_state, blackboard)

            # ---- only run the simulation once started ---------------------
            if M._sim_started:
                if not manual_control:
                    if now - timers["last_leader_tick"] >= M.LEADER_TICK_INTERVAL:
                        for leader in [r for r in world_state["robots"] if r["role"] == LEADER]:
                            if leader["id"] in M._pending_leader:
                                continue
                            M._pending_leader[leader["id"]] = M._decision_executor.submit(
                                leader_decide, leader, world_state, blackboard,
                            )
                        timers["last_leader_tick"] = now

                    if now - timers["last_worker_tick"] >= M.WORKER_TICK_INTERVAL:
                        if M._ALL_AGENTS_LLM:
                            for worker in [r for r in world_state["robots"] if r["role"] == WORKER]:
                                if worker["id"] in M._pending_worker:
                                    continue
                                if worker.get("current_task") is not None:
                                    continue
                                M._pending_worker[worker["id"]] = M._decision_executor.submit(
                                    worker_decide, worker, world_state, blackboard,
                                )
                        else:
                            M._assign_idle_workers(world_state, blackboard)
                        timers["last_worker_tick"] = now

                    if (
                        now - timers["last_strategist_tick"] >= M.STRATEGIST_TICK_INTERVAL
                        and M._pending_strategist is None
                    ):
                        try:
                            retrieved = MEM.retrieve_strategies(
                                world_state["layout"],
                                state_description=MEM.describe_world_state(world_state),
                            )
                        except Exception as exc:
                            print(f"[web] strategy retrieval failed: {exc}",
                                  file=sys.stderr, flush=True)
                            retrieved = []
                        M._pending_strategist = M._decision_executor.submit(
                            strategist_decide, world_state, blackboard, retrieved,
                        )
                        timers["last_strategist_tick"] = now

                # Autonomous fleet dispatcher — drains queued tasks one at a
                # time onto whichever worker is closest and free. Each
                # delegation is gated by NemoClaw (see _fleet_dispatch_step).
                if world_state.get("task_queue"):
                    M._fleet_dispatch_step(world_state, blackboard)

                if now - timers["last_move_tick"] >= M.MOVE_TICK_INTERVAL:
                    move_steps = min(int((now - timers["last_move_tick"]) // M.MOVE_TICK_INTERVAL), 4)
                    for _ in range(max(1, move_steps)):
                        M._advance_robots_coordinated(world_state, blackboard)
                    timers["last_move_tick"] += max(1, move_steps) * M.MOVE_TICK_INTERVAL
                    if world_state.get("task_queue"):
                        M._fleet_dispatch_step(world_state, blackboard)

                world_state["stats"]["elapsed"] = round(sim_elapsed, 1)
                completed = world_state["stats"]["completed"]
                world_state["stats"]["rate"] = (
                    round(completed / sim_elapsed * 60, 1) if sim_elapsed > 0 else 0.0
                )

            # ---- always sync metadata + publish ---------------------------
            world_state["blackboard"] = blackboard.to_list()
            world_state["speed_multiplier"] = speed_multiplier
            world_state["tick"] += 1

            if wall_now - last_publish >= publish_dt:
                info = _build_info(manual_control)
                web_server.update_state(world_state, info)
                last_publish = wall_now

            if max_runtime and (
                sim_elapsed >= max_runtime
                or wall_now - run_start >= max_runtime
            ):
                break

            sleep_for = frame_dt - (time.time() - wall_now)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[web] interrupted, shutting down", flush=True)
    finally:
        M._cancel_pending_decisions()
        M._decision_executor.shutdown(wait=False, cancel_futures=True)


def _build_info(manual_control: bool) -> dict:
    """Best-effort diagnostic block for the browser HUD."""
    info: dict = {
        "started": M._sim_started,
        "manual_control": manual_control,
        "mock": os.getenv("AGENTS_USE_MOCK", "false").lower() == "true",
    }
    try:
        from factorymind import inference  # local import: respects reloads
        info["endpoints"] = inference.describe_endpoints()
        if hasattr(inference, "circuit_status"):
            info["circuit"] = inference.circuit_status()
    except Exception as exc:
        info["endpoints"] = f"(unavailable: {exc})"

    engine = nemoclaw.get_engine()
    if engine is not None:
        info["nemoclaw"] = engine.stats_snapshot()
        info["nemoclaw"]["describe"] = engine.describe()
    return info


if __name__ == "__main__":
    run()
