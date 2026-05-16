"""Headless web entry point for the autonomous FactoryMind 3D demo."""

from __future__ import annotations

import copy
import os
import time

from dotenv import load_dotenv

load_dotenv()

from factorymind import main as M
from factorymind import web_server
from factorymind.agents import Blackboard


def run() -> None:
    """Run the autonomous simulation and serve the browser 3D viewer."""
    inference_target = M._print_startup_banner()
    speed_multiplier = 1
    world_state = M._reset_demo_state(speed_multiplier, "online", inference_target)
    blackboard = Blackboard()

    initial_use_local = os.getenv("USE_LOCAL_NIM", "true").lower() == "true"
    world_state["llm_ready"] = M._local_nim_available() if initial_use_local else True
    if not world_state["llm_ready"]:
        M._thought(
            world_state,
            M.S.WARNING,
            "Leader",
            "Local NIM is not reachable; deterministic policy planner is active.",
            0.0,
            blackboard,
        )
    M._thought(
        world_state,
        M.S.OBSERVE,
        "Leader",
        "FactoryMind web runtime ready. Press START to begin autonomous dispatch.",
        0.0,
        blackboard,
    )

    host = os.getenv("FACTORYMIND_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("FACTORYMIND_WEB_PORT", "8080"))
    target_fps = float(os.getenv("FACTORYMIND_WEB_FPS", "60"))
    publish_dt = float(os.getenv("FACTORYMIND_WEB_PUBLISH_DT", "0.08"))
    max_runtime = float(os.getenv("FACTORYMIND_MAX_RUNTIME", "0") or 0)
    heartbeat_seconds = float(os.getenv("FACTORYMIND_HEARTBEAT_SECONDS", "10"))

    url = web_server.serve_in_background(host=host, port=port)
    print("=" * 72, flush=True)
    print(f"[web] FactoryMind 3D viewer ready -> {url}", flush=True)
    if host in ("0.0.0.0", "::", ""):
        print("[web] Use this machine's LAN IP to open it from another device.", flush=True)
    print("=" * 72, flush=True)

    web_server.update_state(world_state, _build_info(world_state))

    sim_elapsed = 0.0
    run_started_wall = time.time()
    last_frame = run_started_wall
    last_publish = 0.0
    last_heartbeat = run_started_wall
    frames_since_beat = 0
    last_move_tick = -M.MOVE_TICK_INTERVAL
    last_factory_tick = -M.FACTORY_TICK_INTERVAL
    last_leader_tick = -M.LEADER_TICK_INTERVAL
    frame_dt = 1.0 / max(target_fps, 1.0)

    try:
        while True:
            wall_now = time.time()
            real_delta = wall_now - last_frame
            last_frame = wall_now
            if world_state.get("running"):
                sim_elapsed += real_delta * speed_multiplier
            now = sim_elapsed

            frames_since_beat += 1
            if heartbeat_seconds > 0 and wall_now - last_heartbeat >= heartbeat_seconds:
                fps = frames_since_beat / max(wall_now - last_heartbeat, 1e-6)
                print(
                    f"[web] alive fps={fps:.0f} sim_t={sim_elapsed:.1f}s "
                    f"running={world_state.get('running')} "
                    f"completed={world_state['stats'].get('completed', 0)}",
                    flush=True,
                )
                last_heartbeat = wall_now
                frames_since_beat = 0

            for action in web_server.drain_actions():
                result = _handle_web_action(
                    action,
                    world_state,
                    blackboard,
                    now,
                    speed_multiplier,
                    initial_use_local,
                    inference_target,
                )
                if result.get("reset_state") is not None:
                    world_state = result["reset_state"]
                    blackboard = result["blackboard"]
                    inference_target = result["inference_target"]
                    sim_elapsed = 0.0
                    now = 0.0
                    last_move_tick = -M.MOVE_TICK_INTERVAL
                    last_factory_tick = -M.FACTORY_TICK_INTERVAL
                    last_leader_tick = -M.LEADER_TICK_INTERVAL
                speed_multiplier = result.get("speed_multiplier", speed_multiplier)
                inference_target = result.get("inference_target", inference_target)

            if M._pending_leader is not None and M._pending_leader.done():
                future = M._pending_leader
                M._pending_leader = None
                try:
                    plan = future.result()
                except Exception as exc:
                    plan = {
                        "assignments": M._deterministic_assignments(copy.deepcopy(world_state)),
                        "thought": "",
                        "warning": f"Leader planner failed; deterministic policy planner used ({exc}).",
                    }
                M._apply_leader_plan(world_state, plan, blackboard, now)

            if world_state.get("running"):
                if now - last_factory_tick >= M.FACTORY_TICK_INTERVAL:
                    steps = max(1, min(int((now - last_factory_tick) // M.FACTORY_TICK_INTERVAL), 4))
                    for _ in range(steps):
                        M._update_factories(world_state, M.FACTORY_TICK_INTERVAL, now, blackboard)
                    last_factory_tick += steps * M.FACTORY_TICK_INTERVAL

                if now - last_leader_tick >= M.LEADER_TICK_INTERVAL and M._pending_leader is None:
                    snapshot = copy.deepcopy(world_state)
                    world_state["leader_thinking"] = True
                    world_state["leader"]["thinking"] = True
                    M._pending_leader = M._decision_executor.submit(M._leader_decide_autonomous, snapshot)
                    last_leader_tick = now

                if now - last_move_tick >= M.MOVE_TICK_INTERVAL:
                    steps = max(1, min(int((now - last_move_tick) // M.MOVE_TICK_INTERVAL), 4))
                    for _ in range(steps):
                        M._move_workers(world_state, now, blackboard)
                    last_move_tick += steps * M.MOVE_TICK_INTERVAL

            M._update_stats(world_state, now)
            world_state["blackboard"] = blackboard.to_list()
            world_state["speed_multiplier"] = speed_multiplier
            world_state["tick"] += 1

            if wall_now - last_publish >= publish_dt:
                web_server.update_state(world_state, _build_info(world_state))
                last_publish = wall_now

            if max_runtime and (now >= max_runtime or wall_now - run_started_wall >= max_runtime):
                break

            sleep_for = frame_dt - (time.time() - wall_now)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[web] interrupted, shutting down", flush=True)
    finally:
        M._cancel_pending_decisions()
        M._decision_executor.shutdown(wait=False, cancel_futures=True)


def _handle_web_action(
    action: dict,
    world_state: dict,
    blackboard: Blackboard,
    now: float,
    speed_multiplier: int,
    initial_use_local: bool,
    inference_target: str,
) -> dict:
    kind = action.get("type")
    result = {
        "speed_multiplier": speed_multiplier,
        "inference_target": inference_target,
        "reset_state": None,
        "blackboard": blackboard,
    }

    if kind == "start":
        world_state["running"] = True
        M._thought(world_state, M.S.OBSERVE, "Leader", "Autonomous loop started.", now, blackboard)
    elif kind == "stop":
        world_state["running"] = False
        M._cancel_pending_decisions()
        world_state["leader_thinking"] = False
        world_state["leader"]["thinking"] = False
        M._thought(world_state, M.S.RETHINK, "Leader", "Simulation paused in place.", now, blackboard)
    elif kind == "reset":
        M._cancel_pending_decisions()
        new_state = M._reset_demo_state(speed_multiplier, world_state["connection_status"], inference_target)
        new_state["llm_ready"] = M._local_nim_available() if initial_use_local else True
        new_blackboard = Blackboard()
        M._thought(
            new_state,
            M.S.OBSERVE,
            "Leader",
            "State reset to 1 leader, 3 workers, 3 factories, 3 drop boxes.",
            0.0,
            new_blackboard,
        )
        result["reset_state"] = new_state
        result["blackboard"] = new_blackboard
    elif kind == "speed_toggle":
        speed_multiplier = M._cycle_speed(speed_multiplier)
        world_state["speed_multiplier"] = speed_multiplier
        result["speed_multiplier"] = speed_multiplier
    elif kind == "disconnect":
        M._set_inference_mode(True)
        inference_target = M._describe_inference_target()
        world_state["inference_target"] = inference_target
        world_state["connection_status"] = "offline"
        world_state["llm_ready"] = M._local_nim_available()
        M._thought(world_state, M.S.WARNING, "Leader", "Cloud disconnected; GX10 local policy path active.", now, blackboard)
        result["inference_target"] = inference_target
    elif kind == "reconnect":
        M._set_inference_mode(initial_use_local)
        inference_target = M._describe_inference_target()
        world_state["inference_target"] = inference_target
        world_state["connection_status"] = "online"
        world_state["llm_ready"] = M._local_nim_available() if initial_use_local else True
        M._thought(world_state, M.S.OBSERVE, "Leader", "Cloud/demo inference mode restored.", now, blackboard)
        result["inference_target"] = inference_target
    elif kind in {"mode_cursor", "mode_builder"}:
        world_state["ui_mode"] = "builder" if kind == "mode_builder" else "cursor"
    elif isinstance(kind, str) and kind.startswith("builder_"):
        world_state["builder_mode"] = kind.replace("builder_", "")
    elif kind in {"wall_add", "wall_remove", "place_factory", "place_dropbox", "place_worker"}:
        M._handle_builder_action(world_state, action)

    return result


def _build_info(world_state: dict) -> dict:
    info = {
        "started": bool(world_state.get("running")),
        "policy_status": world_state.get("policy_status", "unknown"),
        "policy_path": world_state.get("policy_path", ""),
    }
    try:
        from factorymind import inference
        info["endpoints"] = inference.describe_endpoints()
    except Exception as exc:
        info["endpoints"] = f"Inference target unavailable: {exc}"
    return info


if __name__ == "__main__":
    run()
