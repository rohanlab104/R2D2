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
            M._drain_worker_llm_decisions(world_state, blackboard, now)

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
    elif kind in {"wall_add", "wall_remove", "place_factory", "place_dropbox", "place_worker", "remove_asset"}:
        M._handle_builder_action(world_state, action)
    elif kind == "apply_layout":
        new_state = _state_from_uploaded_layout(
            action,
            speed_multiplier,
            world_state["connection_status"],
            inference_target,
            initial_use_local,
        )
        new_blackboard = Blackboard()
        M._thought(
            new_state,
            M.S.OBSERVE,
            "Leader",
            f"Uploaded layout applied: {new_state.get('layout_name', 'Digital Twin')}.",
            0.0,
            new_blackboard,
        )
        result["reset_state"] = new_state
        result["blackboard"] = new_blackboard
    elif kind == "set_worker_count":
        _set_worker_count(world_state, blackboard, action.get("worker_count"), now)
    elif kind == "set_order_volume":
        _set_order_volume(world_state, blackboard, action.get("order_volume"), now)
    elif kind == "order_spike":
        _trigger_order_spike(world_state, blackboard, now)
    elif kind == "disable_worker":
        _disable_worker(world_state, blackboard, action.get("worker_id"), now)
    elif kind == "enable_worker":
        _enable_worker(world_state, blackboard, action.get("worker_id"), now)

    return result


def _state_from_uploaded_layout(
    action: dict,
    speed_multiplier: int,
    connection_status: str,
    inference_target: str,
    initial_use_local: bool,
) -> dict:
    parsed = action.get("layout") if isinstance(action.get("layout"), dict) else {}
    grid = parsed.get("grid") if isinstance(parsed.get("grid"), list) else []
    order_volume = _coerce_int(action.get("order_volume"), 1, 20, 6)
    worker_count = _coerce_int(action.get("worker_count"), 1, 24, 3)
    world_state = M._reset_demo_state(speed_multiplier, connection_status, inference_target)
    world_state["layout"] = "UPLOADED_DIGITAL_TWIN"
    world_state["layout_id"] = parsed.get("id", "uploaded-layout")
    world_state["layout_name"] = parsed.get("name", "Uploaded Layout")
    world_state["uploaded_layout"] = parsed
    world_state["order_volume"] = order_volume
    world_state["llm_ready"] = M._local_nim_available() if initial_use_local else True

    layout = parsed.get("layout", {}) if isinstance(parsed.get("layout"), dict) else {}
    blocked = _blocked_cells_from_grid(grid)
    blocked.extend(_component_cells((layout.get("shelves") or []) + (layout.get("obstacles") or [])))
    world_state["wall"] = _unique_cells(blocked)
    world_state["factories"] = []
    world_state["dropboxes"] = []
    world_state["workers"] = []
    world_state["packages"] = []
    world_state["tasks"] = []
    world_state["blackboard"] = []
    world_state["thought_log"] = []
    world_state["policy_log"] = world_state.get("policy_log", [])
    world_state["stats"] = {
        "completed": 0,
        "elapsed": 0.0,
        "rate": 0.0,
        "last_route_time": None,
        "avg_route_time": None,
    }

    colors = [item["name"] for item in M.S.PACKAGE_COLORS]
    conveyors = layout.get("conveyors") or []
    bins = layout.get("bins") or []
    chargers = layout.get("chargers") or []
    worker_spawns = layout.get("workerSpawns") or []

    conveyor_refs = []
    for index, conveyor in enumerate(conveyors):
        color = _package_color(conveyor, colors[index % len(colors)])
        pos = _clamp_factory_pos([int(conveyor.get("x", 5)), int(conveyor.get("y", 5))])
        factory = M.S.make_factory(f"Intake-{index + 1}", pos, color, index)
        factory["produce_every"] = _produce_every_for_volume(order_volume)
        factory["produce_timer"] = 0.6 + index * 0.45
        world_state["factories"].append(factory)
        conveyor_refs.append({"color": color, "center": _object_center(conveyor)})
        _clear_asset_cells(world_state["wall"], pos, 8, 3)

    unused_conveyor_refs = list(conveyor_refs)
    for index, bin_obj in enumerate(bins):
        fallback = colors[index % max(1, min(len(colors), len(conveyors) or len(colors)))]
        color = _explicit_package_color(bin_obj)
        if color is None:
            color = _nearest_conveyor_color(bin_obj, unused_conveyor_refs or conveyor_refs, fallback)
            if unused_conveyor_refs:
                used_ref = _nearest_conveyor_ref(bin_obj, unused_conveyor_refs)
                if used_ref:
                    unused_conveyor_refs.remove(used_ref)
        pos = _clamp_dropbox_pos([int(bin_obj.get("x", 40)), int(bin_obj.get("y", 8))])
        box = M.S.make_dropbox(f"DropBox-{index + 1}-{color}", pos, color, index)
        world_state["dropboxes"].append(box)
        _clear_asset_cells(world_state["wall"], pos, 3, 2)

    if not world_state["factories"]:
        factory = M.S.make_factory("Intake-1", [5, 5], colors[0], 0)
        factory["produce_every"] = _produce_every_for_volume(order_volume)
        world_state["factories"].append(factory)
    if not world_state["dropboxes"]:
        world_state["dropboxes"].append(M.S.make_dropbox("DropBox-1-Red", [40, 8], colors[0], 0))

    spawn_cells = _component_cells(worker_spawns) or [[M.GRID_WIDTH // 2, M.GRID_HEIGHT - 7]]
    workers = []
    used: set[tuple[int, int]] = set()
    for index in range(worker_count):
        preferred = spawn_cells[index % len(spawn_cells)]
        pos = _nearest_open_cell(world_state["wall"], preferred, used)
        used.add(tuple(pos))
        workers.append(M.S.make_worker(index + 1, pos))

    leader_pos = _nearest_open_cell(world_state["wall"], [M.GRID_WIDTH // 2, M.GRID_HEIGHT - 3], used)
    leader = {
        "id": 0,
        "name": "Leader",
        "role": M.S.LEADER,
        "pos": leader_pos,
        "spawn_pos": list(leader_pos),
        "path": [],
        "current_task": None,
        "status": "observing",
        "thinking": False,
    }
    world_state["leader"] = leader
    world_state["workers"] = workers
    world_state["robots"] = [leader] + workers
    world_state["workstations"] = copy.deepcopy(world_state["factories"])
    world_state["chargers"] = chargers
    world_state["next_package_id"] = 0
    world_state["next_task_id"] = 0
    world_state["next_factory_id"] = len(world_state["factories"])
    world_state["next_dropbox_id"] = len(world_state["dropboxes"])
    world_state["next_worker_id"] = len(workers) + 1
    return world_state


def _set_worker_count(world_state: dict, blackboard: Blackboard, count, now: float) -> None:
    target = _coerce_int(count, 1, 24, len(world_state.get("workers", [])) or 3)
    current = list(world_state.get("workers", []))
    while len(current) > target:
        worker = current.pop()
        _release_worker_assignment(world_state, worker)
    used = {tuple(worker.get("pos", [0, 0])) for worker in current}
    while len(current) < target:
        worker_id = max([worker.get("id", 0) for worker in current] + [0]) + 1
        pos = _nearest_open_cell(world_state.get("wall", []), [M.GRID_WIDTH // 2, M.GRID_HEIGHT - 7], used)
        used.add(tuple(pos))
        current.append(M.S.make_worker(worker_id, pos))
    world_state["workers"] = current
    world_state["robots"] = [world_state["leader"]] + current
    world_state["next_worker_id"] = max([worker.get("id", 0) for worker in current] + [0]) + 1
    M._thought(world_state, M.S.OBSERVE, "Leader", f"Worker count set to {target}.", now, blackboard)


def _set_order_volume(world_state: dict, blackboard: Blackboard, volume, now: float) -> None:
    value = _coerce_int(volume, 1, 20, 6)
    world_state["order_volume"] = value
    produce_every = _produce_every_for_volume(value)
    for factory in world_state.get("factories", []):
        factory["produce_every"] = produce_every
    M._thought(world_state, M.S.RETHINK, "Leader", f"Order volume set to {value}; intake interval {produce_every:.1f}s.", now, blackboard)


def _trigger_order_spike(world_state: dict, blackboard: Blackboard, now: float) -> None:
    """Demo control: push an immediate backlog onto every factory pad."""
    count = 0
    for factory in world_state.get("factories", []):
        for _ in range(3):
            package = M._make_package(world_state, factory)
            package["status"] = "pad"
            package["progress"] = float(factory.get("belt_length", 5))
            factory.setdefault("pad_packages", []).append(package)
            count += 1
    M._thought(world_state, M.S.WARNING, "Leader", f"Order spike injected {count} packages across intake pads.", now, blackboard)


def _disable_worker(world_state: dict, blackboard: Blackboard, worker_id, now: float) -> None:
    """Demo control: fault a worker and release any active assignment."""
    try:
        worker_id = int(worker_id)
    except (TypeError, ValueError):
        return
    worker = next((w for w in world_state.get("workers", []) if w.get("id") == worker_id), None)
    if not worker:
        return
    _release_worker_assignment(world_state, worker)
    worker["status"] = "disabled"
    worker["path"] = []
    M._thought(world_state, M.S.WARNING, worker.get("name", f"Worker-{worker_id}"), "Worker removed from fleet; tasks redistributed.", now, blackboard)


def _enable_worker(world_state: dict, blackboard: Blackboard, worker_id, now: float) -> None:
    try:
        worker_id = int(worker_id)
    except (TypeError, ValueError):
        return
    worker = next((w for w in world_state.get("workers", []) if w.get("id") == worker_id), None)
    if not worker or worker.get("status") != "disabled":
        return
    worker["status"] = "idle"
    worker["current_task"] = None
    worker["path"] = []
    M._thought(world_state, M.S.OBSERVE, worker.get("name", f"Worker-{worker_id}"), "Worker returned to active fleet.", now, blackboard)


def _release_worker_assignment(world_state: dict, worker: dict) -> None:
    task = next((t for t in world_state.get("tasks", []) if t.get("id") == worker.get("current_task")), None)
    if task and task.get("status") not in {"done", "blocked"}:
        task["status"] = "blocked"
    carrying = worker.get("carrying")
    if carrying:
        factory = next((f for f in world_state.get("factories", []) if f.get("id") == carrying.get("factory_id")), None)
        if factory:
            carrying["status"] = "pad"
            carrying["reserved_by"] = None
            factory.setdefault("pad_packages", []).append(carrying)
    for factory in world_state.get("factories", []):
        for package in factory.get("pad_packages", []):
            if package.get("reserved_by") == worker.get("id"):
                package["reserved_by"] = None
    worker["carrying"] = None
    worker["current_task"] = None
    worker["target_factory_id"] = None
    worker["target_dropbox_id"] = None
    worker["task_started_at"] = None


def _blocked_cells_from_grid(grid: list) -> list[list[int]]:
    blocked = []
    for y, row in enumerate(grid[:M.GRID_HEIGHT]):
        if not isinstance(row, list):
            continue
        for x, tile in enumerate(row[:M.GRID_WIDTH]):
            if tile in {"wall", "shelf", "restricted"}:
                blocked.append([x, y])
    return blocked


def _component_cells(components: list) -> list[list[int]]:
    cells = []
    for component in components:
        for cell in component.get("cells", []) if isinstance(component, dict) else []:
            if isinstance(cell, list) and len(cell) == 2:
                cells.append([int(cell[0]), int(cell[1])])
    return cells


def _unique_cells(cells: list[list[int]]) -> list[list[int]]:
    seen = set()
    unique = []
    for cell in cells:
        key = tuple(cell)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cell)
    return unique


def _clear_asset_cells(walls: list[list[int]], pos: list[int], width: int, height: int) -> None:
    x0, y0 = pos
    occupied = {
        (x, y)
        for x in range(x0, min(M.GRID_WIDTH, x0 + width))
        for y in range(y0, min(M.GRID_HEIGHT, y0 + height))
    }
    walls[:] = [cell for cell in walls if tuple(cell) not in occupied]


def _nearest_open_cell(walls: list[list[int]], preferred: list[int], used: set[tuple[int, int]] | None = None) -> list[int]:
    wall_set = {tuple(cell) for cell in walls}
    used = used or set()
    px = max(0, min(M.GRID_WIDTH - 1, int(preferred[0])))
    py = max(0, min(M.GRID_HEIGHT - 1, int(preferred[1])))
    for radius in range(max(M.GRID_WIDTH, M.GRID_HEIGHT)):
        for x in range(px - radius, px + radius + 1):
            for y in range(py - radius, py + radius + 1):
                if x < 0 or y < 0 or x >= M.GRID_WIDTH or y >= M.GRID_HEIGHT:
                    continue
                if (x, y) in wall_set or (x, y) in used:
                    continue
                return [x, y]
    return [px, py]


def _clamp_factory_pos(pos: list[int]) -> list[int]:
    return [
        max(0, min(M.GRID_WIDTH - 9, int(pos[0]))),
        max(0, min(M.GRID_HEIGHT - 3, int(pos[1]))),
    ]


def _clamp_dropbox_pos(pos: list[int]) -> list[int]:
    return [
        max(0, min(M.GRID_WIDTH - 3, int(pos[0]))),
        max(0, min(M.GRID_HEIGHT - 2, int(pos[1]))),
    ]


def _coerce_int(value, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


def _produce_every_for_volume(order_volume: int) -> float:
    return max(0.8, round(8.0 / max(1, order_volume), 2))


def _package_color(obj: dict, fallback: str) -> str:
    return _explicit_package_color(obj) or fallback


def _explicit_package_color(obj: dict) -> str | None:
    candidate = obj.get("packageColor") or obj.get("acceptsType")
    valid = {item["name"] for item in M.S.PACKAGE_COLORS}
    return str(candidate) if candidate in valid else None


def _object_center(obj: dict) -> tuple[float, float]:
    cells = obj.get("cells") if isinstance(obj, dict) else None
    if isinstance(cells, list) and cells:
        xs = [float(cell[0]) for cell in cells if isinstance(cell, list) and len(cell) == 2]
        ys = [float(cell[1]) for cell in cells if isinstance(cell, list) and len(cell) == 2]
        if xs and ys:
            return (sum(xs) / len(xs), sum(ys) / len(ys))
    x = float(obj.get("x", 0))
    y = float(obj.get("y", 0))
    return (x + float(obj.get("width", 1)) / 2.0, y + float(obj.get("depth", 1)) / 2.0)


def _nearest_conveyor_color(bin_obj: dict, conveyor_refs: list[dict], fallback: str) -> str:
    ref = _nearest_conveyor_ref(bin_obj, conveyor_refs)
    return str(ref["color"]) if ref else fallback


def _nearest_conveyor_ref(bin_obj: dict, conveyor_refs: list[dict]) -> dict | None:
    if not conveyor_refs:
        return None
    bx, by = _object_center(bin_obj)
    return min(
        conveyor_refs,
        key=lambda ref: (bx - ref["center"][0]) ** 2 + (by - ref["center"][1]) ** 2,
    )


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
