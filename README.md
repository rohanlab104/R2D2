# FactoryMind R2D2

Multi-agent factory simulation where autonomous robots coordinate through a shared blackboard, powered by NVIDIA Nemotron models running on an ASUS Ascent GX10.

## Team Roles

| Person | File(s) | Responsibility |
|--------|---------|----------------|
| **A**  | `factorymind/inference.py` | Nemotron API integration, model wrangling, `safe_parse_json` |
| **B**  | `factorymind/render.py` | Pygame renderer, colour scheme, side panel, disconnect banner |
| **C**  | `factorymind/agents.py` | Blackboard class, `leader_decide`, `strategist_decide`, prompts |
| **D**  | `factorymind/main.py`, `factorymind/memory.py` | Main loop, A* pathfinding, task scheduling, SQLite memory |

`factorymind/state.py` is shared ground truth — everyone reads it, nobody changes the schema without a team sync.

---

## Setup

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Copy and fill environment variables
cp .env.example .env
# Edit .env — add your NVIDIA_API_KEY

# 3. Seed the strategy memory database
python3 scripts/seed_memory.py

# 4. Run with mock agents (no API key needed)
AGENTS_USE_MOCK=true python3 -m factorymind.main

# 5. Run with real Nemotron
python3 -m factorymind.main

# 6. Run tests
python3 -m pytest tests/
```

### 3D viewer (Three.js, runs on the GX10)

Same simulation as `factorymind.main`, rendered in 3D in a browser instead of pygame:

```bash
./scripts/run_web.sh
# then open http://<gx10-ip>:8080  (or http://localhost:8080 on the GX10 itself)
```

`scripts/run_web.sh` runs `python -m factorymind.web_main`, which serves a static Three.js page on port 8080 and a JSON snapshot of `world_state` at `/state.json`. The browser polls 10 Hz and posts chat prompts / button clicks back to `/action`.

### Autonomous fleet (OpenClaw leader)

Type any of these into chat and the leader becomes an autonomous fleet manager that delegates one task at a time to the closest free worker:

```text
do 100 deliveries
queue 50 tasks
take Parts to QA 25 times
move 80 Parts to Shipping
```

How it works:

1. The chat parser fills `world_state["task_queue"]` with the requested number of routes (random Parts→Assembly→QA→Shipping cycle if no stations were named).
2. Every loop iteration, `_fleet_dispatch_step` in `factorymind/main.py` picks the **closest idle worker** for the next queued task and assigns it. Workers stream out one-by-one as they finish.
3. The leader stays parked at its dispatch position — it does **not** carry tasks itself in fleet mode. It is the brain; workers are the body.
4. Each delegation is gated by **NemoClaw** (see below). The HUD shows the live queue size, the running allow/deny counter, and every dispatch as a `POLICY` line in the agent log.

### NemoClaw / OpenClaw policy runtime

`factorymind/nemoclaw.py` is the in-process policy engine that gates every autonomous decision the leader makes (and every outbound LLM call). The policy lives in `scripts/nemoclaw_policy.yaml`:

* **`network`** — only `127.0.0.1:8000` and `127.0.0.1:8001` (the local NIM endpoints) are allowed; everything else, including `build.nvidia.com`, is on the deny list. Calls are routed through `inference.py`'s `_nemoclaw_gate`, so flipping `NEMOCLAW_ENFORCE=true` makes the simulation fall back to mock agents if the only reachable endpoint is the cloud.
* **`agents`** — names each role (`leader`, `worker`, `strategist`) and the actions it may take. The leader is marked `autonomous: true` with `delegate_task` and `fleet_dispatch_tick` in its allow list; `exec_shell` and `modify_walls` are explicitly denied. Every fleet dispatch calls `engine.check_action("leader", "delegate_task", ...)`, so denials show up live in the agent log.
* **`logging`** — every allow/deny is appended to `logs/nemoclaw.log` with timestamp, actor, action, target, and reason.

Modes:

```bash
NEMOCLAW_ENFORCE=false   # default — log every decision but never refuse
NEMOCLAW_ENFORCE=true    # refuse denied actions; simulation falls back to mock
```

The `scripts/run_with_nemoclaw.sh` wrapper is still there for **outer** sandboxing — if a real `nemoclaw` binary or `firejail` is installed it will exec under that. The in-process engine here is the **inner** loop the leader is "based on".

### Where does inference happen?

The Python in `factorymind/inference.py` is just an OpenAI-compatible HTTP client. The GPU work happens wherever the URL points:

| `USE_LOCAL_NIM` | `GX10_IP` | Endpoint hit | Where the model actually runs |
|---|---|---|---|
| `false` | (ignored) | `https://integrate.api.nvidia.com/v1` | NVIDIA cloud |
| `true` | `localhost` | `http://localhost:8000/v1` and `:8001/v1` | This GX10's GPU (running Docker NIM) |
| `true` | GX10 IP | `http://<gx10>:8000/v1` and `:8001/v1` | The GX10's GPU (from a remote laptop) |

| Port | Model | Role |
|------|-------|------|
| 8000 | `nemotron-nano-9b-v2` | Leaders, workers, chat task interpreter |
| 8001 | `llama-3_3-nemotron-super-49b-v1_5` | Strategist |

Runtime defaults are tuned for the GX10 demo: `USE_LOCAL_NIM=true`,
`ALL_AGENTS_LLM=true`, `USE_LLM_TASK_INTERPRETER=true`,
`COOPERATIVE_PATHING=true`, and `NEMOTRON_TIMEOUT_SECONDS=60`. If any model
override contains `120B`, the client ignores it for now and falls back to the
configured smaller role model.

### Person A — Inference checklist (running on the DGX Spark GX10)

**On the GX10**

```bash
git clone https://github.com/rohanlab104/R2D2.git
cd R2D2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x scripts/*.sh

cp .env.example .env
# Edit .env: set NVIDIA_API_KEY and NGC_API_KEY (same value works).
# USE_LOCAL_NIM=true and GX10_IP=localhost are already the defaults.

nvidia-smi && docker --version    # confirm Blackwell GPU + Docker
```

**Start NIM (two terminals on the GX10)**

```bash
# terminal 1 — leader (9B) on host port 8000
./scripts/run_nim_nano.sh

# terminal 2 — strategist (49B) on host port 8001
./scripts/run_nim_49b.sh
```

First pull is 20–40 min; weights are cached in `~/.cache/nim` for fast restarts.

**Verify (third terminal on the GX10)**

```bash
./scripts/test_nim_curl.sh localhost 8000
./scripts/test_nim_curl.sh localhost 8001 nvidia/llama-3_3-nemotron-super-49b-v1_5
source .venv/bin/activate
./scripts/verify_local.sh
```

**Run the full sim on the GX10**

```bash
python3 scripts/seed_memory.py
python3 -m factorymind.main
```

**Teammate laptops** — open an SSH tunnel and they can use `localhost:8000/8001`:

```bash
export GX10_USER=... GX10_IP=<gx10-ip>
./scripts/gx10_tunnel.sh           # keep open
USE_LOCAL_NIM=true GX10_IP=localhost ./scripts/verify_local.sh
```

**Cloud fallback** (only if NIM isn't ready, e.g. early dev) — set `USE_LOCAL_NIM=false` and run `./scripts/verify_cloud.sh`.

**Demo controls**

```bash
./scripts/disconnect_demo_block.sh   # block cloud egress; local NIM keeps working
./scripts/reset_demo.sh              # undo + stop NIM containers
```

SSH note: `gx10-d8fb` is a short hostname that won't resolve outside the lab's DNS. Use the GX10's **IP** from event organizers.

---

## world_state Schema

`create_initial_state()` in `state.py` is the canonical source. Every module conforms to this structure:

```python
{
    "robots": [
        {
            "id": int,           # unique robot ID
            "role": str,         # "LEADER" | "WORKER"
            "pos": [x, y],       # current grid cell
            "path": [[x,y],...], # remaining A* path
            "current_task": int | None,  # task id being worked
        },
        ...
    ],
    "tasks": [
        {
            "id": int,
            "status": str,          # "open" | "in_transit" | "done"
            "pickup": [x, y],
            "delivery": [x, y],
            "pickup_name": str,
            "delivery_name": str,
            "assigned_to": int | None,  # robot id
        },
        ...
    ],
    "workstations": [
        {"name": str, "pos": [x, y], "color": [r, g, b]},
        ...
    ],
    "blackboard": [
        {"from": int, "type": str, "content": str, "timestamp": float},
        ...
    ],
    "layout": str,          # "OPEN_FLOOR" | "BOTTLENECK_BRIDGE"
    "wall": [[x, y], ...],  # blocked grid cells (BOTTLENECK_BRIDGE only)
    "stats": {
        "completed": int,
        "elapsed": float,   # seconds
        "rate": float,      # tasks per minute
    },
    "connection_status": str,  # "online" | "offline"
    "tick": int,
}
```

---

## Controls (in the Pygame window)

| Control | Action |
|---------|--------|
| **Open** button | Switch to open floor layout |
| **Bottleneck** button | Switch to bottleneck bridge layout |
| **DISCONNECT** button | Toggle cloud/local mode indicator |
| Window close | Quit |

---

## Models

| Role | Model |
|------|-------|
| Leader / Worker / Chat task interpreter | `nvidia/nvidia-nemotron-nano-9b-v2` |
| Strategist (high-level) | `nvidia/llama-3_3-nemotron-super-49b-v1_5` |

---

## Architecture

```
main.py  ──► state.py   (world_state dict)
         ──► agents.py  (Blackboard, leader_decide, strategist_decide)
         ──► render.py  (Pygame frame)
         ──► memory.py  (SQLite strategy retrieval)

agents.py ──► inference.py  (Nemotron API / local NIM)
```
