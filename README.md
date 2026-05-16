# FactoryMind R2D2

Autonomous master/slave factory simulation powered by NVIDIA Nemotron models on an ASUS Ascent GX10.

The current demo is no longer chat-command driven. Factories continuously produce colored packages, a stationary leader observes the floor and assigns idle workers, and workers move packages from conveyor pads to matching-color drop boxes until the simulation is stopped.

## Setup

```bash
pip3 install -r requirements.txt
cp .env.example .env
# Edit .env: set NVIDIA_API_KEY / NGC_API_KEY as needed.

python3 -m factorymind.web_main
python3 -m pytest tests/
```

Open `http://localhost:8080` after starting the server. Set
`FACTORYMIND_WEB_HOST`, `FACTORYMIND_WEB_PORT`, or `FACTORYMIND_WEB_FPS` to
override the defaults.

## Runtime Model

- 1 stationary leader tower at the bottom center of the floor.
- 3 mobile worker spheres.
- 3 factories with animated conveyors.
- 3 conveyor drop pads where packages accumulate.
- 3 color-matched drop boxes with delivered counters.
- Packages are colored cubes and must be delivered to the same-color drop box.

The leader loop:

1. Observes package counts, worker state, and drop-box locations.
2. Posts worker directives through the OpenClaw/NemoClaw policy path.
3. Assigns each idle worker a pickup pad and matching drop box.
4. Re-evaluates continuously while the simulation is running.

The worker loop:

1. Pathfinds to the assigned conveyor pad.
2. Picks up only from that pad.
3. Carries the colored cube to the matching drop box.
4. Deposits, reports completion, and returns to idle.

## Controls

| Control | Action |
|---|---|
| START | Begin the autonomous factory loop |
| STOP | Pause conveyors, workers, and leader dispatch in place |
| RESET | Restore the initial 1 leader / 3 workers / 3 factories / 3 drop boxes |
| SPEED | Cycle 1x / 2x / 4x simulation speed |
| CURSOR / BUILDER | Toggle inspection vs placement mode |
| WALL / FACTORY / DROPBOX / WORKER | Builder sub-mode |
| DISCONNECT CLOUD | Force the local GX10 resilience-demo path |

Builder mode places objects directly on the 3D simulation floor. Left-click places the active object. Wall mode supports click-drag painting and right-click erase.

## Inference

`factorymind/inference.py` is an OpenAI-compatible HTTP client. GPU execution happens wherever the configured endpoint points.

| Role | Default endpoint | Model |
|---|---|---|
| Leader | `http://localhost:8001/v1` | `nvidia/llama-3_3-nemotron-super-49b-v1_5` |
| Worker | `http://localhost:8000/v1` | `nvidia/nvidia-nemotron-nano-9b-v2` |

Use `USE_LOCAL_NIM=true` on the GX10 or through an SSH tunnel. Use `USE_LOCAL_NIM=false` for NVIDIA cloud.

## NemoClaw Policy

The runtime loads [scripts/nemoclaw_policy.yaml](scripts/nemoclaw_policy.yaml) at startup and writes allow/deny decisions to `logs/nemoclaw.log`.

Policy summary:

- Workers can pick up only from conveyor pads.
- Workers can drop only at matching-color drop boxes.
- Workers cannot leave the simulation area or modify factories.
- The leader can post directives but cannot directly move workers or modify factories.

## Architecture

```text
factorymind/web_main.py  headless web server runtime
factorymind/main.py      autonomous loop helpers, pathfinding, policy logging
factorymind/state.py     canonical world_state schema
factorymind/web/         Three.js browser renderer
factorymind/inference.py OpenAI-compatible Nemotron client
factorymind/agents.py    Blackboard compatibility helpers
```
