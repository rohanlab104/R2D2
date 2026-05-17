# FactoryMind R2D2

**FactoryMind R2D2** is a research and demo stack for orchestrating **general-purpose autonomous robots** in a **factory or warehouse** setting. It combines a real-time **digital twin** simulation, **path planning**, and an **optional AI leader** (NVIDIA **Nemotron** via NIM or cloud) that proposes work assignments—while **NemoClaw-style policies** define what the leader and workers are allowed to do.

The project makes concrete a problem many operations teams face: *how to coordinate many mobile agents safely under one “brain,” with guardrails, without each robot running unconstrained logic.* Here, that brain is explicit, inspectable, and policy-bound.

---

## The problem

Modern warehouses and plants mix **fixed automation** (conveyors, AS/RS) with **mobile platforms** (AMRs, cobot arms on carts, human-assisted fleets). Coordinating them usually requires:

- A **single source of truth** for floor state, orders, and robot status  
- **Safe delegation**: who may assign motion, pick, place, or change equipment  
- **Predictable fallback** when the cloud, GPU host, or model endpoint is slow or down  
- Something you can **demo, replay, and explain** to stakeholders—not only a slide deck  

FactoryMind R2D2 is a **compact simulation** that stresses exactly those ideas: multi-agent dispatch, constraints, and optional LLM reasoning—without requiring a full ROS stack to get started.

---

## What this repository is

| Layer | Role |
|--------|------|
| **Simulation** | Grid world with walls, **factories** (intake), **drop boxes**, colored **packages**, a **leader** role, and **worker** agents. |
| **Planning** | `A*` pathfinding and a **deterministic assignment** planner (nearest idle worker to pad + matching-color drop box) so the system always runs. |
| **Optional AI** | The **leader** can call **Nemotron** (`factorymind/inference.py`) to output JSON assignments; output is **validated** and ignored if illegal. |
| **Governance** | `scripts/nemoclaw_policy.yaml` describes **allow/deny** rules for leader vs worker; the runtime **loads** it and **logs** decisions to `logs/nemoclaw.log`. |
| **Digital twin UI** | **Three.js** viewer served over **stdlib HTTP** (`factorymind/web_server.py`) so you can watch the floor from a browser. |

This is **not** a production WMS or fleet manager. It is a **credible slice** of one: enough to show how policy, models, and visualization fit together on a laptop or on a **DGX Spark / GX10** class box running **local NIM**.

---

## How the demo behaves

When you press **START** in the web UI:

1. **Factories** produce colored packages onto conveyor pads.  
2. The **leader** periodically plans **which idle worker** should serve **which package** and **which matching-color drop box**.  
3. **Workers** pathfind, **pick up only at pads**, **drop only at matching boxes**, and report completion.  
4. If **NIM** is unreachable, the **deterministic planner** keeps dispatch alive; Nemotron is an enhancement, not a single point of failure for the loop.

You can also **build** the layout (walls, factories, drop boxes, workers) from the browser and experiment with connectivity and traffic.

---

## Tech stack

| Area | Technology |
|------|------------|
| **Runtime** | Python 3 |
| **Config** | `python-dotenv`, `.env` (see `.env.example`) |
| **LLM client** | `openai` package → OpenAI-compatible **NIM** or NVIDIA cloud |
| **Models (defaults)** | Leader: `nvidia/llama-3_3-nemotron-super-49b-v1_5` · Worker config: `nvidia/nvidia-nemotron-nano-9b-v2` |
| **Web UI** | Three.js (ES modules), static assets under `factorymind/web/` |
| **HTTP bridge** | `factorymind/web_server.py` (no Flask/FastAPI dependency) |
| **Simulation core** | `factorymind/main.py`, `factorymind/state.py`, pathfinding helpers |
| **Policy** | YAML: `scripts/nemoclaw_policy.yaml` |
| **Optional headless loop** | `factorymind/warehouse_main.py` + `factorymind/warehouse.py` (metrics-oriented backend demo) |
| **Tests** | `pytest` |

---

## Quick start

```bash
pip3 install -r requirements.txt
cp .env.example .env
# Edit .env: API keys for cloud mode; GX10_IP / NIM URLs for local NIM.

python3 -m factorymind.web_main
```

Open **http://localhost:8080** (or your host/port if overridden). Use **START** to run the autonomous loop.

Convenience:

```bash
./scripts/run_web.sh
```

### Running under the policy launcher

To log launcher context and optionally use a **sandbox** (OpenShell / firejail) per `scripts/run_with_nemoclaw.sh`:

```bash
./scripts/run_with_nemoclaw.sh
```

The app still loads `scripts/nemoclaw_policy.yaml` and writes policy-aligned lines to **`logs/nemoclaw.log`**.

### Headless warehouse backend (no 3D UI)

```bash
python3 -m factorymind.warehouse_main
```

Useful for verifying stepping logic and metrics without the browser.

---

## Configuration highlights

| Variable | Meaning |
|----------|---------|
| `USE_LOCAL_NIM` | `true` → talk to **local NIM** on `GX10_IP` (default `localhost`). `false` → **NVIDIA cloud** (needs `NVIDIA_API_KEY`). |
| `GX10_IP` | Host for default `http://{host}:8001/v1` (leader) and `:8000/v1` (worker) in local mode. |
| `NIM_LEADER_BASE_URL` / `NIM_WORKER_BASE_URL` | Override endpoints. |
| `LEADER_MODEL` / `WORKER_MODEL` | Model IDs passed to the API. |
| `FACTORYMIND_WEB_HOST` / `FACTORYMIND_WEB_PORT` | Bind address and port for the viewer. |
| `FACTORYMIND_DISABLE_LLM` | Set to `true` to skip leader LLM calls and rely on deterministic planning only. |

---

## NemoClaw policy (summary)

Full spec: **[scripts/nemoclaw_policy.yaml](scripts/nemoclaw_policy.yaml)**.

- **Leader** may **observe** and **post worker directives**; it may **not** move workers directly, modify factories/drop boxes, or pick/drop packages.  
- **Workers** may **move inside the sim**, **pick from conveyor pads**, **drop at matching-color boxes**, and **report complete**; they may not leave the area, cheat pickups/drops, or edit policy.  
- **Network** and **filesystem** sections document intended sandbox allowances when you use `run_with_nemoclaw.sh`; on a plain host run, enforcement is primarily **application-level** plus logging.

---

## Project layout

```text
factorymind/
  web_main.py       # Default entry: sim loop + HTTP snapshot + drain browser actions
  main.py           # World reset, leader plan, workers, pathfinding, policy hooks
  web_server.py     # Tiny GET/POST server for /state.json and /action
  web/              # index.html, Three.js scene, static JS (layout parser, etc.)
  inference.py      # Nemotron / NIM OpenAI-compatible client
  state.py          # World schema helpers
  agents.py         # In-process Blackboard for messages
  warehouse.py      # Alternate warehouse sim backend
  warehouse_main.py # CLI runner for warehouse backend
  memory.py         # Optional memory / persistence helpers
scripts/
  nemoclaw_policy.yaml
  run_web.sh
  run_with_nemoclaw.sh
  run_nim_*.sh, verify_*.sh   # NIM / tunnel helpers
tests/
  pytest tests for inference, smoke paths, warehouse
```

---

## Testing

```bash
python3 -m pytest tests/
```

---

## Where inference actually runs

`factorymind/inference.py` is only an **HTTP client**. GPUs execute models **wherever the base URL points**: **this machine** (e.g. DGX Spark with NIM on localhost), **another host** on the LAN, or **NVIDIA cloud**. The **Python simulation** itself runs on a laptop or workstation without a GPU.

---

## License / attribution

Add your team’s license and hackathon attribution if required. This README describes the FactoryMind R2D2 demo as shipped in this repository.
