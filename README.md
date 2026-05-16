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

### Local NIM (ASUS Ascent GX10)

```bash
USE_LOCAL_NIM=true AGENTS_USE_MOCK=false python3 -m factorymind.main
```

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
| Leader (fast decisions) | `nvidia/nvidia-nemotron-nano-9b-v2` |
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
