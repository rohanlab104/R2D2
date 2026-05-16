"""Pre-populate the SQLite memory database with hand-crafted factory strategies.

Run this once before starting the simulation:
    python scripts/seed_memory.py
"""

from __future__ import annotations

import sys
import os

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factorymind.memory import init_db
from factorymind.state import OPEN_FLOOR, BOTTLENECK_BRIDGE

SEEDS: list[tuple[str, str, str]] = [
    # (layout, strategy, outcome)

    # --- OPEN_FLOOR strategies ---
    (OPEN_FLOOR,
     "Assign one dedicated leader per workstation to minimise cross-floor travel.",
     "Throughput increased 18% vs round-robin; leaders rarely idle."),

    (OPEN_FLOOR,
     "Batch Parts pickups: collect 3 units before moving to Assembly.",
     "Reduced empty trips by 30% but introduced short starvation at QA."),

    (OPEN_FLOOR,
     "Use a priority queue: QA tasks first, then Assembly, then Parts.",
     "QA bottleneck cleared within 2 minutes; overall rate improved 12%."),

    (OPEN_FLOOR,
     "Deploy all workers in a sweep pattern: Parts→Assembly→QA→Shipping in order.",
     "Predictable throughput but poor adaptability to dynamic task arrival."),

    (OPEN_FLOOR,
     "Reserve one worker as a 'floater' who takes whichever task has waited longest.",
     "Eliminated task starvation; median wait time dropped by 40%."),

    (OPEN_FLOOR,
     "Leaders broadcast INTENT before moving; workers avoid occupied routes.",
     "Collision rate near zero; minor overhead from extra blackboard messages."),

    (OPEN_FLOOR,
     "Cluster workers near the geometric centre of all open tasks.",
     "Fast response to burst arrivals; works best when tasks spread evenly."),

    (OPEN_FLOOR,
     "Let the strategist re-issue directives every 15 s instead of 30 s.",
     "More responsive to state changes but increased LLM token usage by 2x."),

    (OPEN_FLOOR,
     "When queue depth > 5, pull all idle workers regardless of role assignment.",
     "Clears backlogs quickly; leaders must re-establish order afterwards."),

    (OPEN_FLOOR,
     "Pre-position workers at predicted high-demand workstations during ramp-up.",
     "First-minute throughput doubled; requires accurate demand forecasting."),

    # --- BOTTLENECK_BRIDGE strategies ---
    (BOTTLENECK_BRIDGE,
     "Single-file transit through the bridge; enforce strict FIFO queuing.",
     "Deadlock-free; throughput limited by bridge bandwidth (~2 robots/s)."),

    (BOTTLENECK_BRIDGE,
     "Assign one leader to each side of the bridge to regulate flow.",
     "Throughput improved 25% over uncoordinated crossing; fewer collisions."),

    (BOTTLENECK_BRIDGE,
     "Alternate direction: north-to-south for 10 s, then south-to-north for 10 s.",
     "High utilisation of bridge; slight delay for robots waiting to switch direction."),

    (BOTTLENECK_BRIDGE,
     "Maintain a crossing reservation: robots post INTENT before entering bridge.",
     "Zero deadlocks observed; 8% overhead from reservation messages."),

    (BOTTLENECK_BRIDGE,
     "Keep QA and Shipping workers south of the wall; Parts and Assembly workers north.",
     "Eliminated cross-bridge travel for 70% of tasks; requires careful task routing."),

    (BOTTLENECK_BRIDGE,
     "Pair each north-side worker with a south-side worker; they swap tasks at bridge.",
     "Experimental; halved crossing frequency but increased handoff complexity."),

    (BOTTLENECK_BRIDGE,
     "Use the strategist to predict when bridge will be congested and pre-route workers.",
     "Reduced average crossing wait from 4 s to 1.5 s in simulation."),

    (BOTTLENECK_BRIDGE,
     "Limit bridge to 2 robots simultaneously; third robot waits at entry cell.",
     "Safe conservative strategy; throughput 15% below theoretical maximum."),

    (BOTTLENECK_BRIDGE,
     "Post a BOTTLENECK message when bridge queue length > 3; trigger reroute.",
     "Effective early-warning system; leaders redistributed load proactively."),

    (BOTTLENECK_BRIDGE,
     "During high-load periods, promote one worker to temporary leader at bridge.",
     "Dynamic promotion improved crossing rate by 20%; requires careful demotion logic."),

    (BOTTLENECK_BRIDGE,
     "Track each robot's side of the wall; only allow crossing if destination is opposite.",
     "Prevents unnecessary crossings; requires accurate task-side tagging."),

    (BOTTLENECK_BRIDGE,
     "When all tasks are on one side, park bridge-idle robots south to avoid blocking.",
     "Freed bridge for productive crossings; robots re-deploy quickly when needed."),
]


def main() -> None:
    """Insert all seed strategies and report how many were added."""
    conn = init_db()
    # Avoid duplicating seeds if run multiple times
    existing = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
    if existing > 0:
        print(f"DB already has {existing} strategies — clearing and re-seeding.")
        conn.execute("DELETE FROM strategies")
        conn.commit()

    conn.executemany(
        "INSERT INTO strategies (layout, strategy, outcome) VALUES (?, ?, ?)",
        SEEDS,
    )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
    conn.close()
    print(f"Seeded {total} strategies into factorymind_memory.db")


if __name__ == "__main__":
    main()
