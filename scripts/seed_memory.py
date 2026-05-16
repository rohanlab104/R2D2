"""Pre-populate the SQLite memory database with factory coordination strategies.

Run this once before starting the simulation:
    python scripts/seed_memory.py
"""

from __future__ import annotations

import os
import sys

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factorymind.memory import build_embedding, init_db
from factorymind.state import BOTTLENECK_BRIDGE, OPEN_FLOOR

SEEDS: list[tuple[str, str, str]] = [
    # (layout, strategy, outcome)

    # --- OPEN_FLOOR strategies ---
    (OPEN_FLOOR,
     "Assign one dedicated leader per workstation zone and keep workers inside the closest two-zone radius.",
     "Average pickup travel fell 18%; leader idle time stayed below 9% during a 12-minute run."),

    (OPEN_FLOOR,
     "Batch Parts pickups in groups of three before sending robots toward Assembly.",
     "Empty travel fell 30%, but QA starvation appeared after 4 minutes when Assembly output surged."),

    (OPEN_FLOOR,
     "Use a priority queue that clears QA-bound tasks first, then Assembly, then Shipping.",
     "QA backlog cleared within 2 minutes; total completion rate improved from 8.5 to 9.6 tasks/min."),

    (OPEN_FLOOR,
     "Run a clockwise sweep route: Parts to Assembly to QA to Shipping, then return through center.",
     "Throughput was predictable at 7.8 tasks/min, but response time doubled for bursty Shipping work."),

    (OPEN_FLOOR,
     "Reserve one worker as a floater assigned to the oldest open task regardless of workstation.",
     "No task waited longer than 21 seconds; median wait time dropped 40%."),

    (OPEN_FLOOR,
     "Have leaders broadcast INTENT before moving and let workers avoid claimed pickup lanes.",
     "Observed route conflicts fell from 11 per minute to 2 per minute with minor blackboard overhead."),

    (OPEN_FLOOR,
     "Cluster idle workers near the geometric center of all open pickups every 20 seconds.",
     "Burst response improved 22% when tasks were evenly distributed across all stations."),

    (OPEN_FLOOR,
     "Re-issue leader directives every 15 seconds during startup, then slow to 30 seconds after rate stabilizes.",
     "Startup throughput reached target 38 seconds faster while token usage stayed manageable."),

    (OPEN_FLOOR,
     "When open queue depth exceeds five, pull every idle worker to the nearest pickup before leaders claim new work.",
     "Backlogs cleared 26% faster; leader task ownership needed cleanup after the surge."),

    (OPEN_FLOOR,
     "Pre-position two workers between Parts and Assembly during the first minute of a run.",
     "First-minute completions doubled from 4 to 8 when Parts demand was high."),

    (OPEN_FLOOR,
     "Assign fast-turn Shipping tasks to the nearest worker even when a leader has already claimed them.",
     "Shipping dwell time fell from 17 seconds to 9 seconds with no measurable loss in QA throughput."),

    (OPEN_FLOOR,
     "Keep one leader parked near center to broker duplicate claims instead of carrying tasks.",
     "Duplicate claims dropped 72%; total throughput was flat because one leader stopped doing transport."),

    (OPEN_FLOOR,
     "When three or more tasks share a pickup station, send two workers there and hold additional robots at center.",
     "Pickup congestion dropped 35%; station approach lanes stayed clear during high Parts demand."),

    (OPEN_FLOOR,
     "Route Assembly-to-QA traffic along the east edge while Parts-to-Assembly traffic uses the north lane.",
     "Head-on route conflicts fell 48% in open floor trials with crossed demand."),

    (OPEN_FLOOR,
     "Prioritize tasks whose delivery station currently has no robot within eight cells.",
     "Station starvation events fell from 6 to 1 per 10-minute run."),

    (OPEN_FLOOR,
     "Use nearest-worker dispatch for short routes under 20 cells and leader dispatch for longer cross-floor routes.",
     "Mean completion time improved 14%; long-haul work was less likely to strand workers."),

    (OPEN_FLOOR,
     "Hold one worker near QA whenever more than two Assembly-to-QA tasks are active.",
     "QA queue depth stayed under three items for 93% of the run."),

    (OPEN_FLOOR,
     "Pause new Parts pickups for 10 seconds if Assembly has four active inbound deliveries.",
     "Assembly congestion fell 41%; total throughput dipped only 3%."),

    (OPEN_FLOOR,
     "Assign tasks by earliest deadline first, using task age as a stand-in for delivery deadline.",
     "Oldest-task wait time fell from 46 seconds to 19 seconds."),

    (OPEN_FLOOR,
     "Move idle workers to a two-point standby pattern near Parts and Shipping instead of a single center cluster.",
     "Average first pickup response fell 16% when demand alternated between upstream and outbound work."),

    (OPEN_FLOOR,
     "Let leaders claim only when at least two tasks are open, leaving single tasks for the nearest worker.",
     "Leader over-travel fell 23%; worker utilization rose from 71% to 79%."),

    (OPEN_FLOOR,
     "Post COMPLETE messages for every delivery and let the leader react when a station receives three completions.",
     "Directive quality improved in review; strategy changes aligned with real station output spikes."),

    (OPEN_FLOOR,
     "Split the map into northwest and southeast dispatch zones and only break zones for tasks older than 30 seconds.",
     "Cross-map travel fell 21%; no starvation was observed in a 15-minute trial."),

    (OPEN_FLOOR,
     "If two robots are heading to the same pickup, keep the nearer robot and reroute the farther one to the oldest open task.",
     "Duplicate pickup arrivals dropped from 14 to 3 per run."),

    (OPEN_FLOOR,
     "Use a rate target of 8 tasks/min; if rate is below target, lower batching thresholds and dispatch immediately.",
     "Recovery from slow starts improved by 33 seconds on average."),

    (OPEN_FLOOR,
     "Keep workers with active paths moving and avoid reassigning them unless the new pickup is at least 12 cells closer.",
     "Mid-route churn fell 64%; completion time became more stable."),

    (OPEN_FLOOR,
     "Prioritize Shipping deliveries during the final 20 seconds of a demo window.",
     "Visible completed count increased by 2 to 3 tasks in short judge walkthroughs."),

    # --- BOTTLENECK_BRIDGE strategies ---
    (BOTTLENECK_BRIDGE,
     "Enforce single-file transit through the bridge using first-in, first-out crossing order.",
     "Deadlocks were eliminated; bridge throughput stabilized near 2 robots/sec."),

    (BOTTLENECK_BRIDGE,
     "Assign one leader north of the wall and one leader south of the wall to regulate bridge flow.",
     "Throughput improved 25% over uncoordinated crossing and bridge conflicts fell sharply."),

    (BOTTLENECK_BRIDGE,
     "Alternate bridge direction: north-to-south for 10 seconds, then south-to-north for 10 seconds.",
     "Bridge utilization stayed high, but opposite-side tasks waited up to 8 seconds during direction holds."),

    (BOTTLENECK_BRIDGE,
     "Require robots to post INTENT before entering the bridge and wait if a crossing intent is already active.",
     "Zero deadlocks observed; blackboard traffic rose 8%."),

    (BOTTLENECK_BRIDGE,
     "Keep QA and Shipping workers south of the wall while Parts and Assembly workers stay north when possible.",
     "Unnecessary bridge crossings fell 70%; task routing needed side-aware assignment."),

    (BOTTLENECK_BRIDGE,
     "Pair one north-side worker with one south-side worker and hand off tasks at the bridge mouth.",
     "Crossing frequency halved, but handoff delay added 5 seconds to complex routes."),

    (BOTTLENECK_BRIDGE,
     "Use the leader to forecast bridge congestion and pre-route workers before queues exceed three robots.",
     "Average crossing wait fell from 4.0 seconds to 1.5 seconds."),

    (BOTTLENECK_BRIDGE,
     "Limit the bridge to two robots simultaneously and hold the third robot at the nearest entry cell.",
     "Safe conservative behavior; throughput was 15% below theoretical bridge capacity."),

    (BOTTLENECK_BRIDGE,
     "Post a BOTTLENECK message when bridge queue length exceeds three and pause new cross-wall assignments.",
     "Early warning let leaders redistribute same-side work within one leader cycle."),

    (BOTTLENECK_BRIDGE,
     "Promote one idle worker to temporary bridge marshal during high-load periods.",
     "Crossing rate improved 20%; demotion needed to happen once the queue cleared."),

    (BOTTLENECK_BRIDGE,
     "Track each robot side of wall and only allow crossing when pickup and delivery are on opposite sides.",
     "Unproductive crossings were nearly eliminated in low-load runs."),

    (BOTTLENECK_BRIDGE,
     "When all active tasks are on one side, park bridge-idle robots away from the bridge entry.",
     "The bridge stayed open for productive crossings and redeployment took under 6 seconds."),

    (BOTTLENECK_BRIDGE,
     "Use a north staging cell at x=24,y=24 and a south staging cell at x=24,y=26 for bridge queues.",
     "Queue shape became predictable and robots stopped blocking lateral traffic near the wall."),

    (BOTTLENECK_BRIDGE,
     "Do not send leaders through the bridge unless no worker on the destination side is idle.",
     "Leader availability improved; cross-wall leader travel fell 58%."),

    (BOTTLENECK_BRIDGE,
     "If three cross-wall tasks are open, batch them in one direction before releasing opposite-direction traffic.",
     "Bridge direction switches fell 45%; average wait rose slightly for lone opposite-side tasks."),

    (BOTTLENECK_BRIDGE,
     "Reserve bridge access for deliveries already in transit before allowing new pickup trips to cross.",
     "Completed-task rate improved 11% because payload-carrying robots cleared first."),

    (BOTTLENECK_BRIDGE,
     "Keep one worker parked on each side within six cells of the bridge during active cross-wall demand.",
     "Handoff response improved 29%; idle time rose during all-same-side task bursts."),

    (BOTTLENECK_BRIDGE,
     "Use same-side substitutions: if a task pickup is north and an idle north worker exists, block south workers from claiming it.",
     "Bridge entries dropped 36% without reducing completion rate."),

    (BOTTLENECK_BRIDGE,
     "When bridge wait exceeds 5 seconds, reroute idle robots to the oldest same-side task even if it is lower priority.",
     "Visible idling near the bridge fell 52%; oldest cross-wall tasks still completed within 35 seconds."),

    (BOTTLENECK_BRIDGE,
     "Create a 3-cell no-parking zone around both bridge mouths unless a robot is actively crossing.",
     "Entry blockage incidents fell from 9 to 1 per run."),

    (BOTTLENECK_BRIDGE,
     "Give QA deliveries bridge priority over Parts pickups during congestion.",
     "QA starvation disappeared; upstream Parts queue grew by one item on average."),

    (BOTTLENECK_BRIDGE,
     "Use a token rule: only the robot holding the bridge token may enter the gap, and it releases token at exit.",
     "Deadlocks remained zero across 20 simulated runs."),

    (BOTTLENECK_BRIDGE,
     "If two robots approach opposite bridge mouths, the robot carrying an in-transit task keeps priority.",
     "Payload delivery delay fell 24%; empty pickup trips waited longer but did not block flow."),

    (BOTTLENECK_BRIDGE,
     "Pre-stage Shipping-bound robots south of the wall before the final 30 seconds of a demo.",
     "Short-demo completed count improved by 2 tasks on average."),

    (BOTTLENECK_BRIDGE,
     "Use one-way lane windows only when at least four cross-wall tasks are active; otherwise use nearest-first crossing.",
     "Avoided unnecessary waits during light demand while preserving stability during surges."),
]


def main() -> None:
    """Insert all seed strategies and report how many were added."""
    conn = init_db()
    existing = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
    if existing > 0:
        print(f"DB already has {existing} strategies; clearing and re-seeding.")
        conn.execute("DELETE FROM strategies")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'strategies'")
        conn.commit()

    conn.executemany(
        """
        INSERT INTO strategies (layout, strategy, outcome, embedding)
        VALUES (?, ?, ?, ?)
        """,
        [
            (layout, strategy, outcome, build_embedding(strategy, outcome))
            for layout, strategy, outcome in SEEDS
        ],
    )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
    conn.close()
    print(f"Seeded {total} strategies into factorymind_memory.db")


if __name__ == "__main__":
    main()
