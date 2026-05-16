"""Headless autonomous warehouse simulation runner.

Run:
    python3 -m factorymind.warehouse_main

This is a backend/demo loop only. It prints metrics and recent events so the
logic can be verified before connecting a 3D digital twin UI.
"""

from __future__ import annotations

import json
import os
import time

from factorymind.warehouse import create_warehouse_state, public_state, step_warehouse


def run() -> None:
    tick_ms = int(os.getenv("WAREHOUSE_TICK_MS", "100"))
    max_runtime = float(os.getenv("WAREHOUSE_MAX_RUNTIME", "30"))
    print_every = float(os.getenv("WAREHOUSE_PRINT_SECONDS", "2"))
    state = create_warehouse_state()
    started = time.time()
    last_print = started

    print("[warehouse] autonomous backend running", flush=True)
    try:
        while True:
            step_warehouse(state, tick_ms)
            now = time.time()
            if now - last_print >= print_every:
                snapshot = public_state(state)
                metrics = snapshot["metrics"]
                latest = snapshot["events"][-3:]
                print(
                    "[warehouse] "
                    f"t={snapshot['simTimeMs'] / 1000:.1f}s "
                    f"completed={metrics['completedTasks']} "
                    f"pending={metrics['pendingTasks']} "
                    f"active_workers={metrics['activeWorkers']} "
                    f"throughput={metrics['throughput']}/min",
                    flush=True,
                )
                print(json.dumps(latest, indent=2), flush=True)
                last_print = now
            if max_runtime and now - started >= max_runtime:
                break
            time.sleep(max(tick_ms / 1000.0, 0.01))
    except KeyboardInterrupt:
        print("\n[warehouse] interrupted", flush=True)


if __name__ == "__main__":
    run()
