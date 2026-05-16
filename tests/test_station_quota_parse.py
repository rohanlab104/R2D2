"""Tests for per-station task quota parsing."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["USE_LLM_TASK_INTERPRETER"] = "false"

from factorymind.state import create_initial_state, OPEN_FLOOR
from factorymind import main as M


def test_multi_station_for_clauses():
    ws = create_initial_state(OPEN_FLOOR, num_workers=12)
    text = "do 5 tasks for parts, and 2 for assembly"
    requests = M._parse_user_task_requests_regex(text, ws)
    assert len(requests) == 2
    assert requests[0][0]["name"] == "Parts"
    assert requests[0][1]["name"] == "Assembly"
    assert requests[0][2] == 5
    assert requests[1][0]["name"] == "Assembly"
    assert requests[1][1]["name"] == "QA"
    assert requests[1][2] == 2


def test_apply_sets_station_targets():
    ws = create_initial_state(OPEN_FLOOR)
    from factorymind.agents import Blackboard

    bb = Blackboard()
    ok = M._apply_user_task_request("do 3 tasks for QA", ws, bb)
    assert ok
    assert ws["station_quotas"]["QA"]["target"] == 3
    assert ws["station_quotas"]["QA"]["completed"] == 0
