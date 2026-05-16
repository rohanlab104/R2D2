"""Unit tests for inference helpers (no API calls)."""

from factorymind.inference import safe_parse_json


def test_safe_parse_json_strips_fences() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert safe_parse_json(raw) == {"a": 1}


def test_safe_parse_json_extracts_object() -> None:
    raw = 'Here is the plan: {"claim_task_id": 3, "ok": true}'
    parsed = safe_parse_json(raw)
    assert parsed == {"claim_task_id": 3, "ok": True}


def test_safe_parse_json_invalid_returns_none() -> None:
    assert safe_parse_json("not json at all") is None
