"""Persistent memory layer backed by SQLite.

Stores factory strategies and their outcomes so the strategist can retrieve
relevant past knowledge at decision time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_DEFAULT_DB = "factorymind_memory.db"


def init_db(path: str = _DEFAULT_DB) -> sqlite3.Connection:
    """Create (or open) the SQLite database and ensure the strategies table exists.

    Returns an open connection. Callers are responsible for closing it.
    """
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            layout   TEXT    NOT NULL,
            strategy TEXT    NOT NULL,
            outcome  TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def retrieve_strategies(layout: str, n: int = 3, path: str = _DEFAULT_DB) -> list[dict]:
    """Return the n most recently recorded strategies for the given layout.

    Each dict has keys "strategy" and "outcome".
    Returns an empty list if the database does not exist or the table is empty.
    """
    if not Path(path).exists():
        return []
    conn = init_db(path)
    try:
        cur = conn.execute(
            "SELECT strategy, outcome FROM strategies WHERE layout = ? ORDER BY id DESC LIMIT ?",
            (layout.upper(), n),
        )
        rows = cur.fetchall()
        return [{"strategy": row[0], "outcome": row[1]} for row in rows]
    finally:
        conn.close()


def record_strategy(layout: str, strategy: str, outcome: str, path: str = _DEFAULT_DB) -> None:
    """Persist a new strategy and its observed outcome for future retrieval."""
    conn = init_db(path)
    try:
        conn.execute(
            "INSERT INTO strategies (layout, strategy, outcome) VALUES (?, ?, ?)",
            (layout.upper(), strategy, outcome),
        )
        conn.commit()
    finally:
        conn.close()
