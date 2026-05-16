"""Persistent memory layer backed by SQLite.

Stores factory strategies and their outcomes so the leader can retrieve
relevant past knowledge at dispatch time.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

_DEFAULT_DB = "factorymind_memory.db"

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "if",
    "in", "into", "is", "it", "of", "on", "or", "so", "the", "then", "to",
    "use", "when", "with",
}


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
            outcome  TEXT    NOT NULL,
            embedding TEXT
        )
    """)
    _ensure_embedding_column(conn)
    _backfill_embeddings(conn)
    conn.commit()
    return conn


def retrieve_strategies(
    layout: str,
    n: int = 3,
    path: str = _DEFAULT_DB,
    state_description: str | None = None,
) -> list[dict]:
    """Return the n most relevant recorded strategies for the given layout.

    Each dict has keys "strategy", "outcome", and "score".
    Returns an empty list if the database does not exist or the table is empty.
    """
    if n <= 0:
        return []
    if not Path(path).exists():
        return []
    conn = init_db(path)
    try:
        cur = conn.execute(
            """
            SELECT id, strategy, outcome, embedding
            FROM strategies
            WHERE layout = ?
            ORDER BY id DESC
            """,
            (layout.upper(),),
        )
        rows = cur.fetchall()
        query = state_description or layout
        query_vec = _vectorize(f"{layout} {query}")
        doc_entries = [
            (
                row_id,
                strategy,
                outcome,
                _decode_embedding(embedding) or _vectorize(
                    _strategy_document(strategy, outcome)
                ),
            )
            for row_id, strategy, outcome, embedding in rows
        ]
        idf = _idf_weights([entry[3] for entry in doc_entries])
        weighted_query = _apply_idf(query_vec, idf)
        scored = []
        for row_id, strategy, outcome, doc_vec in doc_entries:
            score = _cosine(weighted_query, _apply_idf(doc_vec, idf))
            result = {
                "strategy": strategy,
                "outcome": outcome,
                "score": round(score, 4),
            }
            # Stable recency tiebreaker so equally relevant rows do not look random.
            scored.append((score, row_id, result))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [result for _, _, result in scored[:n]]
    finally:
        conn.close()


def record_strategy(layout: str, strategy: str, outcome: str, path: str = _DEFAULT_DB) -> None:
    """Persist a new strategy and its observed outcome for future retrieval."""
    conn = init_db(path)
    try:
        conn.execute(
            """
            INSERT INTO strategies (layout, strategy, outcome, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (layout.upper(), strategy, outcome, build_embedding(strategy, outcome)),
        )
        conn.commit()
    finally:
        conn.close()


def describe_world_state(world_state: dict) -> str:
    """Create a compact text query for strategy retrieval from live simulation state."""
    tasks = world_state.get("tasks", [])
    open_tasks = [t for t in tasks if t.get("status") == "open"]
    active_tasks = [t for t in tasks if t.get("status") != "done"]
    in_transit = [t for t in tasks if t.get("status") == "in_transit"]
    idle = [
        r for r in world_state.get("robots", [])
        if r.get("current_task") is None and not r.get("path")
    ]
    routes = Counter(
        f"{t.get('pickup_name', '?')} to {t.get('delivery_name', '?')}"
        for t in active_tasks
    )
    route_text = ", ".join(
        f"{count} {route}" for route, count in routes.most_common(4)
    ) or "no active routes"
    stats = world_state.get("stats", {})
    return (
        f"layout {world_state.get('layout')}; "
        f"{len(open_tasks)} open tasks; {len(in_transit)} in transit; "
        f"{len(active_tasks)} active tasks; {len(idle)} idle robots; "
        f"completed {stats.get('completed', 0)}; "
        f"rate {stats.get('rate', 0)} per minute; routes {route_text}"
    )


def build_embedding(strategy: str, outcome: str) -> str:
    """Return a JSON term-frequency vector for a strategy memory row."""
    vector = _vectorize(_strategy_document(strategy, outcome))
    return json.dumps(dict(sorted(vector.items())), separators=(",", ":"))


def _ensure_embedding_column(conn: sqlite3.Connection) -> None:
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(strategies)").fetchall()
    }
    if "embedding" not in columns:
        conn.execute("ALTER TABLE strategies ADD COLUMN embedding TEXT")


def _backfill_embeddings(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, strategy, outcome
        FROM strategies
        WHERE embedding IS NULL OR embedding = ''
        """
    ).fetchall()
    for row_id, strategy, outcome in rows:
        conn.execute(
            "UPDATE strategies SET embedding = ? WHERE id = ?",
            (build_embedding(strategy, outcome), row_id),
        )


def _strategy_document(strategy: str, outcome: str) -> str:
    return f"{strategy} {outcome}"


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOP_WORDS
    ]


def _vectorize(text: str) -> Counter:
    return Counter(_tokenize(text))


def _decode_embedding(text: str | None) -> Counter:
    if not text:
        return Counter()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Counter()
    if not isinstance(data, dict):
        return Counter()
    vector = Counter()
    for key, value in data.items():
        try:
            vector[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return vector


def _idf_weights(doc_vectors: list[Counter]) -> dict[str, float]:
    if not doc_vectors:
        return {}
    total_docs = len(doc_vectors)
    document_frequency = Counter()
    for vector in doc_vectors:
        document_frequency.update(vector.keys())
    return {
        token: math.log((1 + total_docs) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }


def _apply_idf(vector: Counter, idf: dict[str, float]) -> Counter:
    if not idf:
        return vector
    return Counter({
        token: value * idf.get(token, 1.0)
        for token, value in vector.items()
    })


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in shared)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
