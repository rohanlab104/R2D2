"""Tiny stdlib HTTP/JSON bridge for the Three.js viewer.

The headless simulation publishes snapshots to ``/state.json``. The browser
posts UI actions to ``/action``; callers drain those actions with
``drain_actions``.

Pure stdlib — no Flask/FastAPI dependency added to the project.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_STATE_LOCK = threading.Lock()
_STATE: dict = {"world": {}, "info": {}}
_ACTIONS: list[dict] = []

_STATIC_DIR = Path(__file__).parent / "web"


# ---------------------------------------------------------------------------
# Public API used by the simulation loop
# ---------------------------------------------------------------------------

def update_state(world_state: dict, info: dict | None = None) -> None:
    """Publish a fresh world snapshot for the browser to fetch."""
    with _STATE_LOCK:
        _STATE["world"] = world_state
        if info is not None:
            _STATE["info"] = info


def drain_actions() -> list[dict]:
    """Return and clear any actions submitted by the browser since last call."""
    with _STATE_LOCK:
        out = list(_ACTIONS)
        _ACTIONS.clear()
    return out


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg":  "image/svg+xml",
    ".png":  "image/png",
    ".ico":  "image/x-icon",
}


class _Handler(BaseHTTPRequestHandler):
    server_version = "FactoryMindWeb/1.0"

    def log_message(self, format, *args):  # noqa: A003 - stdlib signature
        # Silence default per-request access log; the sim loop has its own
        # heartbeat. Errors still surface via log_error.
        return

    # ------- GET ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_static("index.html")
            return
        if path == "/state.json":
            self._serve_state()
            return
        if path.startswith("/static/"):
            self._serve_static("static/" + path[len("/static/"):])
            return
        self.send_error(404, "not found")

    def _serve_state(self) -> None:
        with _STATE_LOCK:
            try:
                payload = json.dumps(_STATE, default=_json_safe).encode("utf-8")
            except (TypeError, ValueError) as exc:
                self.send_error(500, f"state serialization failed: {exc}")
                return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_static(self, name: str) -> None:
        # Prevent traversal: resolve relative to STATIC_DIR and check parent.
        target = (_STATIC_DIR / name).resolve()
        try:
            target.relative_to(_STATIC_DIR.resolve())
        except ValueError:
            self.send_error(403, "forbidden")
            return
        if not target.is_file():
            self.send_error(404, "not found")
            return
        try:
            data = target.read_bytes()
        except OSError as exc:
            self.send_error(500, f"read failed: {exc}")
            return
        ext = target.suffix.lower()
        ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ------- POST ---------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        if self.path != "/action":
            self.send_error(404, "not found")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return
        if not isinstance(payload, dict) or "type" not in payload:
            self.send_error(400, "missing 'type' field")
            return
        with _STATE_LOCK:
            _ACTIONS.append(payload)
        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


def _json_safe(obj):
    """Best-effort fallback for non-JSON-native types in world_state."""
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve_in_background(host: str = "0.0.0.0", port: int = 8080) -> str:
    """Start the HTTP server in a daemon thread and return a viewer URL."""
    server = _Server((host, port), _Handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="factorymind-web",
        daemon=True,
    )
    thread.start()
    public_host = "localhost" if host in ("0.0.0.0", "::", "") else host
    return f"http://{public_host}:{port}"
