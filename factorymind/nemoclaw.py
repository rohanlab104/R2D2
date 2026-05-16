"""NemoClaw / OpenClaw runtime — in-process policy engine for the agent fleet.

This module is the "OpenClaw runtime" referenced in the demo. It loads the YAML
policy at ``scripts/nemoclaw_policy.yaml`` and gates every autonomous decision
the leader makes (and every outbound LLM call) against an allow/deny list.

Why it exists
-------------
The hackathon brief asks for the leader to be an *autonomous agent that runs
under NemoClaw*. In production NemoClaw is an OS-level sandbox, but for the
in-simulation demo we want the policy to be visible inside the world: every
``delegate_task``, every ``ask_nemotron`` call, and every external network hit
flows through :class:`PolicyEngine`, which:

* checks the action against the YAML allow/deny rules,
* records an entry in ``logs/nemoclaw.log``,
* posts a ``POLICY`` message to the shared blackboard so the side panel /
  Three.js HUD can render the decision live,
* and (optionally, when ``NEMOCLAW_ENFORCE=true``) actually refuses the call.

The wrapper script ``scripts/run_with_nemoclaw.sh`` still launches an
*outer* sandbox (``nemoclaw`` binary or ``firejail``) for filesystem +
process containment. This module is the *inner* loop — the layer the
autonomous leader is "based on".
"""

from __future__ import annotations

import os
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - import guard
    import yaml
except Exception:  # pragma: no cover - falls back to bare-bones policy
    yaml = None  # type: ignore[assignment]

from . import state as state_mod


# ---------------------------------------------------------------------------
# Defaults used if the YAML file is missing or unreadable. These match the
# committed scripts/nemoclaw_policy.yaml so the runtime still behaves
# sensibly on a fresh checkout.
# ---------------------------------------------------------------------------
_DEFAULT_POLICY: dict[str, Any] = {
    "name": "factorymind-default",
    "network": {
        "allow": [
            "127.0.0.1:8000",
            "127.0.0.1:8001",
            "localhost:8000",
            "localhost:8001",
        ],
        "deny": ["*"],
    },
    "agents": {
        "leader": {
            "role": "autonomous_fleet_manager",
            "autonomous": True,
            "actions": {
                "allow": [
                    "delegate_task",
                    "fleet_dispatch_tick",
                    "post_strategy",
                ],
                "deny": ["exec_shell", "modify_walls"],
            },
        },
        "worker": {
            "role": "task_executor",
            "autonomous": False,
            "actions": {
                "allow": ["execute_task", "move", "post_intent"],
                "deny": ["delegate_task", "exec_shell"],
            },
        },
        "strategist": {
            "role": "advisor",
            "autonomous": False,
            "actions": {
                "allow": ["post_strategy", "raise_bottleneck"],
                "deny": ["delegate_task", "exec_shell"],
            },
        },
    },
    "enforce": False,
    "logging": {"path": "./logs/nemoclaw.log", "events": ["allow", "deny"]},
}


@dataclass
class PolicyDecision:
    """Outcome of a single ``check`` call."""

    allowed: bool
    reason: str
    actor: str
    action: str
    target: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class PolicyDenied(RuntimeError):
    """Raised when ``enforce`` is on and a check returns ``allowed=False``."""

    def __init__(self, decision: PolicyDecision) -> None:
        super().__init__(f"NemoClaw denied {decision.actor}.{decision.action}: {decision.reason}")
        self.decision = decision


class PolicyEngine:
    """In-process policy gate. Thread-safe; one global instance per process."""

    def __init__(self, policy: dict[str, Any], *, log_path: Path | None = None) -> None:
        self._policy = policy
        self._lock = threading.Lock()
        self._blackboard = None  # set via attach_blackboard
        self._world_state = None  # set via attach_world_state
        self._log_path = log_path or Path(self._policy.get("logging", {}).get("path", "./logs/nemoclaw.log"))
        self._enforce = bool(self._policy.get("enforce", False))
        env_enforce = os.getenv("NEMOCLAW_ENFORCE", "").strip().lower()
        if env_enforce in {"1", "true", "yes", "on"}:
            self._enforce = True
        elif env_enforce in {"0", "false", "no", "off"}:
            self._enforce = False
        self._counters = {"allow": 0, "deny": 0}
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # -------------------- attachment --------------------

    def attach_blackboard(self, blackboard) -> None:
        """Bind a Blackboard instance so POLICY messages are visible in-world."""
        self._blackboard = blackboard

    def attach_world_state(self, world_state: dict) -> None:
        """Bind world_state so the engine can update stats counters."""
        self._world_state = world_state

    # -------------------- introspection --------------------

    @property
    def enforce(self) -> bool:
        return self._enforce

    @property
    def policy_name(self) -> str:
        return str(self._policy.get("name", "factorymind"))

    def describe(self) -> str:
        net = self._policy.get("network", {})
        allow = ", ".join(net.get("allow", [])[:4]) or "(none)"
        agents = list(self._policy.get("agents", {}).keys())
        mode = "ENFORCE" if self._enforce else "log-only"
        return (
            f"NemoClaw policy '{self.policy_name}' [{mode}]; "
            f"agents={agents}; net allow={allow}"
        )

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.policy_name,
            "enforce": self._enforce,
            "allow": self._counters["allow"],
            "deny": self._counters["deny"],
        }

    # -------------------- gate calls --------------------

    def check_action(
        self,
        actor: str,
        action: str,
        *,
        target: str = "",
        extra: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Decide whether ``actor`` may perform ``action`` on ``target``."""
        agents = self._policy.get("agents", {})
        agent_cfg = agents.get(actor, {})
        actions_cfg = agent_cfg.get("actions", {}) if isinstance(agent_cfg, dict) else {}
        allow_list = set(actions_cfg.get("allow") or [])
        deny_list = set(actions_cfg.get("deny") or [])

        if action in deny_list:
            decision = PolicyDecision(False, f"action '{action}' explicitly denied for '{actor}'", actor, action, target, extra or {})
        elif allow_list and action not in allow_list:
            decision = PolicyDecision(False, f"action '{action}' not in allow list for '{actor}'", actor, action, target, extra or {})
        else:
            decision = PolicyDecision(True, "ok", actor, action, target, extra or {})
        self._record(decision)
        return decision

    def check_network(self, url: str, *, actor: str = "system") -> PolicyDecision:
        """Decide whether ``url`` is reachable per the YAML network rules."""
        host_port = _normalize_host_port(url)
        net = self._policy.get("network", {})
        allow = set(net.get("allow") or [])
        deny = set(net.get("deny") or [])

        host_only = host_port.split(":", 1)[0]

        def _matches(rule: str, target: str) -> bool:
            return rule == "*" or rule == target or rule == host_only

        if any(_matches(rule, host_port) for rule in allow):
            decision = PolicyDecision(True, f"host '{host_port}' in network allow", actor, "net_connect", host_port)
        elif any(_matches(rule, host_port) for rule in deny):
            decision = PolicyDecision(False, f"host '{host_port}' blocked by deny list", actor, "net_connect", host_port)
        else:
            # Neither explicitly allowed nor denied -> default deny (safer).
            decision = PolicyDecision(False, f"host '{host_port}' not in allow list", actor, "net_connect", host_port)

        self._record(decision)
        return decision

    # -------------------- enforce helper --------------------

    def enforce_decision(self, decision: PolicyDecision) -> None:
        """Raise :class:`PolicyDenied` when the engine is in enforce mode."""
        if not decision.allowed and self._enforce:
            raise PolicyDenied(decision)

    # -------------------- internals --------------------

    def _record(self, decision: PolicyDecision) -> None:
        with self._lock:
            key = "allow" if decision.allowed else "deny"
            self._counters[key] += 1
            try:
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {key.upper():5s} "
                        f"actor={decision.actor} action={decision.action} "
                        f"target={decision.target or '-'} reason={decision.reason}\n"
                    )
            except Exception:
                pass

            if self._world_state is not None:
                stats = self._world_state.setdefault("stats", {})
                stats["policy_allowed"] = stats.get("policy_allowed", 0) + (1 if decision.allowed else 0)
                stats["policy_denied"] = stats.get("policy_denied", 0) + (0 if decision.allowed else 1)

            bb = self._blackboard
        # POSTING happens outside the lock to avoid nested lock contention.
        if bb is not None:
            tag = "ALLOW" if decision.allowed else "DENY"
            target_part = f" -> {decision.target}" if decision.target else ""
            try:
                bb.post(
                    -1,
                    state_mod.POLICY,
                    f"NemoClaw {tag}: {decision.actor}.{decision.action}{target_part} ({decision.reason})",
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Module-level singleton + helpers
# ---------------------------------------------------------------------------

_engine: PolicyEngine | None = None
_engine_lock = threading.Lock()


def _normalize_host_port(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port
    if not host:
        # url may be raw "host:port"
        bare = url.split("/", 1)[0]
        return bare or url
    if port is None:
        if parsed.scheme == "https":
            port = 443
        elif parsed.scheme == "http":
            port = 80
    return f"{host}:{port}" if port is not None else host


def _load_policy_from_disk(policy_path: Path | None = None) -> dict[str, Any]:
    if policy_path is None:
        # Default: scripts/nemoclaw_policy.yaml relative to repo root.
        policy_path = Path(__file__).resolve().parent.parent / "scripts" / "nemoclaw_policy.yaml"

    if not policy_path.exists() or yaml is None:
        return _DEFAULT_POLICY

    try:
        with policy_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return _DEFAULT_POLICY


def activate(blackboard=None, world_state: dict | None = None, *, policy_path: Path | None = None) -> PolicyEngine:
    """Create (or return) the global policy engine and bind it to the world."""
    global _engine
    with _engine_lock:
        if _engine is None:
            policy = _load_policy_from_disk(policy_path)
            _engine = PolicyEngine(policy)
        if blackboard is not None:
            _engine.attach_blackboard(blackboard)
        if world_state is not None:
            _engine.attach_world_state(world_state)
        return _engine


def get_engine() -> PolicyEngine | None:
    """Return the active engine, or ``None`` if NemoClaw was never activated."""
    return _engine


def reset_for_tests() -> None:  # pragma: no cover - test helper
    global _engine
    with _engine_lock:
        _engine = None
