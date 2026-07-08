"""fascia-guard integration for the router (defense-in-depth over privacy_mod).

Two hook points in submit_task:
  * task-in    — scan the raw task text; adds prompt-INJECTION detection (which the
                 privacy classifier does not do) plus an independent secret/PII pass.
  * output     — scan a worker/model artifact before it is persisted + summarized.

Design constraints for landing in a mature, gated codebase:
  * Graceful — if fascia-guard is not installed, every function here is a no-op.
  * Dormant by default — scanning runs only when FASCIA_GUARD (or _ENFORCE) is set,
    so existing routing behavior and the test gate are untouched out of the box.
  * Enforcement is opt-in — FASCIA_GUARD_ENFORCE=1 lets the guard actually block a
    secret-bearing task and redact a secret-bearing artifact. Without it the guard
    is advisory (logged only). This keeps privacy_mod authoritative unless an
    operator deliberately turns guard enforcement on.
"""
from __future__ import annotations

import os

try:  # soft dependency
    from fascia_guard import Decision
    from fascia_guard.integrations.agentconnect import guard_artifact, guard_task_input
    _AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when the package is absent
    _AVAILABLE = False
    Decision = None  # type: ignore


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


def available() -> bool:
    return _AVAILABLE


def active() -> bool:
    """Scanning runs at all only when explicitly switched on."""
    return _AVAILABLE and (_flag("FASCIA_GUARD") or _flag("FASCIA_GUARD_ENFORCE"))


def enforcing() -> bool:
    """Enforcement (block / redact) requires the stronger flag."""
    return _AVAILABLE and _flag("FASCIA_GUARD_ENFORCE")


def scan_task(text: str, task_id: str):
    """Return a fascia-guard Verdict for task-in, or None if inactive."""
    if not active():
        return None
    verdict, _action = guard_task_input(text, task_id=task_id)
    return verdict


def scan_output(text: str, task_id: str):
    """Return a fascia-guard Verdict for an output artifact, or None if inactive."""
    if not active():
        return None
    verdict, _action = guard_artifact(text, task_id=task_id)
    return verdict


def describe(verdict, side: str = "task") -> str:
    cats = sorted({f.category.value for f in verdict.findings})
    return (f"fascia-guard[{side}] decision={verdict.decision.name} "
            f"findings={len(verdict.findings)} categories={cats or '-'} "
            f"engines={verdict.engines_run}"
            + (f" errors={len(verdict.errors)}" if verdict.errors else ""))


def is_block(verdict) -> bool:
    return _AVAILABLE and verdict is not None and verdict.decision is Decision.BLOCK


def should_redact_output(verdict) -> bool:
    return (_AVAILABLE and verdict is not None
            and verdict.decision in (Decision.BLOCK, Decision.REDACT))
