"""Subtask lifecycle and sandbox admission (spec §9.8, §21).

A subtask is the bounded unit a manager hands to a worker. It carries its own
privacy tier and its own sandbox demand; both are checked *before* a worker is
selected, never inside the worker.
"""

from __future__ import annotations

from .errors import Conflict
from .models import SandboxSpec, Subtask, SubtaskStatus

TRANSITIONS: dict[SubtaskStatus, frozenset[SubtaskStatus]] = {
    SubtaskStatus.queued: frozenset(
        {SubtaskStatus.running, SubtaskStatus.needs_approval, SubtaskStatus.failed,
         SubtaskStatus.cancelled}
    ),
    SubtaskStatus.needs_approval: frozenset(
        {SubtaskStatus.running, SubtaskStatus.queued, SubtaskStatus.failed,
         SubtaskStatus.cancelled}
    ),
    SubtaskStatus.running: frozenset(
        {SubtaskStatus.succeeded, SubtaskStatus.failed, SubtaskStatus.cancelled}
    ),
    SubtaskStatus.succeeded: frozenset(),
    SubtaskStatus.failed: frozenset(),
    SubtaskStatus.cancelled: frozenset(),
}

TERMINAL = frozenset({SubtaskStatus.succeeded, SubtaskStatus.failed, SubtaskStatus.cancelled})


def check_transition(subtask: Subtask, to: SubtaskStatus) -> None:
    if to not in TRANSITIONS[subtask.status]:
        raise Conflict(
            f"subtask {subtask.id} cannot move {subtask.status.value} -> {to.value}"
        )


def sandbox_satisfied(needed: SandboxSpec, offered: SandboxSpec) -> bool:
    return needed.satisfied_by(offered)


def describe_sandbox(spec: SandboxSpec) -> str:
    bits = [f"filesystem={spec.filesystem.value}"]
    if spec.network:
        bits.append("network")
    if spec.shell:
        bits.append("shell")
    return ", ".join(bits)
